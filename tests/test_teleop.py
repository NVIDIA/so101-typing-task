"""Leader/follower recording and launcher routing without serial access."""

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from so101_typing.hardware import teleop_scene_record as recorder


# Follower recording

@dataclass
class Calibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def calibration():
    return {name: dict(id=index, drive_mode=0, homing_offset=0,
                       range_min=1024 if name in recorder.ARM else 1000,
                       range_max=3072 if name in recorder.ARM else 3000)
            for index, name in enumerate(recorder.JOINT_NAMES, 1)}


class Clock:
    def __init__(self):
        self.now = 10.

    def monotonic(self):
        self.now += .00001
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Bus:
    def __init__(self, *, torque=0):
        self.is_connected = False
        self.connected = 0
        self.closed = []
        self.writes = []
        self.reads = 0
        self.motors = {name: SimpleNamespace(id=index, model='sts3215',
                       norm_mode='degrees' if name in recorder.ARM else 'range_0_100')
                       for index, name in enumerate(recorder.JOINT_NAMES, 1)}
        self.calibration = calibration()
        self.raw = {name: 2048 if name in recorder.ARM else 1000 for name in recorder.JOINT_NAMES}
        self.registers = {name: dict.fromkeys(recorder.JOINT_NAMES, 0) for name in recorder.REGISTERS}
        self.registers['Torque_Enable'] = dict.fromkeys(recorder.JOINT_NAMES, torque)
        self.registers['P_Coefficient'] = dict.fromkeys(recorder.JOINT_NAMES, 16)
        self.registers['D_Coefficient'] = dict.fromkeys(recorder.JOINT_NAMES, 32)
        self.on_position_read = lambda bus: None
        self.on_read = lambda register, bus: None

    def connect(self):
        self.is_connected = True
        self.connected += 1

    def read_calibration(self):
        return {name: Calibration(**row) for name, row in self.calibration.items()}

    def sync_read(self, register, names, *, normalize, num_retry):
        assert normalize is False and num_retry == 0 and self.is_connected
        self.on_read(register, self)
        if register == 'Present_Position':
            self.reads += 1
            self.on_position_read(self)
            return {name: self.raw[name] for name in names}
        return {name: self.registers[register][name] for name in names}

    def sync_write(self, register, values, *, normalize, num_retry):
        assert normalize is False and num_retry == 0 and self.is_connected
        # This assertion makes any extra register/jaw write a test failure.
        assert register in ('Goal_Position', 'Torque_Enable')
        assert set(values) == set(recorder.ARM)
        assert all(type(value) is int and 0 <= value <= 4095 for value in values.values())
        self.writes.append((register, deepcopy(values)))
        self.registers[register].update(values)

    def disconnect(self, *, disable_torque):
        self.closed.append(disable_torque)
        self.is_connected = False


class Keyboard:
    def __init__(self, clock, stop_after=8):
        self.clock = clock
        self.polls = 0
        self.stop_after = stop_after
        self.preflight_polled = False

    def poll(self):
        if not self.preflight_polled:
            self.preflight_polled = True
            return [], False
        self.polls += 1
        edges = []
        if self.polls in (2, 3):
            edges = [dict(char='q', key_name='KEY_Q', edge='down' if self.polls == 2 else 'up',
                          received_monotonic_s=self.clock.monotonic())]
        return edges, self.polls >= self.stop_after


@pytest.fixture
def fixture():
    follower, leader, clock = Bus(), Bus(), Clock()
    report = {'port': '/dev/follower', 'calibration': calibration(),
              'leader': {'port': '/dev/leader', 'calibration': calibration()}}
    keyboard = Keyboard(clock)
    return follower, leader, report, keyboard, clock


def run(fixture, *, execute=True, **kwargs):
    follower, leader, report, keyboard, clock = fixture
    return recorder.run_session(follower, leader, report, keyboard, execute=execute,
                                seconds=1., clock=clock, **kwargs)


def test_measured_follower_is_replay_source_and_goals_are_separate(fixture):
    follower, leader, report, _, _ = fixture
    # After aligned startup the leader moves; the follower deliberately stays put.
    leader.on_position_read = lambda bus: bus.raw.update(
        shoulder_pan=2048 if bus.reads == 1 else 2088, gripper=2000)
    run(fixture)
    assert report['status'] == 'recorded' and report['stop_reason'] == 'esc'
    assert len(report['samples']) == 8 and len(report['events']) == 2
    for row in report['samples']:
        assert row['raw_counts']['shoulder_pan'] == 2048
        assert row['lerobot_position'] == [0.]*6
        np.testing.assert_allclose(row['q_model_rad'], np.deg2rad([0, 0, 0, 0, 0, -10]))
        assert row['leader']['lerobot_position'][0] > 0
        assert row['acquisition_monotonic_start_s'] <= row['monotonic_s'] <= row['acquisition_monotonic_end_s']
    assert any(row['goal_arm_deg'][0] > 0 for row in report['samples'])
    assert any(row['tracking_error_deg'][0] != 0 for row in report['samples'])
    assert report['samples'][-1]['goal_sent'] is False
    assert not leader.writes
    assert follower.closed == leader.closed == [False]
    assert follower.registers['Torque_Enable']['gripper'] == 0
    assert follower.registers['Goal_Position']['gripper'] == 0
    assert report['final_hold_goal_raw_counts'] == dict.fromkeys(recorder.ARM, 2048)
    # Direct mirroring may change by the leader's full per-cycle displacement.
    goals = [values for register, values in follower.writes if register == 'Goal_Position'][:-1]
    for before, after in zip(goals, goals[1:]):
        assert max(abs(before[name]-after[name]) for name in recorder.ARM) <= 4095
    assert all(a['time_s'] < b['time_s'] for a, b in zip(report['samples'], report['samples'][1:]))


@pytest.mark.parametrize('which', ['follower', 'leader'])
def test_calibration_mismatch_never_writes_and_closes_buses(fixture, which):
    follower, leader, _, _, _ = fixture
    bus = follower if which == 'follower' else leader
    bus.calibration['shoulder_pan']['homing_offset'] += 1
    with pytest.raises(ValueError, match='live calibration differs'):
        run(fixture)
    assert not follower.writes and not leader.writes
    assert follower.closed == [False]
    if which == 'leader':
        assert leader.closed == [False]


def test_port_alias_rejected_before_bus_open(fixture, tmp_path):
    follower, leader, report, _, _ = fixture
    port = tmp_path/'device'
    port.touch()
    alias = tmp_path/'alias'
    alias.symlink_to(port)
    report['port'] = str(port)
    report['leader']['port'] = str(alias)
    with pytest.raises(ValueError, match='same serial device'):
        run(fixture)
    assert follower.connected == leader.connected == 0


@pytest.mark.parametrize('bad', ['leader_torque', 'follower_mode', 'follower_normalization',
                                  'leader_normalization', 'jaw_open', 'out_of_range'])
def test_invalid_startup_has_no_writes(fixture, bad):
    follower, leader, report, _, _ = fixture
    if bad == 'leader_torque':
        leader.registers['Torque_Enable']['gripper'] = 1
    elif bad == 'follower_mode':
        follower.registers['Operating_Mode']['shoulder_pan'] = 1
    elif bad.endswith('normalization'):
        bus = leader if bad.startswith('leader') else follower
        bus.motors['shoulder_pan'].norm_mode = 'range_m100_100'
    elif bad == 'jaw_open':
        follower.raw['gripper'] += 200
    else:
        follower.raw['shoulder_pan'] = 1023
    with pytest.raises(ValueError):
        run(fixture)
    assert not follower.writes and not leader.writes
    assert not follower.is_connected and not leader.is_connected


def test_incomplete_torque_snapshot_is_prewrite_failure(fixture):
    follower = fixture[0]
    original = follower.sync_read
    def incomplete(register, names, **kwargs):
        values = original(register, names, **kwargs)
        if register == 'Torque_Enable':
            values.pop('gripper')
        return values
    follower.sync_read = incomplete
    with pytest.raises(ValueError, match='incomplete'):
        run(fixture)
    assert not follower.writes
    assert not follower.is_connected


def test_inspect_reports_misalignment_without_writes(fixture):
    follower, leader, report, _, _ = fixture
    leader.raw['shoulder_pan'] += 100
    run(fixture, execute=False)
    assert report['status'] == 'inspected' and not report['startup_alignment_ok']
    assert report['startup_leader_minus_follower_deg'][0] > 5
    assert not follower.writes and not leader.writes
    assert follower.closed == leader.closed == [False]


def test_large_tracking_transient_is_recorded_and_fresh_pose_is_held(fixture):
    follower, leader, report, _, _ = fixture
    follower.on_position_read = lambda bus: bus.raw.update(
        shoulder_pan=2048 if bus.reads == 1 else 2150 if bus.reads == 2 else 2160)
    run(fixture)
    assert report['status'] == 'recorded'
    assert follower.writes[-1][1]['shoulder_pan'] == 2160
    assert report['final_hold_measured']['raw_counts']['shoulder_pan'] == 2160
    assert follower.closed == leader.closed == [False]
    assert not leader.writes


def test_sampling_gap_aborts_and_holds(fixture):
    follower, leader, report, _, clock = fixture
    leader.on_position_read = lambda bus: clock.sleep(.3) if bus.reads == 2 else None
    with pytest.raises(TimeoutError, match='acquisition exceeded'):
        run(fixture)
    assert report['final_hold_goal_raw_counts'] == dict.fromkeys(recorder.ARM, 2048)
    assert follower.closed == leader.closed == [False]


def test_ctrl_c_stops_without_torque_off(fixture):
    follower, leader, report, keyboard, _ = fixture
    original = keyboard.poll
    def interrupt():
        if keyboard.polls == 3:
            raise KeyboardInterrupt
        return original()
    keyboard.poll = interrupt
    run(fixture)
    assert report['stop_reason'] == 'ctrl_c' and len(report['samples']) == 3
    assert all(follower.registers['Torque_Enable'][name] == 1 for name in recorder.ARM)
    assert follower.closed == leader.closed == [False]


def test_gripper_motion_stops_without_commanding_it(fixture):
    follower, _, report, _, _ = fixture
    follower.on_position_read = lambda bus: bus.raw.update(gripper=1000 if bus.reads == 1 else 1200)
    with pytest.raises(ValueError, match='gripper moved'):
        run(fixture)
    assert report['stop_reason'] == 'error'
    assert all('gripper' not in goals for _, goals in follower.writes)


def test_large_startup_lag_is_recorded_while_follower_catches_up(fixture):
    _, leader, report, _, _ = fixture
    leader.on_position_read = lambda bus: bus.raw.update(shoulder_pan=2048 if bus.reads == 1 else 2448)
    result = run(fixture)
    assert result['status'] == 'recorded'
    assert max(abs(row['leader_minus_follower_deg'][0]) for row in result['samples']) > 30


def test_goals_never_accumulate_outside_measured_tracking_envelope():
    cal = calibration()
    measured = {name: 2048 for name in recorder.JOINT_NAMES}
    previous = dict.fromkeys(recorder.ARM, 2048)
    limited = False
    for _ in range(100):
        goal, _, tracking_limited = recorder.command_step([80.]*6, measured, previous, cal, max_step_counts=3)
        assert all(abs(goal[name]-previous[name]) <= 3 for name in recorder.ARM)
        assert all(abs(goal[name]-measured[name])*recorder.COUNT_DEG < 5 for name in recorder.ARM)
        limited |= tracking_limited
        previous = goal
    assert limited


def test_default_check_does_not_open_buses(monkeypatch, capsys):
    follower, leader = Bus(), Bus()
    monkeypatch.setattr(recorder, 'offline_preflight', lambda args: ({'status': 'offline_checked'}, follower, leader))
    assert recorder.main(['--leader-port', '/dev/leader', '--leader-id', 'my_leader',
                          '--robot-id', 'my_follower', '--port', '/dev/follower',
                          '--keyboard-device', '/dev/test_keyboard',
                          '--task-config', 'configs/tasks/keyboard-pose.json']) == 0
    assert not follower.connected and not leader.connected
    assert 'No serial device was opened' in capsys.readouterr().out


def test_preflight_escape_never_writes(fixture):
    follower, leader, _, keyboard, _ = fixture
    keyboard.poll = lambda: ([], True)
    with pytest.raises(ValueError, match='during preflight'):
        run(fixture)
    assert not follower.writes and not leader.writes
    assert follower.closed == leader.closed == [False]


def test_preflight_ctrl_c_does_not_claim_torque_enable_or_hold(fixture):
    follower, leader, report, keyboard, _ = fixture
    def interrupted():
        raise KeyboardInterrupt
    keyboard.poll = interrupted
    run(fixture)
    assert report['status'] == 'interrupted_before_recording'
    assert not report['arm_torque_enable_verified'] and 'final_hold_measured' not in report
    assert not follower.writes and not leader.writes


def test_final_hold_readback_must_match(fixture):
    follower, _, report, keyboard, _ = fixture
    original = follower.sync_read
    def readback(register, names, **kwargs):
        values = original(register, names, **kwargs)
        if register == 'Goal_Position' and keyboard.polls >= keyboard.stop_after:
            values['shoulder_pan'] += 1
        return values
    follower.sync_read = readback
    run(fixture)
    assert report['status'] == 'failed'
    assert 'final hold goal readback mismatch' in report['hold_error']
    assert 'final_hold_measured' not in report


# Launcher identity routing

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    follower, leader = tmp_path/'follower', tmp_path/'leader'
    follower.touch()
    leader.touch()
    log = tmp_path/'args.json'
    stub = '#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["TEST_LAUNCH_LOG"],"w").write(json.dumps(sys.argv[1:]))\n'
    for name in ('python', 'lerobot-calibrate'):
        path = tmp_path/name
        path.write_text(stub)
        path.chmod(0o755)
    env = {**os.environ, 'SO101_LEROBOT_PYTHON': str(tmp_path/'python'),
           'TEST_LAUNCH_LOG': str(log), 'SO101_LEADER_PORT': '', 'SO101_LEADER_ID': ''}

    def run(mode, *args):
        recorder_args = (['--robot-id', 'test_follower', '--keyboard-device', '/dev/test_keyboard',
                          '--task-config', 'configs/tasks/keyboard-pose.json']
                         if mode in ('check', 'inspect', 'record') else [])
        return subprocess.run([str(ROOT/'scripts/leader_scene_check.sh'), mode,
                               '--leader-port', str(leader), '--leader-id', 'test_leader',
                               '--port', str(follower), *recorder_args, *args], env=env, text=True, capture_output=True)

    return run, log, follower, leader


def test_calibration_routes_only_to_leader(launcher):
    run, log, follower, leader = launcher
    assert run('calibrate').returncode == 0
    assert json.loads(log.read_text()) == ['--teleop.type=so101_leader',
        f'--teleop.port={leader}', '--teleop.id=test_leader']
    assert str(follower) not in log.read_text()


@pytest.mark.parametrize('mode,flag', [('check', '--check'), ('inspect', '--inspect'), ('record', '--execute')])
def test_modes_explicitly_route_to_recorder(launcher, mode, flag):
    run, log, follower, leader = launcher
    assert run(mode).returncode == 0
    argv = json.loads(log.read_text())
    assert argv[:2] == ['-m', 'so101_typing.hardware.teleop_scene_record']
    assert argv[argv.index('--leader-port')+1] == str(leader)
    assert argv[argv.index('--port')+1] == str(follower)
    assert argv[-1] == flag
    assert argv[argv.index('--task-config')+1] == 'configs/tasks/keyboard-pose.json'
    assert argv[argv.index('--robot-id')+1] == 'test_follower'


def test_same_port_alias_refused_before_calibration(launcher):
    run, log, follower, leader = launcher
    leader.unlink()
    leader.symlink_to(follower)
    result = run('calibrate')
    assert result.returncode == 2 and 'different serial devices' in result.stderr
    assert not log.exists()


def test_calibration_refuses_extra_arguments_and_invalid_id(launcher):
    run, log, _, _ = launcher
    assert run('calibrate', '--robot.type=so101_follower').returncode == 2
    assert run('calibrate', '--leader-id', '../follower').returncode == 2
    assert not log.exists()


def test_missing_identity_refused_before_calibration(launcher):
    run, log, _, _ = launcher
    result = run('calibrate', '--port', '')
    assert result.returncode == 2 and 'no serial device is selected automatically' in result.stderr
    assert not log.exists()
