"""Record measured follower poses during bounded, powered leader teleoperation.

Default --check is offline. --inspect reads both buses without changing either.
Only --execute writes follower goals and torque enable. All follower
motor torque is disabled and verified on exit. No leader, gain, calibration,
configuration, or model-offset writes occur.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from scripts.robot_model_identity import WORKSHOP_USD_SHA256
from .fk_probe_trajectory import ROOT, TASK, digest
from .hardware_lock import RobotBusLock
from .keyboard_debounce import DEFAULT_KEYBOARD_DEBOUNCE_MS, KeyboardDebouncer
from .so101_kinematics import JOINT_NAMES, lerobot_position_to_model_rad

ARM = JOINT_NAMES[:5]
COUNT_DEG = 360 / 4095
MAX_GAP_S = .25
TRACKING_ABORT_DEG = 60.
TRACKING_ENVELOPE_DEG = 5.
START_ALIGNMENT_DEG = 5.
LEADER_LAG_ABORT_DEG = 30.
REGISTERS = ('Torque_Enable', 'Operating_Mode', 'Goal_Position',
             'P_Coefficient', 'I_Coefficient', 'D_Coefficient')
TELEMETRY_REGISTERS = ('Present_Position', 'Present_Velocity', 'Present_Load',
                       'Present_Voltage', 'Present_Temperature', 'Status',
                       'Moving', 'Present_Current', 'Torque_Enable',
                       'Goal_Position', 'Goal_Position_2')


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_calibration(data):
    fields = {'id', 'drive_mode', 'homing_offset', 'range_min', 'range_max'}
    if not isinstance(data, dict) or set(data) != set(JOINT_NAMES):
        raise ValueError('calibration must contain exactly the five arm joints and gripper')
    for index, name in enumerate(JOINT_NAMES, 1):
        row = data[name]
        if (not isinstance(row, dict) or set(row) != fields
                or any(type(value) is not int for value in row.values())):
            raise ValueError(f'{name}: calibration requires integer motor fields')
        if row['id'] != index or row['drive_mode'] != 0:
            raise ValueError(f'{name}: expected motor ID {index} and drive_mode 0')
        if not 0 <= row['range_min'] < row['range_max'] <= 4095:
            raise ValueError(f'{name}: invalid single-turn calibration range')
    return data


def require_leader_calibration(leader_id):
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS
    if (not leader_id or leader_id in ('.', '..')
            or '/' in leader_id or '\\' in leader_id):
        raise ValueError('leader ID must be a calibration filename, not a path')
    directory = HF_LEROBOT_CALIBRATION / TELEOPERATORS / 'so_leader'
    path = directory / f'{leader_id}.json'
    if not path.is_file():
        available = ', '.join(sorted(p.stem for p in directory.glob('*.json'))) or 'none'
        raise FileNotFoundError(f'No leader calibration for {leader_id!r}: {path}. '
                                f'Available leader IDs: {available}. Calibrate the leader first.')
    validate_calibration(json.loads(path.read_text()))
    return path


def validate_ports(follower_port, leader_port):
    if os.path.realpath(follower_port) == os.path.realpath(leader_port):
        raise ValueError('leader and follower resolve to the same serial device; no bus was opened')


def positions(raw, calibration):
    """Installed LeRobot DEGREES convention; gripper remains RANGE_0_100."""
    result = []
    for name in ARM:
        cal = calibration[name]
        result.append((raw[name] - (cal['range_min'] + cal['range_max']) / 2) * COUNT_DEG)
    cal = calibration['gripper']
    bounded = min(cal['range_max'], max(cal['range_min'], raw['gripper']))
    result.append((bounded - cal['range_min']) * 100 / (cal['range_max'] - cal['range_min']))
    return result


def read_raw(bus, register, names=JOINT_NAMES):
    values = bus.sync_read(register, list(names), normalize=False, num_retry=2)
    if (set(values) != set(names)
            or any(not math.isfinite(float(v)) or int(v) != v for v in values.values())):
        raise ValueError(f'incomplete or nonfinite {register} read')
    return {name: int(values[name]) for name in names}


def read_pose(bus, calibration, clock):
    before = clock.monotonic()
    raw = read_raw(bus, 'Present_Position')
    after = clock.monotonic()
    if after - before > MAX_GAP_S:
        raise TimeoutError('encoder acquisition exceeded 250 ms')
    if any(not 0 <= value <= 4095 for value in raw.values()):
        raise ValueError('encoder position outside single-turn range 0..4095')
    normalized = positions(raw, calibration)
    return {'acquisition_monotonic_start_s': before, 'acquisition_monotonic_end_s': after,
            'monotonic_s': (before + after) / 2, 'raw_counts': raw,
            'lerobot_position': normalized,
            'q_model_rad': lerobot_position_to_model_rad(normalized).tolist()}


def check_arm_range(raw, calibration):
    for name in ARM:
        cal = calibration[name]
        if not cal['range_min'] <= raw[name] <= cal['range_max']:
            raise ValueError(f'{name}: encoder outside its own calibration range')


def register_snapshot(bus):
    return {name: read_raw(bus, name) for name in REGISTERS}


def telemetry_snapshot(bus, clock):
    before = clock.monotonic()
    registers = {name: read_raw(bus, name) for name in TELEMETRY_REGISTERS}
    after = clock.monotonic()
    electrical = {}
    for name in JOINT_NAMES:
        voltage_v = registers['Present_Voltage'][name] * 0.1
        current_a = registers['Present_Current'][name] * 0.0065
        electrical[name] = {'voltage_v': voltage_v, 'current_a': current_a,
                            'power_estimate_w': voltage_v * current_a,
                            'temperature_c': registers['Present_Temperature'][name]}
    return {'acquisition_monotonic_start_s': before,
            'acquisition_monotonic_end_s': after,
            'monotonic_s': (before + after) / 2,
            'registers_raw': registers, 'electrical_estimates': electrical}


def return_to_start_state(initial_leader, follower, leader, departed,
                          candidate_since, now, *, departure_deg=8.,
                          tolerance_deg=5., hold_s=1.5):
    """Require both arms to leave and then settle near the leader's start pose."""
    reset = initial_leader['lerobot_position'][:5]
    follower_delta = max(abs(a-b) for a, b in zip(follower['lerobot_position'][:5], reset))
    leader_delta = max(abs(a-b) for a, b in zip(leader['lerobot_position'][:5], reset))
    departed = departed or (follower_delta >= departure_deg and leader_delta >= departure_deg)
    near_start = follower_delta <= tolerance_deg and leader_delta <= tolerance_deg
    candidate_since = now if departed and near_start and candidate_since is None else (
        candidate_since if departed and near_start else None)
    return departed, candidate_since, bool(candidate_since is not None and now-candidate_since >= hold_s)


def command_step(leader_position, measured_raw, previous_goal, calibration, *, max_step_counts=None,
                 mirror_gripper=True):
    """Convert leader pose to follower goals; direct mode mirrors each sample."""
    goals, rate_limited, tracking_limited = {}, False, False
    for index, name in enumerate(ARM):
        cal = calibration[name]
        desired = int(leader_position[index] / COUNT_DEG + (cal['range_min'] + cal['range_max']) / 2)
        bounded = min(cal['range_max'], max(cal['range_min'], desired))
        if max_step_counts is None:
            goal = bounded
        else:
            limited = min(previous_goal[name] + max_step_counts,
                          max(previous_goal[name] - max_step_counts, bounded))
            rate_limited |= limited != desired
            tracking_counts = math.floor(TRACKING_ENVELOPE_DEG / COUNT_DEG) - 1
            lower = max(cal['range_min'], measured_raw[name] - tracking_counts)
            upper = min(cal['range_max'], measured_raw[name] + tracking_counts)
            goal = min(upper, max(lower, limited))
            tracking_limited |= goal != limited
        goals[name] = int(goal)
    if mirror_gripper:
        cal = calibration['gripper']
        leader_percent = min(100., max(0., leader_position[5]))
        goals['gripper'] = int(round(cal['range_min'] +
                                     leader_percent * (cal['range_max'] - cal['range_min']) / 100))
    return goals, bool(rate_limited), bool(tracking_limited)


def validate_bus_definition(bus):
    if set(bus.motors) != set(JOINT_NAMES):
        raise ValueError('bus must contain the five arm motors and gripper')
    for index, name in enumerate(JOINT_NAMES, 1):
        motor = bus.motors[name]
        expected_mode = 'degrees' if name in ARM else 'range_0_100'
        mode = getattr(motor.norm_mode, 'value', motor.norm_mode)
        if motor.id != index or motor.model != 'sts3215' or mode != expected_mode:
            raise ValueError(f'{name}: require STS3215 ID {index}, {expected_mode} normalization')


def run_session(follower_bus, leader_bus, report, keyboard, *, seconds=120., hz=50.,
                max_rate_deg_s=60., execute=False, emit=lambda event: None, clock=time,
                telemetry_hz=0., stop_on_return=False, mirror_gripper=True,
                health_check=lambda: None, keyboard_debounce_ms=DEFAULT_KEYBOARD_DEBOUNCE_MS):
    """Own both direct bus connections. Injected buses/clock support hardware-free tests."""
    validate_ports(report['port'], report['leader']['port'])
    follower_cal = validate_calibration(report['calibration'])
    leader_cal = validate_calibration(report['leader']['calibration'])
    if (not 1 <= seconds <= 600 or not 10 <= hz <= 100 or not 0 <= telemetry_hz <= hz
            or not 1 <= max_rate_deg_s <= 60):
        raise ValueError('seconds must be 1..600, hz 10..100, telemetry hz 0..control hz, max rate 1..60 deg/s')
    debouncer = KeyboardDebouncer(keyboard_debounce_ms)
    report['keyboard_debounce_ms'] = keyboard_debounce_ms
    report['debounced_events'] = []
    report.update(samples=[], events=[], telemetry=[], status='failed', started_utc=utc(),
                  arm_torque_enable_verified=False, gripper_torque_enable_verified=False)
    started_writes = False
    try:
        for label, bus, calibration in (('follower', follower_bus, follower_cal),
                                        ('leader', leader_bus, leader_cal)):
            validate_bus_definition(bus)
            bus.connect()
            actual = {name: asdict(value) for name, value in bus.read_calibration().items()}
            report.setdefault('live', {})[label] = {'calibration': actual,
                                                    'calibration_matches_file': actual == calibration}
            if actual != calibration:
                raise ValueError(f'{label} live calibration differs from its file; no motor writes sent')
            report['live'][label]['registers_before'] = register_snapshot(bus)
        initial = read_pose(follower_bus, follower_cal, clock)
        leader = read_pose(leader_bus, leader_cal, clock)
        report['initial_follower'] = initial
        report['initial_leader'] = leader
        deltas = [a-b for a, b in zip(leader['lerobot_position'][:5], initial['lerobot_position'][:5])]
        report['startup_leader_minus_follower_deg'] = deltas
        emit({'kind': 'preflight', 'keyboard_debounce_ms': keyboard_debounce_ms, 'live': report['live'], 'initial_follower': initial,
              'initial_leader': leader, 'leader_minus_follower_deg': deltas})
        follower_regs = report['live']['follower']['registers_before']
        leader_regs = report['live']['leader']['registers_before']
        for label, registers in (('leader', leader_regs), ('follower', follower_regs)):
            if any(value not in (0, 1) for value in registers['Torque_Enable'].values()):
                raise ValueError(f'{label} has invalid torque register state')
        if any(leader_regs['Torque_Enable'].values()):
            raise ValueError('leader torque must already be OFF on all six motors; nothing was changed')
        if any(follower_regs['Operating_Mode'][name] != 0 for name in ARM):
            raise ValueError('follower arm must already be in position mode; no configuration writes permitted')
        if mirror_gripper and follower_regs['Operating_Mode']['gripper'] != 0:
            raise ValueError('follower gripper must already be in position mode; no configuration writes permitted')
        check_arm_range(initial['raw_counts'], follower_cal)
        if mirror_gripper and not (follower_cal['gripper']['range_min'] <= initial['raw_counts']['gripper']
                                   <= follower_cal['gripper']['range_max']):
            raise ValueError('follower gripper starts outside its calibration range')
        report['startup_alignment_ok'] = max(map(abs, deltas)) <= START_ALIGNMENT_DEG
        report['fixed_jaw_closed_ok'] = initial['lerobot_position'][5] <= 5.
        if not execute:
            report['status'] = 'inspected'
            return report
        # Seed goals at the measured pose before torque enable. Subsequent
        # goals mirror the leader directly without a software rate limit.
        if not mirror_gripper and not report['fixed_jaw_closed_ok']:
            raise ValueError('follower moving jaw must already be closed (<=5%); no gripper commands are sent')
        preflight_edges, stop = keyboard.poll()
        report['preflight_key_events'] = preflight_edges
        device = getattr(keyboard, 'device', None)
        if stop or (device is not None and device.active_keys()):
            raise ValueError('Esc or held keyboard key during preflight; no motor writes sent')
        commanded = JOINT_NAMES if mirror_gripper else ARM
        previous_goal = {name: initial['raw_counts'][name] for name in commanded}
        started_writes = True
        follower_bus.sync_write('Goal_Position', previous_goal, normalize=False, num_retry=0)
        if read_raw(follower_bus, 'Goal_Position', commanded) != previous_goal:
            raise ValueError('seed goal readback mismatch; arm torque was not enabled')
        enabled = dict.fromkeys(commanded, 1)
        follower_bus.sync_write('Torque_Enable', enabled, normalize=False, num_retry=0)
        if read_raw(follower_bus, 'Torque_Enable', commanded) != enabled:
            raise ValueError('follower torque enable readback failed')
        report['arm_torque_enable_verified'] = True
        report['gripper_torque_enable_verified'] = mirror_gripper
        emit({'kind': 'seed_and_enable_arm', 'goal_raw_counts': previous_goal,
              'gripper_mirror_enabled': mirror_gripper, 'monotonic_s': clock.monotonic()})
        print('Powered recording. Move the leader slowly; press and release Q, G, P. Esc or Ctrl-C saves. '
              + ('Follower gripper mirrors the leader directly. ' if mirror_gripper
                 else 'Follower gripper is not commanded. ')
              + 'Stopping disables and verifies follower torque.', flush=True)
        start = previous_tick = clock.monotonic()
        departed = False
        return_candidate_since = None
        next_telemetry = start
        report['recording_start_monotonic_s'] = start
        report['stop_reason'] = 'duration'
        while clock.monotonic() - start < seconds:
            health_check()
            tick = clock.monotonic()
            if tick - previous_tick > MAX_GAP_S:
                raise TimeoutError('control gap exceeded 250 ms')
            leader = read_pose(leader_bus, leader_cal, clock)
            measured = read_pose(follower_bus, follower_cal, clock)
            report['latest_control_observation'] = {
                'follower': measured, 'leader': leader, 'previous_goal_raw_counts': previous_goal.copy()}
            if measured['acquisition_monotonic_end_s'] - previous_tick > MAX_GAP_S:
                raise TimeoutError('combined bus acquisition gap exceeded 250 ms')
            check_arm_range(measured['raw_counts'], follower_cal)
            if not mirror_gripper and abs(measured['lerobot_position'][5] - initial['lerobot_position'][5]) > 5.:
                raise ValueError('follower gripper moved more than 5 percentage points; keep the fixed jaw closed')
            if mirror_gripper and not (follower_cal['gripper']['range_min'] - 20
                                       <= measured['raw_counts']['gripper']
                                       <= follower_cal['gripper']['range_max'] + 20):
                raise ValueError('follower gripper moved outside calibration range')
            tracking = [(measured['raw_counts'][name]-previous_goal[name])*COUNT_DEG for name in ARM]
            if max(map(abs, tracking)) > TRACKING_ABORT_DEG:
                details = ', '.join(f'{name}={value:+.2f}' for name, value in zip(ARM, tracking))
                raise ValueError(f'follower tracking error exceeds {TRACKING_ABORT_DEG:g} degrees: {details}')
            lag = [a-b for a, b in zip(leader['lerobot_position'][:5], measured['lerobot_position'][:5])]
            # Large initial leader/follower lag is diagnostic only; do not abort.
            if stop_on_return:
                departed, return_candidate_since, returned = return_to_start_state(
                    report['initial_leader'], measured, leader, departed,
                    return_candidate_since, tick)
            else:
                returned = False
            edges, stop = keyboard.poll()
            edges = [{**event, 'time_s': event['received_monotonic_s']-start} for event in edges]
            for event in edges:
                report['events'].append(event)
                emit({'kind': 'key', **event})
                print(f"REAL {event['key_name']} {event['edge'].upper()} at {event['time_s']:.2f}s", flush=True)
            for event in debouncer.process(edges, now_s=clock.time()):
                report['debounced_events'].append(event)
                emit({'kind': 'key_debounced', **event})
            # Direct teleoperation: send the leader's current pose every cycle.
            # The servo controller, rather than this recorder, determines physical
            # response speed. max_rate is retained in metadata for compatibility.
            goal, rate_limited, tracking_limited = command_step(
                leader['lerobot_position'], measured['raw_counts'], previous_goal, follower_cal,
                max_step_counts=None, mirror_gripper=mirror_gripper)
            row = {**measured, 'time_s': measured['monotonic_s']-start,
                   'leader': leader, 'previous_goal_raw_counts': previous_goal.copy(),
                   'tracking_error_deg': tracking, 'leader_minus_follower_deg': lag,
                   'goal_raw_counts': goal.copy(),
                   'goal_arm_deg': positions({**measured['raw_counts'], **goal}, follower_cal)[:5],
                   'goal_gripper_percent': (positions({**measured['raw_counts'], **goal}, follower_cal)[5]
                                            if mirror_gripper else None),
                   'goal_sent': False, 'rate_limited': rate_limited,
                   'tracking_limited': tracking_limited}
            report['samples'].append(row)
            if stop or returned:
                emit({'kind': 'sample', **row})
                report['stop_reason'] = 'return_to_start' if returned else 'esc'
                break
            follower_bus.sync_write('Goal_Position', goal, normalize=False, num_retry=0)
            row['goal_sent'] = True
            row['goal_write_completed_monotonic_s'] = clock.monotonic()
            emit({'kind': 'sample', **row})
            if telemetry_hz and clock.monotonic() >= next_telemetry:
                telemetry = {'kind': 'telemetry', 'time_s': clock.monotonic()-start,
                             'leader': telemetry_snapshot(leader_bus, clock),
                             'follower': telemetry_snapshot(follower_bus, clock)}
                report['telemetry'].append(telemetry)
                emit(telemetry)
                next_telemetry = clock.monotonic() + 1/telemetry_hz
            previous_goal = goal
            previous_tick = tick
            clock.sleep(max(0., 1/hz-(clock.monotonic()-tick)))
        report['status'] = 'recorded'
    except KeyboardInterrupt:
        report.update(stop_reason='ctrl_c', status='recorded' if report['samples'] else 'interrupted_before_recording')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['stop_reason'] = 'error'
        emit({'kind': 'error', 'error': report['error'], 'monotonic_s': clock.monotonic()})
        raise
    finally:
        if started_writes and follower_bus.is_connected:
            try:
                held = read_pose(follower_bus, follower_cal, clock)
                commanded = JOINT_NAMES if mirror_gripper else ARM
                hold_goal = {name: held['raw_counts'][name] for name in commanded}
                follower_bus.sync_write('Goal_Position', hold_goal, normalize=False, num_retry=0)
                if read_raw(follower_bus, 'Goal_Position', commanded) != hold_goal:
                    raise ValueError('final hold goal readback mismatch')
                report['final_hold_measured'] = held
                report['final_hold_goal_raw_counts'] = hold_goal
                emit({'kind': 'hold_fresh_measured_pose', 'sample': held, 'goal_raw_counts': hold_goal})
            except BaseException as error:
                report['hold_error'] = f'{type(error).__name__}: {error}; existing goal retained'
                report['status'] = 'failed'
            finally:
                if mirror_gripper:
                    try:
                        follower_bus.sync_write('Torque_Enable', {'gripper': 0},
                                                normalize=False, num_retry=0)
                        if read_raw(follower_bus, 'Torque_Enable', ('gripper',)) != {'gripper': 0}:
                            raise ValueError('gripper torque off readback failed')
                        report['gripper_torque_off_verified'] = True
                    except BaseException as error:
                        report['gripper_torque_off_error'] = f'{type(error).__name__}: {error}'
                        report['status'] = 'failed'
        if started_writes and follower_bus.is_connected:
            try:
                off = dict.fromkeys(JOINT_NAMES, 0)
                follower_bus.sync_write('Torque_Enable', off, normalize=False, num_retry=0)
                observed = read_raw(follower_bus, 'Torque_Enable')
                if observed != off:
                    raise ValueError(f'follower torque-off readback mismatch: {observed}')
                report['follower_torque_off_verified'] = True
                emit({'kind': 'follower_torque_off_verified', 'torque_enable': observed,
                      'monotonic_s': clock.monotonic()})
            except BaseException as error:
                report['follower_torque_off_error'] = f'{type(error).__name__}: {error}'
                report['status'] = 'failed'
        for label, bus in (('leader', leader_bus), ('follower', follower_bus)):
            if bus.is_connected:
                try:
                    report.setdefault('live', {}).setdefault(label, {})['registers_after'] = register_snapshot(bus)
                except Exception as error:
                    report.setdefault('cleanup_errors', []).append(f'{label} final snapshot: {error}')
                finally:
                    try:
                        bus.disconnect(disable_torque=False)
                    except Exception as error:
                        report.setdefault('cleanup_errors', []).append(f'{label} disconnect: {error}')
        for event in debouncer.flush(0., final=True):
            report['debounced_events'].append(event)
            emit({'kind': 'key_debounced', **event})
        report['finished_utc'] = utc()
    return report


def offline_preflight(args):
    from scripts.fixture_layout import load_geometry
    from scripts.reinforcement_learning.rsl_rl.public_task import load_config
    from .robot import require_calibration, resolve_robot_port, load_lerobot
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    follower_type, follower_config, _ = load_lerobot()
    port = resolve_robot_port(args.port)
    leader_port = resolve_robot_port(args.leader_port)
    validate_ports(port, leader_port)
    calibration = require_calibration(args.robot_id)
    leader_calibration = require_leader_calibration(args.leader_id)
    follower_data = validate_calibration(json.loads(calibration.read_text()))
    leader_data = validate_calibration(json.loads(leader_calibration.read_text()))
    geometry = load_geometry()
    sources = ['scripts/so101_homing/so101_kinematics.py', *geometry['source_sha256']]
    report = {'schema_version': 1, 'kind': 'so101_powered_teleop_recording', 'task': TASK,
              'measurement_source': 'follower_present_position',
              'task_config': load_config(args.task_config), 'task_config_sha256': digest(args.task_config),
              'source_sha256': {name: digest(ROOT/name) for name in sources},
              'recorder_sha256': digest(Path(__file__)),
              'robot_usd_sha256': WORKSHOP_USD_SHA256,
              'robot_id': args.robot_id, 'port': port, 'calibration': follower_data,
              'calibration_file': str(calibration), 'calibration_sha256': digest(calibration),
              'leader': {'robot_id': args.leader_id, 'port': leader_port, 'calibration': leader_data,
                         'calibration_file': str(leader_calibration), 'calibration_sha256': digest(leader_calibration)},
              'keyboard_device': args.keyboard_device, 'joint_names': list(JOINT_NAMES),
              'joint_zero_offset_deg': [0]*5, 'requested_hz': args.hz,
              'control': {'max_rate_deg_s': args.max_rate_deg_s, 'tracking_abort_deg': TRACKING_ABORT_DEG,
                          'max_gap_s': MAX_GAP_S, 'startup_alignment_check_deg': START_ALIGNMENT_DEG,
                          'leader_lag_abort_deg': None,
                          'gripper': 'not commanded; preserve initial closed pose and torque state',
                          'exit': 'hold fresh measured arm pose, then disable and verify follower torque on all six motors'},
              'limits': 'Encoder-derived poses, not independent calibrated 3D measurements.',
              'samples': [], 'events': [], 'status': 'offline_checked'}
    # Constructors load files only; connect/configure methods are never called here.
    follower = follower_type(follower_config(port=port, id=args.robot_id, calibration_dir=calibration.parent,
                                             use_degrees=True, disable_torque_on_disconnect=False))
    leader = SO101Leader(SO101LeaderConfig(port=leader_port, id=args.leader_id,
                                          calibration_dir=leader_calibration.parent, use_degrees=True))
    return report, follower.bus, leader.bus


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='offline preflight (default): no serial access')
    mode.add_argument('--inspect', action='store_true', help='read-only calibration, torque and pose alignment checks')
    mode.add_argument('--execute', action='store_true', help='enable bounded powered follower teleoperation and record')
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--leader-port', required=True)
    parser.add_argument('--leader-id', required=True)
    parser.add_argument('--keyboard-device', required=True)
    parser.add_argument('--task-config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT/'output/calibration_video_gtc_berlin')
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--hz', type=float, default=50.)
    parser.add_argument('--max-rate-deg-s', type=float, default=60.)
    parser.add_argument('--telemetry-hz', type=float, default=5.,
                        help='sample all STS3215 feedback registers from both buses at this rate')
    parser.add_argument('--stop-on-return', action='store_true',
                        help='stop after both arms depart and settle near the initial leader arm pose')
    parser.add_argument('--keyboard-debounce-ms', type=float, default=DEFAULT_KEYBOARD_DEBOUNCE_MS,
                        help='merge same-key re-presses within this release interval; 0 disables filtering')
    parser.add_argument('--record-cameras', action='store_true',
                        help='record arm MJPEG stream with host receipt timestamps')
    args = parser.parse_args(argv)
    try:
        if (not 1 <= args.seconds <= 600 or not 10 <= args.hz <= 100
                or not 0 <= args.telemetry_hz <= args.hz
                or not 1 <= args.max_rate_deg_s <= 60):
            raise ValueError('seconds must be 1..600, hz 10..100, telemetry hz 0..control hz, max rate 1..60 deg/s')
        KeyboardDebouncer(args.keyboard_debounce_ms)  # validate before opening hardware
        report, follower_bus, leader_bus = offline_preflight(args)
        report['telemetry_capture'] = {
            'requested_hz': args.telemetry_hz, 'registers': list(TELEMETRY_REGISTERS),
            'scope': 'six motors on each of leader and follower',
            'voltage_v_per_count': 0.1, 'current_a_per_count': 0.0065,
            'power_estimate': 'voltage times current from motor feedback; not a supply power meter'}
        report['camera_capture'] = {
            'enabled': bool(args.execute and args.record_cameras),
            'sources': ['typing_calibration'] if args.record_cameras else [],
            'timestamp_semantics': 'host MJPEG frame receipt; not camera exposure time'}
        report['keyboard_capture'] = {'device': args.keyboard_device, 'events': 'keyboard_events.jsonl, kind=key rows in capture.jsonl, and recording.json events', 'debounced_events': 'keyboard_events_debounced.jsonl, kind=key_debounced rows, and recording.json debounced_events', 'debounce_ms': args.keyboard_debounce_ms, 'fields': ['char', 'code', 'key_name', 'value', 'edge', 'evdev_timestamp_s', 'received_monotonic_s', 'time_s']}
        report['return_to_start'] = {
            'enabled': args.stop_on_return, 'target': 'initial leader arm pose',
            'departure_deg': 8., 'tolerance_deg': 5., 'hold_s': 1.5}
        if not (args.inspect or args.execute):
            print(json.dumps(report, indent=2))
            print('OFFLINE CHECK ONLY. No serial device was opened; use --inspect for live alignment checks.')
            return 0
        directory = args.output_root.resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        directory.mkdir(parents=True, exist_ok=False)
        path = directory/('recording.json' if args.execute else 'inspection.json')
        try:
            with ExitStack() as stack:
                cameras = None
                if args.execute and args.record_cameras:
                    from .typing_calibration_camera import DualCameraCapture
                    cameras = stack.enter_context(DualCameraCapture(directory))
                stack.enter_context(RobotBusLock())
                suffix = hashlib.sha256(os.path.realpath(report['leader']['port']).encode()).hexdigest()[:16]
                stack.enter_context(RobotBusLock(Path('/tmp')/f'so101_teleop_leader_{suffix}.lock'))
                keyboard = None
                if args.execute:
                    from .policy_runtime import KeyboardMonitor
                    keyboard = stack.enter_context(KeyboardMonitor(args.keyboard_device))
                    keyboard.poll()
                    if keyboard.device.active_keys():
                        raise ValueError('release keyboard keys before starting')
                stream = stack.enter_context((directory/'capture.jsonl').open('x', buffering=1))
                key_stream = stack.enter_context((directory/'keyboard_events.jsonl').open('x', buffering=1))
                debounced_stream = stack.enter_context((directory/'keyboard_events_debounced.jsonl').open('x', buffering=1))
                def emit_row(row):
                    line = json.dumps(row)+'\n'
                    stream.write(line)
                    if row.get('kind') == 'key':
                        key_stream.write(line)
                    elif row.get('kind') == 'key_debounced':
                        debounced_stream.write(line)
                run_session(follower_bus, leader_bus, report, keyboard,
                            seconds=args.seconds, hz=args.hz, max_rate_deg_s=args.max_rate_deg_s,
                            execute=args.execute, emit=emit_row,
                            telemetry_hz=args.telemetry_hz, stop_on_return=args.stop_on_return,
                            mirror_gripper=False, keyboard_debounce_ms=args.keyboard_debounce_ms,
                            health_check=cameras.check if cameras else lambda: None)
        except BaseException as error:
            report['status'] = 'failed'
            report.setdefault('error', f'{type(error).__name__}: {error}')
            raise
        finally:
            path.write_text(json.dumps(report, indent=2)+'\n')
            print(f'Recording: {path}' if args.execute else f'Inspection: {path}', flush=True)
        if args.inspect:
            print('Joint             Follower deg   Leader deg   Leader-minus-follower')
            for index, name in enumerate(ARM):
                print(f"{name:17} {report['initial_follower']['lerobot_position'][index]:12.2f} "
                      f"{report['initial_leader']['lerobot_position'][index]:12.2f} "
                      f"{report['startup_leader_minus_follower_deg'][index]:+10.2f}")
            print(f"Startup alignment within 5 degrees: {report['startup_alignment_ok']}")
            print(f"Follower gripper: {report['initial_follower']['lerobot_position'][5]:.2f}% open. No registers changed.")
        else:
            if report.get('hold_error'):
                raise RuntimeError(report['hold_error'])
            if report.get('gripper_torque_off_error'):
                raise RuntimeError(report['gripper_torque_off_error'])
            if report.get('follower_torque_off_error'):
                raise RuntimeError(report['follower_torque_off_error'])
            final_torque = report.get('live', {}).get('follower', {}).get('registers_after', {}).get('Torque_Enable', {})
            if report.get('follower_torque_off_verified') and all(final_torque.get(name) == 0 for name in JOINT_NAMES):
                print('Follower torque OFF verified on all six motors after recording.')
            elif report.get('final_hold_measured'):
                raise RuntimeError('Follower torque-off verification missing after recording')
            else:
                print('Stopped before follower commands; no motor state was changed.')
            if len(report['samples']) >= 2:
                print(f'Replay without offsets: ./scripts/scene_check.sh replay --recording {path}')
            else:
                print('Stopped before two measured samples were recorded; this file is not replayable.')
        return 0
    except (OSError, ValueError, RuntimeError, ConnectionError, TimeoutError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
