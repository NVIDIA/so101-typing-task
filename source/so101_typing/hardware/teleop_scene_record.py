"""Record measured follower poses during bounded, powered leader teleoperation.

Default --check is offline. --inspect reads both buses without changing either.
Only --execute writes: follower arm goal positions and arm torque enable. No
leader, gripper, gain, calibration, configuration, or model-offset writes occur.
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
    values = bus.sync_read(register, list(names), normalize=False, num_retry=0)
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


def command_step(leader_position, measured_raw, previous_goal, calibration, *, max_step_counts=None):
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
                max_rate_deg_s=60., execute=False, emit=lambda event: None, clock=time):
    """Own both direct bus connections. Injected buses/clock support hardware-free tests."""
    validate_ports(report['port'], report['leader']['port'])
    follower_cal = validate_calibration(report['calibration'])
    leader_cal = validate_calibration(report['leader']['calibration'])
    if (not 1 <= seconds <= 600 or not 10 <= hz <= 100
            or not 1 <= max_rate_deg_s <= 60):
        raise ValueError('seconds must be 1..600, hz 10..100, max rate 1..60 deg/s')
    report.update(samples=[], events=[], status='failed', started_utc=utc(), arm_torque_enable_verified=False)
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
        emit({'kind': 'preflight', 'live': report['live'], 'initial_follower': initial,
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
        check_arm_range(initial['raw_counts'], follower_cal)
        report['startup_alignment_ok'] = max(map(abs, deltas)) <= START_ALIGNMENT_DEG
        report['fixed_jaw_closed_ok'] = initial['lerobot_position'][5] <= 5.
        if not execute:
            report['status'] = 'inspected'
            return report
        # Initial separation is allowed. The follower starts from its measured
        # pose and the rate limiter brings it toward the leader progressively.
        if not report['fixed_jaw_closed_ok']:
            raise ValueError('follower moving jaw must already be closed (<=5%); no gripper commands are sent')
        preflight_edges, stop = keyboard.poll()
        report['preflight_key_events'] = preflight_edges
        device = getattr(keyboard, 'device', None)
        if stop or (device is not None and device.active_keys()):
            raise ValueError('Esc or held keyboard key during preflight; no motor writes sent')
        previous_goal = {name: initial['raw_counts'][name] for name in ARM}
        started_writes = True
        follower_bus.sync_write('Goal_Position', previous_goal, normalize=False, num_retry=0)
        if read_raw(follower_bus, 'Goal_Position', ARM) != previous_goal:
            raise ValueError('seed goal readback mismatch; arm torque was not enabled')
        follower_bus.sync_write('Torque_Enable', dict.fromkeys(ARM, 1), normalize=False, num_retry=0)
        if read_raw(follower_bus, 'Torque_Enable', ARM) != dict.fromkeys(ARM, 1):
            raise ValueError('follower arm torque enable readback failed')
        report['arm_torque_enable_verified'] = True
        emit({'kind': 'seed_and_enable_arm', 'goal_raw_counts': previous_goal, 'monotonic_s': clock.monotonic()})
        print('Powered recording. Move the leader slowly; press and release Q, G, P. Esc or Ctrl-C saves. '
              'Follower gripper is not commanded. Stopping holds the current arm pose.', flush=True)
        start = previous_tick = clock.monotonic()
        credit = 0.
        report['recording_start_monotonic_s'] = start
        report['stop_reason'] = 'duration'
        while clock.monotonic() - start < seconds:
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
            if abs(measured['lerobot_position'][5] - initial['lerobot_position'][5]) > 5.:
                raise ValueError('follower gripper moved more than 5 percentage points; keep the fixed jaw closed')
            tracking = [(measured['raw_counts'][name]-previous_goal[name])*COUNT_DEG for name in ARM]
            if max(map(abs, tracking)) > TRACKING_ABORT_DEG:
                details = ', '.join(f'{name}={value:+.2f}' for name, value in zip(ARM, tracking))
                raise ValueError(f'follower tracking error exceeds {TRACKING_ABORT_DEG:g} degrees: {details}')
            lag = [a-b for a, b in zip(leader['lerobot_position'][:5], measured['lerobot_position'][:5])]
            # Large initial leader/follower lag is diagnostic only; do not abort.
            edges, stop = keyboard.poll()
            for event in edges:
                event = {**event, 'time_s': event['received_monotonic_s']-start}
                report['events'].append(event)
                emit({'kind': 'key', **event})
                print(f"REAL {event['key_name']} {event['edge'].upper()} at {event['time_s']:.2f}s", flush=True)
            # Direct teleoperation: send the leader's current pose every cycle.
            # The servo controller, rather than this recorder, determines physical
            # response speed. max_rate is retained in metadata for compatibility.
            goal, rate_limited, tracking_limited = command_step(
                leader['lerobot_position'], measured['raw_counts'], previous_goal, follower_cal,
                max_step_counts=None)
            row = {**measured, 'time_s': measured['monotonic_s']-start,
                   'leader': leader, 'previous_goal_raw_counts': previous_goal.copy(),
                   'tracking_error_deg': tracking, 'leader_minus_follower_deg': lag,
                   'goal_raw_counts': goal.copy(),
                   'goal_arm_deg': positions({**measured['raw_counts'], **goal}, follower_cal)[:5],
                   'goal_sent': False, 'rate_limited': rate_limited,
                   'tracking_limited': tracking_limited}
            report['samples'].append(row)
            if stop:
                emit({'kind': 'sample', **row})
                report['stop_reason'] = 'esc'
                break
            follower_bus.sync_write('Goal_Position', goal, normalize=False, num_retry=0)
            row['goal_sent'] = True
            row['goal_write_completed_monotonic_s'] = clock.monotonic()
            emit({'kind': 'sample', **row})
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
                hold_goal = {name: held['raw_counts'][name] for name in ARM}
                follower_bus.sync_write('Goal_Position', hold_goal, normalize=False, num_retry=0)
                if read_raw(follower_bus, 'Goal_Position', ARM) != hold_goal:
                    raise ValueError('final hold goal readback mismatch')
                report['final_hold_measured'] = held
                report['final_hold_goal_raw_counts'] = hold_goal
                emit({'kind': 'hold_fresh_measured_pose', 'sample': held, 'goal_raw_counts': hold_goal})
            except BaseException as error:
                report['hold_error'] = f'{type(error).__name__}: {error}; existing goal retained'
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
    parser.add_argument('--output-root', type=Path, default=ROOT/'output/leader-recordings')
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--hz', type=float, default=50.)
    parser.add_argument('--max-rate-deg-s', type=float, default=60.)
    args = parser.parse_args(argv)
    try:
        if (not 1 <= args.seconds <= 600 or not 10 <= args.hz <= 100
                or not 1 <= args.max_rate_deg_s <= 60):
            raise ValueError('seconds must be 1..600, hz 10..100, max rate 1..60 deg/s')
        report, follower_bus, leader_bus = offline_preflight(args)
        if not (args.inspect or args.execute):
            print(json.dumps(report, indent=2))
            print('OFFLINE CHECK ONLY. No serial device was opened; use --inspect for live alignment checks.')
            return 0
        directory = args.output_root.resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        directory.mkdir(parents=True, exist_ok=False)
        path = directory/('recording.json' if args.execute else 'inspection.json')
        try:
            with ExitStack() as stack:
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
                run_session(follower_bus, leader_bus, report, keyboard,
                            seconds=args.seconds, hz=args.hz, max_rate_deg_s=args.max_rate_deg_s,
                            execute=args.execute, emit=lambda row: stream.write(json.dumps(row)+'\n'))
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
            print(f"Follower jaw closed: {report['fixed_jaw_closed_ok']}. No registers changed.")
        else:
            if report.get('hold_error'):
                raise RuntimeError(report['hold_error'])
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
