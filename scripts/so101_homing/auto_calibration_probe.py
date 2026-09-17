"""Supervised, bounded automatic endpoint measurement; never installs calibration.

Uses the installed LeRobot SDK without patching it. Unlike the full PR-3282
workflow, this probes one of the four mapping joints at a time, retains homing
and PID, and checks stops against an existing full-range calibration. It is an
endpoint-repeatability experiment, not first-time or model-zero calibration.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import time

from .hardware_lock import RobotBusLock

ARM = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
JOINTS = ARM[:4]
DEG_PER_TICK = 360 / 4095
# Every writable register touched by this runner, plus invariants and health.
CONFIG = ('Homing_Offset', 'Min_Position_Limit', 'Max_Position_Limit',
          'Operating_Mode', 'Torque_Limit', 'Max_Torque_Limit',
          'P_Coefficient', 'I_Coefficient', 'D_Coefficient', 'Phase',
          'Acceleration', 'Goal_Time', 'Goal_Velocity', 'Lock')
REGISTERS = (*CONFIG, 'Torque_Enable', 'Goal_Position', 'Present_Position',
             'Present_Velocity', 'Present_Load', 'Present_Current',
             'Present_Voltage', 'Present_Temperature', 'Status')


@dataclass(frozen=True)
class Settings:
    repeats: int = 3
    speed_raw: int = 100
    torque_limit_raw: int = 200
    endpoint_margin_deg: float = 12.
    timeout_s: float = 55.
    period_s: float = .04
    settle_samples: int = 7
    repeatability_deg: float = 1.

    def validate(self):
        if not 3 <= self.repeats <= 5:
            raise ValueError('use 3 to 5 repeats')
        if not 30 <= self.speed_raw <= 150 or not 50 <= self.torque_limit_raw <= 300:
            raise ValueError('speed must be 30..150 and torque limit 50..300 in raw units')
        if not 0 < self.endpoint_margin_deg <= 15 or not 5 <= self.timeout_s <= 90:
            raise ValueError('endpoint margin must be (0,15] degrees; timeout 5..90 seconds')


def read(bus, register, motor):
    return int(bus.read(register, motor, normalize=False, num_retry=0))


def write(bus, register, motor, value):
    bus.write(register, motor, int(value), normalize=False, num_retry=0)


def checked_write(bus, register, motor, value):
    write(bus, register, motor, value)
    if read(bus, register, motor) != int(value):
        raise RuntimeError(f'{motor}: {register} write/read mismatch')


def snapshot(bus):
    return {motor: {reg: read(bus, reg, motor) for reg in REGISTERS} for motor in bus.motors}


def bounds(calibration, settings):
    margin = math.ceil(settings.endpoint_margin_deg / DEG_PER_TICK)
    lo, hi = calibration['range_min'], calibration['range_max']
    if lo - margin < 0 or hi + margin > 4095:
        raise ValueError('expanded endpoints cross encoder wrap; this bounded probe cannot handle this joint')
    return lo - margin, hi + margin


def validate_baseline(before, calibration, joint, settings):
    settings.validate()
    if joint not in JOINTS or set(before) != set(calibration):
        raise ValueError('expected the calibrated six-motor follower and a supported joint')
    bounds(calibration[joint], settings)
    for motor, row in before.items():
        for register, key in (('Homing_Offset', 'homing_offset'),
                              ('Min_Position_Limit', 'range_min'), ('Max_Position_Limit', 'range_max')):
            if row[register] != calibration[motor][key]:
                raise ValueError(f'{motor}: file/register mismatch for {register}')
        if row['Operating_Mode'] != 0 or row['Torque_Enable'] not in (0, 1):
            raise ValueError(f'{motor}: expected position mode and ordinary torque state')
        if row['Phase'] & 0x10:
            raise ValueError(f'{motor}: multi-turn phase is unsupported by this probe')
        if abs(row['Present_Velocity']) > 3 or row['Status']:
            raise ValueError(f'{motor}: motor is moving or reports a fault')
        if row['Present_Temperature'] >= 50 or not 100 <= row['Present_Voltage'] <= 140:
            raise ValueError(f'{motor}: temperature/voltage outside the probe limits')
        if calibration[motor]['drive_mode'] != 0:
            raise ValueError(f'{motor}: inverted drive convention is unsupported')
        if motor in ARM and not calibration[motor]['range_min'] <= row['Present_Position'] <= calibration[motor]['range_max']:
            raise ValueError(f'{motor}: starting position is outside the existing calibration limits')


def summarize(runs, original, settings):
    """Both ends must repeat; midpoint alone could hide span changes."""
    if len(runs) != settings.repeats:
        raise ValueError('incomplete automatic endpoint repetitions')
    lows = [r['range_min'] for r in runs]
    highs = [r['range_max'] for r in runs]
    mids = [(lo + hi) / 2 for lo, hi in zip(lows, highs)]
    spread = {name: (max(values) - min(values)) * DEG_PER_TICK
              for name, values in (('minimum', lows), ('maximum', highs), ('midpoint', mids))}
    stable = max(spread.values()) <= settings.repeatability_deg
    # Endpoint medians stay in the original encoder frame; homing never changes.
    lo, hi = sorted(lows)[len(lows)//2], sorted(highs)[len(highs)//2]
    old_mid, new_mid = (original['range_min'] + original['range_max']) / 2, (lo + hi) / 2
    shift = (old_mid - new_mid) * DEG_PER_TICK
    return {'repeatable': stable, 'spread_deg': spread, 'candidate_range_min': lo,
            'candidate_range_max': hi, 'original_midpoint_ticks': old_mid,
            'candidate_midpoint_ticks': new_mid, 'q_auto_minus_q_manual_deg': shift,
            'conversion': 'q_auto_deg = q_manual_deg + q_auto_minus_q_manual_deg; homing unchanged',
            'model_zero_alignment_established': False}


def require_fresh_camera(path, clock=time.monotonic):
    if path is None:
        return
    state = json.loads(path.read_text())
    age = clock() - state['host_monotonic_received_s']
    if state.get('status') != 'streaming' or not 0 <= age <= .8:
        raise RuntimeError(f'camera stream unavailable/stale ({age:.3f}s); motion stopped')


class Probe:
    def __init__(self, bus, before, calibration, joint, settings, emit,
                 clock=time.monotonic, sleep=time.sleep, check_camera=lambda: None):
        self.bus, self.before, self.calibration, self.joint = bus, before, calibration, joint
        self.settings, self.emit, self.clock, self.sleep = settings, emit, clock, sleep
        self.touched = set()
        self.held = {}
        self.check_camera = check_camera
        self.velocity_frame_shift = None
        self.last_position = None

    def switch_mode(self, mode):
        checked_write(self.bus, 'Operating_Mode', self.joint, mode)
        # A register acknowledgement does not mean Present_Position has already
        # refreshed in the new mode. Immediate reads returned the old frame on
        # hardware; wait before using any position reading or enabling torque.
        self.sleep(.02)

    def hold_arm(self):
        # Refresh after the operator prompt: never enable torque toward a stale goal.
        positions = {m: read(self.bus, 'Present_Position', m) for m in ARM}
        for m, q in positions.items():
            if not self.calibration[m]['range_min'] <= q <= self.calibration[m]['range_max']:
                raise ValueError(f'{m}: pose outside original limits after operator confirmation')
        for m in ARM:
            self.touched.add(m)
            checked_write(self.bus, 'Goal_Position', m, positions[m])
            checked_write(self.bus, 'Torque_Enable', m, 1)
        self.held = positions

    def detect_velocity_convention(self):
        """Check mode-0/1/0 readings without a velocity command or homing write.

        Some STS firmware ignores Homing_Offset in velocity mode. Accept only
        identity or the documented position-mode homing correction, and require
        the flanking position-mode samples to demonstrate a stationary pose.
        """
        m = self.joint
        shifts = []
        h = self.before[m]['Homing_Offset']
        for repeat in range(3):
            self.check_camera()
            # Let the position hold settle between brief torque releases; without
            # this, consecutive trials can sample the same gravity-driven drift.
            self.sleep(.4)
            checked_write(self.bus, 'Goal_Velocity', m, 0)
            checked_write(self.bus, 'Torque_Enable', m, 0)
            checked_write(self.bus, 'Lock', m, 0)
            q0 = read(self.bus, 'Present_Position', m)
            self.switch_mode(1)
            q1 = read(self.bus, 'Present_Position', m)
            self.switch_mode(0)
            q2 = read(self.bus, 'Present_Position', m)
            checked_write(self.bus, 'Goal_Position', m, q2)
            checked_write(self.bus, 'Torque_Enable', m, 1)
            # Compare circular encoder coordinates, not artificial wrap jumps.
            circular = lambda delta: (delta+2048) % 4096 - 2048
            drift = circular(q2-q0)
            reference = q0+drift/2
            options = {shift: abs(circular(q1+shift-reference)) for shift in (0,-h)}
            chosen = min(options, key=options.get)
            self.emit({'event':'mode_coordinate_check', 'repeat':repeat+1,
                       'position_mode_before_ticks':q0, 'velocity_mode_ticks':q1,
                       'position_mode_after_ticks':q2, 'homing_offset_ticks':h,
                       'physical_drift_ticks':drift, 'selected_shift_ticks':chosen,
                       'residual_ticks':options[chosen]})
            if abs(drift)>4 or options[chosen]>4:
                raise RuntimeError('mode convention check inconclusive: physical drift or unexpected encoder frame')
            shifts.append(chosen)
            self.last_position = q2
        if len(set(shifts)) != 1:
            raise RuntimeError('velocity-mode coordinate convention did not repeat')
        self.velocity_frame_shift = shifts[0]
        print(f'Velocity-mode read conversion: position-mode ticks = velocity-mode ticks + {shifts[0]} (unwrap near previous pose).', flush=True)
        return {'velocity_to_position_shift_ticks':shifts[0], 'repeats':3,
                'extra_model_joint_offset':False}

    def velocity_position(self):
        if self.velocity_frame_shift is None or self.last_position is None:
            raise RuntimeError('velocity-mode coordinate convention has not been measured')
        raw = read(self.bus, 'Present_Position', self.joint)
        q = raw + self.velocity_frame_shift
        q += 4096 * round((self.last_position-q)/4096)
        self.last_position = q
        return raw, q

    def seek(self, direction):
        m, cfg = self.joint, self.settings
        lo, hi = bounds(self.calibration[m], cfg)
        expected = self.calibration[m]['range_min' if direction < 0 else 'range_max']
        margin = math.ceil(cfg.endpoint_margin_deg / DEG_PER_TICK)
        _, start = self.velocity_position()
        started = previous_time = self.clock()
        previous = start
        window = deque(maxlen=cfg.settle_samples)
        endpoint = None
        try:
            self.check_camera()
            checked_write(self.bus, 'Goal_Velocity', m, direction * cfg.speed_raw)
            while self.clock() - started < cfg.timeout_s:
                self.sleep(cfg.period_s)
                self.check_camera()
                now = self.clock()
                raw, q = self.velocity_position()
                vel = read(self.bus, 'Present_Velocity', m)
                status = read(self.bus, 'Status', m)
                temp = read(self.bus, 'Present_Temperature', m)
                moving = read(self.bus, 'Moving', m)
                self.emit({'event': 'sample', 'joint': m, 'direction': direction,
                           'elapsed_s': now-started, 'q_ticks': q, 'velocity_raw': vel,
                           'velocity_mode_position_raw':raw,
                           'load_raw':read(self.bus, 'Present_Load', m),
                           'current_raw':read(self.bus, 'Present_Current', m),
                           'status': status, 'temperature_c': temp, 'moving': moving})
                if now - previous_time > .3:
                    raise RuntimeError('control gap exceeded 300 ms; endpoint rejected')
                if status or temp >= 50:
                    raise RuntimeError(f'{m}: controller fault/temperature limit; endpoint rejected')
                if not lo < q < hi or abs(q-previous) > 50:
                    raise RuntimeError(f'{m}: exceeded bounded travel or jumped/wrapped; endpoint rejected')
                if direction * (q-start) < -15:
                    raise RuntimeError(f'{m}: moved opposite to commanded direction')
                for other in ARM:
                    if other != m and abs(read(self.bus, 'Present_Position', other)-self.held[other]) * DEG_PER_TICK > 5:
                        raise RuntimeError(f'{other}: holding joint drifted by more than 5 degrees')
                window.append((q, vel, moving))
                if (now-started >= .75 and len(window) == cfg.settle_samples and
                        max(p[0] for p in window)-min(p[0] for p in window) <= 3 and
                        all(abs(p[1]) <= 3 and p[2] == 0 for p in window)):
                    if abs(q-expected) > margin or direction*(q-start) < 20:
                        raise RuntimeError(f'{m}: early stall, not a verified endpoint (position {q})')
                    endpoint = sorted(p[0] for p in window)[len(window)//2]
                    break
                previous, previous_time = q, now
            if endpoint is None:
                raise RuntimeError(f'{m}: endpoint timeout; no calibration accepted')
            return endpoint
        finally:
            # Stop immediately, including on read/communication failure or Ctrl-C.
            # Full cleanup also disables this motor before restoring position mode.
            try:
                write(self.bus, 'Goal_Velocity', m, 0)
            except BaseException:
                # Velocity-stop packet failure must trigger a second kind of stop.
                write(self.bus, 'Torque_Enable', m, 0)
                raise

    def run(self):
        self.check_camera()
        self.hold_arm()
        self.detect_velocity_convention()
        m = self.joint
        lo, hi = bounds(self.calibration[m], self.settings)
        # Only this motor changes mode/limits. Homing, Phase and PID never change.
        checked_write(self.bus, 'Torque_Enable', m, 0)
        checked_write(self.bus, 'Lock', m, 0)
        checked_write(self.bus, 'Goal_Velocity', m, 0)
        checked_write(self.bus, 'Min_Position_Limit', m, lo)
        checked_write(self.bus, 'Max_Position_Limit', m, hi)
        checked_write(self.bus, 'Torque_Limit', m,
                      min(self.settings.torque_limit_raw, self.before[m]['Torque_Limit']))
        self.switch_mode(1)
        checked_write(self.bus, 'Torque_Enable', m, 1)
        runs = []
        for i in range(self.settings.repeats):
            _, q = self.velocity_position()
            midpoint = (self.calibration[m]['range_min']+self.calibration[m]['range_max'])/2
            first = 1 if q < midpoint else -1  # Start toward the farther end.
            ends = {}
            for direction in (first, -first):
                print(f'{m}: repeat {i+1}/{self.settings.repeats}, seeking {"minimum" if direction < 0 else "maximum"}', flush=True)
                ends['range_min' if direction < 0 else 'range_max'] = self.seek(direction)
                self.emit({'event': 'endpoint', 'repeat': i+1, 'direction': direction, 'position_ticks': ends['range_min' if direction < 0 else 'range_max']})
                self.sleep(.15)
            if ends['range_max'] <= ends['range_min']:
                raise RuntimeError('endpoint order invalid')
            runs.append(ends)
        self.back_off()
        return runs

    def back_off(self):
        """On success only, leave the stop slowly before restoring original limits."""
        m, cfg = self.joint, self.settings
        _, q = self.velocity_position()
        interior = math.ceil(3 / DEG_PER_TICK)
        target = min(max(q, self.calibration[m]['range_min']+interior),
                     self.calibration[m]['range_max']-interior)
        checked_write(self.bus, 'Torque_Enable', m, 0)
        checked_write(self.bus, 'Goal_Position', m, q)
        self.switch_mode(0)
        checked_write(self.bus, 'Goal_Velocity', m, cfg.speed_raw)
        checked_write(self.bus, 'Torque_Enable', m, 1)
        started = self.clock()
        goal = q
        while self.clock()-started < 8:
            self.check_camera()
            current = read(self.bus, 'Present_Position', m)
            if read(self.bus, 'Status', m) or read(self.bus, 'Present_Temperature', m) >= 50 or abs(current-goal) > 30:
                raise RuntimeError('endpoint backoff fault or tracking error')
            for other in ARM:
                if other != m and abs(read(self.bus, 'Present_Position', other)-self.held[other]) * DEG_PER_TICK > 5:
                    raise RuntimeError(f'{other}: holding joint drift during backoff')
            if goal == target and abs(current-target) <= 3:
                self.emit({'event': 'backoff_complete', 'joint': m, 'q_ticks': current})
                return
            step = max(1, int(cfg.speed_raw * cfg.period_s / 2))
            goal += max(-step, min(step, target-goal))
            write(self.bus, 'Goal_Position', m, goal)
            self.sleep(cfg.period_s)
        raise RuntimeError('endpoint backoff timed out')

    def restore(self):
        """Restore configuration; hold fresh goals, never replay stale pre-test goals.

        Torque remains ON on arm joints after a successful cleanup, including
        when it was originally OFF. This avoids dropping the unsupported arm.
        A motor that cannot be restored/read is left torque-off where possible.
        """
        errors = {}
        # Deal with the potentially active velocity-mode joint before any others.
        for m in (self.joint, *(name for name in ARM if name != self.joint)):
            if m not in self.touched:
                continue
            try:
                if m == self.joint:
                    write(self.bus, 'Goal_Velocity', m, 0)
                    checked_write(self.bus, 'Torque_Enable', m, 0)
                    checked_write(self.bus, 'Lock', m, 0)
                    # Restore mode before restoring the position-mode speed value.
                    self.switch_mode(self.before[m]['Operating_Mode'])
                    for reg in ('Min_Position_Limit', 'Max_Position_Limit', 'Torque_Limit'):
                        checked_write(self.bus, reg, m, self.before[m][reg])
                    checked_write(self.bus, 'Goal_Velocity', m, self.before[m]['Goal_Velocity'])
                    checked_write(self.bus, 'Lock', m, self.before[m]['Lock'])
                    q = read(self.bus, 'Present_Position', m)
                    if not self.before[m]['Min_Position_Limit'] <= q <= self.before[m]['Max_Position_Limit']:
                        raise RuntimeError('measured endpoint outside original limits; support this joint; torque stays off')
                    checked_write(self.bus, 'Goal_Position', m, q)
                    checked_write(self.bus, 'Torque_Enable', m, 1)
                else:
                    q = read(self.bus, 'Present_Position', m)
                    checked_write(self.bus, 'Goal_Position', m, q)
                for reg in CONFIG:
                    if read(self.bus, reg, m) != self.before[m][reg]:
                        raise RuntimeError(f'{reg} not restored')
            except BaseException as exc:
                errors[m] = str(exc)
                try:
                    write(self.bus, 'Torque_Enable', m, 0)
                except BaseException as stop_error:
                    errors[m] += '; torque-off failed: '+str(stop_error)
        return {'configuration_restored': not errors, 'errors': errors,
                'shutdown': 'Fresh measured goals with arm torque ON; failed joints torque OFF where possible; gripper untouched.'}


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--joint', choices=JOINTS, default='elbow_flex')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--speed-raw', type=int, default=100)
    parser.add_argument('--torque-limit-raw', type=int, default=200)
    parser.add_argument('--output-root', type=Path, default=Path('output/auto-calibration-comparison'))
    parser.add_argument('--camera-state', type=Path, help='Optional live RGB heartbeat; stale/missing frames abort motion')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--snapshot', action='store_true', help='read-only controller backup')
    mode.add_argument('--execute', action='store_true', help='supervised automatic endpoint probe; no calibration installation')
    mode.add_argument('--check-mode', action='store_true', help='check encoder mode conventions without endpoint sweeps')
    args = parser.parse_args()
    from .robot import require_calibration, load_lerobot
    settings = Settings(repeats=args.repeats, speed_raw=args.speed_raw, torque_limit_raw=args.torque_limit_raw)
    try:
        path = require_calibration(args.robot_id)
        raw = path.read_bytes()
        calibration = json.loads(raw)
        settings.validate()
        bounds(calibration[args.joint], settings)
        moving = args.execute or args.check_mode
        if not args.snapshot and not moving:
            print(json.dumps({'mode': 'offline_plan', 'joint': args.joint, 'settings': asdict(settings),
                'bounded_ticks': bounds(calibration[args.joint], settings),
                'calibration_sha256': hashlib.sha256(raw).hexdigest(),
                'candidate_only': True, 'hardware_access': False}, indent=2))
            return 0
        if moving:
            if args.check_mode:
                print('Stationary mode-0/1/0 encoder check. No velocity command; torque briefly releases on the selected joint.')
                input('With the arm supported and operator at power, press Enter to start: ')
            else:
                print(f'Automatic endpoint probe: {args.joint}, {settings.repeats} repeats. No policy or calibration installation.')
                print('Clear the keyboard and the selected joint\'s complete sweep; arrange the arm so table/clamp/cables do not obstruct it.')
                print('Keep motor power within reach. Ctrl-C aborts. Other arm joints hold position; normal exit keeps arm torque ON.')
                input('With the arm supported and the full sweep clear, press Enter to start: ')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        out = args.output_root.resolve()/f'{stamp}-{args.joint}'
        out.mkdir(parents=True, exist_ok=False)
        (out/'original-calibration.json').write_bytes(raw)
        report = {'schema_version': 1, 'status': 'started', 'robot_id': args.robot_id,
            'port': args.port, 'joint': args.joint, 'settings': asdict(settings),
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'calibration_sha256': hashlib.sha256(raw).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'calibration_installed': False, 'register_writes_requested': bool(moving),
            'mode': 'stationary_encoder_convention_check' if args.check_mode else 'endpoint_probe' if args.execute else 'snapshot'}
        report['camera_state'] = str(args.camera_state.resolve()) if args.camera_state else None
        robot_type, config_type, _ = load_lerobot()
        robot = robot_type(config_type(port=args.port, id=args.robot_id, use_degrees=True))
        probe = None
        exit_code = 0
        print(f'Output: {out}', flush=True)
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'signal {signum}')
        old_signal = signal.signal(signal.SIGTERM, interrupt)
        with RobotBusLock(), (out/'events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps(event)+'\n'); stream.flush()
            try:
                robot.bus.connect()
                before = snapshot(robot.bus)
                save(out/'controller-before.json', before)
                validate_baseline(before, calibration, args.joint, settings)
                if moving:
                    probe = Probe(robot.bus, before, calibration, args.joint, settings, emit,
                                  check_camera=lambda: require_fresh_camera(args.camera_state))
                    if args.check_mode:
                        probe.hold_arm()
                        report['mode_convention'] = probe.detect_velocity_convention()
                        report['status'] = 'mode_convention_verified'
                    else:
                        runs = probe.run()
                        report['velocity_to_position_shift_ticks'] = probe.velocity_frame_shift
                        report['runs'] = runs
                        report['comparison'] = summarize(runs, calibration[args.joint], settings)
                        report['status'] = 'repeatable_candidate' if report['comparison']['repeatable'] else 'not_repeatable'
                        if not report['comparison']['repeatable']:
                            exit_code = 1
                else:
                    report['status'] = 'read_only_snapshot'
            except BaseException as exc:
                report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            finally:
                if probe is not None:
                    report['restoration'] = probe.restore()
                    if not report['restoration']['configuration_restored']:
                        report['status'] = 'restore_failed'
                        exit_code = 1
                if robot.bus.is_connected:
                    try:
                        save(out/'controller-after.json', snapshot(robot.bus))
                    except Exception as exc:
                        report['post_snapshot_error'] = str(exc)
                        exit_code = 1
                    finally:
                        robot.bus.disconnect(disable_torque=False)
                report['original_calibration_file_unchanged'] = path.read_bytes() == raw
                if not report['original_calibration_file_unchanged']:
                    report['status'] = 'calibration_file_changed'
                    exit_code = 1
                report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
                save(out/'report.json', report)
                signal.signal(signal.SIGTERM, old_signal)
        if report['status'] == 'repeatable_candidate' and exit_code == 0:
            candidate = json.loads(raw)
            for key in ('range_min', 'range_max'):
                candidate[args.joint][key] = report['comparison']['candidate_'+key]
            save(out/'candidate-calibration-NOT-INSTALLED.json', candidate)
        print(json.dumps(report, indent=2))
        print(f'Report: {out/"report.json"}')
        return exit_code
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
