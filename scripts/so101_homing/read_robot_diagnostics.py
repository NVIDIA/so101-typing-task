"""Read calibration and controller registers without changing motor state.

This does not call the follower's connect/configure methods, write registers,
load a policy, enable/disable torque, or issue position commands.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

from .hardware_lock import RobotBusLock
from .robot import load_lerobot, require_calibration

REGISTERS = (
    'Present_Position', 'Goal_Position', 'P_Coefficient', 'I_Coefficient',
    'D_Coefficient', 'Operating_Mode', 'Torque_Enable', 'Present_Voltage',
    'Present_Temperature', 'Present_Load', 'Present_Current',
)


def snapshot(bus, expected):
    """Only ping/read operations; close the port without changing torque."""
    try:
        bus.connect()
        actual = {name: asdict(value) for name, value in bus.read_calibration().items()}
        mismatches = {}
        for name in sorted(set(expected) | set(actual)):
            for field in ('id', 'homing_offset', 'range_min', 'range_max'):
                wanted = expected.get(name, {}).get(field)
                observed = actual.get(name, {}).get(field)
                if wanted != observed:
                    mismatches.setdefault(name, {})[field] = {'file': wanted, 'motor': observed}
        registers, errors = {}, {}
        for register in REGISTERS:
            registers[register] = {}
            for motor in bus.motors:
                try:
                    registers[register][motor] = bus.read(register, motor, normalize=False)
                except (ConnectionError, RuntimeError, OSError, ValueError) as error:
                    errors.setdefault(register, {})[motor] = str(error)
        return {'motor_calibration': actual, 'calibration_mismatches': mismatches,
                'calibration_matches_file': not mismatches, 'registers_raw': registers,
                'register_read_errors': errors}
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        calibration = require_calibration(args.robot_id)
        raw = calibration.read_bytes()
        expected = json.loads(raw)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        out = args.out or Path('output/robot-diagnostics') / f'{stamp}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        robot_type, config_type, _ = load_lerobot()
        robot = robot_type(config_type(port=args.port, id=args.robot_id, use_degrees=True))
        # Reserve output before accessing hardware; never overwrite an earlier record.
        with out.open('x') as stream:
            report = {'schema_version': 1, 'kind': 'so101_read_only_diagnostics',
                      'robot_id': args.robot_id, 'port': args.port,
                      'calibration_file': str(calibration),
                      'calibration_file_sha256': hashlib.sha256(raw).hexdigest(),
                      'motor_register_writes': False,
                      'limits': 'Raw controller values. File/register agreement does not prove simulator alignment.'}
            try:
                with RobotBusLock():
                    report.update(snapshot(robot.bus, expected))
            except Exception as error:
                report['error'] = str(error)
                raise
            finally:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        print(f"Calibration file matches motor registers: {report['calibration_matches_file']}")
        print(f"Individual register read errors: {sum(len(v) for v in report['register_read_errors'].values())}")
        print(f'Diagnostic report: {out.resolve()}')
        print('No motor settings, torque state, or position commands were changed.')
        return 0 if report['calibration_matches_file'] and not report['register_read_errors'] else 1
    except (OSError, ValueError, RuntimeError, ConnectionError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
