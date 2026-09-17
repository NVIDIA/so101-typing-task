"""The diagnostic must remain read-only, including its failure cleanup."""
from dataclasses import dataclass, asdict
import unittest
from scripts.so101_homing.read_robot_diagnostics import snapshot


@dataclass
class Calibration:
    id: int = 1
    homing_offset: int = 4
    range_min: int = 100
    range_max: int = 3000


class ReadOnlyBus:
    motors = {'shoulder_pan': object()}
    is_connected = False
    disconnected_with = None

    def connect(self):
        self.is_connected = True

    def read_calibration(self):
        return {'shoulder_pan': Calibration()}

    def read(self, register, motor, *, normalize):
        assert normalize is False
        return 123

    def disconnect(self, *, disable_torque):
        assert disable_torque is False
        self.disconnected_with = disable_torque
        self.is_connected = False

    # Any write, configure or torque method is deliberately absent.


class DiagnosticsTests(unittest.TestCase):
    def test_mismatch_is_reported_without_writing_to_motor(self):
        bus = ReadOnlyBus()
        report = snapshot(bus, {'shoulder_pan': asdict(Calibration(homing_offset=999))})
        self.assertFalse(report['calibration_matches_file'])
        self.assertEqual(report['calibration_mismatches']['shoulder_pan']['homing_offset'],
                         {'file':999, 'motor':4})
        self.assertEqual(report['register_read_errors'], {})
        self.assertIs(bus.disconnected_with, False)

    def test_calibration_read_failure_closes_without_disabling_torque(self):
        class BrokenBus(ReadOnlyBus):
            def read_calibration(self):
                raise RuntimeError('read failed')
        bus = BrokenBus()
        with self.assertRaisesRegex(RuntimeError, 'read failed'):
            snapshot(bus, {})
        self.assertIs(bus.disconnected_with, False)
