"""Robot connection, calibration checks and read-only diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
import unittest

import pytest

from so101_typing.hardware import robot as robot_module
from so101_typing.hardware.read_robot_diagnostics import snapshot


# Calibration checks before connecting

def setup_connection(monkeypatch, *, calibrated=True, read_error=False):
    calls = []

    class Bus:
        def connect(self):
            calls.append('connect')

        @property
        def is_calibrated(self):
            calls.append('check_calibration')
            if read_error:
                raise ConnectionError('calibration read failed')
            return calibrated

        def sync_read(self, *args, **kwargs):
            calls.append('read_position')
            return {'shoulder_pan': 0.0}

        def sync_write(self, *args, **kwargs):
            calls.append('write_position')

        def enable_torque(self, *args, **kwargs):
            calls.append('enable_torque')

        def disconnect(self, *, disable_torque):
            assert disable_torque is False
            calls.append('disconnect_without_torque_change')

    bus = Bus()
    follower = SimpleNamespace(calibration={'shoulder_pan': object()}, bus=bus,
                               disconnect=lambda: bus.disconnect(disable_torque=False))

    class Lock:
        def acquire(self):
            calls.append('lock')
            return self

        def release(self):
            calls.append('unlock')

    monkeypatch.setattr(robot_module, 'load_lerobot', lambda: (
        lambda config: follower, lambda **kwargs: SimpleNamespace(**kwargs), None))
    monkeypatch.setattr(robot_module, 'resolve_robot_port', lambda port: '/unused/mock-port')
    monkeypatch.setattr(robot_module, 'RobotBusLock', Lock)
    return calls


@pytest.mark.parametrize('enable_torque', [False, True])
def test_mismatched_calibration_never_commands_robot(monkeypatch, enable_torque):
    calls = setup_connection(monkeypatch, calibrated=False)
    with pytest.raises(ValueError, match='registers do not match'):
        robot_module.connect_robot(robot_id='test', enable_torque=enable_torque)
    assert calls == ['lock', 'connect', 'check_calibration',
                     'disconnect_without_torque_change', 'unlock']


def test_unreadable_calibration_never_commands_robot(monkeypatch):
    calls = setup_connection(monkeypatch, read_error=True)
    with pytest.raises(ConnectionError, match='calibration read failed'):
        robot_module.connect_robot(robot_id='test')
    assert calls == ['lock', 'connect', 'check_calibration',
                     'disconnect_without_torque_change', 'unlock']


@pytest.mark.parametrize('enable_torque', [False, True])
def test_matching_calibration_checked_before_optional_hold(monkeypatch, enable_torque):
    calls = setup_connection(monkeypatch)
    robot = robot_module.connect_robot(robot_id='test', enable_torque=enable_torque)
    expected = ['lock', 'connect', 'check_calibration']
    if enable_torque:
        expected += ['read_position', 'write_position', 'enable_torque']
    assert calls == expected
    robot.disconnect()
    assert calls[-2:] == ['disconnect_without_torque_change', 'unlock']


# Read-only diagnostics

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


# Serial port selection

def test_resolve_robot_port_requires_explicit_port_when_multiple_controllers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports_by_pattern = {
        "/dev/serial/by-id/*": [
            "/dev/serial/by-id/usb-SO101_TEST_A-if00",
            "/dev/serial/by-id/usb-SO101_TEST_B-if00",
        ],
        "/dev/ttyACM*": ["/dev/ttyACM0", "/dev/ttyACM1"],
        "/dev/ttyUSB*": [],
    }
    real_paths = {
        "/dev/serial/by-id/usb-SO101_TEST_A-if00": "/dev/ttyACM0",
        "/dev/serial/by-id/usb-SO101_TEST_B-if00": "/dev/ttyACM1",
    }

    monkeypatch.setattr(robot_module.glob, "glob", lambda pattern: ports_by_pattern[pattern])
    monkeypatch.setattr(robot_module.os.path, "realpath", lambda path: real_paths.get(path, path))

    with pytest.raises(FileNotFoundError, match="multiple robot serial ports"):
        robot_module.resolve_robot_port()


def test_resolve_robot_port_returns_only_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    port = "/dev/serial/by-id/usb-SO101_TEST_B-if00"
    ports_by_pattern = {
        "/dev/serial/by-id/*": [port],
        "/dev/ttyACM*": ["/dev/ttyACM1"],
        "/dev/ttyUSB*": [],
    }

    monkeypatch.setattr(robot_module.glob, "glob", lambda pattern: ports_by_pattern[pattern])
    monkeypatch.setattr(robot_module.os.path, "realpath", lambda path: "/dev/ttyACM1" if path == port else path)

    assert robot_module.resolve_robot_port() == port
