"""Calibration mismatch must be detected before direct-bus motion or torque writes."""
from types import SimpleNamespace

import pytest

from so101_typing.hardware import robot as module


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

    monkeypatch.setattr(module, 'load_lerobot', lambda: (
        lambda config: follower, lambda **kwargs: SimpleNamespace(**kwargs), None))
    monkeypatch.setattr(module, 'resolve_robot_port', lambda port: '/unused/mock-port')
    monkeypatch.setattr(module, 'RobotBusLock', Lock)
    return calls


@pytest.mark.parametrize('enable_torque', [False, True])
def test_mismatched_calibration_never_commands_robot(monkeypatch, enable_torque):
    calls = setup_connection(monkeypatch, calibrated=False)
    with pytest.raises(ValueError, match='registers do not match'):
        module.connect_robot(robot_id='test', enable_torque=enable_torque)
    assert calls == ['lock', 'connect', 'check_calibration',
                     'disconnect_without_torque_change', 'unlock']


def test_unreadable_calibration_never_commands_robot(monkeypatch):
    calls = setup_connection(monkeypatch, read_error=True)
    with pytest.raises(ConnectionError, match='calibration read failed'):
        module.connect_robot(robot_id='test')
    assert calls == ['lock', 'connect', 'check_calibration',
                     'disconnect_without_torque_change', 'unlock']


@pytest.mark.parametrize('enable_torque', [False, True])
def test_matching_calibration_checked_before_optional_hold(monkeypatch, enable_torque):
    calls = setup_connection(monkeypatch)
    robot = module.connect_robot(robot_id='test', enable_torque=enable_torque)
    expected = ['lock', 'connect', 'check_calibration']
    if enable_torque:
        expected += ['read_position', 'write_position', 'enable_torque']
    assert calls == expected
    robot.disconnect()
    assert calls[-2:] == ['disconnect_without_torque_change', 'unlock']
