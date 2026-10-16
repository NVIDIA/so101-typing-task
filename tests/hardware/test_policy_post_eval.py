"""Post-policy homing and torque-off are ordered and verified."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from so101_typing.hardware.constants import ARM_JOINT_NAMES
from so101_typing.hardware.run_fixed_cartesian_policy_handoff import (
    _load_post_eval_home,
    _return_home_and_disable,
    _validate_policy_keyboard_device,
)
from so101_typing.hardware.policy_runtime import PolicyContractError

NAMES = (*ARM_JOINT_NAMES, 'gripper')


class Bus:
    def __init__(self):
        self.calls = []
        self.torque = dict.fromkeys(NAMES, 1)

    def sync_read(self, register, motors, **kwargs):
        self.calls.append(('read', register))
        if register == 'Present_Position':
            return dict.fromkeys(motors, 2048)
        if register == 'Torque_Enable':
            return self.torque.copy()
        raise AssertionError(register)

    def sync_write(self, register, values, **kwargs):
        self.calls.append(('write', register))
        if register == 'Torque_Enable':
            self.torque.update(values)


def test_home_then_torque_off(monkeypatch):
    bus = Bus()
    target = np.asarray([1., 2., 3., 4., 5.])
    calls = bus.calls
    monkeypatch.setattr('so101_typing.hardware.run_fixed_cartesian_policy_handoff.smooth_move_arm',
                        lambda robot, q, **kwargs: calls.append(('move', tuple(q))))
    monkeypatch.setattr('so101_typing.hardware.run_fixed_cartesian_policy_handoff.read_arm_q_deg',
                        lambda robot: target.copy())
    report = _return_home_and_disable(SimpleNamespace(bus=bus), target, 6.)
    assert report['home_converged'] and report['torque_off_verified']
    assert calls.index(('move', tuple(target))) < calls.index(('write', 'Torque_Enable'))
    assert bus.torque == dict.fromkeys(NAMES, 0)


def test_motion_error_still_disables_torque(monkeypatch):
    bus = Bus()
    def fail_move(*args, **kwargs):
        raise RuntimeError('motion failed')
    monkeypatch.setattr('so101_typing.hardware.run_fixed_cartesian_policy_handoff.smooth_move_arm', fail_move)
    report = _return_home_and_disable(SimpleNamespace(bus=bus), np.zeros(5), 6.)
    assert 'motion failed' in report['home_error']
    assert report['torque_off_verified']
    assert bus.torque == dict.fromkeys(NAMES, 0)


def test_saved_home_requires_matching_robot_calibration(monkeypatch, tmp_path):
    calibration = tmp_path / 'calibration.json'
    calibration.write_text('{"sample": 1}')
    monkeypatch.setattr('so101_typing.hardware.run_fixed_cartesian_policy_handoff.require_calibration',
                        lambda robot_id: calibration)
    path = tmp_path / 'home.json'
    data = {'schema_version': 1, 'kind': 'so101_post_eval_home', 'robot_id': 'robot',
            'follower_calibration_sha256': hashlib.sha256(calibration.read_bytes()).hexdigest(),
            'arm_joint_names': list(ARM_JOINT_NAMES),
            'pose_deg': dict(zip(ARM_JOINT_NAMES, [1, 2, 3, 4, 5], strict=True))}
    path.write_text(json.dumps(data))
    np.testing.assert_allclose(_load_post_eval_home(path, 'robot'), [1, 2, 3, 4, 5])
    calibration.write_text('{"sample": 2}')
    with pytest.raises(PolicyContractError, match='different follower calibration'):
        _load_post_eval_home(path, 'robot')


def test_wrong_keyboard_rejected_before_robot_connection(tmp_path):
    expected = tmp_path / 'typing-keyboard'
    wrong = tmp_path / 'desktop-keyboard'
    expected.touch()
    wrong.touch()
    setup = tmp_path / 'setup.json'
    setup.write_text(json.dumps({'typing_keyboard_device': str(expected)}))
    assert _validate_policy_keyboard_device(str(expected), setup) == str(expected)
    with pytest.raises(PolicyContractError, match='differs from saved typing keyboard'):
        _validate_policy_keyboard_device(str(wrong), setup)
