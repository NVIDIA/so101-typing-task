# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Cleanup failures must override a successful policy result, without hardware I/O."""
from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from so101_typing.hardware import run_vision_policy_handoff as runner


@pytest.fixture
def completed_run(monkeypatch, tmp_path):
    out = tmp_path/'run'
    monkeypatch.setattr('sys.argv', ['vision-runner', '--out', str(out), '--no-external-camera'])
    profile = SimpleNamespace(q_reset_rad=np.zeros(5))
    state = SimpleNamespace(complete=True, target='NEWTON', typed='NEWTON',
                            clearance_tick=lambda **kwargs: 'target_complete')
    actor = SimpleNamespace(action=lambda *args: np.zeros(5))
    monkeypatch.setattr(runner, '_prepare', lambda *args: (
        actor, profile, np.zeros(5), state, np.zeros(5), {'keyboard_event_node': 'fixture'}))
    cameras = SimpleNamespace(check=lambda: None, sources={'wrist': 'fixture'},
        wrist_rgb_history=lambda previous: (np.zeros((6,120,160)), None, {}))
    monkeypatch.setattr(runner, 'CameraRecorder', lambda *args: nullcontext(cameras))
    keyboard = SimpleNamespace(poll=lambda: ([], False), pressed_codes=set())
    monkeypatch.setattr(runner, 'KeyboardMonitor', lambda *args: nullcontext(keyboard))
    robot = SimpleNamespace(bus=Mock(), disconnect=Mock())
    connect = Mock(return_value=robot)
    monkeypatch.setattr(runner, 'connect_robot', connect)
    monkeypatch.setattr(runner, 'resolve_robot_port', lambda *args: 'fixture')
    monkeypatch.setattr(runner, 'prepare_closed_typing_jaw', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, '_append_telemetry_register', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, '_move_to_fixed_reset_with_retries',
        lambda *args, **kwargs: ({'converged': True}, np.zeros(6)))
    monkeypatch.setattr(runner, '_start_error_deg', lambda *args: 0)
    monkeypatch.setattr(runner, '_read_closed_typing_q_rad', lambda *args: np.zeros(6))
    monkeypatch.setattr(runner, 'vision_observation', lambda *args: np.zeros(45))
    monkeypatch.setattr(runner, '_suppress_terminal_echo', lambda **kwargs: nullcontext())
    monkeypatch.setattr(runner.time, 'sleep', lambda *args: None)
    cleanup = Mock(return_value={'torque_off_verified': True, 'home_converged': True})
    monkeypatch.setattr(runner, '_return_home_and_disable', cleanup)
    return SimpleNamespace(out=out, cleanup=cleanup, robot=robot, connect=connect)


def test_successful_cleanup_preserves_success(completed_run):
    assert runner.main() == 0
    summary = json.loads((completed_run.out/'summary.json').read_text())
    assert summary['status'] == 'completed'
    completed_run.cleanup.assert_called_once()
    completed_run.robot.disconnect.assert_called_once()


@pytest.mark.parametrize('failure', ['home', 'hold', 'torque', 'cleanup_exception', 'disconnect'])
def test_cleanup_failure_overrides_success_and_is_recorded(completed_run, failure):
    if failure in ('home', 'hold'):
        completed_run.cleanup.return_value[failure+'_error'] = 'fixture failure'
    elif failure == 'torque':
        completed_run.cleanup.return_value['torque_off_verified'] = False
    elif failure == 'cleanup_exception':
        completed_run.cleanup.side_effect = RuntimeError('fixture cleanup failure')
    else:
        completed_run.robot.disconnect.side_effect = RuntimeError('fixture disconnect failure')
    with pytest.raises(RuntimeError, match='post-evaluation'):
        runner.main()
    summary = json.loads((completed_run.out/'summary.json').read_text())
    assert summary['status'] == 'cleanup_failed'
    completed_run.cleanup.assert_called_once()
    completed_run.robot.disconnect.assert_called_once()


def test_preflight_does_not_connect_or_require_motor_cleanup(completed_run, monkeypatch):
    monkeypatch.setattr('sys.argv', ['vision-runner', '--out', str(completed_run.out), '--preflight-only'])
    assert runner.main() == 0
    completed_run.connect.assert_not_called()
    completed_run.cleanup.assert_not_called()
    assert json.loads((completed_run.out/'summary.json').read_text())['status'] == 'preflight_passed'
