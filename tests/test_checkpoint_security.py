# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint compatibility and rejection of executable pickle payloads."""

import os
from pathlib import Path
import pickle
import shlex

import pytest
import torch

from so101_typing.checkpoints import load_runner_checkpoint
from so101_typing.hardware.fixed_cartesian_policy import FixedCartesianPolicy

ROOT = Path(__file__).resolve().parents[1]


class _WriteMarker:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        # This harmless payload creates a marker only if a loader executes pickle.
        return os.system, (f'touch {shlex.quote(str(self.marker))}',)


class _UnsafeThirdPartyRunner:
    def load(self, path, **kwargs):
        # Mirrors the explicit unsafe flag in the pinned RSL-RL loader.
        return torch.load(path, map_location='cpu', weights_only=False), kwargs


@pytest.fixture(autouse=True)
def neutral_load_environment(monkeypatch):
    monkeypatch.setenv('TORCH_FORCE_WEIGHTS_ONLY_LOAD', '0')
    monkeypatch.delenv('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', raising=False)


@pytest.fixture
def malicious_checkpoint(tmp_path):
    marker = tmp_path / 'pickle-executed'
    path = tmp_path / 'untrusted.pt'
    torch.save({'actor_state_dict': {}, 'payload': _WriteMarker(marker)}, path)
    return path, marker


def test_hardware_policy_rejects_executable_pickle(malicious_checkpoint):
    path, marker = malicious_checkpoint
    with pytest.raises(pickle.UnpicklingError):
        FixedCartesianPolicy(path)
    assert not marker.exists()


def test_runner_rejects_executable_pickle(malicious_checkpoint, monkeypatch):
    path, marker = malicious_checkpoint
    monkeypatch.setenv('TORCH_FORCE_WEIGHTS_ONLY_LOAD', '0')
    with pytest.raises(pickle.UnpicklingError):
        load_runner_checkpoint(_UnsafeThirdPartyRunner(), path)
    assert not marker.exists()
    assert os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] == '1'


@pytest.mark.parametrize('model', ('newton_vision/newton_ckpt.pt', 'isaac_lab_state/isaac_lab_ckpt.pt'))
def test_bundled_checkpoint_safe_load_preserves_tensor_state(model, monkeypatch):
    monkeypatch.setenv('TORCH_FORCE_WEIGHTS_ONLY_LOAD', '0')
    path = ROOT / 'models' / model
    expected = torch.load(path, map_location='cpu', weights_only=True)
    actual, kwargs = load_runner_checkpoint(
        _UnsafeThirdPartyRunner(), path, load_cfg={'actor': True, 'optimizer': False})
    assert kwargs == {'load_cfg': {'actor': True, 'optimizer': False}}
    assert actual['iter'] == expected['iter']
    for group in ('actor_state_dict', 'critic_state_dict'):
        assert actual[group].keys() == expected[group].keys()
        for key, tensor in expected[group].items():
            assert torch.equal(actual[group][key], tensor), key
