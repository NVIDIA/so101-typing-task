# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise exported CPU actors with the current and archived task identifiers."""
import hashlib
import json

import numpy as np
import pytest
import torch

from so101_typing.hardware.vision_policy import ALPHABET, PHRASE_CHARACTERS, VisionActor
from so101_typing.vision_contract import (
    NEWTON_VISION_TASK_CONTRACT, PHRASE_VISION_TASK_CONTRACT,
    LEGACY_PHRASE_VISION_TASK_CONTRACT,
)


class ZeroActor(torch.nn.Module):
    def forward(self, proprio, rgb):
        return proprio[:, :5] * 0 + rgb.sum() * 0


def exported_actor(root, characters, contract):
    dim = 19 + len(characters)
    actor = root/'vision_actor.pt'
    torch.jit.trace(ZeroActor(), (torch.zeros(1, dim), torch.zeros(1, 6, 120, 160))).save(str(actor))
    checkpoint = root/'model_1.pt'
    checkpoint.write_bytes(b'identity fixture, never deserialized')
    metadata = {
        'kind': 'vision_cpu_export', 'actor_file': actor.name,
        'actor_sha256': hashlib.sha256(actor.read_bytes()).hexdigest(),
        'source_checkpoint': str(checkpoint),
        'source_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'goal_characters': characters, 'proprio_dim': dim,
        'rgb_shape': [6, 120, 160], 'action_dim': 5, 'control_hz': 25,
        'rgb_order': 'RGB float32 [0,1]; previous frame followed by current frame',
    }
    meta = root/'vision_actor.json'
    meta.write_text(json.dumps(metadata))
    env = root/'environment.json'
    env.write_text(json.dumps({'task_contract': contract, 'actuator_profile': 'gtc_berlin'}))
    return actor, meta, env


@pytest.mark.parametrize('characters,contract', [
    (ALPHABET, NEWTON_VISION_TASK_CONTRACT),
    (PHRASE_CHARACTERS, PHRASE_VISION_TASK_CONTRACT),
    (PHRASE_CHARACTERS, LEGACY_PHRASE_VISION_TASK_CONTRACT),
])
@pytest.mark.parametrize('suffix', ['', '_task_'+'a'*64])
def test_exported_actor_loads_supported_training_contract(tmp_path, characters, contract, suffix):
    actor = VisionActor(*exported_actor(tmp_path, characters, contract+suffix))
    action = actor.action(np.zeros(19+len(characters), dtype=np.float32),
                          np.zeros((6,120,160), dtype=np.float32))
    np.testing.assert_array_equal(action, np.zeros(5))


@pytest.mark.parametrize('contract', [NEWTON_VISION_TASK_CONTRACT, PHRASE_VISION_TASK_CONTRACT+'_unknown'])
def test_exported_actor_rejects_wrong_vocabulary_or_unknown_version(tmp_path, contract):
    with pytest.raises(ValueError, match='does not match'):
        VisionActor(*exported_actor(tmp_path, PHRASE_CHARACTERS, contract))
