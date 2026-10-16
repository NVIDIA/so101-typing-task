# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Portable asset identities for replay compatibility checks."""
import hashlib
import json
from pathlib import Path


def save_asset_identity(output, cfg):
    path = Path(cfg.scene.robot.spawn.usd_path)
    identity = {'robot_usd':path.name, 'robot_usd_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'task_contract':cfg.task_contract}
    (Path(output)/'asset_identity.json').write_text(json.dumps(identity,indent=2)+'\n')


def verify_asset_identity(checkpoint, cfg):
    record = Path(checkpoint).parent/'asset_identity.json'
    if record.exists():
        expected = json.loads(record.read_text())['robot_usd_sha256']
        actual = hashlib.sha256(Path(cfg.scene.robot.spawn.usd_path).read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError('Robot USD differs from the checkpoint asset identity; use its matching runtime image')
