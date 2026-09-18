"""Metadata cleanup preserves measured evidence but never accepts geometry drift."""
import copy
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from so101_typing.hardware import source_identity as identity
from so101_typing.hardware import scene_check, scene_alignment, fk_probe_trajectory
from so101_typing.hardware.so101_kinematics import JOINT_NAMES, lerobot_position_to_model_rad

ROOT = Path(__file__).resolve().parents[1]
ASSET = 'source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/keyboards/data/mx_keys/mx_keys_asset.npz'
ORIGINAL = '976c858b82a99b11876c8846a844398a7f0841d7635af0d0974a930c21ebb8dc'
PUBLIC = '20331e2f67ab7202af4c87e1e661df10172100bdeadad2804bd6c4a02776ddbf'
ORIGINAL_DRAWING = 'd918589b9dc0fbcf220406d57a9ff4eff12ac3b83c118ccf1ee01066af770603'


def test_sanitized_asset_preserves_every_numeric_npy_payload():
    path = ROOT / ASSET
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PUBLIC
    payloads = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == 21
        for name in archive.namelist():
            if name != 'manifest_json.npy':
                payloads.update(name.encode() + b'\0' + archive.read(name))
    # Measured on the original archive before its source filename was removed.
    assert payloads.hexdigest() == '35f262b2d4b3c2eeed84600ac892c0d3cd591afbd893ce5f8725af9dae3531fc'
    with np.load(path, allow_pickle=False) as asset:
        manifest = json.loads(asset['manifest_json'].tobytes())
    assert manifest == json.loads(path.with_name('manifest.json').read_text())
    assert manifest['source'] == 'Logitech_MX_Keys.usdz'


def test_identity_alias_is_exact_and_scoped_to_one_asset():
    assert identity.source_hash_matches(ASSET, ORIGINAL, PUBLIC)
    assert identity.source_hash_matches(ASSET, PUBLIC, ORIGINAL)
    assert not identity.source_hash_matches(ASSET, 'changed geometry', ORIGINAL)
    assert not identity.source_hash_matches('robot.usd', PUBLIC, ORIGINAL)
    assert not identity.identity_matches('robot_usd_sha256', PUBLIC, ORIGINAL)
    old = {ASSET: ORIGINAL, 'robot.py': 'unchanged'}
    new = {ASSET: PUBLIC, 'robot.py': 'unchanged'}
    assert identity.identity_matches('source_sha256', old, new)
    assert not identity.identity_matches('source_sha256', old, {**new, 'extra.py': 'x'})
    assert not identity.identity_matches('source_sha256', old, {ASSET: PUBLIC})
    assert not identity.identity_matches('source_sha256', old, {**new, 'robot.py': 'changed'})
    assert not identity.identity_matches('source_sha256', old, {**new, ASSET: 'unknown'})
    assert not identity.identity_matches('source_sha256', None, None)


def test_old_recording_loads_without_rewriting_its_original_hash(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    asset = root / ASSET
    asset.parent.mkdir(parents=True)
    asset.write_bytes((ROOT / ASSET).read_bytes())
    monkeypatch.setattr(scene_check, 'ROOT', root)
    positions = [0.] * 6
    data = {'schema_version': 1, 'kind': 'so101_torque_off_recording',
            'joint_names': list(JOINT_NAMES), 'joint_zero_offset_deg': [0] * 5,
            'source_sha256': {ASSET: ORIGINAL}, 'events': [],
            'samples': [{'time_s': t, 'lerobot_position': positions,
                         'q_model_rad': lerobot_position_to_model_rad(positions).tolist()}
                        for t in (0., .02)]}
    path = tmp_path / 'recording.json'
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    assert scene_check.load_recording(path)['source_sha256'][ASSET] == ORIGINAL
    assert path.read_bytes() == before
    asset.write_bytes(asset.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='source changed'):
        scene_check.load_recording(path)


def test_old_fk_plan_accepts_only_the_equivalent_asset(tmp_path):
    task = ROOT / 'configs/tasks/physical-reference-20260915.json'
    plan = fk_probe_trajectory.generate(task, tmp_path / 'trajectory.json')
    plan['source_sha256'][ASSET] = ORIGINAL
    before = copy.deepcopy(plan)
    fk_probe_trajectory.validate(plan)
    assert plan == before
    plan['source_sha256'][ASSET] = 'changed geometry'
    with pytest.raises(ValueError, match='model source changed'):
        fk_probe_trajectory.validate(plan)


def test_old_alignment_preserves_contract_and_rejects_other_changes(tmp_path, monkeypatch):
    calibration = tmp_path / 'calibration.json'
    calibration.write_text('{}')
    monkeypatch.setattr(scene_alignment, 'require_calibration', lambda _: calibration)
    task = ROOT / 'configs/tasks/physical-reference-20260915.json'
    data = scene_alignment.prepare(task, 'test_robot', tmp_path / 'alignment', convention='lerobot')
    data['contract']['source_sha256'][ASSET] = ORIGINAL
    data['contract']['geometry_sha256'] = ORIGINAL_DRAWING
    data['contract_sha256'] = scene_alignment.fingerprint(data['contract'])
    before = copy.deepcopy(data)
    assert scene_alignment.validate_candidate(data, 'test_robot') == data['contract']
    assert data == before
    for field in ('source_sha256', 'geometry_sha256', 'robot_usd_sha256'):
        broken = copy.deepcopy(data)
        if field == 'source_sha256':
            broken['contract'][field]['scripts/so101_homing/so101_kinematics.py'] = 'changed'
        else:
            broken['contract'][field] = 'changed'
        broken['contract_sha256'] = scene_alignment.fingerprint(broken['contract'])
        with pytest.raises(ValueError, match='model/jaw identity changed'):
            scene_alignment.validate_candidate(broken, 'test_robot')
