# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Artifact downloads, checkpoint pairing and portable source integrity."""

import ast
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import check_public_release as check
from so101_typing.hardware import (
    checkpoint_identity,
    fk_probe_trajectory,
    scene_alignment,
    scene_check,
    source_identity,
)
from so101_typing.hardware.so101_kinematics import JOINT_NAMES, lerobot_position_to_model_rad


# Artifact downloads

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'releases/calibrated-20260917/'
HISTORICAL = {
    'checkpoints/mjwarp-anchorbench-19k/model_19000.pt',
    'configs/mjwarp-anchorbench-19k.env.yaml',
    'evaluation/mjwarp-anchorbench-19k.corpus_100x6_seed1307.json',
}


@pytest.fixture
def mirror(tmp_path):
    if not shutil.which('curl') or not shutil.which('sha256sum'):
        pytest.skip('local mirror test requires curl and sha256sum')
    source = (ROOT / 'scripts/download_benchmark_artifacts.sh').read_text()
    entries = re.findall(r"'([^'|]+)\|([0-9a-f]{64})'", source)
    assert entries
    origin = tmp_path / 'mirror'
    for relative, released_hash in entries:
        payload = relative.encode() + b'\n'
        path = origin / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        # Retain actual downloader/checksum logic, substituting tiny mirror payloads.
        source = source.replace(f'{relative}|{released_hash}',
                                f'{relative}|{hashlib.sha256(payload).hexdigest()}')
    script = tmp_path / 'download.sh'
    script.write_text(source)
    return script, origin, {relative for relative, _ in entries}


def run_download(mirror, destination, *flags):
    script, origin, _ = mirror
    return subprocess.run(['bash', str(script), *flags, str(destination)],
                          env={**os.environ, 'SO101_HF_BASE_URL': origin.as_uri()},
                          text=True, capture_output=True)


@pytest.mark.parametrize('flag', [None, '--calibrated', '--historical', '--all'])
def test_selects_complete_bundle_and_checksums(mirror, tmp_path, flag):
    destination = tmp_path / 'download'
    result = run_download(mirror, destination, *([flag] if flag else []))
    assert result.returncode == 0, result.stderr
    all_paths = mirror[2]
    expected = (all_paths if flag == '--all' else HISTORICAL if flag == '--historical'
                else {path for path in all_paths if path.startswith(RELEASE)})
    downloaded = {str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}
    assert downloaded == expected | {'DOWNLOAD_SHA256SUMS'}
    checksums = (destination / 'DOWNLOAD_SHA256SUMS').read_text().splitlines()
    assert {line.split('  ', 1)[1] for line in checksums} == expected
    for path in downloaded:
        assert (destination / path).stat().st_mode & 0o777 == 0o644
    verified = subprocess.run(['sha256sum', '-c', 'DOWNLOAD_SHA256SUMS'], cwd=destination,
                              text=True, capture_output=True)
    assert verified.returncode == 0, verified.stderr


def test_rejects_corrupt_release_payload(mirror, tmp_path):
    relative = next(path for path in mirror[2] if path.startswith(RELEASE))
    (mirror[1] / relative).write_bytes(b'corrupt')
    destination = tmp_path / 'download'
    result = run_download(mirror, destination)
    assert result.returncode != 0
    assert 'checksum mismatch' in result.stderr
    assert not (destination / relative).exists()
    assert not (destination / 'DOWNLOAD_SHA256SUMS').exists()
    assert not list(destination.rglob('*.partial.*'))


@pytest.mark.parametrize('flags', list(itertools.combinations(['--all', '--calibrated', '--historical'], 2)))
def test_conflicting_selections_create_no_output(mirror, tmp_path, flags):
    destination = tmp_path / 'download'
    result = run_download(mirror, destination, *flags)
    assert result.returncode == 2
    assert not destination.exists()


# Checkpoint and environment pairing

def test_published_pairs_match_downloader_and_reject_crossed_environments():
    downloader = (ROOT/'scripts/download_benchmark_artifacts.sh').read_text()
    assert len(checkpoint_identity.RELEASED_ENVIRONMENTS) == 5
    for checkpoint, environment in checkpoint_identity.RELEASED_ENVIRONMENTS.items():
        assert checkpoint in downloader and environment in downloader
        checkpoint_identity.validate_pair(checkpoint, environment)
        for other in set(checkpoint_identity.RELEASED_ENVIRONMENTS.values())-{environment}:
            with pytest.raises(ValueError, match='published checkpoint requires its matching environment'):
                checkpoint_identity.validate_pair(checkpoint, other)
    # Actor format alone cannot establish training provenance for a new policy.
    checkpoint_identity.validate_pair('unknown-checkpoint', 'caller-supplied-environment')


@pytest.mark.parametrize('entrypoint', ['evaluation', 'legacy_handoff'])
def test_runtime_entrypoints_reject_known_mismatch_before_environment_validation(tmp_path, monkeypatch, entrypoint):
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'synthetic-known-actor')
    environment = tmp_path/'env.yaml'
    environment.write_text('not a training environment')
    monkeypatch.setitem(checkpoint_identity.RELEASED_ENVIRONMENTS,
                        hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'expected-environment-hash')
    if entrypoint == 'legacy_handoff':
        from so101_typing.hardware.run_fixed_cartesian_policy_handoff import _validated_training_environment
        invoke = lambda: _validated_training_environment(checkpoint=checkpoint, env_config=environment,
            profile=SimpleNamespace(name='physical-calibrated-20260718'), stage='p1d')
    else:
        # Load the actual pure validation function without importing Isaac Lab or
        # parsing the simulator entrypoint's global command-line arguments.
        source = ROOT/'scripts/reinforcement_learning/rsl_rl/evaluate_so101_fixed_cartesian.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_validate_artifact_contract')
        namespace = {'Path': Path, 'ManagerBasedRLEnvCfg': object,
                     'args_cli': SimpleNamespace(checkpoint=checkpoint, env_config=environment)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        invoke = lambda: namespace['_validate_artifact_contract'](None)
    with pytest.raises(ValueError, match='published checkpoint requires its matching environment'):
        invoke()


# Source identity and geometry preservation

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
    assert source_identity.source_hash_matches(ASSET, ORIGINAL, PUBLIC)
    assert source_identity.source_hash_matches(ASSET, PUBLIC, ORIGINAL)
    assert not source_identity.source_hash_matches(ASSET, 'changed geometry', ORIGINAL)
    assert not source_identity.source_hash_matches('robot.usd', PUBLIC, ORIGINAL)
    assert not source_identity.identity_matches('robot_usd_sha256', PUBLIC, ORIGINAL)
    old = {ASSET: ORIGINAL, 'robot.py': 'unchanged'}
    new = {ASSET: PUBLIC, 'robot.py': 'unchanged'}
    assert source_identity.identity_matches('source_sha256', old, new)
    assert not source_identity.identity_matches('source_sha256', old, {**new, 'extra.py': 'x'})
    assert not source_identity.identity_matches('source_sha256', old, {ASSET: PUBLIC})
    assert not source_identity.identity_matches('source_sha256', old, {**new, 'robot.py': 'changed'})
    assert not source_identity.identity_matches('source_sha256', old, {**new, ASSET: 'unknown'})
    assert not source_identity.identity_matches('source_sha256', None, None)


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


def test_alignment_accepts_equivalent_keyboard_asset_but_rejects_old_robot_geometry(tmp_path, monkeypatch):
    calibration = tmp_path / 'calibration.json'
    calibration.write_text('{}')
    monkeypatch.setattr(scene_alignment, 'require_calibration', lambda _: calibration)
    task = ROOT / 'configs/tasks/physical-reference-20260915.json'
    data = scene_alignment.prepare(task, 'test_robot', tmp_path / 'alignment', convention='lerobot')
    data['contract']['source_sha256'][ASSET] = ORIGINAL
    data['contract_sha256'] = scene_alignment.fingerprint(data['contract'])
    before = copy.deepcopy(data)
    assert scene_alignment.validate_candidate(data, 'test_robot') == data['contract']
    assert data == before
    old = copy.deepcopy(data)
    old['contract']['geometry_sha256'] = ORIGINAL_DRAWING
    old['contract_sha256'] = scene_alignment.fingerprint(old['contract'])
    with pytest.raises(ValueError, match='geometry_sha256'):
        scene_alignment.validate_candidate(old, 'test_robot')
    for field in ('source_sha256', 'geometry_sha256', 'robot_usd_sha256'):
        broken = copy.deepcopy(data)
        if field == 'source_sha256':
            broken['contract'][field]['scripts/so101_homing/so101_kinematics.py'] = 'changed'
        else:
            broken['contract'][field] = 'changed'
        broken['contract_sha256'] = scene_alignment.fingerprint(broken['contract'])
        with pytest.raises(ValueError, match='model/jaw identity changed'):
            scene_alignment.validate_candidate(broken, 'test_robot')


# Portable release checks

def npy_bytes(payload, dtype, shape, version=1):
    """Build standard array bytes without adding NumPy to the checker tests."""
    header = repr({'descr': dtype, 'fortran_order': False, 'shape': shape}).encode()
    prefix_size = 10 if version == 1 else 12
    header += b' ' * (-(prefix_size + len(header) + 1) % 64) + b'\n'
    return (b'\x93NUMPY' + bytes((version, 0))
            + len(header).to_bytes(prefix_size - 8, 'little') + header + payload)


def compressed_array(path, payload):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest_json.npy', payload)


@pytest.mark.parametrize('storage', ['uint8', 'bytes', 'unicode-le', 'unicode-be', 'object'])
def test_compressed_npz_metadata_is_rejected_without_echoing_values(tmp_path, monkeypatch, capsys, storage):
    private = '/' + 'home' + '/example/private-recording'
    metadata = '{"source": "' + private + '"}'
    if storage == 'uint8':
        encoded = metadata.encode()
        payload = npy_bytes(encoded, '|u1', (len(encoded),))
    elif storage == 'bytes':
        encoded = metadata.encode()
        payload = npy_bytes(encoded, '|S' + str(len(encoded)), ())
    elif storage.startswith('unicode'):
        little = storage.endswith('le')
        payload = npy_bytes(metadata.encode('utf-32-le' if little else 'utf-32-be'),
                            ('<' if little else '>') + 'U' + str(len(metadata)), (), version=2)
    else:
        # An object-array body is only inspected as bytes, never unpickled.
        payload = npy_bytes(b'not executable pickle: ' + metadata.encode(), '|O', (1,))
    compressed_array(tmp_path/'asset.npz', payload)
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_public_release.py'])
    assert check.main() == 1
    output = capsys.readouterr()
    assert 'asset.npz: workstation path' in output.err
    assert private not in output.err + output.out


def test_numeric_npz_and_clean_unicode_metadata_pass(tmp_path, monkeypatch, capsys):
    with zipfile.ZipFile(tmp_path/'asset.npz', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('vertices.npy', npy_bytes(struct.pack('<6d', 0, 1, 2, 3, 4, 5), '<f8', (2, 3)))
        metadata = 'portable geometry \N{GREEK SMALL LETTER ALPHA}'
        archive.writestr('description.npy', npy_bytes(metadata.encode('utf-32-le'), '<U' + str(len(metadata)), ()))
        archive.writestr('empty.npy', npy_bytes(b'', '|u1', (0,)))
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_public_release.py'])
    assert check.main() == 0
    assert not capsys.readouterr().err


@pytest.mark.parametrize('payload', [None, b'not NPY', b'\x93NUMPY\x01\x00\xff\xff',
                                    npy_bytes(b'\0', '<U1', (1,))])
def test_malformed_npz_fails_cleanly(tmp_path, monkeypatch, capsys, payload):
    path = tmp_path/'broken.npz'
    if payload is None:
        path.write_bytes(b'not a ZIP archive')
    else:
        compressed_array(path, payload)
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_public_release.py'])
    assert check.main() == 1
    assert capsys.readouterr().err == 'broken.npz: malformed or unreadable NPZ archive\n'


def test_corrupt_compressed_npz_member_fails_cleanly(tmp_path, monkeypatch, capsys):
    path = tmp_path/'broken.npz'
    compressed_array(path, npy_bytes(b'portable metadata', '|u1', (17,)))
    contents = bytearray(path.read_bytes())
    filename_size, extra_size = struct.unpack_from('<HH', contents, 26)
    contents[30 + filename_size + extra_size] = 7  # Invalid DEFLATE block type.
    path.write_bytes(contents)
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_public_release.py'])
    assert check.main() == 1
    assert capsys.readouterr().err == 'broken.npz: malformed or unreadable NPZ archive\n'


def test_checkpoint_pickle_strings_still_rejected(tmp_path, monkeypatch, capsys):
    with zipfile.ZipFile(tmp_path/'model.pt', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('model/data.pkl', ('/' + 'home' + '/example/checkpoint').encode())
        archive.writestr('model/data/0', b'numeric tensor storage')
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_public_release.py'])
    assert check.main() == 1
    assert 'model.pt: workstation path' in capsys.readouterr().err


@pytest.mark.parametrize('label,payload', [
    ('hardware serial', '/dev/serial/by-id/usb-controller_Serial_' + '12AB34CD' + '56-if00'),
    ('credential material', 'github' + '_pat_' + 'a' * 40),
    ('credential material', 'hf_' + 'a' * 30),
])
def test_identifiers_and_tokens_are_rejected_without_echoing_values(tmp_path, monkeypatch, capsys, label, payload):
    (tmp_path/'new-note.md').write_text(payload)
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_public_release.py'])
    assert check.main() == 1
    output = capsys.readouterr()
    assert f'new-note.md: {label}' in output.err
    assert payload not in output.err + output.out


def test_untracked_private_path_is_rejected_before_staging(tmp_path, monkeypatch, capsys):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path/'README.md').write_text('clean source')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'README.md'], check=True)
    (tmp_path/'new-note.md').write_text('/'+'home'+'/example/private-recording')
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_public_release.py'])
    assert check.main() == 1
    assert 'new-note.md: workstation path' in capsys.readouterr().err


def test_existing_ignored_evidence_is_not_a_distributable_link(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path/'.gitignore').write_text('output/\nlocal/\n')
    (tmp_path/'output').mkdir()
    (tmp_path/'output/result.json').write_text('{}')
    (tmp_path/'README.md').write_text('[Evidence](output/result.json)')
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    assert len(check.source_link_failures(check.source_files())) == 1
    (tmp_path/'README.md').write_text('Local artifact identifier: `output/result.json`')
    assert check.source_link_failures(check.source_files()) == []


def test_source_links_survive_gitless_distribution(tmp_path, monkeypatch):
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/guide.md').write_text('[Home](../README.md#install)')
    (tmp_path/'README.md').write_text('[Guide](docs/guide.md)')
    (tmp_path/'local').mkdir()
    (tmp_path/'local/machine.md').write_text('/'+'home'+'/example/private')
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_public_release.py'])
    assert check.main() == 0
    (tmp_path/'docs/guide.md').unlink()
    assert check.main() == 1
