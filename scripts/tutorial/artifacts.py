# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint selection and portable, verified bundles. Never uploads artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

SIDECARS = ('environment.json', 'agent.json', 'run.json')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
        return checksum.hexdigest()


def checkpoint(run, selection='best', suite='phrase'):
    run = Path(run).resolve()
    if run.is_file():
        chosen = run
    elif selection == 'best':
        record_path = run / f'best_{suite}.json'
        if not record_path.is_file():
            raise ValueError(f'No {suite} evaluation selection in {run}; evaluate checkpoints first or select an explicit model_N.pt')
        record = read(record_path)
        chosen = (run / record['checkpoint']).resolve()
        if not chosen.is_relative_to(run):
            raise ValueError('Best-checkpoint record escapes its run')
        if digest(chosen) != record['checkpoint_sha256']:
            raise ValueError('Best-checkpoint hash mismatch')
    elif selection == 'latest':
        choices = list(run.glob('model_*.pt'))
        if not choices:
            raise ValueError(f'No model_*.pt checkpoints in {run}')
        chosen = max(choices, key=lambda p: p.stat().st_mtime_ns)
    else:
        chosen = (run / selection).resolve()
    if not chosen.is_file():
        raise ValueError(f'Checkpoint missing: {chosen}')
    for name in SIDECARS:
        if not (chosen.parent / name).is_file():
            raise ValueError(f'Missing {name} beside {chosen}. Use a tutorial run or a complete bundle.')
    return chosen


def record_evaluation(run, chosen, report, suite):
    run, chosen = Path(run).resolve(), Path(chosen).resolve()
    if not chosen.is_relative_to(run):
        raise ValueError('Evaluation checkpoint must belong to the selected run')
    record = dict(report)
    record.update(checkpoint=str(chosen.relative_to(run)), checkpoint_sha256=digest(chosen), suite=suite)
    path = run / f'best_{suite}.json'
    # Do not compare different protocols as though their rates were equivalent.
    protocol = ('target', 'episodes', 'seed', 'domain_strength', 'num_envs')
    if path.exists():
        old = read(path)
        if any(old.get(k) != record.get(k) for k in protocol):
            raise ValueError(f'Evaluation protocol differs from {path}; use a separate run/selection record')
        if old['success_rate'] >= record['success_rate']:
            return
    write(path, record)


def pack(chosen, output, camera, image=None):
    chosen, output = Path(chosen).resolve(), Path(output).resolve()
    checkpoint(chosen)
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    files = {'checkpoint.pt': chosen, **{n: chosen.parent / n for n in SIDECARS},
             'evaluation_replay_camera.json': Path(camera)}
    for name in ('tutorial.json', 'task.json', 'asset_identity.json', 'runtime.json'):
        if (chosen.parent / name).is_file():
            files[name] = chosen.parent / name
    manifest = {'schema_version': 1, 'checkpoint': 'checkpoint.pt',
                'runtime_image': image, 'files': {n: digest(p) for n, p in files.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr('bundle.json', json.dumps(manifest, indent=2) + '\n')
    write(output.with_suffix(output.suffix + '.sha256.json'), {'sha256': digest(output)})
    return manifest


def unpack(archive, destination, expected_sha256):
    archive, destination = Path(archive), Path(destination)
    if len(expected_sha256) != 64 or digest(archive) != expected_sha256:
        raise ValueError('Bundle SHA256 mismatch')
    if destination.exists():
        raise ValueError(f'Destination already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / 'bundle'
        staging.mkdir()
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(set(names)) != len(names) or any(PurePosixPath(n).name != n or '\\' in n for n in names):
                raise ValueError('Bundle contains duplicate or non-flat paths')
            if sum(i.file_size for i in z.infolist()) > 4 * 1024**3:
                raise ValueError('Bundle exceeds 4 GiB limit')
            manifest = json.loads(z.read('bundle.json'))
            if manifest.get('schema_version') != 1 or manifest.get('checkpoint') != 'checkpoint.pt':
                raise ValueError('Unsupported bundle schema')
            if set(names) != set(manifest['files']) | {'bundle.json'}:
                raise ValueError('Bundle members differ from manifest')
            if not set(('checkpoint.pt', *SIDECARS)).issubset(manifest['files']):
                raise ValueError('Incomplete model bundle')
            for name in names:
                with z.open(name) as src, (staging / name).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            for name, checksum in manifest['files'].items():
                if digest(staging / name) != checksum:
                    raise ValueError(f'Bundle member hash mismatch: {name}')
        staging.rename(destination)
    return destination / 'checkpoint.pt'


def download(url, sha256, destination):
    if not url.startswith('https://'):
        raise ValueError('Bundle downloads require HTTPS')
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'bundle.zip'
        with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as output:
            if not response.url.startswith('https://'):
                raise ValueError('Download redirected away from HTTPS')
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > 4 * 1024**3:
                    raise ValueError('Download exceeds 4 GiB limit')
                output.write(block)
        return unpack(archive, destination, sha256)
