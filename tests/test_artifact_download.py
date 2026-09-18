"""Exercise artifact selection and verification against a local release mirror."""
import hashlib
import itertools
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

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
