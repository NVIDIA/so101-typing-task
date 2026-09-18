"""Publication checks must work before git add and without local run artifacts."""
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import pytest

from scripts import check_public_release as check


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
