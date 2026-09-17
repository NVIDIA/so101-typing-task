"""Publication checks must work before git add and without local run artifacts."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import check_public_release as check


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
