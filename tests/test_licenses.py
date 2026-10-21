# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep the source attribution inventory consistent with distributed files."""
import hashlib
import json
from pathlib import Path


def test_attribution_inventory_and_notices():
    root = Path(__file__).resolve().parents[1]
    notice = b'Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.'
    assert notice in (root / 'LICENSE').read_bytes()
    inventory = json.loads((root / 'licenses/isaac-lab-source.json').read_text())
    assert len(inventory['files']) == 57
    for entry in inventory['files']:
        contents = (root / entry['path']).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == entry['sha256'], entry['path']
        if entry['classification'] != 'unmodified':
            assert notice in contents, entry['path']
        assert b'The Isaac Lab Project Developers' in contents, entry['path']


def test_runtime_patch_attribution_inventory():
    root = Path(__file__).resolve().parents[1]
    notice = 'Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.'
    inventory = json.loads((root / 'licenses/runtime-patch-source.json').read_text())
    assert len(inventory['files']) == 16
    patch = (root / inventory['patch']).read_text()
    sections = {}
    current = None
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            current = line.split(' b/', 1)[0].removeprefix('diff --git a/')
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    assert len(sections) == 17
    for entry in inventory['files']:
        body = '\n'.join(sections[entry['path']])
        assert notice in body, entry['path']
        assert entry['classification'] in {'modified', 'added'}
        assert entry['before_notice_sha256'] != entry['sha256']
        assert len(entry['sha256']) == 64
    changelog = 'source/isaaclab_newton/changelog.d/anchored_model.rst'
    assert changelog in sections
    assert notice not in '\n'.join(sections[changelog])
