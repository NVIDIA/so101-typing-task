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
