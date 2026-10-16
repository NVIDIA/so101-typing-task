# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Preserve existing evidence only for the exact restored notice changes."""
import hashlib
import json
from pathlib import Path

import pytest

from so101_typing.hardware import source_identity as identity

ROOT = Path(__file__).resolve().parents[1]
NOTICE = b'# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n'


@pytest.mark.parametrize('paths,hashes,header', [
    (identity.CALIBRATION_NOTICE_PATHS, identity.CALIBRATION_NOTICE_HASHES, NOTICE),
    (identity.KINEMATICS_NOTICE_PATHS, identity.KINEMATICS_NOTICE_HASHES,
     NOTICE+b'# SPDX-License-Identifier: BSD-3-Clause\n\n'),
])
def test_notice_aliases_are_exact_and_path_scoped(paths, hashes, header):
    for name in paths:
        current = (ROOT/name).read_bytes()
        before = current.replace(header, b'', 1)
        old, new = (hashlib.sha256(value).hexdigest() for value in (before,current))
        assert {old,new} == hashes
        assert identity.source_hash_matches(name, old, new)
        assert identity.source_hash_matches(name, new, old)
        assert not identity.source_hash_matches('unrelated.py', old, new)
        changed = hashlib.sha256(current+b'\n# code changed\n').hexdigest()
        assert not identity.source_hash_matches(name, changed, old)


def test_notice_drawing_alias_changes_only_source_hashes():
    path = ROOT/'docs/fixtures/model_geometry.json'
    current = path.read_bytes()
    data = json.loads(current)
    for name, value in data['source_sha256'].items():
        for paths, hashes in ((identity.CALIBRATION_NOTICE_PATHS, identity.CALIBRATION_NOTICE_HASHES),
                              (identity.KINEMATICS_NOTICE_PATHS, identity.KINEMATICS_NOTICE_HASHES)):
            if name in paths:
                data['source_sha256'][name] = next(h for h in hashes if h != value)
    before = (json.dumps(data, separators=(',', ':'))+'\n').encode()
    old,new = (hashlib.sha256(value).hexdigest() for value in (before,current))
    assert {old,new} == identity.NOTICE_DRAWING_HASHES
    assert identity.identity_matches('geometry_sha256', old, new)
    assert not identity.identity_matches('geometry_sha256', new, next(iter(identity.DRAWING_HASHES)))
