# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact identities for metadata-only asset and copyright updates.

The public NPZ removes the converter's workstation source path. All other NPY
member payloads are byte-identical. The drawing JSON changes only that asset's
source hash. Preserve original evidence hashes; accept only these two known
representations when comparing model identity, never arbitrary asset changes.
The calibration configuration also has two exact representations differing only
by its added NVIDIA copyright notice. Historical evidence keeps its original hash.
"""

CALIBRATION_SOURCE_PATHS = frozenset((
    'source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/physical_calibrated_20260718_env_cfg.py',
    'source/so101_typing/tasks/physical_calibrated_20260718_env_cfg.py',
))
CALIBRATION_SOURCE_HASHES = frozenset((
    'edb76ea369fd29426fbfe45db9759ed98a5b10bf5b6389e0bbb95721cdf9dd8e',
    '9e28338432f7cd79ef62f0185e1da0fd0243e771aa957c3070504f8c82abc48e',
))

KEYBOARD_ASSET_PATHS = frozenset((
    'source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/keyboards/data/mx_keys/mx_keys_asset.npz',
    'source/so101_typing/tasks/keyboards/data/mx_keys/mx_keys_asset.npz',
))
KEYBOARD_ASSET_HASHES = frozenset((
    '976c858b82a99b11876c8846a844398a7f0841d7635af0d0974a930c21ebb8dc',
    '20331e2f67ab7202af4c87e1e661df10172100bdeadad2804bd6c4a02776ddbf',
))
DRAWING_HASHES = frozenset((
    'd918589b9dc0fbcf220406d57a9ff4eff12ac3b83c118ccf1ee01066af770603',
    'f95d8c93f6b42e1e55fc3c0da86ab2161969a9472beb321207a473518082dd29',
))


def source_hash_matches(name, actual, expected):
    if actual == expected:
        return True
    if (name in CALIBRATION_SOURCE_PATHS and isinstance(actual, str)
            and isinstance(expected, str)):
        return actual in CALIBRATION_SOURCE_HASHES and expected in CALIBRATION_SOURCE_HASHES
    return (name in KEYBOARD_ASSET_PATHS and isinstance(actual, str)
            and isinstance(expected, str)
            and actual in KEYBOARD_ASSET_HASHES and expected in KEYBOARD_ASSET_HASHES)


def identity_matches(field, actual, expected):
    """Compare an identity field without normalizing or rewriting its evidence."""
    if field == 'source_sha256':
        return (isinstance(actual, dict) and isinstance(expected, dict)
                and actual.keys() == expected.keys()
                and all(source_hash_matches(name, value, expected[name])
                        for name, value in actual.items()))
    if field == 'geometry_sha256' and actual != expected:
        return (isinstance(actual, str) and isinstance(expected, str)
                and actual in DRAWING_HASHES and expected in DRAWING_HASHES)
    return actual == expected
