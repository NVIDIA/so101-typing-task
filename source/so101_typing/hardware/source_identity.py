# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact identities for metadata-only asset cleanup and restored copyright notices.

The public NPZ removes the converter's workstation source path. All other NPY
member payloads are byte-identical. The drawing JSON changes only that asset's
source hash. Preserve original evidence hashes; accept only these two known
representations when comparing model identity, never arbitrary asset changes.
"""

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

# Exact pre/post-notice identities; these do not authorize code or geometry changes.
CALIBRATION_NOTICE_PATHS = frozenset(('source/so101_typing/tasks/physical_calibrated_20260718_env_cfg.py', 'source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/physical_calibrated_20260718_env_cfg.py'))
CALIBRATION_NOTICE_HASHES = frozenset(('e7670aeb2f0b205c9ccf30ab96b7d226cf9ff1d51f0faab2115085cb99a26d61', '4dfebfd353c8382323e93672f40b87652a638cfdee84fd20f938d9270c6f2958'))
KINEMATICS_NOTICE_PATHS = frozenset(('source/so101_typing/hardware/so101_kinematics.py', 'scripts/so101_homing/so101_kinematics.py'))
KINEMATICS_NOTICE_HASHES = frozenset(('afdf4e062bd8ef463f27570b5a10bf3887d0dc701260c0730889bafb46de8d5e', '54d68c013123b85ede1b99456056ab161f97099d927e412ae9342e0ea5d7ace7'))
NOTICE_DRAWING_HASHES = frozenset(('f7b2bd53ae0b30193fc77c4a5dd2760312b947f64bad5d9a3f61855203e20395', 'ffef246879eaec7fe86341af5ae339a5bedee73517f41f9fb29a4c48590cde9a'))


def source_hash_matches(name, actual, expected):
    if actual == expected:
        return True
    if isinstance(actual, str) and isinstance(expected, str):
        for paths, hashes in ((CALIBRATION_NOTICE_PATHS, CALIBRATION_NOTICE_HASHES),
                              (KINEMATICS_NOTICE_PATHS, KINEMATICS_NOTICE_HASHES)):
            if name in paths and actual in hashes and expected in hashes:
                return True
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
                and any(actual in hashes and expected in hashes
                        for hashes in (DRAWING_HASHES, NOTICE_DRAWING_HASHES)))
    return actual == expected
