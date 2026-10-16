# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen A-Z targets for the registered physical MX Keys fixture.

The runtime table is intentionally self contained.  The source JSON and its
manifest are re-read only by deterministic validation code/tests; training
does not depend on a path outside the task package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .registered_mx_env_cfg import (
    REGISTERED_KEYBOARD_POSITION_B_M,
    REGISTERED_KEYBOARD_ROTATION_XYZW,
    REGISTERED_MX_KEYS_VARIANT_INDEX,
)

FROZEN_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FROZEN_REFERENCE_SHA256 = "4217e1c7257af0b90cb2284349723d82c92baddbde780aaf5d57ee71a3594e03"
FROZEN_REFERENCE_MANIFEST_SHA256 = "5b04f77014923811d7116b172b6f33bffe0d1263ba55892c33639ebc105d4e10"
FROZEN_REFERENCE_SOURCE_VARIANT_INDEX = 32
# Physical face measurement supplied for the demo keyboard.  Switch edges,
# not this footprint, decide success; the value constrains probe/audit margins.
FROZEN_ALPHABET_KEY_FACE_XY_M = (0.016, 0.016)

# Alphabetical slot order in the 108-key MX Keys articulation.
FROZEN_LETTER_SLOTS = (
    29,
    46,
    44,
    31,
    17,
    32,
    33,
    34,
    22,
    35,
    36,
    37,
    48,
    47,
    23,
    24,
    15,
    18,
    30,
    19,
    21,
    45,
    16,
    43,
    20,
    42,
)

# Registered key-body centers in the robot-base frame [m].  Row i is
# ``FROZEN_LETTERS[i]``.  These are geometric goals, not press depths or joint
# targets; PPO remains responsible for seek, press, release, and clearance.
FROZEN_AZ_XYZ_B_M = (
    (0.2283641701720075, 0.16267415535493321, -0.020819183751136415),  # A
    (0.19695143473215612, 0.081357871586966404, -0.02297117928221297),  # B
    (0.20264210531790192, 0.11894956938497298, -0.022971179282212963),  # C
    (0.22267349902867606, 0.12508245387360131, -0.020819183751136422),  # D
    (0.24235652249371037, 0.12723536465080104, -0.018638560758507557),  # E
    (0.21982816373580316, 0.10628660497459803, -0.020819183751136422),  # F
    (0.21698282816413744, 0.087490754233932072, -0.020819183751136425),  # G
    (0.21413749259247172, 0.068694903493266138, -0.020819183751136425),  # H
    (0.2281298440777961, 0.033256107264145976, -0.018638560758507564),  # I
    (0.21129215757839165, 0.049899056435925522, -0.020819183751136429),  # J
    (0.20844682256431155, 0.031103209378584892, -0.020819183751136432),  # K
    (0.2056014875502315, 0.012307362321244283, -0.020819183751136432),  # L
    (0.19126076470399597, 0.043766177472285159, -0.022971179282212973),  # M
    (0.19410609971807605, 0.062562024529625782, -0.02297117928221297),  # N
    (0.22528451017888732, 0.014460267573456, -0.018638560758507568),  # O
    (0.22243917516480724, -0.0043355794838846251, -0.018638560758507571),  # P
    (0.24804719287036156, 0.16482706106756059, -0.01863856075850755),  # Q
    (0.23951118636445901, 0.10843951022680977, -0.018638560758507557),  # R
    (0.22551883460034178, 0.14387830461426729, -0.020819183751136418),  # S
    (0.23666585135037893, 0.089643663169469145, -0.018638560758507561),  # T
    (0.23097518020704749, 0.052051961688137242, -0.018638560758507564),  # U
    (0.1997967697462362, 0.10015371864430703, -0.022971179282212966),  # V
    (0.24520185750779044, 0.14603121170814165, -0.018638560758507554),  # W
    (0.20548744088956764, 0.13774542012563892, -0.022971179282212963),  # X
    (0.23382051577871321, 0.070847812428803197, -0.018638560758507561),  # Y
    (0.20833288539952882, 0.15654199049598982, -0.022971179282212959),  # Z
)

FROZEN_AZ_XYZ_MEAN_B_M = (0.21937903048029955, 0.08811349835436841, -0.020559866012338238)
FROZEN_AZ_XYZ_STD_B_M = (0.015972329715743194, 0.04878220342215096, 0.0017344958454968311)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rotation_matrix_xyzw(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def validate_frozen_az_source(reference_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Rebuild and validate the frozen table from its checked-in source files.

    Raises ``ValueError`` on a hash, label, slot, variant, or transformed-center
    mismatch.  This is a deterministic build/probe gate, not a rollout-time
    operation.
    """

    if _sha256(reference_path) != FROZEN_REFERENCE_SHA256:
        raise ValueError(f"MX Keys reference hash mismatch: {reference_path}")
    if _sha256(manifest_path) != FROZEN_REFERENCE_MANIFEST_SHA256:
        raise ValueError(f"MX Keys manifest hash mismatch: {manifest_path}")

    reference = json.loads(reference_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    keyboard_manifest = manifest.get("keyboard", {})
    if keyboard_manifest.get("reference_sha256") != FROZEN_REFERENCE_SHA256:
        raise ValueError("MX Keys manifest does not name the frozen reference hash")
    if keyboard_manifest.get("reference_source_variant_index") != FROZEN_REFERENCE_SOURCE_VARIANT_INDEX:
        raise ValueError("MX Keys manifest variant index changed")
    if REGISTERED_MX_KEYS_VARIANT_INDEX != FROZEN_REFERENCE_SOURCE_VARIANT_INDEX:
        raise ValueError("registered simulator variant no longer matches the frozen map source")

    by_letter: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for key in reference.get("keys", []):
        label = str(key.get("label", "")).upper()
        if len(label) == 1 and label.isalpha():
            if label in by_letter:
                duplicates.append(label)
            by_letter[label] = key
    if duplicates or set(by_letter) != set(FROZEN_LETTERS):
        raise ValueError(f"reference must contain exactly one A-Z key; duplicates={duplicates}")

    slots = tuple(int(by_letter[letter]["slot"]) for letter in FROZEN_LETTERS)
    if slots != FROZEN_LETTER_SLOTS:
        raise ValueError(f"frozen A-Z slot order changed: {slots}")

    local = np.asarray([by_letter[letter]["center"] for letter in FROZEN_LETTERS], dtype=np.float64)
    rotation = _rotation_matrix_xyzw(REGISTERED_KEYBOARD_ROTATION_XYZW)
    translation = np.asarray(REGISTERED_KEYBOARD_POSITION_B_M, dtype=np.float64)
    rebuilt = local @ rotation.T + translation
    frozen = np.asarray(FROZEN_AZ_XYZ_B_M, dtype=np.float64)
    max_error = float(np.linalg.norm(rebuilt - frozen, axis=1).max())
    if max_error > 1.0e-12:
        raise ValueError(f"frozen A-Z table differs from source transform by {max_error:.3e} m")

    return {
        "reference_sha256": FROZEN_REFERENCE_SHA256,
        "manifest_sha256": FROZEN_REFERENCE_MANIFEST_SHA256,
        "variant_index": FROZEN_REFERENCE_SOURCE_VARIANT_INDEX,
        "letter_count": len(FROZEN_LETTERS),
        "max_rebuild_error_m": max_error,
    }


def frozen_az_manifest() -> dict[str, Any]:
    """Return the serializable target-map portion of a checkpoint contract."""

    return {
        "letters": "".join(FROZEN_LETTERS),
        "letter_slots": list(FROZEN_LETTER_SLOTS),
        "xyz_b_m": [list(row) for row in FROZEN_AZ_XYZ_B_M],
        "xyz_mean_b_m": list(FROZEN_AZ_XYZ_MEAN_B_M),
        "xyz_std_b_m": list(FROZEN_AZ_XYZ_STD_B_M),
        "reference_sha256": FROZEN_REFERENCE_SHA256,
        "reference_manifest_sha256": FROZEN_REFERENCE_MANIFEST_SHA256,
        "reference_source_variant_index": FROZEN_REFERENCE_SOURCE_VARIANT_INDEX,
        "physical_alphabet_key_face_xy_m": list(FROZEN_ALPHABET_KEY_FACE_XY_M),
        "keyboard_position_b_m": list(REGISTERED_KEYBOARD_POSITION_B_M),
        "keyboard_rotation_xyzw": list(REGISTERED_KEYBOARD_ROTATION_XYZW),
    }
