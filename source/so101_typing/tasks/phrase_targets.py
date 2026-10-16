# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrated main-block targets for the fixed ISAAC LAB 3.0 task."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .keyboards import TYPING_KEYBOARD_POOL
from .physical_calibrated_20260718_env_cfg import (
    PHYSICAL_CALIBRATED_AZ_XYZ_B_M,
    PHYSICAL_CALIBRATED_KEYBOARD_POSITION_B_M,
    PHYSICAL_CALIBRATED_KEYBOARD_ROTATION_XYZW,
)

TARGET_TEXT = "ISAAC LAB 3.0"
MAPPED_CHARACTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .")
_ASSET = Path(__file__).parent / "keyboards/data/mx_keys/mx_keys_asset.npz"


def _build_map():
    with np.load(_ASSET) as asset:
        labels = json.loads(bytes(asset["manifest_json"]))["slot_labels"]
        centers = asset["keys_center"].astype(np.float64)
    numpad = set(TYPING_KEYBOARD_POOL.numpad_slots)
    slots = []
    for char in MAPPED_CHARACTERS:
        label = "Space" if char == " " else char
        matches = [i for i, value in enumerate(labels)
                   if value.casefold() == label.casefold() and i not in numpad]
        if len(matches) != 1:
            raise ValueError(f"main-block key {char!r} maps to {matches}")
        slots.append(matches[0])
    q = np.asarray(PHYSICAL_CALIBRATED_KEYBOARD_ROTATION_XYZW, dtype=np.float64)
    local = centers[slots]
    rotated = local + 2 * np.cross(q[:3], np.cross(q[:3], local) + q[3] * local)
    world = rotated + np.asarray(PHYSICAL_CALIBRATED_KEYBOARD_POSITION_B_M)
    if not np.allclose(world[:26], PHYSICAL_CALIBRATED_AZ_XYZ_B_M, atol=1e-10, rtol=0):
        raise ValueError("MX Keys geometry no longer matches the calibrated A-Z map")
    return tuple(slots), tuple(tuple(float(v) for v in row) for row in world)


KEY_SLOTS, KEY_XYZ_B_M = _build_map()
SLOT_BY_CHARACTER = dict(zip(MAPPED_CHARACTERS, KEY_SLOTS, strict=True))
TARGET_SLOTS = tuple(SLOT_BY_CHARACTER[char] for char in TARGET_TEXT)
