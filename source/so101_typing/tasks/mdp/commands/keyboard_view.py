# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic gather maps from global key-slot order into the keyboard articulation's layout."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def _capture_int(pattern: str, text: str) -> int:
    """Return the first integer capture group of ``pattern`` in ``text`` (must match)."""
    match = re.search(pattern, text)
    assert match is not None, f"{text!r} does not match {pattern!r}"
    return int(match.group(1))


def build_keyboard_slot_gather(
    keyboard: Articulation, num_envs: int, device: torch.device | str
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-env gather indices into the articulation's ``(num_instances, dof | num_bodies)`` data.

    Maps the canonical global key-slot order (``slot = part * dof + local``) onto the physics view's
    instance/column layout, so reads become plain advanced indexing:
    ``joint_pos[inst_idx, joint_col_idx]`` and ``body_pos_w[inst_idx, body_col_idx]``.

    Works for both physics backends, which order the keyboard's part instances differently:

    * **PhysX** exposes ``root_view.prim_paths`` (non-numeric instance order); ``(env, part)`` is
      recovered by parsing each instance's prim path.
    * **Newton** exposes no ``prim_paths`` but lays instances out as a regular world-major grid
      (``world_count`` x ``count_per_world``), so ``instance = env * parts_per_env + part`` with the
      parts in authoring order (``part_000`` .. ``part_NNN``), which matches the global slot order.

    Args:
        keyboard: The keyboard articulation (``root_view`` already initialized).
        num_envs: Number of environments.
        device: Device for the returned index tensors.

    Returns:
        ``(num_keys, inst_idx, joint_col_idx, body_col_idx)`` where the three tensors have shape
        ``(num_envs, num_keys)``.
    """
    view = keyboard.root_view
    dof = len(keyboard.joint_names)
    # local slot per joint column (e.g. "key_003_joint" -> 3) and the body column of each key body.
    local_of_col = [_capture_int(r"key_(\d+)_joint", name) for name in keyboard.joint_names]
    body_col_of_local = {local: keyboard.body_names.index(f"key_{local:03d}") for local in local_of_col}

    prim_paths = getattr(view, "prim_paths", None)
    if prim_paths is not None:
        # PhysX: recover (env, part) from each instance's prim path (instance order is non-numeric).
        num_instances = len(prim_paths)
        env_part: list[tuple[int, int]] = []
        for path in prim_paths:
            part_match = re.search(r"/part_(\d+)", path)
            env_part.append((_capture_int(r"/env_(\d+)", path), int(part_match.group(1)) if part_match else 0))
    else:
        # Newton: regular world-major layout, parts in authoring order -> arithmetic (env, part).
        num_instances = int(view.count)
        per_world = max(num_instances // num_envs, 1)
        env_part = [(inst // per_world, inst % per_world) for inst in range(num_instances)]

    num_keys = (num_instances // num_envs) * dof
    inst_np = np.zeros((num_envs, num_keys), dtype=np.int64)
    joint_col_np = np.zeros((num_envs, num_keys), dtype=np.int64)
    body_col_np = np.zeros((num_envs, num_keys), dtype=np.int64)
    for inst, (env_id, part) in enumerate(env_part):
        for col, local in enumerate(local_of_col):
            slot = part * dof + local
            inst_np[env_id, slot] = inst
            joint_col_np[env_id, slot] = col
            body_col_np[env_id, slot] = body_col_of_local[local]
    return (
        num_keys,
        torch.as_tensor(inst_np, device=device),
        torch.as_tensor(joint_col_np, device=device),
        torch.as_tensor(body_col_np, device=device),
    )
