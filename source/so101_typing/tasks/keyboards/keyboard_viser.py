# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viser drawing helpers for no-sim keyboard inspection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .keyboard_geometry import (
    base_profile_visuals,
    is_identity_quat,
    mat3_rotate,
    quat_rotate,
    quat_to_matrix,
)
from .keyboard_labels import label_mesh_data
from .keyboard_schema import ResolvedKeyboard


def draw_keyboard_grid_viser(
    server: Any,
    keyboards: Sequence[ResolvedKeyboard],
    *,
    spacing: float | None = None,
    show_press_guides: bool = False,
) -> None:
    if not keyboards:
        return
    columns = math.ceil(math.sqrt(len(keyboards)))
    rows = math.ceil(len(keyboards) / columns)
    if spacing is None:
        max_width = max(keyboard.case_size[0] for keyboard in keyboards)
        max_depth = max(keyboard.case_size[1] for keyboard in keyboards)
        spacing = max(max_width, max_depth) + 0.08
    if spacing <= 0.0:
        raise ValueError("inspect spacing must be > 0.")

    for index, keyboard in enumerate(keyboards):
        row, col = divmod(index, columns)
        offset = (
            (col - (columns - 1) * 0.5) * spacing,
            -((row - (rows - 1) * 0.5) * spacing),
            0.0,
        )
        root = f"/keyboards/{index:02d}_{keyboard.family}_{keyboard.style_name}_seed{keyboard.seed}"
        draw_keyboard_viser(server, keyboard, root=root, offset=offset, show_press_guides=show_press_guides)


def draw_keyboard_viser(
    server: Any,
    keyboard: ResolvedKeyboard,
    *,
    root: str = "/keyboard",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    show_press_guides: bool = False,
) -> None:
    def translate(position: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(position[i] + offset[i] for i in range(3))

    try:
        server.scene.add_frame(f"{root}/frame", position=offset, axes_length=0.05, axes_radius=0.0015)
    except TypeError:
        server.scene.add_frame(f"{root}/frame", axes_length=0.05, axes_radius=0.0015)
    _viser_box(
        server,
        f"{root}/case",
        translate(keyboard.case_center),
        keyboard.case_size,
        keyboard.deck_quat_xyzw,
        keyboard.style.case_color,
    )
    _viser_box(
        server,
        f"{root}/plate",
        translate(keyboard.plate_center),
        keyboard.plate_size,
        keyboard.deck_quat_xyzw,
        keyboard.style.plate_color,
    )
    for visual in base_profile_visuals(keyboard):
        _viser_box(
            server,
            f"{root}/base/{visual.name}",
            translate(visual.center),
            visual.size,
            visual.quat_xyzw,
            visual.color,
        )
    for key in keyboard.keys:
        if not key.active:
            continue
        key_center = translate(key.center)
        _viser_box(server, f"{root}/keys/{key.slot:03d}_{key.name}", key_center, key.size, key.quat_xyzw, key.color)
        if show_press_guides:
            press_vec = quat_rotate(key.quat_xyzw, (0.0, 0.0, key.lower_limit))
            pressed_center = tuple(key_center[i] + press_vec[i] for i in range(3))
            _viser_box(
                server,
                f"{root}/pressed/{key.slot:03d}_{key.name}",
                pressed_center,
                key.collision_size,
                key.quat_xyzw,
                tuple(min(c + 0.25, 1.0) for c in key.color),
                opacity=0.25,
            )
            axis = quat_rotate(key.quat_xyzw, (0.0, 0.0, -key.travel))
            try:
                server.scene.add_spline_catmull_rom(
                    f"{root}/axes/{key.slot:03d}",
                    positions=[key_center, tuple(key_center[i] + axis[i] for i in range(3))],
                    color=(255, 210, 80),
                    line_width=1.5,
                )
            except AttributeError:
                pass
    _viser_labels_mesh(server, f"{root}/labels", keyboard, offset)

def _viser_labels_mesh(
    server: Any,
    name: str,
    keyboard: ResolvedKeyboard,
    offset: tuple[float, float, float],
) -> None:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for key in keyboard.active_keys:
        local_vertices, faces = label_mesh_data(key, keyboard.style)
        if not local_vertices or not faces:
            continue
        base = len(vertices)
        center = tuple(key.center[i] + offset[i] for i in range(3))
        if is_identity_quat(key.quat_xyzw):
            vertices.extend(tuple(center[i] + vertex[i] for i in range(3)) for vertex in local_vertices)
        else:
            rotation = quat_to_matrix(key.quat_xyzw)
            vertices.extend(
                tuple(center[i] + mat3_rotate(rotation, vertex)[i] for i in range(3))
                for vertex in local_vertices
            )
        for a, b, c, d in faces:
            triangles.append((base + a, base + b, base + c))
            triangles.append((base + a, base + c, base + d))
    if not vertices or not triangles:
        return
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return

    color = tuple(int(max(0.0, min(1.0, c)) * 255) for c in keyboard.style.label_color)
    server.scene.add_mesh_simple(
        name,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(triangles, dtype=np.uint32),
        color=color,
        flat_shading=True,
        side="double",
        cast_shadow=False,
        receive_shadow=False,
    )

def _viser_box(
    server: Any,
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
    color: tuple[float, float, float],
    *,
    opacity: float = 1.0,
) -> None:
    rgba = tuple(int(max(0.0, min(1.0, c)) * 255) for c in color)
    wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
    try:
        server.scene.add_box(name, dimensions=size, position=center, wxyz=wxyz, color=rgba, opacity=opacity)
    except TypeError:
        server.scene.add_box(name, dimensions=size, position=center, wxyz=wxyz, color=rgba)
