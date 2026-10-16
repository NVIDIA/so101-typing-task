# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime spawner for the remodeled Logitech MX Keys keyboard.

Consumes the ``mx_keys_asset.npz`` produced by :mod:`convert_mx_keys` and authors a keyboard
articulation through the shared authoring path in :mod:`keyboard_usd`, so the physics structure
(link/joint naming, prismatic joints, drives) is identical to the procedural generator's. Only the
gprims differ: segmented scan meshes for the keycap and case visuals plus box colliders sized from
the scan, with gprim counts matching the generator exactly (required by Newton's shared
ArticulationView across the heterogeneous pool).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from .keyboard_geometry import _BASE_VISUAL_SLOTS, _validate_resolved, quat_rotate
from .keyboard_schema import KeyboardStyle, ResolvedKey, ResolvedKeyboard
from .keyboard_usd import _define_box, author_keyboard

if TYPE_CHECKING:
    from .keyboard_gen_cfg import MXKeysKeyboardSpawnerCfg

_MM = 1.0e-3
_KEY_SLOT_COUNT = 108
# Extra travel clearance [m] between a fully pressed keycap collider and the case colliders.
_TRAVEL_CLEARANCE = 5.0e-4


@dataclass(frozen=True)
class MXKeysAsset:
    """In-memory converted MX Keys asset (see :mod:`convert_mx_keys` for the frame conventions)."""

    asset_dir: str
    manifest: dict
    deck_quat_xyzw: tuple[float, float, float, float]
    deck_tilt: float
    case_min: np.ndarray
    case_max: np.ndarray
    keys_center: np.ndarray
    keys_center_deck: np.ndarray
    keys_size: np.ndarray
    keys_color: np.ndarray
    key_points: np.ndarray
    key_st: np.ndarray
    key_tris: np.ndarray
    key_point_offsets: np.ndarray
    key_tri_offsets: np.ndarray
    case_meshes: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]

    def key_mesh(self, slot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(points, st, triangles)`` of the keycap visual for a global slot."""
        p0, p1 = self.key_point_offsets[slot], self.key_point_offsets[slot + 1]
        t0, t1 = self.key_tri_offsets[slot], self.key_tri_offsets[slot + 1]
        return self.key_points[p0:p1], self.key_st[p0:p1], self.key_tris[t0:t1]


@lru_cache(maxsize=2)
def load_mx_keys_asset(asset_dir: str) -> MXKeysAsset:
    """Load and cache the converted MX Keys asset from ``asset_dir``.

    Raises:
        FileNotFoundError: If the asset has not been generated (run :mod:`convert_mx_keys`).
    """
    path = f"{asset_dir}/mx_keys_asset.npz"
    try:
        data = np.load(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"MX Keys asset not found at '{path}'. Generate it with: ./isaaclab.sh -p -m "
            "isaaclab_tasks.core.dexsuite.config.so101.keyboards.convert_mx_keys --source <usdz>"
        ) from exc
    manifest = json.loads(bytes(data["manifest_json"]))
    if len(manifest["slot_labels"]) != _KEY_SLOT_COUNT:
        raise ValueError(f"MX Keys asset has {len(manifest['slot_labels'])} slots; expected {_KEY_SLOT_COUNT}.")
    return MXKeysAsset(
        asset_dir=asset_dir,
        manifest=manifest,
        deck_quat_xyzw=tuple(float(v) for v in data["deck_quat_xyzw"]),
        deck_tilt=float(manifest["deck_tilt_rad"]),
        case_min=data["case_min"],
        case_max=data["case_max"],
        keys_center=data["keys_center"],
        keys_center_deck=data["keys_center_deck"],
        keys_size=data["keys_size"],
        keys_color=data["keys_color"],
        key_points=data["key_points"],
        key_st=data["key_st"],
        key_tris=data["key_tris"],
        key_point_offsets=data["key_point_offsets"],
        key_tri_offsets=data["key_tri_offsets"],
        case_meshes=(
            (data["case_points_0"], data["case_st_0"], data["case_tris_0"], data["case_color_0"]),
            (data["case_points_1"], data["case_st_1"], data["case_tris_1"], data["case_color_1"]),
        ),
    )


def mx_keys_slot_labels(asset_dir: str) -> tuple[str, ...]:
    """Per-slot legend labels of the converted asset, for cross-checks against the pool reference."""
    return tuple(load_mx_keys_asset(asset_dir).manifest["slot_labels"])


def resolve_mx_keys_keyboard(cfg: MXKeysKeyboardSpawnerCfg) -> ResolvedKeyboard:
    """Resolve the spawner cfg plus the converted asset into generator-format keyboard geometry."""
    asset = load_mx_keys_asset(cfg.asset_dir)
    if cfg.partition_mode == "fixed_dof":
        if _KEY_SLOT_COUNT % cfg.partition_dof != 0:
            raise ValueError(f"partition_dof={cfg.partition_dof} must divide {_KEY_SLOT_COUNT} for MX Keys.")
        partition_count = _KEY_SLOT_COUNT // cfg.partition_dof
    elif cfg.partition_mode == "single":
        partition_count = 1
    else:
        raise ValueError(f"Unsupported partition_mode '{cfg.partition_mode}'.")

    manifest = asset.manifest
    case_size = tuple(float(v) for v in (asset.case_max - asset.case_min))
    style = KeyboardStyle(
        name="mx_keys",
        pitch=0.01905,
        gap=0.0026,
        cap_height=float(asset.keys_size[:, 2].mean()),
        cap_top_scale=1.0,
        travel=cfg.travel,
        upper_limit=cfg.rest_upper_limit,
        cap_clearance=0.0,
        case_margin_u=0.0,
        base_thickness=case_size[2],
        plate_thickness=0.001,
        key_mass=cfg.key_mass,
        base_mass=cfg.base_mass,
        stiffness=cfg.stiffness,
        damping=cfg.damping,
        max_force=cfg.max_force,
        case_color=tuple(asset.case_meshes[0][3]),
        plate_color=tuple(asset.case_meshes[1][3]),
        alpha_color=(0.08, 0.08, 0.08),
        modifier_color=(0.08, 0.08, 0.08),
        accent_color=(0.08, 0.08, 0.08),
        label_color=(0.9, 0.9, 0.9),
        label_scale=1.0,
        label_pixel_fill=1.0,
        label_mode="none",
        label_spacebar=False,
        base_profile="slab",
        deck_tilt=asset.deck_tilt,
    )

    keys = []
    for slot in range(_KEY_SLOT_COUNT):
        size = tuple(float(v) for v in asset.keys_size[slot])
        collision_size = (
            max(size[0] - 2.0 * cfg.key_collision_margin, size[0] * 0.25),
            max(size[1] - 2.0 * cfg.key_collision_margin, size[1] * 0.25),
            cfg.key_collision_thickness,
        )
        keys.append(
            ResolvedKey(
                slot=slot,
                active=True,
                name=manifest["slot_names"][slot],
                label=manifest["slot_labels"][slot],
                group=manifest["slot_groups"][slot],
                center=tuple(float(v) for v in asset.keys_center[slot]),
                size=size,
                collision_size=collision_size,
                quat_xyzw=asset.deck_quat_xyzw,
                travel=cfg.travel,
                lower_limit=-cfg.travel,
                upper_limit=cfg.rest_upper_limit,
                mass=cfg.key_mass,
                stiffness=cfg.stiffness,
                damping=cfg.damping,
                max_force=cfg.max_force,
                color=tuple(float(v) for v in asset.keys_color[slot]),
            )
        )

    warnings = tuple(_validate_resolved(keys, case_size))
    if cfg.strict_validation and warnings:
        raise ValueError("Invalid MX Keys keyboard:\n  - " + "\n  - ".join(warnings))
    return ResolvedKeyboard(
        family="mx_keys",
        style_name="mx_keys",
        seed=0,
        topology_mode="exact",
        active_key_count=_KEY_SLOT_COUNT,
        slot_count=_KEY_SLOT_COUNT,
        partition_mode=cfg.partition_mode,
        partition_dof=cfg.partition_dof,
        partition_tail_policy=cfg.partition_tail_policy,
        partition_count=partition_count,
        keys=tuple(keys),
        case_center=(0.0, 0.0, 0.0),
        case_size=case_size,
        plate_center=(0.0, 0.0, 0.0),
        plate_size=case_size,
        deck_quat_xyzw=asset.deck_quat_xyzw,
        style=style,
        warnings=warnings,
    )


def spawn_mx_keys_keyboard(
    prim_path: str,
    cfg: MXKeysKeyboardSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Any:
    """Author the remodeled MX Keys keyboard articulation into a USD stage.

    Drop-in analog of :func:`keyboard_usd.spawn_keyboard` for the converted scan asset; supports the
    same ``single`` and ``fixed_dof`` partition modes.
    """
    asset = load_mx_keys_asset(cfg.asset_dir)
    resolved = resolve_mx_keys_keyboard(cfg)
    visuals = _MXKeysVisuals(asset, cfg, prim_path)
    return author_keyboard(
        prim_path, resolved, cfg, translation=translation, orientation=orientation, visuals=visuals, **kwargs
    )


class _MXKeysVisuals:
    """Authors MX Keys scan meshes and colliders with generator-identical gprim counts."""

    def __init__(self, asset: MXKeysAsset, cfg: MXKeysKeyboardSpawnerCfg, root_path: str):
        self._asset = asset
        self._cfg = cfg
        self._root_path = root_path

    # -- case ------------------------------------------------------------------------------------

    def author_base(self, stage, base_path: str, resolved: ResolvedKeyboard, cfg: Any, *, collision: bool) -> None:
        # MJWarp requires homogeneous worlds, and Newton's importer orders visual shapes and
        # collision shapes separately - so BOTH sub-sequences must match the generator base
        # position-by-position: visuals [mesh, mesh, box x 7] and exactly one collision box.
        asset = self._asset
        caps_mat, case_mat = _author_materials(stage, self._root_path, asset)
        _define_trimesh(stage, f"{base_path}/case_visual", *asset.case_meshes[0][:3], asset.case_meshes[0][3], case_mat)
        _define_trimesh(stage, f"{base_path}/trim_visual", *asset.case_meshes[1][:3], asset.case_meshes[1][3], caps_mat)

        center, size, quat = _case_collision_box(asset, self._cfg)
        case_color = tuple(float(v) for v in asset.case_meshes[0][3])
        _define_box(stage, f"{base_path}/case_collision", center, size, quat, case_color, collision=collision)

        # Tiny visual-only boxes filling the generator's plate + trim visual slots.
        pad_count = 1 + _BASE_VISUAL_SLOTS
        bottom = (0.0, 0.0, float(asset.case_min[2]))
        for index in range(pad_count):
            _define_box(stage, f"{base_path}/trim_pad_{index:02d}", bottom, (1.0e-5,) * 3, _IDENTITY, case_color)

    # -- keys ------------------------------------------------------------------------------------

    def author_key(self, stage, link_path: str, key: ResolvedKey, resolved: ResolvedKeyboard, cfg: Any) -> None:
        caps_mat, _ = _author_materials(stage, self._root_path, self._asset)
        points, st, tris = self._asset.key_mesh(key.slot)
        _define_trimesh(stage, f"{link_path}/visual", points, st, tris, key.color, caps_mat)
        # Collider top face flush with the cap top: the visual is a thin shell, so the collider
        # extends below it to give the fingertip a solid press target.
        offset_z = key.size[2] / 2.0 - key.collision_size[2] / 2.0
        _define_box(
            stage,
            f"{link_path}/collision",
            (0.0, 0.0, offset_z),
            key.collision_size,
            _IDENTITY,
            key.color,
            collision=True,
        )
        _define_tiny_label(stage, f"{link_path}/label", key.size[2] / 2.0, resolved.style.label_color)


_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _case_collision_box(asset: MXKeysAsset, cfg: MXKeysKeyboardSpawnerCfg):
    """Compute the deck-tilted case collision slab from the cap field and the spawner's travel.

    Returns:
        ``(center, size, quat_xyzw)`` in the keyboard frame. The slab is deck-aligned and its top
        face sits just below the pressed keycap colliders' underside plane, so it backs a press
        into the gaps between keys while leaving the full key travel free. The keyboard base is
        fixed-jointed to the world (it never rests under gravity), so a single tilted slab is
        sufficient - mirroring the generator's single ``case_collision`` box.
    """
    centers, sizes = asset.keys_center_deck, asset.keys_size
    # Deck-frame plane of the pressed cap colliders' undersides.
    collider_bottom = centers[:, 2] + sizes[:, 2] / 2.0 - cfg.key_collision_thickness
    pressed_bottom = float(collider_bottom.min()) - cfg.travel - _TRAVEL_CLEARANCE
    if pressed_bottom < float(asset.case_min[2]):
        raise ValueError(f"MX Keys travel {cfg.travel} presses keycaps below the case underside.")

    # Deck-frame span covering the whole case footprint (the small tilt makes kb-y ~ deck-y).
    inset = max(cfg.case_collision_margin, 2.0 * _MM)
    x_lo, x_hi = float(asset.case_min[0]) + inset, float(asset.case_max[0]) - inset
    y_lo, y_hi = float(asset.case_min[1]) + inset, float(asset.case_max[1]) - inset
    # Thick enough to fill the case below the deck; poking slightly under the case bottom at the
    # front is fine since the fixed-base board never contacts the ground.
    thickness = 6.0 * _MM
    center_deck = ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0, pressed_bottom - thickness / 2.0)
    center_kb = quat_rotate(asset.deck_quat_xyzw, center_deck)
    return center_kb, (x_hi - x_lo, y_hi - y_lo, thickness), asset.deck_quat_xyzw


@lru_cache(maxsize=1)
def _shade_modules():
    from pxr import Sdf, UsdGeom, UsdShade, Vt  # noqa: PLC0415

    return Sdf, UsdGeom, UsdShade, Vt


def _define_trimesh(stage, path: str, points: np.ndarray, st: np.ndarray, tris: np.ndarray, color, material) -> Any:
    """Author a triangle mesh with a ``st`` primvar, display color, and material binding."""
    from pxr import Gf  # noqa: PLC0415

    Sdf, UsdGeom, UsdShade, Vt = _shade_modules()
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(tris.reshape(-1), dtype=np.int32)))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    primvar.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(st, dtype=np.float32)))
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*(float(v) for v in color))])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh.GetPrim()


def _define_tiny_label(stage, path: str, top_z: float, color) -> Any:
    """Author the generator's effectively-invisible label quad (legends are baked in the texture)."""
    from pxr import Gf  # noqa: PLC0415

    _, UsdGeom, _, Vt = _shade_modules()
    eps = 1.0e-5
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-eps, -eps, top_z),
                Gf.Vec3f(eps, -eps, top_z),
                Gf.Vec3f(eps, eps, top_z),
                Gf.Vec3f(-eps, eps, top_z),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*(float(v) for v in color))])
    return mesh.GetPrim()


def _author_materials(stage, root_path: str, asset: MXKeysAsset):
    """Author (once) and return the caps/case ``UsdPreviewSurface`` materials under the keyboard root."""
    Sdf, UsdGeom, UsdShade, _ = _shade_modules()
    looks_path = f"{root_path}/Looks"
    caps_path, case_path = f"{looks_path}/mx_caps", f"{looks_path}/mx_case"
    if stage.GetPrimAtPath(caps_path).IsValid():
        return UsdShade.Material(stage.GetPrimAtPath(caps_path)), UsdShade.Material(stage.GetPrimAtPath(case_path))

    UsdGeom.Scope.Define(stage, looks_path)
    materials = []
    for mat_path, prefix, with_normal in ((caps_path, "caps", True), (case_path, "case", False)):
        material = UsdShade.Material.Define(stage, mat_path)
        # Neutral tint for Newton's viewer: its GL shader multiplies the per-shape color with the
        # albedo texture, and shapes whose diffuse is texture-connected otherwise fall back to
        # Newton's hashed per-shape palette (randomly tinted keycaps). Newton's USD parser merges
        # this unconnected Material-level input as the shape color; Hydra ignores it.
        material.CreateInput("displayColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
        surface = UsdShade.Shader.Define(stage, f"{mat_path}/pbr")
        surface.CreateIdAttr("UsdPreviewSurface")
        reader = UsdShade.Shader.Define(stage, f"{mat_path}/st_reader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader_out = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        def _texture(name: str, file_name: str, color_space: str):
            tex = UsdShade.Shader.Define(stage, f"{mat_path}/{name}")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"{asset.asset_dir}/textures/{file_name}")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader_out)
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            return tex

        base = _texture("tex_base", f"{prefix}_baseColor.jpg", "sRGB")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            base.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
        rough = _texture("tex_roughness", f"{prefix}_roughness.jpg", "raw")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
            rough.CreateOutput("r", Sdf.ValueTypeNames.Float)
        )
        metal = _texture("tex_metallic", f"{prefix}_metallic.jpg", "raw")
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
            metal.CreateOutput("r", Sdf.ValueTypeNames.Float)
        )
        if with_normal:
            normal = _texture("tex_normal", f"{prefix}_normal.jpg", "raw")
            normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 2.0))
            normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, -1.0))
            surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            )
        material.CreateSurfaceOutput().ConnectToSource(surface.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        materials.append(material)
    return materials[0], materials[1]
