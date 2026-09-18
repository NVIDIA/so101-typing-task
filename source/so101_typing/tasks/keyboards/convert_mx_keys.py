# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline converter: remodel the Logitech MX Keys USDZ scan into the SO101 keyboard asset format.

The source asset (a Sketchfab-style FBX->USDZ conversion) bakes all 108 keycaps into one merged mesh
and the case into a handful of loose meshes. This script segments the merged keycap mesh into
individual caps, matches each cap to the generator's canonical ``ansi_full`` key-slot order, extracts
the case visuals, and writes a compact ``mx_keys_asset.npz`` (plus the source textures) that
:mod:`keyboard_mx_keys` consumes to author a generator-format articulation at spawn time.

Run once, offline (needs ``pxr``, ``numpy``, and optionally ``Pillow`` for display colors):

.. code-block:: bash

    ./isaaclab.sh -p -m isaaclab_tasks.core.dexsuite.config.so101.keyboards.convert_mx_keys \
        --source ~/Downloads/Logitech_MX_Keys.usdz

Conventions of the output (matching the procedural generator):

* Keyboard frame: Z-up, X = width (right positive), Y = depth (back positive), meters.
  Origin at the case footprint center in XY and at the case's mid-height in Z.
* Slot order: identical to ``keyboard_layouts._layout_ansi_full`` (the pool reference), matched
  band-by-band in x-order. The physical board has one extra key in the function row and one fewer
  right of the spacebar than the reference layout, so the reference ``menu`` slot is mapped onto the
  extra function-row key (recorded in the manifest).
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np

from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_layouts import LAYOUT_BUILDERS
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_schema import UnitKey

# Source mesh names inside the USDZ.
_CAPS_MESH = "Cube_003_Texture1_0"
# Case meshes bound to the "Texture2" material (shells, back strip, feet, power-switch cluster).
_CASE_MESHES_T2 = (
    "Cube_001_Texture2_0",
    "Circle_Texture2_0",
    "Dimensions_001_Texture2_0",
    "Plane_002_Texture2_0",
    "Cylinder_001_Texture2_0",
    "Cylinder_Texture2_0",
    "Cube_004_Texture2_0",
    "Circle_007_Texture2_0",
)
# Case meshes bound to the "Texture1" material (thin trim strip).
_CASE_MESHES_T1 = ("Cube_002_Texture1_0",)

# Texture roles: (usdz member, output name). Texture1 maps the keycaps, Texture2 the case.
_TEXTURES = (
    ("0/Texture1_baseColor.jpg", "caps_baseColor.jpg"),
    ("0/Texture1_normal.jpg", "caps_normal.jpg"),
    ("0/Texture1_metallicRoughness_rough.jpg", "caps_roughness.jpg"),
    ("0/Texture1_metallicRoughness_metal_scale0.jpg", "caps_metallic.jpg"),
    ("0/Texture2_baseColor.jpg", "case_baseColor.jpg"),
    ("0/Texture2_metallicRoughness_rough.jpg", "case_roughness.jpg"),
    ("0/Texture2_metallicRoughness_metal.jpg", "case_metallic.jpg"),
)

# Per-band key counts, back row first: reference layout vs the physical board.
_REFERENCE_BAND_COUNTS = (20, 21, 21, 16, 17, 13)
_MODEL_BAND_COUNTS = (21, 21, 21, 16, 17, 12)
# Reference slot moved to the physical board's extra function-row key (see module docstring).
_CROSS_MAPPED_KEY = "main_menu"
# x-order position of the extra key inside the physical back band: esc + 12 F keys precede it.
_CROSS_MAPPED_BAND = 0
_CROSS_MAPPED_INDEX = 13

_MM = 1.0e-3


def _load_meshes(source: str) -> dict[str, dict[str, np.ndarray]]:
    """Load all named meshes from the USDZ in the keyboard frame (Z-up meters, unshifted)."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(source)
    cache = UsdGeom.XformCache()
    meshes: dict[str, dict[str, np.ndarray]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName() in meshes:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        fvc = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        fvi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not (fvc == 3).all():
            raise ValueError(f"{prim.GetName()}: expected a pure-triangle mesh.")
        m = cache.GetLocalToWorldTransform(prim)
        m_np = np.array([[m[i][j] for j in range(4)] for i in range(4)])
        world = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ m_np
        # Stage frame is Y-up centimeters; keyboard frame is Z-up meters: (x, y, z) -> (x, -z, y) / 100.
        local = np.stack([world[:, 0], -world[:, 2], world[:, 1]], axis=1) * 0.01
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st0")
        st = np.asarray(pv.Get(), dtype=np.float64)
        if pv.IsIndexed():
            st = st[np.asarray(pv.GetIndices(), dtype=np.int64)]
        if len(st) != len(local):
            raise ValueError(f"{prim.GetName()}: expected vertex-interpolated st0.")
        meshes[prim.GetName()] = {"points": local, "st": st, "tris": fvi.reshape(-1, 3)}
    missing = {_CAPS_MESH, *_CASE_MESHES_T2, *_CASE_MESHES_T1} - set(meshes)
    if missing:
        raise ValueError(f"Source USD is missing expected meshes: {sorted(missing)}")
    return meshes


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, a: int) -> int:
        parent = self.parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _connected_caps(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Label each vertex with its keycap cluster id (position-welded connected components)."""
    quant = np.round(points / (0.01 * _MM)).astype(np.int64)
    _, weld = np.unique(quant, axis=0, return_inverse=True)
    uf = _UnionFind(int(weld.max()) + 1)
    for tri in weld[tris]:
        uf.union(int(tri[0]), int(tri[1]))
        uf.union(int(tri[0]), int(tri[2]))
    comp = np.array([uf.find(int(w)) for w in weld])

    # Merge components whose AABBs nearly touch (normal-split islands within one cap).
    ids = np.unique(comp)
    lo = np.stack([points[comp == c].min(axis=0) for c in ids])
    hi = np.stack([points[comp == c].max(axis=0) for c in ids])
    merge = _UnionFind(len(ids))
    gap = 0.8 * _MM
    for i in range(len(ids)):
        sep = np.maximum(lo[i + 1 :] - hi[i], lo[i] - hi[i + 1 :]).max(axis=1)
        for j in np.nonzero(sep < gap)[0]:
            merge.union(i, int(j) + i + 1)
    cluster_of_id = {c: merge.find(i) for i, c in enumerate(ids)}
    return np.array([cluster_of_id[c] for c in comp])


def _reference_bands() -> list[list[tuple[int, UnitKey]]]:
    """Return ``(slot, key)`` pairs of ``ansi_full`` grouped into 6 physical bands (back first), x-sorted."""
    keys = LAYOUT_BUILDERS["ansi_full"](random.Random(0))
    if len(keys) != 108:
        raise ValueError(f"ansi_full layout produced {len(keys)} keys; expected 108.")
    bands: dict[int, list[tuple[int, UnitKey]]] = {}
    for slot, key in enumerate(keys):
        # 2-unit-tall keys (numpad plus/enter) belong to the band of their upper half.
        band_y = key.y + key.h - 0.5 if key.h > 1.5 else key.y + 0.5
        bands.setdefault(int(round(5.5 - band_y)), []).append((slot, key))
    ordered = [sorted(bands[b], key=lambda sk: sk[1].x) for b in sorted(bands)]
    if tuple(len(b) for b in ordered) != _REFERENCE_BAND_COUNTS:
        raise ValueError(f"Unexpected ansi_full band sizes: {[len(b) for b in ordered]}")
    return ordered


def _match_slots(centers_deck: np.ndarray, sizes_deck: np.ndarray) -> tuple[list[int], list[list[tuple[int, UnitKey]]]]:
    """Match physical caps to reference slots.

    Args:
        centers_deck: Deck-frame cap centers [m], shape [num_caps, 3].
        sizes_deck: Deck-frame cap bbox sizes [m], shape [num_caps, 3].

    Returns:
        ``(cap_of_slot, bands)`` where ``cap_of_slot[slot]`` is the cap index serving reference slot
        ``slot`` and ``bands`` is the adjusted reference banding used for the match.
    """
    bands = _reference_bands()
    # Move the cross-mapped reference key into the physical back band at its known x-position.
    cross = [sk for sk in bands[5] if sk[1].name == _CROSS_MAPPED_KEY]
    if len(cross) != 1:
        raise ValueError(f"Reference layout has no unique '{_CROSS_MAPPED_KEY}' key.")
    bands[5] = [sk for sk in bands[5] if sk[1].name != _CROSS_MAPPED_KEY]
    bands[_CROSS_MAPPED_BAND] = (
        bands[_CROSS_MAPPED_BAND][:_CROSS_MAPPED_INDEX] + cross + bands[_CROSS_MAPPED_BAND][_CROSS_MAPPED_INDEX:]
    )
    if tuple(len(b) for b in bands) != _MODEL_BAND_COUNTS:
        raise ValueError(f"Adjusted reference band sizes {[len(b) for b in bands]} != {_MODEL_BAND_COUNTS}")

    # Band the physical caps by deck-frame y. Regular caps cluster cleanly onto 6 row centers;
    # 2-unit-tall caps (numpad plus/enter) sit between rows and attach to their upper band.
    tall = sizes_deck[:, 1] > 25.0 * _MM
    row_groups: list[list[float]] = []
    for y in sorted(centers_deck[~tall, 1], reverse=True):
        if not row_groups or row_groups[-1][-1] - y > 9.0 * _MM:
            row_groups.append([y])
        else:
            row_groups[-1].append(y)
    if len(row_groups) != 6:
        raise ValueError(f"Expected 6 key rows, found {len(row_groups)}.")
    row_y = np.array([np.mean(r) for r in row_groups])

    band_members: list[list[int]] = [[] for _ in range(6)]
    for idx, (center, is_tall) in enumerate(zip(centers_deck, tall)):
        if is_tall:
            above = np.nonzero(row_y > center[1])[0]
            if len(above) == 0:
                raise ValueError("Tall keycap found above the back row.")
            band = int(above[-1])
        else:
            band = int(np.argmin(np.abs(row_y - center[1])))
        band_members[band].append(idx)
    counts = tuple(len(b) for b in band_members)
    if counts != _MODEL_BAND_COUNTS:
        raise ValueError(f"Physical band sizes {counts} != expected {_MODEL_BAND_COUNTS}")

    cap_of_slot = [-1] * 108
    for band, members in zip(bands, band_members):
        for (slot, _), cap_idx in zip(band, sorted(members, key=lambda i: centers_deck[i, 0])):
            cap_of_slot[slot] = cap_idx
    if sorted(cap_of_slot) != list(range(108)):
        raise ValueError("Slot matching is not a bijection onto the 108 caps.")
    return cap_of_slot, bands


def _rot_x(tilt: float) -> np.ndarray:
    c, s = np.cos(tilt), np.sin(tilt)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _sample_color(img: np.ndarray | None, st: np.ndarray, floor: float) -> np.ndarray:
    """Mean texture color at the given UVs (V flipped), clamped to a display-friendly floor."""
    if img is None or len(st) == 0:
        return np.full(3, max(0.5, floor))
    h, w, _ = img.shape
    px = np.clip((st[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
    py = np.clip(((1.0 - st[:, 1]) % 1.0) * (h - 1), 0, h - 1).astype(int)
    return np.maximum(img[py, px].mean(axis=0), floor)


def _load_texture(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        print("Pillow not available; falling back to flat display colors.")
        return None
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


# Keycap base-color remap: the scan's bake is crushed (cap body ~0.0001, legends ~0.37 gray), so a
# faithful render shows near-black caps with dim gray legends. Remap to the physical product's look.
_CAPS_BODY_LEVEL = 0.16
"""Remapped keycap body brightness (graphite plastic)."""
_CAPS_LEGEND_RANGE = (0.03, 0.45)
"""Source luminance range mapped onto body->white; legends saturate to white above the top end."""


def _remap_caps_basecolor(path: Path) -> None:
    """Rewrite the keycap base-color texture: neutral graphite body with white legends.

    The remap is luminance-based (the source legends are already chroma-free), so glyph antialiasing
    is preserved while the caps render as graphite with white legends in any textured viewer.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        print("Pillow not available; keeping the source keycap base color.")
        return
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    luminance = img @ (0.2126, 0.7152, 0.0722)
    lo, hi = _CAPS_LEGEND_RANGE
    t = np.clip((luminance - lo) / (hi - lo), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)  # smoothstep keeps the glyph edge antialiasing
    out = _CAPS_BODY_LEVEL + (1.0 - _CAPS_BODY_LEVEL) * t
    Image.fromarray((np.repeat(out[..., None], 3, axis=2) * 255.0).round().astype(np.uint8)).save(path, quality=95)


def convert(source: str, out_dir: Path) -> None:
    """Convert the MX Keys USDZ at ``source`` into ``out_dir/mx_keys_asset.npz`` plus textures."""
    meshes = _load_meshes(source)

    # Extract textures first so display colors can be sampled from them.
    tex_dir = out_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as zf:
        for member, dst_name in _TEXTURES:
            with zf.open(member) as f:
                (tex_dir / dst_name).write_bytes(f.read())
    _remap_caps_basecolor(tex_dir / "caps_baseColor.jpg")
    caps_img = _load_texture(tex_dir / "caps_baseColor.jpg")
    case_img = _load_texture(tex_dir / "case_baseColor.jpg")

    caps = meshes[_CAPS_MESH]
    cluster_of_pt = _connected_caps(caps["points"], caps["tris"])
    cluster_ids = np.unique(cluster_of_pt)
    if len(cluster_ids) != 108:
        raise ValueError(f"Keycap segmentation produced {len(cluster_ids)} clusters; expected 108.")

    # Deck plane: fit the tilt (about X) from the plane of all cap centers.
    centers_raw = np.stack(
        [
            (caps["points"][cluster_of_pt == c].min(axis=0) + caps["points"][cluster_of_pt == c].max(axis=0)) / 2
            for c in cluster_ids
        ]
    )
    fit = centers_raw - centers_raw.mean(axis=0)
    _, _, vt = np.linalg.svd(fit, full_matrices=False)
    normal = vt[2] if vt[2][2] > 0 else -vt[2]
    deck_tilt = float(np.arctan2(-normal[1], normal[2]))
    rot = _rot_x(deck_tilt)  # deck frame -> keyboard frame
    deck_quat = np.array([np.sin(deck_tilt / 2.0), 0.0, 0.0, np.cos(deck_tilt / 2.0)])

    # Keyboard origin: case footprint center in XY, case mid-height in Z.
    case_pts_all = np.concatenate([meshes[name]["points"] for name in (*_CASE_MESHES_T2, *_CASE_MESHES_T1)])
    origin = (case_pts_all.min(axis=0) + case_pts_all.max(axis=0)) / 2

    # Per-cap deck-frame geometry (relative to the shifted origin).
    centers_deck_list, sizes_deck_list = [], []
    for c in cluster_ids:
        p = (caps["points"][cluster_of_pt == c] - origin) @ rot
        lo, hi = p.min(axis=0), p.max(axis=0)
        centers_deck_list.append((lo + hi) / 2)
        sizes_deck_list.append(hi - lo)
    centers_deck = np.stack(centers_deck_list)
    sizes_deck = np.stack(sizes_deck_list)

    cap_of_slot, _ = _match_slots(centers_deck, sizes_deck)
    flat_keys = LAYOUT_BUILDERS["ansi_full"](random.Random(0))

    key_points, key_st, key_tris = [], [], []
    key_point_offsets, key_tri_offsets = [0], [0]
    keys_center = np.zeros((108, 3))
    keys_center_deck = np.zeros((108, 3))
    keys_size = np.zeros((108, 3))
    keys_color = np.zeros((108, 3))
    for slot in range(108):
        cap = cap_of_slot[slot]
        mask = cluster_of_pt == cluster_ids[cap]
        keys_center_deck[slot] = centers_deck[cap]
        keys_center[slot] = rot @ centers_deck[cap]
        keys_size[slot] = sizes_deck[cap]
        # Key-link LOCAL frame: the link is authored with the deck orientation, so the mesh points
        # must be expressed in the deck frame relative to the cap center (not keyboard-frame offsets,
        # which would pitch every cap by the deck tilt a second time when the link pose is applied).
        pts = (caps["points"][mask] - origin) @ rot - centers_deck[cap]
        # Remap this cluster's triangles to cluster-local vertex indices.
        tri_mask = mask[caps["tris"]].all(axis=1)
        remap = np.full(len(mask), -1, dtype=np.int64)
        remap[np.nonzero(mask)[0]] = np.arange(int(mask.sum()))
        tris = remap[caps["tris"][tri_mask]]
        st = caps["st"][mask]
        top = pts[:, 2] > pts[:, 2].max() - 0.6 * _MM
        keys_color[slot] = _sample_color(caps_img, st[top], floor=0.08)
        key_points.append(pts.astype(np.float32))
        key_st.append(st.astype(np.float32))
        key_tris.append(tris.astype(np.int32))
        key_point_offsets.append(key_point_offsets[-1] + len(pts))
        key_tri_offsets.append(key_tri_offsets[-1] + len(tris))

    # Case visual groups (points relative to the keyboard origin).
    def _merge(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points, st, tris, base = [], [], [], 0
        for name in names:
            mesh = meshes[name]
            points.append((mesh["points"] - origin).astype(np.float32))
            st.append(mesh["st"].astype(np.float32))
            tris.append((mesh["tris"] + base).astype(np.int32))
            base += len(mesh["points"])
        return np.concatenate(points), np.concatenate(st), np.concatenate(tris)

    case_points_0, case_st_0, case_tris_0 = _merge(_CASE_MESHES_T2)
    case_points_1, case_st_1, case_tris_1 = _merge(_CASE_MESHES_T1)
    top_sel = case_points_0[:, 2] > case_points_0[:, 2].max() - 3.0 * _MM
    case_color_0 = _sample_color(case_img, case_st_0[top_sel], floor=0.10)
    case_color_1 = _sample_color(caps_img, case_st_1, floor=0.10)

    case_lo = case_pts_all.min(axis=0) - origin
    case_hi = case_pts_all.max(axis=0) - origin

    manifest = {
        "source": str(source),
        "slot_names": [key.name for key in flat_keys],
        "slot_labels": [key.label for key in flat_keys],
        "slot_groups": [key.group for key in flat_keys],
        "cross_mapped": {
            _CROSS_MAPPED_KEY: f"extra function-row key (band {_CROSS_MAPPED_BAND}, x-index {_CROSS_MAPPED_INDEX})"
        },
        "deck_tilt_rad": deck_tilt,
        "case_min_m": case_lo.tolist(),
        "case_max_m": case_hi.tolist(),
    }

    np.savez_compressed(
        out_dir / "mx_keys_asset.npz",
        manifest_json=np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8),
        deck_quat_xyzw=deck_quat,
        case_min=case_lo,
        case_max=case_hi,
        keys_center=keys_center,
        keys_center_deck=keys_center_deck,
        keys_size=keys_size,
        keys_color=keys_color,
        key_points=np.concatenate(key_points),
        key_st=np.concatenate(key_st),
        key_tris=np.concatenate(key_tris),
        key_point_offsets=np.asarray(key_point_offsets, dtype=np.int64),
        key_tri_offsets=np.asarray(key_tri_offsets, dtype=np.int64),
        case_points_0=case_points_0,
        case_st_0=case_st_0,
        case_tris_0=case_tris_0,
        case_color_0=case_color_0,
        case_points_1=case_points_1,
        case_st_1=case_st_1,
        case_tris_1=case_tris_1,
        case_color_1=case_color_1,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    _report(manifest, keys_center, keys_size, case_lo, case_hi, deck_tilt)


def _report(manifest: dict, keys_center: np.ndarray, keys_size: np.ndarray, case_lo, case_hi, deck_tilt) -> None:
    size = (case_hi - case_lo) * 1000
    print(f"deck tilt: {np.degrees(deck_tilt):.2f} deg")
    z_span = f"z {case_lo[2] * 1000:.1f}..{case_hi[2] * 1000:.1f}"
    print(f"case size: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm  ({z_span})")
    print(f"key centers z: {keys_center[:, 2].min() * 1000:.1f}..{keys_center[:, 2].max() * 1000:.1f} mm")
    labels = manifest["slot_labels"]
    print("sample slots:")
    for slot in (0, 13, 27, 40, 53, 61, 66, 74, 78, 87, 91, 107):
        c = keys_center[slot] * 1000
        s = keys_size[slot] * 1000
        print(
            f"  slot {slot:3d} {labels[slot]:>10s}: center=({c[0]:7.1f},{c[1]:6.1f},{c[2]:5.1f})mm"
            f" size=({s[0]:.1f} x {s[1]:.1f} x {s[2]:.1f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Logitech MX Keys USDZ into mx_keys_asset.npz")
    parser.add_argument("--source", required=True, help="Path to Logitech_MX_Keys.usdz")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "data" / "mx_keys"),
        help="Output directory for mx_keys_asset.npz and textures/",
    )
    args = parser.parse_args()
    convert(args.source, Path(args.out))


if __name__ == "__main__":
    main()
