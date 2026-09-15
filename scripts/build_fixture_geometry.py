"""Regenerate the small host-side drawing asset inside the pinned container.

Requires the container's numpy and pxr; ordinary `so101 fixture` is stdlib-only.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/'


def hull(points):
    points = sorted(set(tuple(float(v) for v in p) for p in points))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def half(seq):
        result = []
        for p in seq:
            while len(result) >= 2 and cross(result[-2], result[-1], p) <= 0:
                result.pop()
            result.append(p)
        return result
    return half(points)[:-1] + half(reversed(points))[:-1]


def rotation(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-usd', type=Path, default=Path('/opt/so101/assets/so101.usd'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = ROOT / CONFIG / 'physical_calibrated_20260718_env_cfg.py'
    constants = {}
    for node in ast.parse(cfg.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    position = np.array(constants['PHYSICAL_CALIBRATED_KEYBOARD_POSITION_B_M'])
    quat = constants['PHYSICAL_CALIBRATED_KEYBOARD_ROTATION_XYZW']
    rot = rotation(quat)
    asset_path = ROOT / CONFIG / 'keyboards/data/mx_keys/mx_keys_asset.npz'
    asset = np.load(asset_path)
    manifest = json.loads(asset['manifest_json'].tobytes().decode('utf-8'))
    deck = rotation(asset['deck_quat_xyzw'])
    stage = Usd.Stage.Open(str(args.robot_usd))
    cache = UsdGeom.XformCache()
    base = None
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and '/base_link/visuals/base_so101_v2/' in str(prim.GetPath()):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            base = (np.c_[points, np.ones(len(points))] @ np.array(cache.GetLocalToWorldTransform(prim)))[:, :3]
    if base is None or UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('expected pinned Z-up metre SO-101 base mesh')
    datum = [float(base[:, 0].max()), 0.0, float(base[:, 2].min())]
    keys = []
    for i, label in enumerate(manifest['slot_labels']):
        # Rim-plane centre, excluding the tactile F/J nub and the depressed bowl.
        center = asset['keys_center'][i]
        size = asset['keys_size'][i]
        rim = center + deck @ [0, 0, size[2]/2]
        corners = np.array([[x*size[0]/2, y*size[1]/2, size[2]/2]
                            for x, y in [(-1,-1), (1,-1), (1,1), (-1,1)]])
        corners = (corners @ deck.T + center) @ rot.T + position
        keys.append({'label': label, 'slot': manifest['slot_names'][i],
                     'rim_center_m': (rot @ rim + position).tolist(),
                     'outline_xy_m': corners[:, :2].tolist()})
    cases = np.concatenate([asset['case_points_0'], asset['case_points_1']]) @ rot.T + position
    data = {'schema_version': 1, 'robot_usd_sha256': hashlib.sha256(args.robot_usd.read_bytes()).hexdigest(),
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (cfg, asset_path)},
            'reference_position_m': position.tolist(), 'reference_rotation_xyzw': quat,
            'datum_m': datum, 'base_outline_xy_m': hull(base[:, :2]),
            'case_outline_xy_m': hull(cases[:, :2]), 'keys': keys}
    args.output.write_text(json.dumps(data, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
