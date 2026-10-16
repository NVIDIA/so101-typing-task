"""Place two captured typing replays on adjacent tabletops in one USD stage."""

import argparse
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


LEFT_ENV = Sdf.Path('/World/envs/env_0')
RIGHT_ENV = Sdf.Path('/World/envs/env_1')


def _remap(path):
    text = str(path)
    source = str(LEFT_ENV)
    if text == source or text.startswith(source + '/') or text.startswith(source + '.'):
        return Sdf.Path(str(RIGHT_ENV) + text[len(source):])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--center-spacing', type=float, default=0.88)
    parser.add_argument('--camera-distance-scale', type=float, default=1.7)
    parser.add_argument('--resolution', nargs=2, type=int, default=(2560, 1440))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    if args.center_spacing < 0.8 or args.camera_distance_scale <= 1:
        parser.error('Benches need at least 0.8 m spacing and a pulled-back camera')
    reports = [json.loads(path.with_name('evaluation.json').read_text()) for path in (args.left, args.right)]
    if any(report['num_envs'] != 1 or report['successes'] != 1 for report in reports):
        parser.error('Both source replays must be successful single-environment evaluations')
    if reports[0]['replay_export']['fps'] != reports[1]['replay_export']['fps']:
        parser.error('Source replay frame rates differ')
    right = Usd.Stage.Open(str(args.right))
    if right is None:
        parser.error('Cannot open right replay')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = Usd.Stage.Open(str(args.left))
    if source is None or not source.GetRootLayer().Export(str(args.output)):
        parser.error('Cannot copy left replay')
    stage = Usd.Stage.Open(str(args.output))
    if not Sdf.CopySpec(right.GetRootLayer(), LEFT_ENV, stage.GetRootLayer(), RIGHT_ENV):
        raise RuntimeError('Could not copy the right robot and tabletop into the shared stage')
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(args.output))
    right_root = stage.GetPrimAtPath(RIGHT_ENV)
    if not right_root.IsValid():
        raise RuntimeError('Copied right setup is missing')
    remapped = 0
    for prim in Usd.PrimRange(right_root):
        for rel in prim.GetRelationships():
            old = rel.GetTargets()
            new = [_remap(path) for path in old]
            if new != old:
                rel.SetTargets(new)
                remapped += 1
        for attr in prim.GetAttributes():
            old = attr.GetConnections()
            new = [_remap(path) for path in old]
            if new != old:
                attr.SetConnections(new)
                remapped += 1
    half = args.center_spacing / 2
    for path, y in ((LEFT_ENV, -half), (RIGHT_ENV, half)):
        env = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        ops = env.GetOrderedXformOps()
        if len(ops) != 1 or ops[0].GetOpName() != 'xformOp:translate':
            raise RuntimeError(f'Unexpected environment transform: {path}')
        ops[0].Set(Gf.Vec3d(0, y, 0))
        # Newton exports each rigid body as a world-space transform with
        # !resetXformStack!, so moving the environment parent moves only the
        # static tabletop. Translate all of those recorded world poses too.
        moved_bodies = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            xform = UsdGeom.Xformable(prim)
            if not xform or not xform.GetResetXformStack():
                continue
            transform_ops = [op for op in xform.GetOrderedXformOps()
                             if op.GetOpName() == 'xformOp:transform']
            if len(transform_ops) != 1:
                raise RuntimeError(f'Unexpected reset transform on {prim.GetPath()}')
            op = transform_ops[0]
            samples = op.GetTimeSamples()
            for time in samples or [None]:
                matrix = Gf.Matrix4d(op.Get(time))
                matrix.SetTranslateOnly(matrix.ExtractTranslation() + Gf.Vec3d(0, y, 0))
                if time is None:
                    op.Set(matrix)
                else:
                    op.Set(matrix, time)
            moved_bodies += 1
        if moved_bodies < 20:
            raise RuntimeError(f'Only {moved_bodies} world-space bodies moved for {path}')
    camera = UsdGeom.Xformable(stage.GetPrimAtPath('/RenderCamera'))
    ops = camera.GetOrderedXformOps()
    if len(ops) != 1 or ops[0].GetOpName() != 'xformOp:transform':
        raise RuntimeError('Unexpected overview camera transform')
    matrix = Gf.Matrix4d(ops[0].Get())
    old_eye = matrix.ExtractTranslation()
    focus = Gf.Vec3d(0.17, 0.02, 0.15)
    eye = focus + (old_eye - focus) * args.camera_distance_scale
    matrix.SetTranslateOnly(eye)
    ops[0].Set(matrix)
    product = stage.GetPrimAtPath('/Render/Tabletop')
    product.GetAttribute('resolution').Set(Gf.Vec2i(*args.resolution))
    stage.SetEndTimeCode(max(report['steps_executed'] for report in reports))
    check = stage.GetPrimAtPath('/World/envs/env_1/Robot/base/visuals/base_so101_v2/mesh_1')
    material = UsdShade.MaterialBindingAPI(check).ComputeBoundMaterial()[0]
    if not material or '/env_1/' not in str(material.GetPath()):
        raise RuntimeError(f'Right robot material did not remap: {material.GetPath() if material else None}')
    stage.GetRootLayer().Save()
    metadata = {'scene': args.output.name, 'layout': 'two adjacent white tabletops in one USD stage',
                'left_target': reports[0]['target_descriptor']['target'],
                'right_target': reports[1]['target_descriptor']['target'],
                'center_spacing_m': args.center_spacing, 'gap_m': args.center_spacing - 0.8,
                'camera_eye_m': list(eye), 'camera_focus_m': list(focus),
                'camera_distance_scale': args.camera_distance_scale,
                'resolution': args.resolution, 'fps': reports[0]['replay_export']['fps'],
                'frames': max(report['steps_executed'] for report in reports) + 1,
                'source_frames': [report['steps_executed'] + 1 for report in reports],
                'right_relationships_remapped': remapped,
                'color_rgb': {'left_black': [0.035, 0.045, 0.045], 'right_lime': [0.46, 0.73, 0.03]},
                'source': 'two successful recorded Newton motion replays; final right pose held after completion'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'SHARED SCENE: {args.output} ({remapped} path relationships remapped)', flush=True)


if __name__ == '__main__':
    main()
