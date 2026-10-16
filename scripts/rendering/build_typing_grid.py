"""Build a camera-visible grid of independently colored SO-101 typing replays."""

import argparse
import colorsys
import json
import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdShade


HERO_COLORS = (
    ('black', (0.035, 0.045, 0.045)),
    ('lime', (0.46, 0.73, 0.03)),
    ('single_env_yellow', (1.0, 0.82, 0.12)),
    ('blue', (0.10, 0.36, 0.90)),
)


def _color(index):
    if index < len(HERO_COLORS):
        return HERO_COLORS[index]
    hue = ((index - 4) * 0.618033988749895 + 0.07) % 1
    saturation = (0.73, 0.86, 0.65, 0.92)[index % 4]
    brightness = (0.92, 0.81, 0.98, 0.72)[(index // 4) % 4]
    return f'color_{index + 1:02d}', colorsys.hsv_to_rgb(hue, saturation, brightness)


def _paint(stage, root, rgb):
    changed = 0
    for material_name in ('material_a_d_printed', 'material_a_3d_printed'):
        path = f'{root}/Robot/Looks/{material_name}'
        material = stage.GetPrimAtPath(path)
        if not material.IsValid():
            raise RuntimeError(f'Missing Workshop paint material: {path}')
        for child in Usd.PrimRange(material):
            if not child.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(child)
            for input_name in ('diffuseColor', 'diffuse_color_constant'):
                value = shader.GetInput(input_name)
                if value:
                    value.Set(Gf.Vec3f(*rgb))
                    changed += 1
    if changed != 4:
        raise RuntimeError(f'Expected four paint shader inputs, got {changed} at {root}')


def _place(stage, root, dx, dy):
    env = UsdGeom.Xformable(stage.GetPrimAtPath(root))
    if not env:
        raise RuntimeError(f'Missing grid cell: {root}')
    ops = env.GetOrderedXformOps()
    if len(ops) != 1 or ops[0].GetOpName() != 'xformOp:translate':
        raise RuntimeError(f'Unexpected grid cell transform: {root}')
    ops[0].Set(Gf.Vec3d(dx, dy, 0))
    moved = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        body = UsdGeom.Xformable(prim)
        if not body or not body.GetResetXformStack():
            continue
        original = body.GetOrderedXformOps()
        if len(original) != 1 or original[0].GetOpName() != 'xformOp:transform':
            raise RuntimeError(f'Unexpected recorded world pose: {prim.GetPath()}')
        offset = body.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix='gridOffset')
        offset.Set(Gf.Vec3d(dx, dy, 0))
        # This op must precede the captured pose. Appending it rotates the
        # offset with each body and leaves robots off their own tabletops.
        body.SetXformOpOrder([offset] + original, resetXformStack=True)
        moved += 1
    if moved < 20:
        raise RuntimeError(f'Only {moved} recorded bodies were placed at {root}')
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--cols', type=int, required=True)
    parser.add_argument('--resolution', nargs=2, type=int, required=True)
    parser.add_argument('--x-spacing', type=float, default=0.77)
    parser.add_argument('--y-spacing', type=float, default=0.88)
    parser.add_argument('--camera-height-factor', type=float, default=0.94,
                        help='Camera height as a fraction of grid span for layouts larger than 2x2')
    parser.add_argument('--camera-planar-factor', type=float, default=0.56,
                        help='Camera horizontal distance per axis as a fraction of grid span')
    parser.add_argument('--camera-focus-z', type=float, default=0.08,
                        help='Camera look-at height for layouts larger than 2x2')
    args = parser.parse_args()
    if (args.rows < 1 or args.cols < 1 or args.camera_height_factor <= 0
            or args.camera_planar_factor <= 0 or args.output.exists()):
        parser.error('Rows and cols must be positive and output must be new')
    report = json.loads(args.source.with_name('evaluation.json').read_text())
    if 'episode' in report and 'replay' in report:
        # The tutorial records the exact typed result beside each episode USD.
        report = {'num_envs': 1, 'successes': int(report['success']),
                  'target_descriptor': {'target': report['typed']},
                  'replay_export': report['replay'], 'steps_executed': report['steps']}
    if (report['num_envs'] != 1 or report['successes'] != 1
            or report['target_descriptor']['target'] != 'ISAAC LAB 3.0'):
        parser.error('Source must be a successful ISAAC LAB 3.0 replay')
    source = Usd.Stage.Open(str(args.source))
    if source is None:
        parser.error('Cannot open source replay')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not source.GetRootLayer().Export(str(args.output)):
        parser.error('Cannot copy source USD')
    stage = Usd.Stage.Open(str(args.output))
    reference = os.path.relpath(args.source.resolve(), args.output.parent.resolve())
    cells = []
    for index in range(args.rows * args.cols):
        row, col = divmod(index, args.cols)
        root = f'/World/envs/env_{index}'
        if index:
            prim = stage.DefinePrim(root, 'Xform')
            prim.GetReferences().AddReference(reference, '/World/envs/env_0')
        dx = (row - (args.rows - 1) / 2) * args.x_spacing
        dy = (col - (args.cols - 1) / 2) * args.y_spacing
        bodies = _place(stage, root, dx, dy)
        label, rgb = _color(index)
        _paint(stage, root, rgb)
        material = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(
            root + '/Robot/base/visuals/base_so101_v2/mesh_1')).ComputeBoundMaterial()[0]
        if not material or root not in str(material.GetPath()):
            raise RuntimeError(f'Paint material binding did not resolve in {root}')
        cells.append({'row': row, 'col': col, 'root': root, 'offset_m': [dx, dy],
                      'color_name': label, 'color_rgb': rgb, 'recorded_bodies': bodies})
    camera = UsdGeom.Xformable(stage.GetPrimAtPath('/RenderCamera'))
    ops = camera.GetOrderedXformOps()
    if len(ops) != 1 or ops[0].GetOpName() != 'xformOp:transform':
        raise RuntimeError('Unexpected source camera')
    focus = Gf.Vec3d(0.18, 0.03, -0.15 if args.rows == 2 and args.cols == 2 else args.camera_focus_z)
    if args.rows == 2 and args.cols == 2:
        eye = focus + Gf.Vec3d(1.05, -1.05, 1.25)
    else:
        span = max(args.rows * args.x_spacing, args.cols * args.y_spacing)
        eye = focus + Gf.Vec3d(span * args.camera_planar_factor,
                               -span * args.camera_planar_factor,
                               span * args.camera_height_factor)
    matrix = Gf.Matrix4d().SetLookAt(eye, focus, Gf.Vec3d(0, 0, 1)).GetInverse()
    ops[0].Set(matrix)
    stage.GetPrimAtPath('/Render/Tabletop').GetAttribute('resolution').Set(Gf.Vec2i(*args.resolution))
    stage.GetRootLayer().Save()
    metadata = {'scene': args.output.name, 'layout': f'{args.rows}x{args.cols}',
                'rows': args.rows, 'cols': args.cols, 'num_envs': len(cells),
                'target': 'ISAAC LAB 3.0', 'cell_spacing_m': [args.x_spacing, args.y_spacing],
                'camera_eye_m': list(eye), 'camera_focus_m': list(focus),
                'camera_height_factor': args.camera_height_factor,
                'camera_planar_factor': args.camera_planar_factor,
                'resolution': args.resolution, 'fps': report['replay_export']['fps'],
                'frames': report['steps_executed'] + 1, 'cells': cells,
                'source_replay': str(args.source),
                'motion': 'same successful recorded Newton rollout in every cell, synchronized'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'GRID SCENE: {args.output} ({len(cells)} colored robots)', flush=True)


if __name__ == '__main__':
    main()
