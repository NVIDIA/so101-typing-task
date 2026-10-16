# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Recolor the printed SO-101 parts in a captured replay without changing motion."""

import argparse
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene')
    parser.add_argument('--rgb', nargs=3, type=float, required=True)
    args = parser.parse_args()
    color = tuple(args.rgb)
    if any(not 0 <= value <= 1 for value in color):
        parser.error('RGB components must be in [0, 1]')
    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        parser.error(f'Cannot open {args.scene}')
    material = UsdShade.Material.Define(stage, '/World/TwoEnvsRobotPaint')
    shader = UsdShade.Shader.Define(stage, '/World/TwoEnvsRobotPaint/PreviewSurface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.76)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    # The Workshop USD binds this parent material with strongerThanDescendants.
    # Change its render-context shaders as well; a child mesh binding alone is
    # intentionally ignored by the USD material resolver.
    changed_shaders = 0
    for material_name in ('material_a_d_printed', 'material_a_3d_printed'):
        printed = stage.GetPrimAtPath('/World/envs/env_0/Robot/Looks/' + material_name)
        if not printed.IsValid():
            raise RuntimeError(f'Workshop material is missing: {material_name}')
        for prim in Usd.PrimRange(printed):
            if not prim.IsA(UsdShade.Shader):
                continue
            for name in ('diffuseColor', 'diffuse_color_constant'):
                value = UsdShade.Shader(prim).GetInput(name)
                if value:
                    value.Set(Gf.Vec3f(*color))
                    changed_shaders += 1
    if changed_shaders < 4:
        raise RuntimeError(f'Expected body and camera mount shader colors, got {changed_shaders}')
    painted = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (not prim.IsA(UsdGeom.Gprim) or '/Robot/' not in path
                or '/visuals/' not in path or 'sts3215' in path
                or 'camera_mount' in path):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
        painted.append(path)
    if len(painted) < 10:
        raise RuntimeError(f'Unexpected robot visual mesh count: {len(painted)}')
    stage.GetRootLayer().Save()
    print(f'Painted {len(painted)} printed robot meshes and {changed_shaders} shader inputs {color} in {args.scene}', flush=True)


if __name__ == '__main__':
    main()
