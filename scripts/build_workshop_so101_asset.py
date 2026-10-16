# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Build a pinned, offline SO-101 asset from the Workshop's public USD sources."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def build(manifest_path, overrides_path, cache, output):
    manifest = json.loads(manifest_path.read_text())
    for url, record in manifest['files'].items():
        path = cache / record['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            path.write_bytes(urlopen(url, timeout=120).read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Source hash mismatch: {url}')
    root = cache / manifest['root_path']
    layer = Sdf.Layer.CreateAnonymous('workshop.usda')
    layer.ImportFromString(overrides_path.read_text())
    layer.subLayerPaths = [str(root.resolve())]
    stage = Usd.Stage.Open(layer)
    prim = stage.GetDefaultPrim()
    for key, value in manifest['variants'].items():
        prim.GetVariantSet(key).SetVariantSelection(value)
    # Embed all dependencies so runtime spawning needs neither the sibling repo nor network.
    while True:
        instances = [p for p in stage.Traverse() if p.IsInstance()]
        if not instances:
            break
        for p in instances:
            p.SetInstanceable(False)
    # Convert constant MDL materials for Newton's built-in OpenGL/Warp viewers.
    for p in list(stage.Traverse()):
        if not p.IsA(UsdShade.Material):
            continue
        inputs = {}
        for child in Usd.PrimRange(p):
            if child.IsA(UsdShade.Shader):
                inputs.update({str(i.GetBaseName()): i.Get() for i in UsdShade.Shader(child).GetInputs()})
        color = inputs.get('diffuse_color_constant', inputs.get('diffuse_tint'))
        if color is None:
            continue
        mat = UsdShade.Material(p)
        shader = UsdShade.Shader.Define(stage, str(p.GetPath())+'/NewtonPreview')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(tuple(color))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(inputs.get('reflection_roughness_constant', .65)))
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(inputs.get('metallic_constant', 0)))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    camera = UsdGeom.Camera(stage.GetPrimAtPath(str(prim.GetPath())+'/gripper/wowrobo_2MP_camera'))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(.001, 5))
    # Camera transform scale is an authoring artifact, not a scale on the optical frame.
    camera.GetPrim().GetAttribute('xformOp:scale').Set(Gf.Vec3d(1))
    prim.CreateAttribute('newton:selfCollisionEnabled', Sdf.ValueTypeNames.Bool).Set(False)
    flat = stage.Flatten()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = Usd.Stage.CreateNew(str(output))
    Sdf.CopySpec(flat, prim.GetPath(), result.GetRootLayer(), prim.GetPath())
    result.SetDefaultPrim(result.GetPrimAtPath(prim.GetPath()))
    UsdGeom.SetStageMetersPerUnit(result, 1)
    UsdGeom.SetStageUpAxis(result, 'Z')
    result.GetRootLayer().comment = 'SO-101 SysID asset composed with Workshop calibration; see workshop_asset_sources.json for pinned public dependencies.'
    result.GetRootLayer().Save()
    actual_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    if actual_sha256 != manifest['expected_asset_sha256']:
        raise ValueError(f'Built Workshop USD hash mismatch: {actual_sha256}')
    metadata = dict(workshop_commit=manifest['workshop_commit'], variants=manifest['variants'],
                    source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    asset_sha256=actual_sha256,
                    overrides_sha256=hashlib.sha256(overrides_path.read_bytes()).hexdigest(),
                    adaptations=['flatten USD dependencies', 'constant MDL to UsdPreviewSurface', 'unit camera transform scale',
                                 'camera clipping 1 mm to 5 m', 'disable robot self-collision'])
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Built {output}: {output.stat().st_size} bytes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--overrides', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest, args.overrides, args.cache, args.out)
