# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""White tabletop and three backdrop panels estimated from the trimmed camera takes.

All user dimensions are millimetres in the robot-base frame. The table surface
uses the fitted keyboard support height. Robot, keyboard, and joint maps retain
their calibrated poses. Apply after public_task.apply_config, before launch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def load_config(path):
    data = json.loads(Path(path).read_text())
    if set(data) != {'schema_version', 'table', 'panels', 'color_rgb', 'robot_color_rgb', 'camera'} or data['schema_version'] != 1:
        raise ValueError('tabletop config requires schema_version=1, table, panels, color_rgb, camera')
    fields = {'table': {'center_xy_mm': 2, 'size_mm': 3},
              'panels': {name: 1 for name in ('front_x_mm', 'back_x_mm', 'left_y_mm', 'right_y_mm',
                                             'height_mm', 'thickness_mm')},
              'camera': {'eye_mm': 3, 'lookat_mm': 3}}
    for group, spec in fields.items():
        if not isinstance(data[group], dict) or set(data[group]) != set(spec):
            raise ValueError(f'invalid tabletop {group} fields')
        for name, count in spec.items():
            values = [data[group][name]] if count == 1 else data[group][name]
            if not isinstance(values, list) or len(values) != count or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError(f'{group}.{name} requires {count} finite millimetre values')
    p = data['panels']
    if (min(data['table']['size_mm']) <= 0 or p['height_mm'] <= 0 or p['thickness_mm'] <= 0
            or p['front_x_mm'] >= p['back_x_mm'] or p['right_y_mm'] >= p['left_y_mm']):
        raise ValueError('tabletop dimensions must be positive and panel bounds ordered')
    for field in ('color_rgb', 'robot_color_rgb'):
        color = data[field]
        if (not isinstance(color, list) or len(color) != 3
                or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in color)):
            raise ValueError(f'{field} requires three values between 0 and 1')
    return data


def _material(stage, path, color):
    """Author a USD material without Kit, which Newton does not require."""
    from pxr import Sdf, UsdShade
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/PreviewSurface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(tuple(color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def _paint(prim, material, color):
    from pxr import Gf, UsdGeom, UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _spawn_box(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom
    color = cfg.visual_material.diffuse_color
    prim = sim_utils.spawn_cuboid(prim_path, cfg.replace(visual_material=None), translation, orientation, **kwargs)
    material = _material(prim.GetStage(), str(prim.GetPath()) + '/WhiteMaterial', color)
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Gprim):
            _paint(child, material, color)
    return prim


def _spawn_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdShade
    color = cfg.visual_material.diffuse_color
    prim = sim_utils.spawn_from_usd(prim_path, cfg.replace(visual_material=None), translation, orientation, **kwargs)
    stage = prim.GetStage()
    # The Workshop binds printed-plastic materials through geometry subsets.
    # Update those original shader inputs as well as direct mesh bindings.
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdShade.Shader) and 'printed' in str(child.GetPath()).lower():
            shader = UsdShade.Shader(child)
            for name in ('diffuseColor', 'diffuse_color_constant', 'diffuse_tint'):
                value = shader.GetInput(name)
                if value:
                    value.Set(tuple(color) if name != 'diffuse_tint' else (1., 1., 1.))
    orange = _material(stage, str(prim.GetPath()) + '/RecordedOrange', color)
    dark_color = (0.055, 0.055, 0.055)
    dark = _material(stage, str(prim.GetPath()) + '/RecordedServo', dark_color)
    for child in Usd.PrimRange(prim):
        path = str(child.GetPath()).lower()
        if not child.IsA(UsdGeom.Gprim) or 'camera' in path:
            continue
        if '/visuals/' in path or '/collisions/' in path:
            is_servo = 'sts3215' in path
            _paint(child, dark if is_servo else orange, dark_color if is_servo else color)
    return prim


def apply_config(env_cfg, data):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    surface_z = float(env_cfg.scene.plane.init_state.pos[2])
    env_cfg.scene.robot.spawn = env_cfg.scene.robot.spawn.replace(
        func=_spawn_robot, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(data['robot_color_rgb'])))
    color = tuple(data['color_rgb'])

    def box(name, size, center):
        return AssetBaseCfg(
            prim_path='{ENV_REGEX_NS}/' + name,
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(center)),
            spawn=sim_utils.CuboidCfg(
                func=_spawn_box, size=tuple(size),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.85),
            ),
        )

    sx, sy, thickness = [v / 1000 for v in data['table']['size_mm']]
    cx, cy = [v / 1000 for v in data['table']['center_xy_mm']]
    env_cfg.scene.plane = box('WhiteTable', (sx, sy, thickness), (cx, cy, surface_z - thickness / 2))
    p = {name: value / 1000 for name, value in data['panels'].items()}
    x0, x1, yl, yr = p['front_x_mm'], p['back_x_mm'], p['left_y_mm'], p['right_y_mm']
    h, t = p['height_mm'], p['thickness_mm']
    z = surface_z + h / 2
    env_cfg.scene.tabletop_back_wall = box('WhiteBackPanel', (t, yl-yr+t, h), (x1, (yl+yr)/2, z))
    env_cfg.scene.tabletop_left_wall = box('WhiteLeftPanel', (x1-x0, t, h), ((x0+x1)/2, yl, z))
    env_cfg.scene.tabletop_right_wall = box('WhiteRightPanel', (x1-x0, t, h), ((x0+x1)/2, yr, z))
    # The video shows diffuse indoor illumination, without an outdoor sky.
    env_cfg.scene.sky_light = AssetBaseCfg(
        prim_path='/World/skyLight', spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(1., 1., 1.)))
    eye = tuple(v / 1000 for v in data['camera']['eye_mm'])
    lookat = tuple(v / 1000 for v in data['camera']['lookat_mm'])
    env_cfg.viewer.eye, env_cfg.viewer.lookat = eye, lookat
    env_cfg.video_recorder.eye, env_cfg.video_recorder.lookat = eye, lookat
    return {'kind': 'recorded_white_tabletop', 'dimensions': data,
            'table_surface_z_mm': surface_z * 1000,
            'dimensions_estimated_from_video': True,
            'robot_and_keyboard_pose_changed': False,
            'walls_and_table_have_collision': True}


def _spawn_ground_contact(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Keep the training ground collider; hide only its decorative grid meshes."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdPhysics
    prim = sim_utils.spawn_ground_plane(prim_path, cfg, translation, orientation, **kwargs)
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh) and not child.HasAPI(UsdPhysics.CollisionAPI):
            UsdGeom.Imageable(child).MakeInvisible()
    return prim


def apply_white_base(env_cfg, data):
    """Add just the table rectangle for policy viewing, with no new colliders."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    surface_z = float(env_cfg.scene.plane.init_state.pos[2])
    env_cfg.scene.plane.spawn = env_cfg.scene.plane.spawn.replace(func=_spawn_ground_contact)
    sx, sy, thickness = [v / 1000 for v in data['table']['size_mm']]
    cx, cy = [v / 1000 for v in data['table']['center_xy_mm']]
    env_cfg.scene.white_base = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/WhiteBase',
        init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, surface_z - thickness / 2)),
        spawn=sim_utils.CuboidCfg(
            func=_spawn_box, size=(sx, sy, thickness), collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.94, 0.94, 0.94), roughness=0.85),
        ),
    )
    return {'kind': 'white_rectangular_base', 'dimensions_mm': data['table'],
            'table_surface_z_mm': surface_z * 1000, 'visual_only': True,
            'training_ground_collision_preserved': True, 'panels_added': False}
