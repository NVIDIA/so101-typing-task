# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Data-only task configuration for nominal fixture placement and blind pose DR."""
import hashlib
import json
import math
from pathlib import Path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_config(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) - {'schema_version', 'keyboard', 'domain_randomization'}:
        raise ValueError('unknown task configuration fields')
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('task configuration requires schema_version=1')
    keyboard = data.get('keyboard', {})
    if not isinstance(keyboard, dict) or set(keyboard) - {'position_m', 'yaw_offset_deg', 'rotation_xyzw'}:
        raise ValueError('keyboard accepts position_m and either yaw_offset_deg or rotation_xyzw')
    if 'position_m' in keyboard:
        position = keyboard['position_m']
        if not isinstance(position, list) or len(position) != 3 or not all(map(finite, position)):
            raise ValueError('position_m must contain three finite robot-base coordinates in metres')
    if not finite(keyboard.get('yaw_offset_deg', 0)):
        raise ValueError('yaw_offset_deg must be finite')
    if 'rotation_xyzw' in keyboard:
        rotation = keyboard['rotation_xyzw']
        if 'yaw_offset_deg' in keyboard:
            raise ValueError('use either absolute rotation_xyzw or relative yaw_offset_deg, not both')
        if (not isinstance(rotation, list) or len(rotation) != 4 or not all(map(finite, rotation))
                or abs(math.hypot(*rotation)-1) > 1e-6):
            raise ValueError('rotation_xyzw must be a unit quaternion [x, y, z, w]')
    dr = data.get('domain_randomization', {})
    if not isinstance(dr, dict) or set(dr) - {'keyboard_x_m', 'keyboard_y_m', 'keyboard_yaw_deg', 'reset_joint_noise_rad'}:
        raise ValueError('unknown domain randomization field')
    for key, bounds in dr.items():
        if key == 'reset_joint_noise_rad':
            if not finite(bounds) or bounds < 0:
                raise ValueError('reset_joint_noise_rad must be finite and nonnegative')
        elif not isinstance(bounds, list) or len(bounds) != 2 or not all(map(finite, bounds)) or bounds[0] > bounds[1]:
            raise ValueError(f'{key} requires ordered finite [minimum, maximum]')
    return data


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_perturbation(path):
    data = load_config(path)
    if 'keyboard' in data or not data.get('domain_randomization'):
        raise ValueError('perturbation requires domain_randomization only; nominal geometry must remain frozen')
    return data


def apply_perturbation(env_cfg, data):
    """Apply an explicitly labelled fixed-policy reset experiment after contract validation."""
    points = env_cfg.commands.typing.letter_xyz_b_m
    contract = env_cfg.task_contract
    reference = env_cfg.target_reference_sha256
    apply_config(env_cfg, data)
    env_cfg.commands.typing.letter_xyz_b_m = points
    env_cfg.task_contract = contract
    env_cfg.target_reference_sha256 = reference


def evaluation_reset_metadata(env_cfg):
    """Describe the resolved reset protocol, including nonzero fixed offsets."""
    robot = env_cfg.events.reset_robot_rest.params
    keyboard = env_cfg.events.reset_keyboard.params
    pose = keyboard.get('pose_range', {})
    bounds = [*pose.values(), *(keyboard.get(key, (0, 0))
              for key in ('x_range_m', 'y_range_m', 'yaw_range_rad'))]
    return {
        'exact_model_rest': not any(robot.get('position_range', (0, 0))),
        'zero_joint_velocity': not any(robot.get('velocity_range', (0, 0))),
        'keyboard_pose_randomization': any(any(values) for values in bounds),
        'reset_joint_position_range_rad': list(robot.get('position_range', (0, 0))),
        'keyboard_reset_ranges': {key: value for key, value in keyboard.items()
                                  if key in ('pose_range', 'x_range_m', 'y_range_m', 'yaw_range_rad')},
    }


def rotate_xyzw(q, v):
    x, y, z, w = q
    # q v q^-1 for a unit quaternion.
    tx, ty, tz = 2*(y*v[2]-z*v[1]), 2*(z*v[0]-x*v[2]), 2*(x*v[1]-y*v[0])
    return [v[0]+w*tx+y*tz-z*ty, v[1]+w*ty+z*tx-x*tz, v[2]+w*tz+x*ty-y*tx]


def nominal_geometry(position, rotation, points, keyboard):
    new_position = keyboard.get('position_m', position)
    yaw = math.radians(keyboard.get('yaw_offset_deg', 0))
    c, s = math.cos(yaw), math.sin(yaw)
    transformed = []
    for point in points:
        x, y, z = [a-b for a,b in zip(point, position)]
        transformed.append([new_position[0]+c*x-s*y, new_position[1]+s*x+c*y, new_position[2]+z])
    # Left-multiply by a robot-base-Z rotation; quaternion order is xyzw.
    h, k = math.sin(yaw/2), math.cos(yaw/2)
    x,y,z,w = rotation
    new_rotation = [k*x-h*y, k*y+h*x, k*z+h*w, k*w-h*z]
    local = [rotate_xyzw([-rotation[0],-rotation[1],-rotation[2],rotation[3]],
                         [a-b for a,b in zip(point, position)]) for point in points]
    if 'rotation_xyzw' in keyboard:
        new_rotation = keyboard['rotation_xyzw']
        norm = math.hypot(*new_rotation)
        new_rotation = [v/norm for v in new_rotation]
        transformed = [[a+b for a,b in zip(new_position, rotate_xyzw(new_rotation, p))] for p in local]
    return list(new_position), new_rotation, transformed, local


def apply_config(env_cfg, data):
    from isaaclab_tasks.core.dexsuite.config.so101.mdp.public_events import reset_public_keyboard
    state = env_cfg.scene.keyboard.init_state
    old_position = list(state.pos)
    position, rotation, points, local = nominal_geometry(old_position, state.rot,
                                                        env_cfg.commands.typing.letter_xyz_b_m, data.get('keyboard', {}))
    state.pos, state.rot = tuple(position), tuple(rotation)
    plane = env_cfg.scene.plane.init_state
    plane.pos = (plane.pos[0], plane.pos[1], plane.pos[2]+position[2]-old_position[2])
    if 'rotation_xyzw' in data.get('keyboard', {}):
        # Absolute tilt changes the case's lowest point as well as its origin.
        # Keep the horizontal support plane at the transformed case underside.
        import numpy as np
        asset_path = (Path(__file__).resolve().parents[3] / 'source/isaaclab_tasks/isaaclab_tasks/core/'
                      'dexsuite/config/so101/keyboards/data/mx_keys/mx_keys_asset.npz')
        with np.load(asset_path) as asset:
            case = asset['case_points_0']
        x, y, z, w = rotation
        z_axis = np.array([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)])
        plane.pos = (plane.pos[0], plane.pos[1], position[2]+float((case @ z_axis).min()))
    cfg = env_cfg.commands.typing
    cfg.letter_xyz_b_m = tuple(tuple(p) for p in points)
    # Preserve the original coordinate normalization: changed position remains observable.
    cfg.public_pose_randomization = True
    env_cfg.target_reference_sha256 = fingerprint({'letter_xyz_b_m': points})
    dr = data.get('domain_randomization', {})
    env_cfg.events.reset_keyboard.func = reset_public_keyboard
    env_cfg.events.reset_keyboard.params = {
        'x_range_m': tuple(dr.get('keyboard_x_m', [0,0])),
        'y_range_m': tuple(dr.get('keyboard_y_m', [0,0])),
        'yaw_range_rad': tuple(math.radians(x) for x in dr.get('keyboard_yaw_deg', [0,0])),
        'letter_xyz_keyboard_m': local,
    }
    noise = dr.get('reset_joint_noise_rad', 0)
    env_cfg.events.reset_robot_rest.params['position_range'] = (-noise, noise)
    # q_reset_ref remains fixed; only reset arrival is randomized.
    env_cfg.task_contract += '_task_' + fingerprint(data)
    return env_cfg
