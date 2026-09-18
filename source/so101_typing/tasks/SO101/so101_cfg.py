# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO-ARM101 (SO101) follower arm.

The articulation is defined by the remote Hugging Face ``so101_no_camera_new_calib.usd`` asset,
which carries the ground-truth kinematics, masses, inertias, and joint limits generated from the
URDF/onshape export.
The implicit actuator parameters below override the USD drive gains with SO101 SysID values.

The arm has six revolute joints: ``shoulder_pan``, ``shoulder_lift``, ``elbow_flex``,
``wrist_flex``, ``wrist_roll`` (arm) and ``gripper`` (single-jaw gripper). The base is fixed to the
world via the USD ``root_joint``.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_SO101_USD_PATH = os.environ.get(
    "SO101_ROBOT_USD",
    "https://huggingface.co/datasets/nvidia/Anchor-Lab/resolve/main/robot_assets/so101_no_camera_new_calib.usd",
)

_SO101_SYSID_STIFFNESS: dict[str, float] = {
    "shoulder_pan": 48.419899,
    "shoulder_lift": 47.782635,
    "elbow_flex": 14.715496,
    "wrist_flex": 43.607166,
    "wrist_roll": 54.873917,
    "gripper": 68.250793,
}
_SO101_SYSID_DAMPING: dict[str, float] = {
    "shoulder_pan": 3.334595,
    "shoulder_lift": 2.563165,
    "elbow_flex": 0.343250,
    "wrist_flex": 4.139243,
    "wrist_roll": 2.968392,
    "gripper": 2.818730,
}
_SO101_SYSID_ARMATURE: dict[str, float] = {
    "shoulder_pan": 0.067620,
    "shoulder_lift": 0.027645,
    "elbow_flex": 0.037720,
    "wrist_flex": 0.050714,
    "wrist_roll": 0.054898,
    "gripper": 0.077625,
}
# Anchor Bench calls this fitted Coulomb term `dry_friction`. Newton consumes Isaac Lab
# `friction`, while PhysX consumes `dynamic_friction`, so the same values feed both fields.
_SO101_SYSID_DRY_FRICTION: dict[str, float] = {
    "shoulder_pan": 0.347432,
    "shoulder_lift": 0.344793,
    "elbow_flex": 0.411190,
    "wrist_flex": 0.248233,
    "wrist_roll": 0.221761,
    "gripper": 0.083458,
}
_SO101_SYSID_VISCOUS_FRICTION: dict[str, float] = {
    "shoulder_pan": 1.590709,
    "shoulder_lift": 0.561425,
    "elbow_flex": 0.796639,
    "wrist_flex": 1.071871,
    "wrist_roll": 1.637671,
    "gripper": 0.958565,
}

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_SO101_USD_PATH,
        # Reference the remote layer directly so the source of truth stays in the Hugging Face dataset.
        copy_from_source=False,
        # Off by default (matches FRANKA_PANDA_CFG); a scene enables this only when it adds contact sensors.
        activate_contact_sensors=False,
        # Enable self-collision so the gripper can't pass through the arm. This writes the PhysX attr
        # (``physxArticulation:enabledSelfCollisions``); Newton reads ``newton:selfCollisionEnabled``
        # first, which the calibrated USD authors as ``1`` (see payloads/Physics/physics.usda).
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
    actuators={
        # Previous SO101 actuator settings before SysID params. ``stiffness``, ``damping``,
        # and omitted friction fields inherited their USD values.
        # "all": ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     stiffness=None,
        #     damping=None,
        #     armature=0.028,
        #     effort_limit_sim=None,
        #     velocity_limit_sim=5.0,
        # ),
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=_SO101_SYSID_STIFFNESS,
            damping=_SO101_SYSID_DAMPING,
            armature=_SO101_SYSID_ARMATURE,
            friction=_SO101_SYSID_DRY_FRICTION,
            dynamic_friction=_SO101_SYSID_DRY_FRICTION,
            viscous_friction=_SO101_SYSID_VISCOUS_FRICTION,
            effort_limit_sim=None,
            velocity_limit_sim=5.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the SO-ARM101 (SO101) follower arm using its calibrated USD asset."""
