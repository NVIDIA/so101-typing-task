# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal MX Keys scene and reset base for dated production tasks."""

from __future__ import annotations

from copy import deepcopy

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import MultiAssetSpawnerCfg, SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .keyboards import TYPING_KEYBOARD_POOL
from .SO101.actuator_profiles import make_implicit_actuator_cfg
from .so101_env_cfg import KeyboardAssetCfg, PhysicsCfg, SO101KeyboardEnvCfg, SO101SceneCfg

# Frozen physical MX Keys pose for the IRL benchmark scene. The same scene has
# passed F/Y/N scripted IK parity against the physical keyboard.
REGISTERED_KEYBOARD_POSITION_B_M = (0.2171728850035096, -0.004831984023350194, -0.021933385254502818)
REGISTERED_KEYBOARD_ROTATION_XYZW = (0.0, 0.0, -0.7581806382038836, 0.6520445689159229)
REGISTERED_KEYBOARD_CASE_HEIGHT_M = 0.02007388377235258
REGISTERED_KEYBOARD_CASE_BOTTOM_Z_M = REGISTERED_KEYBOARD_POSITION_B_M[2] - 0.5 * REGISTERED_KEYBOARD_CASE_HEIGHT_M

REGISTERED_TYPING_GRIPPER_RAD = -0.17453292519943295
# Captured on the physical robot after successful F/Y/N homing (2026-07-14).
# This is the common policy handoff posture, not a key pose. Arm values are the
# captured LeRobot degrees converted to the matching USD joint convention.
REGISTERED_POLICY_REST_JOINT_POS_RAD = {
    "shoulder_pan": -0.08285519086390664,
    "shoulder_lift": -0.17184780327328783,
    "elbow_flex": 0.20867233254613524,
    "wrist_flex": 1.1323542751400573,
    "wrist_roll": 0.03759170696603171,
    "gripper": REGISTERED_TYPING_GRIPPER_RAD,
}
_REGISTERED_POLICY_ARM_JOINT_NAMES = tuple(name for name in REGISTERED_POLICY_REST_JOINT_POS_RAD if name != "gripper")
# The pool has 32 procedural boards followed by the measured MX Keys asset.
REGISTERED_MX_KEYS_VARIANT_INDEX = len(TYPING_KEYBOARD_POOL.spawners_single) - 1
REGISTERED_MX_KEYS_VARIANT_INDICES = (REGISTERED_MX_KEYS_VARIANT_INDEX,)
REGISTERED_LETTER_SLOTS = tuple(
    slot
    for slot in TYPING_KEYBOARD_POOL.active_slots
    if len(TYPING_KEYBOARD_POOL.slot_labels[slot]) == 1 and TYPING_KEYBOARD_POOL.slot_labels[slot].isalpha()
)


def registered_keyboard_pose_range() -> dict[str, list[float]]:
    """Return an exact keyboard pose range for the fixed IRL benchmark scene."""

    return {
        "x": [0.0, 0.0],
        "y": [0.0, 0.0],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [0.0, 0.0],
    }


_BASE_KEYBOARD_ASSET_CFG = KeyboardAssetCfg()
_BASE_PHYSICS_CFG = PhysicsCfg()
_REGISTERED_NEWTON_CFG = deepcopy(_BASE_PHYSICS_CFG.newton_mjwarp)
_REGISTERED_NEWTON_CFG.solver_cfg.njmax = 1024


@configclass
class MXKeysKeyboardAssetCfg(PresetCfg):
    """Measured MX Keys articulation used by the physical task family."""

    physx = _BASE_KEYBOARD_ASSET_CFG.physx.replace(
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=[
                TYPING_KEYBOARD_POOL.spawners_partitioned[index] for index in REGISTERED_MX_KEYS_VARIANT_INDICES
            ]
        ),
        init_state=_BASE_KEYBOARD_ASSET_CFG.physx.init_state.replace(
            pos=REGISTERED_KEYBOARD_POSITION_B_M,
            rot=REGISTERED_KEYBOARD_ROTATION_XYZW,
        ),
    )
    newton_mjwarp = _BASE_KEYBOARD_ASSET_CFG.newton_mjwarp.replace(
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=[TYPING_KEYBOARD_POOL.spawners_single[index] for index in REGISTERED_MX_KEYS_VARIANT_INDICES]
        ),
        init_state=_BASE_KEYBOARD_ASSET_CFG.newton_mjwarp.init_state.replace(
            pos=REGISTERED_KEYBOARD_POSITION_B_M,
            rot=REGISTERED_KEYBOARD_ROTATION_XYZW,
        ),
    )
    default = newton_mjwarp


@configclass
class MXKeysSO101SceneCfg(SO101SceneCfg):
    """SO-101 scene containing only the measured MX Keys articulation."""

    keyboard: ArticulationCfg = MXKeysKeyboardAssetCfg()
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, REGISTERED_KEYBOARD_CASE_BOTTOM_Z_M)),
        spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
        collision_group=-1,
    )


@configclass
class MXKeysPhysicsCfg(PresetCfg):
    """Physics with enough Newton equality-constraint capacity for table contact."""

    physx = _BASE_PHYSICS_CFG.physx
    newton_mjwarp = _REGISTERED_NEWTON_CFG
    default = newton_mjwarp


def _reset_gripper_closed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> None:
    """Reset the gripper state and drive targets to its configured closed position."""
    asset: Articulation = env.scene[asset_cfg.name]
    iter_env_ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids
    joint_pos = asset.data.default_joint_pos.torch[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_position_target_index(target=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target_index(target=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def _apply_mx_keys_contract(cfg: SO101KeyboardEnvCfg, actuator_profile: str, *, rest_joint_noise_rad: float) -> None:
    if rest_joint_noise_rad < 0.0:
        raise ValueError("rest_joint_noise_rad must be non-negative")
    # Backspace remains the command's reserved recovery action and is intentionally not a sampled target.
    cfg.commands.typing.typeable_slots = REGISTERED_LETTER_SLOTS
    cfg.commands.typing.letter_length = (3, 3)
    cfg.commands.typing.max_len = 3
    cfg.commands.typing.target_slots = REGISTERED_LETTER_SLOTS
    cfg.commands.typing.reset.enabled = False
    cfg.commands.typing.reset.ik = None
    cfg.commands.typing.reset.pre_solve_reset = None
    cfg.events.reset_keyboard.params["pose_range"] = registered_keyboard_pose_range()
    cfg.scene.robot.init_state.joint_pos = REGISTERED_POLICY_REST_JOINT_POS_RAD.copy()
    cfg.events.reset_robot_rest = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(_REGISTERED_POLICY_ARM_JOINT_NAMES)),
            "position_range": (-rest_joint_noise_rad, rest_joint_noise_rad),
            "velocity_range": (0.0, 0.0),
        },
    )
    cfg.events.reset_gripper_closed = EventTerm(
        func=_reset_gripper_closed,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"])},
    )
    cfg.scene.robot.actuators = {"all": make_implicit_actuator_cfg(actuator_profile)}


@configclass
class SO101KeyboardMXFixedBaseEnvCfg(SO101KeyboardEnvCfg):
    """Unregistered internal base for fixed-layout MX Keys policies."""

    actuator_profile: str = "anchorbench"
    keyboard_profile: str = "internal_mx_keys_base"
    source_keyboard_variant_indices: tuple[int, ...] = REGISTERED_MX_KEYS_VARIANT_INDICES
    rest_joint_noise_rad: float = 0.0
    scene: InteractiveSceneCfg = MXKeysSO101SceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(physics=MXKeysPhysicsCfg(), dt=0.01)

    def __post_init__(self):
        super().__post_init__()
        _apply_mx_keys_contract(self, self.actuator_profile, rest_joint_noise_rad=self.rest_joint_noise_rad)
