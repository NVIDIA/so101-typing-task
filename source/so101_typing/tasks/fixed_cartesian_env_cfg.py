# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-layout Cartesian SO-101 typing with AnchorBench dynamics."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .fixed_cartesian_targets import (
    FROZEN_ALPHABET_KEY_FACE_XY_M,
    FROZEN_AZ_XYZ_B_M,
    FROZEN_AZ_XYZ_MEAN_B_M,
    FROZEN_AZ_XYZ_STD_B_M,
    FROZEN_LETTER_SLOTS,
    FROZEN_REFERENCE_MANIFEST_SHA256,
    FROZEN_REFERENCE_SHA256,
)
from .keyboards import TYPING_KEYBOARD_POOL
from .registered_mx_env_cfg import (
    REGISTERED_LETTER_SLOTS,
    REGISTERED_POLICY_REST_JOINT_POS_RAD,
    SO101KeyboardMXFixedBaseEnvCfg,
)

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
POLICY_Q_RESET_RAD = tuple(REGISTERED_POLICY_REST_JOINT_POS_RAD[name] for name in ARM_JOINT_NAMES)
FIXED_JAW_TIP_OFFSET_M = (-0.0106769063594562, -0.0005212273838322845, -0.10413855748835771)

# Rounded-up 95th-percentile absolute finite-difference rates from the safe,
# torque-enabled F/Y/N homing path (50 Hz).  This is the common physical speed
# envelope, not an actuator-profile-dependent tuning knob.
JOINT_RATE_LIMITS_RAD_S = (0.30, 1.10, 0.75, 0.20, 0.10)
JOINT_RATE_SOURCE_SHA256 = "15dc8fcf17dfafd274f84de53d75e085240d5e44c5aab2f5b741ee170896fb04"
JOINT_RATE_SOURCE_KIND = "recorded_fyn_homing_current.json:p95_abs_finite_difference_rounded_up"


@configclass
class FixedCartesianActionsCfg:
    action = mdp.JointRatePositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        velocity_limits_rad_s=JOINT_RATE_LIMITS_RAD_S,
        q_reset_ref=POLICY_Q_RESET_RAD,
        command_name="typing",
        control_dt_s=0.04,
        saturation_threshold=0.95,
    )


@configclass
class FixedCartesianCommandsCfg:
    typing = mdp.FixedCartesianTypingCommandCfg(
        asset_name="robot",
        object_name="keyboard",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        q_reset_ref=POLICY_Q_RESET_RAD,
        letter_slots=FROZEN_LETTER_SLOTS,
        letter_xyz_b_m=FROZEN_AZ_XYZ_B_M,
        target_xyz_mean_b_m=FROZEN_AZ_XYZ_MEAN_B_M,
        target_xyz_std_b_m=FROZEN_AZ_XYZ_STD_B_M,
        tip_body="gripper_link",
        tip_offset_m=FIXED_JAW_TIP_OFFSET_M,
        key_down_fraction=0.50,
        key_up_fraction=0.25,
        stable_samples=2,
        clearance_m=0.004,
        clearance_control_ticks=2,
        map_tolerance_m=0.001,
        phase_timeouts_s=(6.0, 2.0, 2.0),
        letter_length=(1, 1),
        max_len=3,
        command_mode="letter_left",
        typeable_slots=REGISTERED_LETTER_SLOTS,
        target_slots=REGISTERED_LETTER_SLOTS,
        backspace_slot=TYPING_KEYBOARD_POOL.backspace_slot,
        slot_labels=TYPING_KEYBOARD_POOL.slot_labels,
        reset=mdp.LetterTypingCommandCfg.ResetCfg(
            enabled=False,
            ik=None,
            ik_rpy_deg=(0.0, 0.0, 0.0),
            ik_hover_height=0.0,
            ik_iters=1,
            pre_solve_reset=None,
            match_prob=0.0,
        ),
    )


@configclass
class FixedCartesianObservationsCfg:
    """Exact 22-D deployment observation in manifest order."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel_reset = ObsTerm(func=mdp.fixed_joint_pos_rel_reset, params={"command_name": "typing"})
        joint_vel = ObsTerm(func=mdp.fixed_joint_vel, params={"command_name": "typing"})
        joint_command_rel_reset = ObsTerm(
            func=mdp.fixed_joint_command_rel_reset,
            params={"action_name": "action", "command_name": "typing"},
        )
        target_xyz_b = ObsTerm(func=mdp.fixed_target_xyz_b, params={"command_name": "typing"})
        phase_onehot = ObsTerm(func=mdp.fixed_typing_phase_onehot, params={"command_name": "typing"})
        phase_elapsed_s = ObsTerm(func=mdp.fixed_phase_elapsed_s, params={"command_name": "typing"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FixedCartesianRewardsCfg:
    """Full P0C reward stack; event weights are net values after dt scaling."""

    seek_progress = RewTerm(
        func=mdp.fixed_cartesian_seek_progress,
        weight=6.25,  # 0.25 / 0.04
        params={"command_name": "typing", "distance_scale": 0.02},
    )
    lift_progress = RewTerm(
        func=mdp.fixed_cartesian_lift_progress,
        weight=12.5,  # 0.5 / 0.04
        params={"command_name": "typing", "height_scale": 0.004},
    )
    target_down = RewTerm(
        func=mdp.fixed_cartesian_target_down,
        weight=50.0,  # 2.0 / 0.04
        params={"command_name": "typing"},
    )
    target_up = RewTerm(
        func=mdp.fixed_cartesian_target_up,
        weight=25.0,  # 1.0 / 0.04
        params={"command_name": "typing"},
    )
    clearance = RewTerm(
        func=mdp.fixed_cartesian_clearance,
        weight=25.0,  # 1.0 / 0.04
        params={"command_name": "typing"},
    )
    success = RewTerm(
        func=mdp.fixed_cartesian_success,
        weight=125.0,  # 5.0 / 0.04
        params={"command_name": "typing"},
    )
    action_change = RewTerm(
        func=mdp.fixed_cartesian_action_change_l2,
        weight=-0.02,
        params={"action_name": "action"},
    )
    measured_joint_velocity = RewTerm(
        func=mdp.fixed_cartesian_joint_velocity_l2,
        weight=-0.01,
        params={"command_name": "typing", "action_name": "action"},
    )
    low_clearance_lateral_motion = RewTerm(
        func=mdp.fixed_jaw_low_clearance_lateral_motion,
        weight=-0.5,
        params={
            "command_name": "typing",
            "tip_body": "gripper_link",
            "tip_offset": FIXED_JAW_TIP_OFFSET_M,
            "clearance_m": 0.004,
            "lateral_speed_m_s": 0.04,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    failure = RewTerm(
        func=mdp.is_terminated_term,
        weight=-125.0,  # -5.0 / 0.04
        params={
            "term_keys": [
                "invalid_key",
                "map_invalid",
                "action_contract",
                "phase_timeout",
                "scrape",
                "excessive_contact",
                "abnormal_robot",
            ]
        },
    )
    timeout = RewTerm(
        func=mdp.is_terminated_term,
        weight=-50.0,  # -2.0 / 0.04
        params={"term_keys": ["time_out"]},
    )


@configclass
class FixedCartesianTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    abnormal_robot = DoneTerm(func=mdp.joint_vel_out_of_limit)
    excessive_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot_contact"), "threshold": 10.0},
    )
    invalid_key = DoneTerm(func=mdp.fixed_cartesian_invalid_key, params={"command_name": "typing"})
    map_invalid = DoneTerm(func=mdp.fixed_cartesian_map_invalid, params={"command_name": "typing"})
    action_contract = DoneTerm(
        func=mdp.fixed_cartesian_action_contract_violation,
        params={"action_name": "action"},
    )
    phase_timeout = DoneTerm(func=mdp.fixed_cartesian_phase_timeout, params={"command_name": "typing"})
    scrape = DoneTerm(
        func=mdp.fixed_jaw_scrape,
        params={
            "command_name": "typing",
            "tip_body": "gripper_link",
            "tip_offset": FIXED_JAW_TIP_OFFSET_M,
            "clearance_m": 0.004,
            "lateral_speed_m_s": 0.04,
            "persistence_steps": 3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    success = DoneTerm(func=mdp.fixed_cartesian_typing_complete, params={"command_name": "typing"})


@configclass
class SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg(SO101KeyboardMXFixedBaseEnvCfg):
    """Single-letter A-Z P0C from exact physical handoff pose."""

    actuator_profile: str = "anchorbench"
    task_contract: str = "fixed_cartesian_anchorbench_p0c_v0"
    target_reference_sha256: str = FROZEN_REFERENCE_SHA256
    target_manifest_sha256: str = FROZEN_REFERENCE_MANIFEST_SHA256
    physical_alphabet_key_face_xy_m: tuple[float, float] = FROZEN_ALPHABET_KEY_FACE_XY_M
    joint_rate_limits_rad_s: tuple[float, ...] = JOINT_RATE_LIMITS_RAD_S
    joint_rate_source_sha256: str = JOINT_RATE_SOURCE_SHA256
    joint_rate_source_kind: str = JOINT_RATE_SOURCE_KIND
    rest_joint_noise_rad: float = 0.0

    actions: FixedCartesianActionsCfg = FixedCartesianActionsCfg()
    commands: FixedCartesianCommandsCfg = FixedCartesianCommandsCfg()
    observations: FixedCartesianObservationsCfg = FixedCartesianObservationsCfg()
    rewards: FixedCartesianRewardsCfg = FixedCartesianRewardsCfg()
    terminations: FixedCartesianTerminationsCfg = FixedCartesianTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.typing.typeable_slots = REGISTERED_LETTER_SLOTS
        self.commands.typing.target_slots = REGISTERED_LETTER_SLOTS
        self.commands.typing.letter_length = (1, 1)
        self.commands.typing.max_len = 3
        self.commands.typing.reset.enabled = False
        self.commands.typing.reset.ik = None
        self.commands.typing.reset.pre_solve_reset = None
        self.events.reset_keyboard.params["pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
        self.episode_length_s = 10.0


@configclass
class SO101KeyboardMXFixedCartesianAnchorBenchP1AEnvCfg(SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg):
    """Exactly two A-Z characters with no reset between characters."""

    task_contract: str = "fixed_cartesian_anchorbench_p1a_letters2_v0"

    def __post_init__(self):
        super().__post_init__()
        self.commands.typing.letter_length = (2, 2)
        self.commands.typing.max_len = 3
        # Each character has 6 s seek, 2 s release, and 2 s clearance phase
        # budgets.  Keep a one-second margin so phase timeouts, not the global
        # episode timer, identify a stalled transition at the boundary.
        self.episode_length_s = 21.0
