# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    # SO101 keyboard letter-typing terms (defined in this package).
    "LetterTypingCommand",
    "LetterTypingCommandCfg",
    "FixedCartesianTypingCommand",
    "FixedCartesianTypingCommandCfg",
    "JointRatePositionAction",
    "JointRatePositionActionCfg",
    "letter_typing_progress",
    "typing_success",
    "typing_mistake",
    "typing_complete",
    "reach_key",
    "current_key_position_b",
    "fixed_jaw_low_clearance_lateral_motion",
    "fixed_jaw_scrape",
    "key_positions_b",
    "next_key_position_b",
    "target_keys_onehot",
    "typed_keys_onehot",
    "fixed_joint_pos_rel_reset",
    "fixed_joint_vel",
    "fixed_joint_command_rel_reset",
    "fixed_target_xyz_b",
    "fixed_typing_phase_onehot",
    "fixed_phase_elapsed_s",
    "fixed_cartesian_seek_progress",
    "fixed_cartesian_lift_progress",
    "fixed_cartesian_target_down",
    "fixed_cartesian_target_up",
    "fixed_cartesian_clearance",
    "fixed_cartesian_success",
    "fixed_cartesian_action_change_l2",
    "fixed_cartesian_joint_velocity_l2",
    "fixed_cartesian_invalid_key",
    "fixed_cartesian_map_invalid",
    "fixed_cartesian_phase_timeout",
    "fixed_cartesian_action_contract_violation",
    "fixed_cartesian_typing_complete",
    # Shared dexsuite terms reused by the SO101 env cfg.
    "abnormal_robot_state",
    "action_l2_clamped",
    "action_rate_l2_clamped",
    "body_state_b",
    "mechanical_power",
]

from .commands import (
    FixedCartesianTypingCommand,
    FixedCartesianTypingCommandCfg,
    LetterTypingCommand,
    LetterTypingCommandCfg,
)
from .joint_rate_actions import JointRatePositionAction, JointRatePositionActionCfg
from .observations import (
    current_key_position_b,
    fixed_joint_command_rel_reset,
    fixed_joint_pos_rel_reset,
    fixed_joint_vel,
    fixed_phase_elapsed_s,
    fixed_target_xyz_b,
    fixed_typing_phase_onehot,
    key_positions_b,
    next_key_position_b,
    target_keys_onehot,
    typed_keys_onehot,
)
from .rewards import (
    fixed_cartesian_action_change_l2,
    fixed_cartesian_clearance,
    fixed_cartesian_joint_velocity_l2,
    fixed_cartesian_lift_progress,
    fixed_cartesian_seek_progress,
    fixed_cartesian_success,
    fixed_cartesian_target_down,
    fixed_cartesian_target_up,
    fixed_jaw_low_clearance_lateral_motion,
    letter_typing_progress,
    reach_key,
    typing_success,
)
from .terminations import (
    fixed_cartesian_action_contract_violation,
    fixed_cartesian_invalid_key,
    fixed_cartesian_map_invalid,
    fixed_cartesian_phase_timeout,
    fixed_cartesian_typing_complete,
    fixed_jaw_scrape,
    typing_complete,
    typing_mistake,
)
from isaaclab_tasks.core.dexsuite.mdp import (
    abnormal_robot_state,
    action_l2_clamped,
    action_rate_l2_clamped,
    body_state_b,
    mechanical_power,
)
from isaaclab.envs.mdp import *
