# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for fixed-layout Cartesian SO-101 typing."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .fixed_cartesian_typing_commands import FixedCartesianTypingCommand
from .typing_commands_cfg import LetterTypingCommandCfg


@configclass
class FixedCartesianTypingCommandCfg(LetterTypingCommandCfg):
    """Three-phase A-Z command with hysteretic simulated evdev semantics."""

    class_type: type = FixedCartesianTypingCommand

    q_reset_ref: tuple[float, float, float, float, float] = MISSING
    letter_slots: tuple[int, ...] = MISSING
    letter_xyz_b_m: tuple[tuple[float, float, float], ...] = MISSING
    target_xyz_mean_b_m: tuple[float, float, float] = MISSING
    target_xyz_std_b_m: tuple[float, float, float] = MISSING
    action_name: str = "action"

    tip_body: str = "gripper_link"
    tip_offset_m: tuple[float, float, float] = (
        -0.0106769063594562,
        -0.0005212273838322845,
        -0.10413855748835771,
    )
    key_down_fraction: float = 0.50
    key_up_fraction: float = 0.25
    stable_samples: int = 2
    clearance_m: float = 0.004
    clearance_control_ticks: int = 2
    public_pose_randomization: bool = False
    map_tolerance_m: float = 0.001
    phase_timeouts_s: tuple[float, float, float] = (6.0, 2.0, 2.0)
