# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persistent, physically timed joint-rate action for SO-101 typing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JointRatePositionAction(ActionTerm):
    """Integrate one bounded rate command at the actor period and hold it.

    Unlike ``RelativeJointPositionAction``, this term never rebuilds a target
    from the measured joint state at every physics substep.  The latest issued
    command is persistent and is therefore observable and reproducible by the
    physical runner.
    """

    cfg: JointRatePositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: JointRatePositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(self._joint_ids) != len(cfg.velocity_limits_rad_s):
            raise ValueError("velocity_limits_rad_s must contain one value per controlled joint")
        if len(self._joint_ids) != len(cfg.q_reset_ref):
            raise ValueError("q_reset_ref must contain one value per controlled joint")
        if abs(float(env.step_dt) - float(cfg.control_dt_s)) > 1.0e-9:
            raise ValueError(
                f"joint-rate action requires control_dt={cfg.control_dt_s:.9f}s, got env.step_dt={env.step_dt:.9f}s"
            )
        if min(cfg.velocity_limits_rad_s) <= 0.0:
            raise ValueError("all joint-rate velocity limits must be positive")
        if not torch.isfinite(torch.tensor(cfg.velocity_limits_rad_s)).all():
            raise ValueError("all joint-rate velocity limits must be finite")
        if not torch.isfinite(torch.tensor(cfg.q_reset_ref)).all():
            raise ValueError("q_reset_ref must be finite")

        shape = (self.num_envs, len(self._joint_ids))
        self._raw_actions = torch.zeros(shape, device=self.device)
        self._bounded_actions = torch.zeros_like(self._raw_actions)
        self._previous_bounded_actions = torch.zeros_like(self._raw_actions)
        self._joint_command = torch.tensor(cfg.q_reset_ref, device=self.device).expand(shape).clone()
        self._velocity_limits = torch.tensor(cfg.velocity_limits_rad_s, device=self.device).expand(shape)
        self._q_reset_ref = torch.tensor(cfg.q_reset_ref, device=self.device).expand(shape)
        self._near_saturation = torch.zeros_like(self._raw_actions, dtype=torch.bool)
        self._out_of_bounds = torch.zeros_like(self._raw_actions, dtype=torch.bool)
        self._episode_action_samples = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_action_mean_abs = torch.zeros(self.num_envs, device=self.device)
        self._episode_action_max_abs = torch.zeros(self.num_envs, device=self.device)
        self._episode_near_saturation_fraction = torch.zeros(self.num_envs, device=self.device)
        self._episode_near_saturation_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_out_of_bounds_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Bounded dimensionless rates consumed by the integrator."""

        return self._bounded_actions

    @property
    def bounded_actions(self) -> torch.Tensor:
        return self._bounded_actions

    @property
    def previous_bounded_actions(self) -> torch.Tensor:
        return self._previous_bounded_actions

    @property
    def joint_command(self) -> torch.Tensor:
        """Latest persistent arm-joint position command [rad]."""

        return self._joint_command

    @property
    def velocity_limits_rad_s(self) -> torch.Tensor:
        return self._velocity_limits

    @property
    def near_saturation(self) -> torch.Tensor:
        """Per-joint ``|a| >= saturation_threshold`` telemetry."""

        return self._near_saturation

    @property
    def out_of_bounds(self) -> torch.Tensor:
        """Defensive-contract violations before the final safety clamp."""

        return self._out_of_bounds

    @property
    def episode_action_mean_abs(self) -> torch.Tensor:
        """Running episode mean of ``|a|`` over control ticks and joints."""

        return self._episode_action_mean_abs

    @property
    def episode_action_max_abs(self) -> torch.Tensor:
        """Running episode maximum of ``|a|`` after the safety bound."""

        return self._episode_action_max_abs

    @property
    def episode_near_saturation_fraction(self) -> torch.Tensor:
        """Running fraction of joint actions at the saturation threshold."""

        return self._episode_near_saturation_fraction

    @property
    def episode_near_saturation_any(self) -> torch.Tensor:
        return self._episode_near_saturation_any

    @property
    def episode_out_of_bounds_any(self) -> torch.Tensor:
        return self._episode_out_of_bounds_any

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"expected action shape {self._raw_actions.shape}, got {actions.shape}")
        command = self._env.command_manager.get_term(self.cfg.command_name)
        if hasattr(command, "begin_control_transition"):
            command.begin_control_transition()

        self._previous_bounded_actions.copy_(self._bounded_actions)
        finite = torch.isfinite(actions)
        safe_actions = torch.where(finite, actions, torch.zeros_like(actions))
        # Keep every downstream action buffer finite. The violation mask retains
        # the fact that the actor supplied NaN/Inf so termination can reject the
        # transition instead of silently accepting the zero-motion fallback.
        self._raw_actions.copy_(safe_actions)
        self._out_of_bounds.copy_(~finite | (torch.abs(safe_actions) > 1.0 + self.cfg.action_bound_epsilon))
        self._bounded_actions.copy_(torch.clamp(safe_actions, -1.0, 1.0))
        self._near_saturation.copy_(torch.abs(self._bounded_actions) >= self.cfg.saturation_threshold)

        samples = self._episode_action_samples + 1
        weight = samples.float().reciprocal()
        step_mean_abs = torch.abs(self._bounded_actions).mean(dim=1)
        step_near_fraction = self._near_saturation.float().mean(dim=1)
        self._episode_action_mean_abs.add_((step_mean_abs - self._episode_action_mean_abs) * weight)
        self._episode_near_saturation_fraction.add_(
            (step_near_fraction - self._episode_near_saturation_fraction) * weight
        )
        self._episode_action_max_abs.copy_(
            torch.maximum(
                self._episode_action_max_abs,
                torch.abs(self._bounded_actions).max(dim=1).values,
            )
        )
        self._episode_near_saturation_any |= self._near_saturation.any(dim=1)
        self._episode_out_of_bounds_any |= self._out_of_bounds.any(dim=1)
        self._episode_action_samples.copy_(samples)

        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        command_was_finite = torch.isfinite(self._joint_command)
        safe_previous_command = torch.where(command_was_finite, self._joint_command, self._q_reset_ref)
        candidate = self.integrate_command(
            safe_previous_command,
            self._bounded_actions,
            self._velocity_limits,
            self.cfg.control_dt_s,
            limits,
        )
        candidate_is_finite = torch.isfinite(candidate)
        self._out_of_bounds |= ~command_was_finite | ~candidate_is_finite
        self._episode_out_of_bounds_any |= self._out_of_bounds.any(dim=1)
        # A corrupt internal command or simulator limit must never poison the
        # persistent command. Fall back to the last finite command, then to the
        # validated reset reference if even the clamped fallback is non-finite.
        candidate = torch.where(candidate_is_finite, candidate, safe_previous_command)
        candidate = torch.where(torch.isfinite(candidate), candidate, self._q_reset_ref)
        self._joint_command.copy_(candidate)

    @staticmethod
    def integrate_command(
        joint_command: torch.Tensor,
        bounded_action: torch.Tensor,
        velocity_limits_rad_s: torch.Tensor,
        control_dt_s: float,
        joint_limits: torch.Tensor,
    ) -> torch.Tensor:
        """Pure implementation of the deployment-identical rate integration."""

        safe_action = torch.where(torch.isfinite(bounded_action), bounded_action, torch.zeros_like(bounded_action))
        safe_command = torch.where(torch.isfinite(joint_command), joint_command, torch.zeros_like(joint_command))
        increment = safe_action * velocity_limits_rad_s * control_dt_s
        return torch.clamp(safe_command + increment, min=joint_limits[..., 0], max=joint_limits[..., 1])

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._joint_command, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._bounded_actions[env_ids] = 0.0
        self._previous_bounded_actions[env_ids] = 0.0
        self._near_saturation[env_ids] = False
        self._out_of_bounds[env_ids] = False
        self._joint_command[env_ids] = self._q_reset_ref[env_ids]
        self._episode_action_samples[env_ids] = 0
        self._episode_action_mean_abs[env_ids] = 0.0
        self._episode_action_max_abs[env_ids] = 0.0
        self._episode_near_saturation_fraction[env_ids] = 0.0
        self._episode_near_saturation_any[env_ids] = False
        self._episode_out_of_bounds_any[env_ids] = False


@configclass
class JointRatePositionActionCfg(ActionTermCfg):
    """Configuration for :class:`JointRatePositionAction`."""

    class_type: type = JointRatePositionAction
    asset_name: str = MISSING
    joint_names: tuple[str, ...] = MISSING
    velocity_limits_rad_s: tuple[float, ...] = MISSING
    q_reset_ref: tuple[float, ...] = MISSING
    command_name: str = "typing"
    control_dt_s: float = 0.04
    saturation_threshold: float = 0.95
    action_bound_epsilon: float = 1.0e-6
