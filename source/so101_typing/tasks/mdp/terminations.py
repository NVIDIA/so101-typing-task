# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import LetterTypingCommand


def typing_mistake(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate the episode when a wrong (or extra) key has been typed.

    Fires as soon as the typed buffer extends past the correct prefix (``typed_len > prefix_len``),
    i.e. the agent typed a key that does not continue the target word. Note this makes the backspace
    recovery path unreachable - a single mistyped key ends the episode.

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read typing state from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return command.typed_len > command.prefix_len


def typing_complete(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate the episode once the whole target word has been typed correctly.

    Fires when the :class:`LetterTypingCommand` edit distance reaches zero (the typed buffer exactly
    matches the target), mirroring the :func:`~...mdp.rewards.typing_success` bonus. This is a genuine
    goal-reached terminal state, so it should be registered without ``time_out=True``.

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read typing state from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return command.distance == 0


def fixed_cartesian_invalid_key(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Consume this transition's switch sample and reject wrong/overlap/held starts."""

    command = env.command_manager.get_term(command_name)
    command.finalize_control_transition()
    return command.wrong_key | command.overlap | command.initially_held


def fixed_cartesian_map_invalid(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Block rollout when live A-Z key centers disagree with the frozen map."""

    command = env.command_manager.get_term(command_name)
    command.finalize_control_transition()
    return command.map_invalid


def fixed_cartesian_phase_timeout(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate a seek/release/clearance phase that exceeds its real-time budget."""

    command = env.command_manager.get_term(command_name)
    command.finalize_control_transition()
    return command.phase_elapsed_s > command.phase_timeout_s


def fixed_cartesian_action_contract_violation(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Reject an actor/runtime that emits outside the declared bounded support."""

    action = env.action_manager.get_term(action_name)
    return action.out_of_bounds.any(dim=1)


def fixed_cartesian_typing_complete(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Strict completion after the final release and two-tick clearance."""

    command = env.command_manager.get_term(command_name)
    command.finalize_control_transition()
    return command.completed


class fixed_jaw_scrape(ManagerTermBase):
    """Terminate sustained lateral motion while the fixed jaw holds a key down."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.robot = env.scene[asset_cfg.name]
        self._tip_body = self.robot.body_names.index(cfg.params.get("tip_body", "gripper_link"))
        self._tip_offset = torch.tensor(cfg.params["tip_offset"], device=env.device).expand(env.num_envs, 3)
        self._clearance_m = float(cfg.params.get("clearance_m", 0.004))
        self._lateral_speed_m_s = float(cfg.params.get("lateral_speed_m_s", 0.04))
        self._persistence_steps = int(cfg.params.get("persistence_steps", 3))
        if self._persistence_steps < 1:
            raise ValueError("persistence_steps must be positive")
        self._consecutive = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._consecutive[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        tip_body: str = "gripper_link",
        tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        clearance_m: float = 0.004,
        lateral_speed_m_s: float = 0.04,
        persistence_steps: int = 3,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del env, tip_body, tip_offset, clearance_m, lateral_speed_m_s, persistence_steps, asset_cfg
        command = self._env.command_manager.get_term(command_name)
        body_pos = self.robot.data.body_pos_w.torch[:, self._tip_body]
        body_quat = self.robot.data.body_quat_w.torch[:, self._tip_body]
        lever_w = quat_apply(body_quat, self._tip_offset)
        tip_pos = body_pos + lever_w
        tip_vel = self.robot.data.body_lin_vel_w.torch[:, self._tip_body] + torch.cross(
            self.robot.data.body_ang_vel_w.torch[:, self._tip_body], lever_w, dim=-1
        )
        key_quat = command.pressed_key_quat_w()
        tip_to_key = quat_apply(quat_conjugate(key_quat), tip_pos - command.pressed_key_pos_w())
        tip_vel_key = quat_apply(quat_conjugate(key_quat), tip_vel)
        lateral_speed = torch.linalg.vector_norm(tip_vel_key[:, :2], dim=-1)
        scraping = (
            command.must_release & (tip_to_key[:, 2] < self._clearance_m) & (lateral_speed > self._lateral_speed_m_s)
        )
        self._consecutive = torch.where(scraping, self._consecutive + 1, torch.zeros_like(self._consecutive))
        return self._consecutive >= self._persistence_steps
