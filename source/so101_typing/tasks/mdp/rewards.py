# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import LetterTypingCommand


def letter_typing_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""High-water-mark typing-progress reward: a per-episode ratchet on the correct-prefix length.

    Fetches the :class:`LetterTypingCommand` term and returns a sparse per-step signal driven only by
    *records* in the length of the contiguous correct prefix (``prefix_len``):

    * ``+1`` on any step that sets a new episode maximum prefix length (genuine forward progress),
    * ``-1`` on any step that sets a new episode minimum prefix length (destroying correct progress),
    * ``0`` otherwise - staying put, or re-reaching a prefix length already seen this episode.

    Since one control step registers at most one keystroke, ``prefix_len`` moves by at most one, so each
    integer level is crossed cleanly. Rewarding only *new* maxima means the total positive reward per
    episode is bounded by ``target_len - prefix_0`` and the total penalty by ``prefix_0`` (with
    ``prefix_0`` the correct prefix of the reset buffer); consequently ``type-wrong -> backspace`` and
    ``backspace -> retype`` loops cannot farm reward - a re-reached level is neither a new max nor a new
    min. The marks are episode state seeded at reset, so they live on the command (see
    :meth:`~...typing_commands.LetterTypingCommand._update_metrics`); this term only reads the resulting
    per-step flags. With single-letter targets ``prefix_0`` is always ``0``, so the new-min penalty is
    inert and only the new-max bonus fires; it starts mattering for multi-letter targets.

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read progress from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return command.new_high.float() - command.new_low.float()


def typing_success(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""Sparse success bonus: ``1`` when the whole target word is typed correctly.

    Fires when the :class:`LetterTypingCommand` edit distance is zero, i.e. the typed buffer exactly
    matches the target with no missing and no extra/wrong keys. It is returned on every step the env
    stays in that completed state (until the command resamples a new word).

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read completion from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return (command.distance == 0).float()


class reach_key(ManagerTermBase):
    r"""Reach reward toward the next target key using a tanh kernel on either jaw tip.

    Each gripper jaw has a tip at a fixed offset in its link frame. The reward uses the *closer* of
    the two tips to the next target key, so reaching with either jaw counts:

    .. math::

        r = 1 - \tanh\!\left(\frac{\min(d_\text{moving}, d_\text{fixed})}{\text{std}}\right)

    The value is ``1`` when a tip is on the (lowered) target and decays toward ``0`` with distance
    (``std`` [m] sets the falloff scale), so it stays in ``(0, 1]``. The target is placed
    ``press_depth`` [m] below the key (along world ``-Z``) so the optimum sits past the actuation point,
    encouraging the policy to push down rather than just hover on the cap. Body indices, tip offsets, and
    the press offset are resolved once at construction so each step only does the gather and kernel.

    Args (from ``cfg.params``):
        command_name: Name of the :class:`LetterTypingCommand` term providing the next target key.
        moving_jaw_body: Body name of the moving jaw link.
        moving_jaw_offset: Tip offset [m] in the moving jaw link frame.
        fixed_jaw_body: Body name of the fixed jaw link.
        fixed_jaw_offset: Tip offset [m] in the fixed jaw link frame.
        std: Distance [m] falloff scale of the tanh kernel. Defaults to ``0.2``.
        press_depth: Distance [m] to lower the reach target below the key along world ``-Z`` so reaching
            it requires pressing the key down. Defaults to ``0.0`` (reach the key itself).
        asset_cfg: Scene entity for the robot whose jaw links are read. Defaults to ``SceneEntityCfg("robot")``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.robot = env.scene[asset_cfg.name]
        self.std: float = cfg.params.get("std", 0.2)  # type: ignore
        # resolve body indices and broadcast tip offsets once (they never change at runtime).
        self._moving = self.robot.body_names.index(cfg.params["moving_jaw_body"])
        self._fixed = self.robot.body_names.index(cfg.params["fixed_jaw_body"])
        self._off_moving = torch.tensor(cfg.params["moving_jaw_offset"], device=env.device).expand(env.num_envs, 3)
        self._off_fixed = torch.tensor(cfg.params["fixed_jaw_offset"], device=env.device).expand(env.num_envs, 3)
        # world -Z offset that lowers the reach target below the key to encourage pressing.
        self._press_offset = torch.tensor((0.0, 0.0, cfg.params.get("press_depth", 0.0)), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        moving_jaw_body: str,
        moving_jaw_offset: tuple[float, float, float],
        fixed_jaw_body: str,
        fixed_jaw_offset: tuple[float, float, float],
        std: float = 0.2,
        press_depth: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
        # aim slightly below the key (world -Z) so the tip is rewarded for pressing, not just touching.
        target_pos_w = command.target_key_pos_w() - self._press_offset  # (N, 3)

        body_pos_w = self.robot.data.body_pos_w.torch
        body_quat_w = self.robot.data.body_quat_w.torch
        tip_moving = body_pos_w[:, self._moving] + quat_apply(body_quat_w[:, self._moving], self._off_moving)
        tip_fixed = body_pos_w[:, self._fixed] + quat_apply(body_quat_w[:, self._fixed], self._off_fixed)

        distance = torch.minimum(
            torch.linalg.norm(tip_moving - target_pos_w, dim=-1),
            torch.linalg.norm(tip_fixed - target_pos_w, dim=-1),
        )
        return 1.0 - torch.tanh(distance / self.std)


class fixed_jaw_low_clearance_lateral_motion(ManagerTermBase):
    """Normalized fixed-jaw scrape signal while a validated key is held.

    This is privileged simulator shaping. The actor never receives key poses,
    contact state, clearance, or tip velocity. A press must be released before
    moving laterally to the next key, so lateral motion close to the pressed key
    is the unwanted drag behavior seen in hardware rollouts.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.robot = env.scene[asset_cfg.name]
        self._tip_body = self.robot.body_names.index(cfg.params.get("tip_body", "gripper_link"))
        self._tip_offset = torch.tensor(cfg.params["tip_offset"], device=env.device).expand(env.num_envs, 3)
        self._clearance_m = float(cfg.params.get("clearance_m", 0.004))
        self._speed_m_s = float(cfg.params.get("lateral_speed_m_s", 0.04))

    def _scrape_level(self, command_name: str) -> torch.Tensor:
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
        low_clearance = torch.clamp((self._clearance_m - tip_to_key[:, 2]) / self._clearance_m, min=0.0, max=1.0)
        lateral_speed = torch.linalg.vector_norm(tip_vel_key[:, :2], dim=-1)
        normalized_speed = torch.clamp(lateral_speed / self._speed_m_s, min=0.0, max=1.0)
        return torch.where(command.must_release, low_clearance * normalized_speed, torch.zeros_like(lateral_speed))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        tip_body: str = "gripper_link",
        tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        clearance_m: float = 0.004,
        lateral_speed_m_s: float = 0.04,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del env, tip_body, tip_offset, clearance_m, lateral_speed_m_s, asset_cfg
        return self._scrape_level(command_name)


class fixed_cartesian_seek_progress(ManagerTermBase):
    """Bounded potential change toward the current key center during seek."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._distance_scale = float(cfg.params.get("distance_scale", 0.02))
        self._previous_distance = torch.zeros(env.num_envs, device=env.device)
        self._previous_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._seen_reset_serial = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, distance_scale: float = 0.02) -> torch.Tensor:
        del distance_scale
        command = env.command_manager.get_term(command_name)
        distance = torch.linalg.vector_norm(command.fixed_tip_pos_w() - command.target_key_pos_w(), dim=-1)
        transition = (command.reset_serial != self._seen_reset_serial) | (command.phase != self._previous_phase)
        progress = (self._previous_distance - distance) / self._distance_scale
        active = (command.phase == 0) & ~transition
        self._previous_distance.copy_(distance)
        self._previous_phase.copy_(command.phase)
        self._seen_reset_serial.copy_(command.reset_serial)
        return torch.where(active, progress.clamp(-1.0, 1.0), torch.zeros_like(progress))


class fixed_cartesian_lift_progress(ManagerTermBase):
    """Potential change in vertical jaw clearance during release/clearance."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._height_scale = float(cfg.params.get("height_scale", 0.004))
        self._previous_height = torch.zeros(env.num_envs, device=env.device)
        self._previous_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._seen_reset_serial = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, height_scale: float = 0.004) -> torch.Tensor:
        del height_scale
        command = env.command_manager.get_term(command_name)
        height = command.fixed_tip_pos_w()[:, 2] - command.target_key_pos_w()[:, 2]
        transition = (command.reset_serial != self._seen_reset_serial) | (command.phase != self._previous_phase)
        progress = (height - self._previous_height) / self._height_scale
        active = (command.phase != 0) & ~transition
        self._previous_height.copy_(height)
        self._previous_phase.copy_(command.phase)
        self._seen_reset_serial.copy_(command.reset_serial)
        return torch.where(active, progress.clamp(-1.0, 1.0), torch.zeros_like(progress))


def fixed_cartesian_target_down(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-shot correct target down event."""

    return env.command_manager.get_term(command_name).target_down_event.float()


def fixed_cartesian_target_up(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-shot matching target up event."""

    return env.command_manager.get_term(command_name).target_up_event.float()


def fixed_cartesian_clearance(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-shot two-control-tick clearance event."""

    return env.command_manager.get_term(command_name).clearance_event.float()


def fixed_cartesian_success(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Strict success after final target down, up, and clearance."""

    return env.command_manager.get_term(command_name).completed.float()


def fixed_cartesian_action_change_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Mean squared change of bounded joint-rate action."""

    action = env.action_manager.get_term(action_name)
    delta = action.bounded_actions - action.previous_bounded_actions
    return torch.mean(torch.square(delta), dim=1)


def fixed_cartesian_joint_velocity_l2(env: ManagerBasedRLEnv, command_name: str, action_name: str) -> torch.Tensor:
    """Mean squared measured velocity normalized by the hard rate envelope."""

    command = env.command_manager.get_term(command_name)
    action = env.action_manager.get_term(action_name)
    robot: Articulation = env.scene[command.cfg.asset_name]
    velocity = robot.data.joint_vel.torch[:, command._arm_joint_ids]
    return torch.mean(torch.square(velocity / action.velocity_limits_rad_s), dim=1)
