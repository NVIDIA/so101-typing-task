# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .commands.keyboard_view import build_keyboard_slot_gather

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg

    from .commands import LetterTypingCommand


def _slots_onehot(slots: torch.Tensor, num_keys: int) -> torch.Tensor:
    """One-hot a ``(num_envs, max_len)`` slot-id buffer (``-1`` marks empty) into ``(num_envs, max_len * num_keys)``.

    Empty/padding positions (slot ``-1``) become all-zero rows, so they contribute nothing.
    """
    valid = slots >= 0
    onehot = torch.nn.functional.one_hot(slots.clamp(min=0), num_classes=num_keys).to(torch.float)
    onehot = onehot * valid.unsqueeze(-1).to(onehot.dtype)
    return onehot.reshape(slots.shape[0], -1)


def target_keys_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target word as a one-hot sequence over key slots, shape ``(num_envs, max_len * num_keys)``.

    Each sequence position holds a one-hot of that target key (all-zero past the word's length). The
    compact perception terms provide current and next key positions for the actionable target.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return _slots_onehot(command.target, command.num_keys)


def typed_keys_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Typed buffer as a one-hot sequence over key slots, shape ``(num_envs, max_len * num_keys)``.

    Each sequence position holds a one-hot of the key typed there so far (all-zero for not-yet-typed
    positions). Comparing it against :func:`target_keys_onehot` yields the typing progress and any
    mistake the agent must backspace.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return _slots_onehot(command.typed, command.num_keys)


def fixed_joint_pos_rel_reset(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Measured five-arm-joint position relative to the frozen handoff pose."""

    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[command.cfg.asset_name]
    return robot.data.joint_pos.torch[:, command._arm_joint_ids] - command.q_reset_ref


def fixed_joint_vel(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Measured five-arm-joint velocity [rad/s] in action-joint order."""

    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[command.cfg.asset_name]
    return robot.data.joint_vel.torch[:, command._arm_joint_ids]


def fixed_joint_command_rel_reset(env: ManagerBasedRLEnv, action_name: str, command_name: str) -> torch.Tensor:
    """Latest persistent issued joint command relative to the frozen handoff pose."""

    command = env.command_manager.get_term(command_name)
    action = env.action_manager.get_term(action_name)
    return action.joint_command - command.q_reset_ref


def fixed_target_xyz_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Frozen current-letter XYZ, normalized by frozen A-Z map statistics."""

    command = env.command_manager.get_term(command_name)
    return command.target_xyz_b_normalized


def fixed_typing_phase_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-hot ``seek_press``, ``release_key``, or ``clearance`` phase."""

    command = env.command_manager.get_term(command_name)
    return torch.nn.functional.one_hot(command.phase, num_classes=3).to(torch.float)


def fixed_phase_elapsed_s(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Real seconds elapsed since the current phase began."""

    command = env.command_manager.get_term(command_name)
    return command.phase_elapsed_s.unsqueeze(-1)


def _typing_action_key_slots(command: LetterTypingCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the current and next actionable key slots for the typing command."""
    rows = torch.arange(command.target.shape[0], device=command.target.device)
    invalid = torch.full_like(command.prefix_len, -1)
    backspace = torch.full_like(command.prefix_len, int(command.cfg.backspace_slot))

    prefix = command.prefix_len
    wrong_count = command.typed_len - prefix

    current_target_col = prefix.clamp(max=command.max_len - 1)
    current_target = command.target[rows, current_target_col]
    has_current_target = prefix < command.target_len
    current_target = torch.where(has_current_target, current_target, invalid)

    current_slot = torch.where(wrong_count > 0, backspace, current_target)

    next_target_col = (prefix + 1).clamp(max=command.max_len - 1)
    next_target = command.target[rows, next_target_col]
    has_next_target = (prefix + 1) < command.target_len
    next_after_typing = torch.where(has_next_target, next_target, invalid)
    next_after_backspace = torch.where(wrong_count > 1, backspace, current_target)
    next_slot = torch.where(wrong_count > 0, next_after_backspace, next_after_typing)

    return current_slot, next_slot


class _CommandKeyPositionB(ManagerTermBase):
    """Base class for selected typing-key positions [m] in the robot base frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        keyboard_cfg: SceneEntityCfg = cfg.params.get("keyboard_cfg", SceneEntityCfg("keyboard"))
        self.keyboard: Articulation = env.scene[keyboard_cfg.name]
        self.num_slots, self._inst_idx, _, self._body_idx = build_keyboard_slot_gather(
            self.keyboard, self.num_envs, self.device
        )
        self._rows = torch.arange(self.num_envs, device=self.device)

    def _key_positions_b(
        self, env: ManagerBasedRLEnv, slots: torch.Tensor, base_asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        """Return selected key positions [m] in the base asset frame, shape ``(num_envs, 3)``."""
        valid = (slots >= 0) & (slots < self.num_slots)
        safe_slots = slots.clamp(min=0, max=self.num_slots - 1)
        inst_idx = self._inst_idx[self._rows, safe_slots]
        body_idx = self._body_idx[self._rows, safe_slots]

        base: Articulation = env.scene[base_asset_cfg.name]
        key_pos_w = self.keyboard.data.body_pos_w.torch[inst_idx, body_idx]
        key_pos_b, _ = subtract_frame_transforms(
            base.data.root_link_pos_w.torch,
            base.data.root_link_quat_w.torch,
            key_pos_w,
        )
        return key_pos_b * valid.unsqueeze(-1).to(key_pos_b.dtype)


class current_key_position_b(_CommandKeyPositionB):
    """Current actionable key position ``(x, y, z)`` [m] in the robot base frame.

    The selected key is Backspace while the typed buffer contains a wrong suffix; otherwise it is the
    next untyped target key. Completed episodes return zeros.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        keyboard_cfg: SceneEntityCfg = SceneEntityCfg("keyboard"),
        base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
        current_slot, _ = _typing_action_key_slots(command)
        return self._key_positions_b(env, current_slot, base_asset_cfg)


class next_key_position_b(_CommandKeyPositionB):
    """Next actionable key position ``(x, y, z)`` [m] in the robot base frame.

    For wrong typed suffixes longer than one key, this remains Backspace. For a single wrong key, this is
    the target key that becomes current after Backspace. For normal progress, this is the following target
    key. Missing next keys return zeros.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        keyboard_cfg: SceneEntityCfg = SceneEntityCfg("keyboard"),
        base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
        _, next_slot = _typing_action_key_slots(command)
        return self._key_positions_b(env, next_slot, base_asset_cfg)


class key_positions_b(ManagerTermBase):
    """Per-key keyboard positions ``(x, y, z)`` [m] in the base asset's root frame, in global slot order.

    A gather map is built once from the keyboard articulation's ``(instance, body)`` layout into the
    canonical global key-slot order (``slot = part * dof + local``), mirroring the mapping in
    :class:`~isaaclab_tasks.core.dexsuite.config.so101.mdp.commands.LetterTypingCommand`. This makes the
    observation layout-stable: column ``s`` always refers to the same logical key slot regardless of how
    the keyboard is partitioned into articulation instances. Inactive slots are zero-padded, so a
    keyboard with fewer than :attr:`num_slots` real keys fills only its active subset of columns and the
    remainder stays ``0``.

    The map is agnostic to the partition mode:

    * ``single`` -> one instance per env; key body ``key_{slot:03d}``.
    * ``fixed_dof`` -> ``parts/part_*`` instances per env; key body ``key_{local:03d}``.

    Args (from ``cfg.params``):
        keyboard_cfg: Scene entity for the keyboard articulation. Defaults to ``SceneEntityCfg("keyboard")``.
        base_asset_cfg: Scene entity providing the reference root frame. Defaults to ``SceneEntityCfg("robot")``.
        active_slots: Global key-slot ids that hold a real key; all other slots are zero-padded. ``None``
            keeps every slot that mapped to a key body.

    Returns (from :meth:`__call__`):
        Tensor of shape ``(num_envs, num_slots * 3)`` with per-slot key positions [m] in the base frame,
        flattened in global slot order and with inactive slots zeroed.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        keyboard_cfg: SceneEntityCfg = cfg.params.get("keyboard_cfg", SceneEntityCfg("keyboard"))
        self.keyboard: Articulation = env.scene[keyboard_cfg.name]

        # Backend-agnostic gather into the articulation's (num_instances, num_bodies) body data so reads
        # are plain advanced indexing: body_pos_w[inst_idx, body_idx]. Uses the same global slot order
        # (slot = part * dof + local) as the typing command, and handles PhysX vs Newton instance order.
        self.num_slots, self._inst_idx, _, self._body_idx = build_keyboard_slot_gather(
            self.keyboard, self.num_envs, self.device
        )

        # Zero-padding mask: every gathered slot maps to a real key body, so keep them all unless
        # active_slots restricts the observation to a subset of slots (the rest stay zeroed).
        active_slots: tuple[int, ...] | None = cfg.params.get("active_slots", None)  # type: ignore
        if active_slots is None:
            active = torch.ones(self.num_envs, self.num_slots, dtype=torch.bool, device=self.device)
        else:
            active = torch.zeros(self.num_envs, self.num_slots, dtype=torch.bool, device=self.device)
            active[:, torch.as_tensor(tuple(active_slots), dtype=torch.long, device=self.device)] = True
        self._active = active.unsqueeze(-1).float()  # (num_envs, num_slots, 1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        keyboard_cfg: SceneEntityCfg = SceneEntityCfg("keyboard"),
        base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        active_slots: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        base: Articulation = env.scene[base_asset_cfg.name]
        # (num_envs, num_slots, 3) key body world positions gathered into global slot order.
        key_pos_w = self.keyboard.data.body_pos_w.torch[self._inst_idx, self._body_idx]
        # broadcast the base root pose over slots and transform world -> base frame (position only).
        root_pos_w = base.data.root_link_pos_w.torch.unsqueeze(1).expand(-1, self.num_slots, -1).reshape(-1, 3)
        root_quat_w = base.data.root_link_quat_w.torch.unsqueeze(1).expand(-1, self.num_slots, -1).reshape(-1, 4)
        key_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, key_pos_w.reshape(-1, 3))
        key_pos_b = key_pos_b.view(env.num_envs, self.num_slots, 3) * self._active
        return key_pos_b.reshape(env.num_envs, -1)
