# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-map, press/release/clearance typing command."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from .typing_commands import LetterTypingCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .fixed_cartesian_typing_commands_cfg import FixedCartesianTypingCommandCfg


SEEK_PRESS = 0
RELEASE_KEY = 1
CLEARANCE = 2
PHASE_COUNT = 3


class FixedCartesianTypingCommand(LetterTypingCommand):
    """A-Z target command with a strict three-phase switch state machine.

    The actor receives a frozen Cartesian target.  Exact live key geometry is
    retained only for validation, rewards, safety, and clearance detection.
    No F/Y/N anchor or reset-time IK solve is used.
    """

    cfg: FixedCartesianTypingCommandCfg

    def __init__(self, cfg: FixedCartesianTypingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._arm_joint_ids, _ = self.robot.find_joints(
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            preserve_order=True,
        )
        if len(cfg.q_reset_ref) != 5:
            raise ValueError("q_reset_ref must contain the five controlled arm joints")
        if len(cfg.letter_slots) != 26 or len(cfg.letter_xyz_b_m) != 26:
            raise ValueError("fixed Cartesian command requires exactly 26 A-Z slots and XYZ rows")
        if cfg.stable_samples < 1 or cfg.clearance_control_ticks < 1:
            raise ValueError("stable_samples and clearance_control_ticks must be positive")
        if not 0.0 <= cfg.key_up_fraction < cfg.key_down_fraction <= 1.0:
            raise ValueError("key hysteresis requires 0 <= up_fraction < down_fraction <= 1")

        self.q_reset_ref = torch.tensor(cfg.q_reset_ref, device=self.device).expand(self.num_envs, -1)
        self._letter_slots = torch.tensor(cfg.letter_slots, dtype=torch.long, device=self.device)
        self._letter_xyz_b = torch.tensor(cfg.letter_xyz_b_m, device=self.device)
        self._target_xyz_mean_b = torch.tensor(cfg.target_xyz_mean_b_m, device=self.device)
        self._target_xyz_std_b = torch.tensor(cfg.target_xyz_std_b_m, device=self.device)
        self._slot_to_letter = torch.full((self.num_keys,), -1, dtype=torch.long, device=self.device)
        self._slot_to_letter[self._letter_slots] = torch.arange(26, device=self.device)

        body_ids, _ = self.robot.find_bodies(cfg.tip_body)
        self._tip_body_idx = body_ids[0]
        self._tip_offset = torch.tensor(cfg.tip_offset_m, device=self.device).expand(self.num_envs, 3)

        n, k = self.num_envs, self.num_keys
        self.phase = torch.full((n,), SEEK_PRESS, dtype=torch.long, device=self.device)
        self.character_index = torch.zeros(n, dtype=torch.long, device=self.device)
        self.pressed_slot = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.held = torch.zeros((n, k), dtype=torch.bool, device=self.device)
        self._down_count = torch.zeros((n, k), dtype=torch.int16, device=self.device)
        self._up_count = torch.zeros((n, k), dtype=torch.int16, device=self.device)
        self._initial_sample_count = torch.zeros(n, dtype=torch.int16, device=self.device)
        self._initialized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._clear_ticks = torch.zeros(n, dtype=torch.int16, device=self.device)
        self._phase_start_sim_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self._last_global_finalized_control_step = -1

        self.target_down_event = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.target_up_event = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.clearance_event = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.wrong_key = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.overlap = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.initially_held = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.completed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.reset_serial = torch.zeros(n, dtype=torch.long, device=self.device)

        self._map_checked = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._map_validation_complete = False
        self.map_invalid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.map_max_error_m = torch.zeros(n, device=self.device)
        self._press_threshold: torch.Tensor | None = None
        self._release_threshold: torch.Tensor | None = None
        self._key_upper_limit: torch.Tensor | None = None
        self._key_travel: torch.Tensor | None = None
        self._episode_max_target_depression_fraction = torch.zeros(n, device=self.device)

        self.metrics["phase"] = torch.zeros(n, device=self.device)
        self.metrics["map_max_error_m"] = torch.zeros(n, device=self.device)
        self.metrics["held_key_count"] = torch.zeros(n, device=self.device)
        self.metrics["wrong_key"] = torch.zeros(n, device=self.device)
        self.metrics["overlap"] = torch.zeros(n, device=self.device)
        self.metrics["sequence/first_character_complete"] = torch.zeros(n, device=self.device)
        self.metrics["sequence/second_character_reached"] = torch.zeros(n, device=self.device)
        self.metrics["sequence/second_character_down"] = torch.zeros(n, device=self.device)
        self.metrics["sequence/second_character_complete"] = torch.zeros(n, device=self.device)
        self.metrics["sequence/full_complete"] = torch.zeros(n, device=self.device)
        self.metrics["key/max_target_depression_fraction"] = torch.zeros(n, device=self.device)
        self.metrics["action/mean_abs"] = torch.zeros(n, device=self.device)
        self.metrics["action/max_abs"] = torch.zeros(n, device=self.device)
        self.metrics["action/near_saturation_fraction"] = torch.zeros(n, device=self.device)
        self.metrics["action/near_saturation_any"] = torch.zeros(n, device=self.device)
        self.metrics["action/out_of_bounds_any"] = torch.zeros(n, device=self.device)

    def __str__(self) -> str:
        return "FixedCartesianTypingCommand(A-Z, seek/release/clearance)"

    @property
    def phase_elapsed_s(self) -> torch.Tensor:
        sim_step = int(getattr(self._env, "_sim_step_counter", 0))
        elapsed_steps = (sim_step - self._phase_start_sim_step).clamp(min=0)
        return elapsed_steps.float() * float(self._env.physics_dt)

    @property
    def phase_timeout_s(self) -> torch.Tensor:
        timeouts = torch.tensor(self.cfg.phase_timeouts_s, device=self.device)
        return timeouts[self.phase]

    @property
    def target_xyz_b(self) -> torch.Tensor:
        letter = self._slot_to_letter[self.target_key_slot()].clamp(min=0)
        return self._letter_xyz_b[letter]

    @property
    def target_xyz_b_normalized(self) -> torch.Tensor:
        return (self.target_xyz_b - self._target_xyz_mean_b) / self._target_xyz_std_b

    @property
    def must_release(self) -> torch.Tensor:
        """Compatibility mask for held/post-release scrape guards."""

        return (self.phase != SEEK_PRESS) & ~self.completed

    def target_key_slot(self) -> torch.Tensor:
        rows = torch.arange(self.num_envs, device=self.device)
        column = self.character_index.clamp(max=self.max_len - 1)
        sampled = self.target[rows, column].clamp(min=0)
        # pressed_slot is retained through release and clearance so the next
        # character cannot leak through the Cartesian observation.
        return torch.where(self.phase == SEEK_PRESS, sampled, self.pressed_slot.clamp(min=0))

    def target_key_pos_w(self) -> torch.Tensor:
        rows = torch.arange(self.num_envs, device=self.device)
        slot = self.target_key_slot()
        return self.keyboard.data.body_pos_w.torch[self._inst_idx[rows, slot], self._body_col_idx[rows, slot]]

    def pressed_key_pos_w(self) -> torch.Tensor:
        rows = torch.arange(self.num_envs, device=self.device)
        slot = self.pressed_slot.clamp(min=0)
        return self.keyboard.data.body_pos_w.torch[self._inst_idx[rows, slot], self._body_col_idx[rows, slot]]

    def pressed_key_quat_w(self) -> torch.Tensor:
        rows = torch.arange(self.num_envs, device=self.device)
        slot = self.pressed_slot.clamp(min=0)
        return self.keyboard.data.body_quat_w.torch[self._inst_idx[rows, slot], self._body_col_idx[rows, slot]]

    def fixed_tip_pos_w(self) -> torch.Tensor:
        body_pos = self.robot.data.body_pos_w.torch[:, self._tip_body_idx]
        body_quat = self.robot.data.body_quat_w.torch[:, self._tip_body_idx]
        return body_pos + quat_apply(body_quat, self._tip_offset)

    def begin_control_transition(self) -> None:
        """Clear one-transition event flags before a new actor action."""

        self.target_down_event.zero_()
        self.target_up_event.zero_()
        self.clearance_event.zero_()

    def sync_key_state(self) -> None:
        """Sample all 108 key joints and update the hysteretic edge state.

        Newton currently exposes the folded decimation block at the manager
        boundary, so this is a 25 Hz logical sample under the primary backend.
        The two-sample dwell and hard joint-rate envelope prevent sub-tick
        bounce from being interpreted as a clean press/release.
        """

        if self._press_threshold is None or self._release_threshold is None:
            limits = self.keyboard.data.joint_pos_limits.torch[self._inst_idx, self._col_idx]
            lower, upper = limits[..., 0], limits[..., 1]
            travel = upper - lower
            self._key_upper_limit = upper
            self._key_travel = travel
            self._press_threshold = upper - self.cfg.key_down_fraction * travel
            self._release_threshold = upper - self.cfg.key_up_fraction * travel

        pos = self.keyboard.data.joint_pos.torch[self._inst_idx, self._col_idx]
        rows = torch.arange(self.num_envs, device=self.device)
        target_slot = self.target_key_slot()
        target_depression = (self._key_upper_limit[rows, target_slot] - pos[rows, target_slot]) / self._key_travel[
            rows, target_slot
        ].clamp_min(1.0e-9)
        self._episode_max_target_depression_fraction = torch.maximum(
            self._episode_max_target_depression_fraction,
            target_depression.clamp_min(0.0),
        )
        down_condition = pos <= self._press_threshold
        up_condition = pos >= self._release_threshold

        was_initialized = self._initialized.clone()
        initializing = ~was_initialized
        self._down_count = torch.where(
            initializing[:, None],
            torch.where(
                down_condition,
                self._down_count + 1,
                torch.zeros_like(self._down_count),
            ),
            self._down_count,
        )
        self._initial_sample_count = torch.where(
            initializing,
            self._initial_sample_count + 1,
            self._initial_sample_count,
        )
        ready = initializing & (self._initial_sample_count >= self.cfg.stable_samples)
        initial_down = ready[:, None] & (self._down_count >= self.cfg.stable_samples)
        self.held |= initial_down
        self.initially_held |= initial_down.any(dim=1)
        self._initialized |= ready
        self._down_count = torch.where(ready[:, None], torch.zeros_like(self._down_count), self._down_count)

        held_before = self.held.clone()
        eligible_down = was_initialized[:, None] & ~held_before
        eligible_up = was_initialized[:, None] & held_before
        self._down_count = torch.where(
            eligible_down & down_condition,
            self._down_count + 1,
            torch.zeros_like(self._down_count),
        )
        self._up_count = torch.where(
            eligible_up & up_condition,
            self._up_count + 1,
            torch.zeros_like(self._up_count),
        )
        new_down = eligible_down & (self._down_count >= self.cfg.stable_samples)
        new_up = eligible_up & (self._up_count >= self.cfg.stable_samples)
        self.held = (self.held | new_down) & ~new_up
        self._down_count = torch.where(new_down, torch.zeros_like(self._down_count), self._down_count)
        self._up_count = torch.where(new_up, torch.zeros_like(self._up_count), self._up_count)

        self._consume_edges(new_down, new_up, held_before)
        self._validate_map_once()

    def _consume_edges(self, new_down: torch.Tensor, new_up: torch.Tensor, held_before: torch.Tensor) -> None:
        rows = torch.arange(self.num_envs, device=self.device)
        target_slot = self.target_key_slot()
        down_count = new_down.sum(dim=1)
        expected_down = new_down[rows, target_slot]
        any_held_before = held_before.any(dim=1)
        overlap_now = (down_count > 1) | ((down_count > 0) & any_held_before)

        seek = self.phase == SEEK_PRESS
        release = self.phase == RELEASE_KEY
        clear = self.phase == CLEARANCE
        valid_target_down = seek & (down_count == 1) & expected_down & ~any_held_before
        invalid_down = (down_count > 0) & ~valid_target_down
        self.wrong_key |= invalid_down
        self.overlap |= overlap_now

        self.target_down_event |= valid_target_down
        self.pressed_slot = torch.where(valid_target_down, target_slot, self.pressed_slot)
        self._set_phase(valid_target_down, RELEASE_KEY)

        pressed = self.pressed_slot.clamp(min=0)
        expected_up = new_up[rows, pressed]
        non_target_down = new_down.clone()
        non_target_down[rows, pressed] = False
        release_invalid = release & non_target_down.any(dim=1)
        self.wrong_key |= release_invalid
        self.overlap |= release_invalid
        valid_target_up = release & expected_up & ~release_invalid
        self.target_up_event |= valid_target_up
        self._set_phase(valid_target_up, CLEARANCE)
        self._clear_ticks = torch.where(valid_target_up, torch.zeros_like(self._clear_ticks), self._clear_ticks)

        # Any down in clearance is a wrong key even when no previous key is
        # held. It must not reveal or advance the next target.
        self.wrong_key |= clear & (down_count > 0)

    def finalize_control_transition(self) -> None:
        """Consume the post-physics sample and one 25 Hz clearance tick."""

        control_step = int(getattr(self._env, "common_step_counter", 0))
        if self._last_global_finalized_control_step == control_step:
            return
        self._last_global_finalized_control_step = control_step
        self.sync_key_state()

        in_clearance = self.phase == CLEARANCE
        target_z = self.target_key_pos_w()[:, 2]
        tip_z = self.fixed_tip_pos_w()[:, 2]
        clear_now = in_clearance & ~self.held.any(dim=1) & (tip_z >= target_z + self.cfg.clearance_m)
        self._clear_ticks = torch.where(
            clear_now,
            self._clear_ticks + 1,
            torch.where(in_clearance, torch.zeros_like(self._clear_ticks), self._clear_ticks),
        )
        complete_clearance = in_clearance & (self._clear_ticks >= self.cfg.clearance_control_ticks)
        self.clearance_event |= complete_clearance
        rows = torch.arange(self.num_envs, device=self.device)
        write_col = self.character_index.clamp(max=self.max_len - 1)
        prior = self.typed[rows, write_col]
        self.typed[rows, write_col] = torch.where(complete_clearance, self.pressed_slot, prior)
        self.character_index = self.character_index + complete_clearance.long()
        self.typed_len.copy_(self.character_index)
        self.distance.copy_((self.target_len - self.character_index).clamp(min=0).float())
        final = complete_clearance & (self.character_index >= self.target_len)
        self.completed |= final
        next_character = complete_clearance & ~final
        self.pressed_slot = torch.where(next_character, torch.full_like(self.pressed_slot, -1), self.pressed_slot)
        self._set_phase(next_character, SEEK_PRESS)
        self._clear_ticks = torch.where(complete_clearance, torch.zeros_like(self._clear_ticks), self._clear_ticks)

    def _set_phase(self, mask: torch.Tensor, phase: int) -> None:
        self.phase = torch.where(mask, torch.full_like(self.phase, phase), self.phase)
        sim_step = int(getattr(self._env, "_sim_step_counter", 0))
        self._phase_start_sim_step = torch.where(
            mask, torch.full_like(self._phase_start_sim_step, sim_step), self._phase_start_sim_step
        )

    def _validate_map_once(self) -> None:
        if self._map_validation_complete:
            return
        pending = self._initialized & ~self._map_checked
        if not bool(pending.any()):
            return
        all_key_pos_w = self.key_pos_w()[:, self._letter_slots]
        n = self.num_envs * len(self._letter_slots)
        root_pos = self.robot.data.root_pos_w.torch[:, None, :].expand(-1, 26, -1).reshape(n, 3)
        root_quat = self.robot.data.root_quat_w.torch[:, None, :].expand(-1, 26, -1).reshape(n, 4)
        key_pos_b, _ = subtract_frame_transforms(root_pos, root_quat, all_key_pos_w.reshape(n, 3))
        key_pos_b = key_pos_b.reshape(self.num_envs, 26, 3)
        expected = self._letter_xyz_b[None, :, :]
        if self.cfg.public_pose_randomization:
            expected = self._env._public_expected_key_xyz_b
        error = torch.linalg.norm(key_pos_b - expected, dim=-1)
        max_error = error.max(dim=1).values
        self.map_max_error_m = torch.where(pending, max_error, self.map_max_error_m)
        invalid = pending & (max_error > self.cfg.map_tolerance_m)
        self.map_invalid |= invalid
        self._map_checked |= pending
        self._map_validation_complete = bool(self._map_checked.all())
        if bool(invalid.any()):
            invalid_env_ids = torch.nonzero(invalid, as_tuple=False).flatten()
            displayed_env_ids = invalid_env_ids[:8].detach().cpu().tolist()
            omitted_count = max(0, int(invalid_env_ids.numel()) - len(displayed_env_ids))
            omitted_suffix = f" (+{omitted_count} more)" if omitted_count else ""
            worst_error_m = float(max_error[invalid].max().detach().cpu())
            raise RuntimeError(
                "Frozen A-Z map validation failed after key-state initialization: "
                "live key centers exceed "
                f"map_tolerance_m={self.cfg.map_tolerance_m:.6f} in environment(s) "
                f"{displayed_env_ids}{omitted_suffix}; worst error={worst_error_m:.6f} m. "
                "Refusing to continue because resetting cannot repair a geometry/map mismatch."
            )

    def _update_command(self) -> None:
        self.finalize_control_transition()

    def _update_metrics(self) -> None:
        self.metrics["phase"] = self.phase.float()
        self.metrics["map_max_error_m"] = self.map_max_error_m
        self.metrics["held_key_count"] = self.held.float().sum(dim=1)
        self.metrics["wrong_key"] = self.wrong_key.float()
        self.metrics["overlap"] = self.overlap.float()
        first_complete = self.character_index >= 1
        has_second = self.target_len >= 2
        rows = torch.arange(self.num_envs, device=self.device)
        second_slot = self.target[rows, torch.ones_like(rows)].clamp(min=0)
        second_down = has_second & first_complete & (self.pressed_slot == second_slot)
        self.metrics["sequence/first_character_complete"].copy_(first_complete.float())
        self.metrics["sequence/second_character_reached"].copy_((has_second & first_complete).float())
        self.metrics["sequence/second_character_down"].copy_(second_down.float())
        self.metrics["sequence/second_character_complete"].copy_((has_second & (self.character_index >= 2)).float())
        self.metrics["sequence/full_complete"].copy_(self.completed.float())
        self.metrics["key/max_target_depression_fraction"] = self._episode_max_target_depression_fraction
        action = self._env.action_manager.get_term(self.cfg.action_name)
        # These are command-owned snapshots, not aliases. ActionManager resets
        # before CommandManager, so aliasing the action tensors would erase the
        # terminal episode values before CommandTerm.reset() logs them.
        self.metrics["action/mean_abs"].copy_(action.episode_action_mean_abs)
        self.metrics["action/max_abs"].copy_(action.episode_action_max_abs)
        self.metrics["action/near_saturation_fraction"].copy_(action.episode_near_saturation_fraction)
        self.metrics["action/near_saturation_any"].copy_(action.episode_near_saturation_any.float())
        self.metrics["action/out_of_bounds_any"].copy_(action.episode_out_of_bounds_any.float())

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)

        extras = super().reset(env_ids)
        self.typed[ids] = -1
        self.typed_len[ids] = 0
        self.character_index[ids] = 0
        self.prefix_len[ids] = 0
        self.distance[ids] = self.target_len[ids].float()
        self.max_prefix[ids] = 0
        self.min_prefix[ids] = 0
        self._start_distance[ids] = self.target_len[ids]
        self.phase[ids] = SEEK_PRESS
        self.pressed_slot[ids] = -1
        self.held[ids] = False
        self._down_count[ids] = 0
        self._up_count[ids] = 0
        self._initial_sample_count[ids] = 0
        self._initialized[ids] = False
        self._clear_ticks[ids] = 0
        self._phase_start_sim_step[ids] = int(getattr(self._env, "_sim_step_counter", 0))
        self.target_down_event[ids] = False
        self.target_up_event[ids] = False
        self.clearance_event[ids] = False
        self.wrong_key[ids] = False
        self.overlap[ids] = False
        self.initially_held[ids] = False
        self.completed[ids] = False
        self._episode_max_target_depression_fraction[ids] = 0.0
        self.reset_serial[ids] += 1
        if self.cfg.public_pose_randomization:
            self._map_validation_complete = False
        if not self._map_validation_complete:
            self._map_checked[ids] = False
            self.map_invalid[ids] = False
            self.map_max_error_m[ids] = 0.0
        return extras
