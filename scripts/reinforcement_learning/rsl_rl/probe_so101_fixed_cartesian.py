#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic contract probes for fixed-Cartesian SO-101 typing.

This script runs four separate checks:

1. A real two-second zero-action simulation rollout.  It verifies the exact
   22-D actor observation, persistent reset command, arm drift, key-event
   silence, map validity, absence of robot contact, and absence of resets.
2. A privileged *logical* switch-state probe.  It injects the sampled target
   key's joint state directly into the command's live keyboard-state tensor and
   exercises press -> release -> clearance.  This proves the state-machine
   contract, but intentionally does not claim jaw/key physical contact.
3. A bounded nonzero-action probe.  It sends one action through the real
   ``JointRatePositionAction`` term and verifies the persistent command
   integral, raw/bounded telemetry, near-bound mask, and absence of clipping.
4. A privileged physical-contact probe.  Position-only DLS is used offline to
   compute robot waypoints, after which the real bounded joint-rate action
   drives the fixed jaw through the robot/key collision model.  The probe never
   writes a keyboard joint.  It requires a true target down, matching up, and
   clearance with no neighboring-key, wrong-key, or overlap event.

Run from the repository root, for example::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/probe_so101_fixed_cartesian.py \
        --task Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P0 \
        --num-envs 1 --headless physics=newton_mjwarp
"""

from __future__ import annotations

from pathlib import Path

import argparse
import contextlib
import json
import sys
from typing import Any

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.math import quat_apply, quat_apply_inverse, skew_symmetric_matrix

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks.core.dexsuite.config.so101  # noqa: F401 -- register the typing tasks
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


DEFAULT_TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P0"
EXPECTED_OBSERVATION_DIM = 22
EXPECTED_ACTION_DIM = 5


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reset-audit-count", type=int, default=0)
parser.add_argument("--task-config", type=Path, help="Matching public task JSON")
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--zero-seconds", type=float, default=2.0)
parser.add_argument(
    "--max-zero-drift-rad",
    type=float,
    default=0.02,
    help="Maximum absolute per-joint physical drift from q_reset during the zero-action window.",
)
parser.add_argument(
    "--max-zero-contact-force-n",
    type=float,
    default=1.0e-3,
    help="Maximum robot contact-force norm allowed during the zero-action window.",
)
parser.add_argument("--seed", type=int, default=1307)
parser.add_argument(
    "--skip-logical-switch-probe",
    action="store_true",
    help="Skip the privileged logical key-joint-state probe.",
)
parser.add_argument(
    "--skip-action-contract-probe",
    action="store_true",
    help="Skip the real bounded nonzero joint-rate action probe.",
)
parser.add_argument(
    "--skip-physical-contact-probe",
    action="store_true",
    help="Skip the privileged IK-planned, action-driven physical jaw/key contact probe.",
)
parser.add_argument(
    "--contact-target-label",
    default="F",
    help="Single A-Z key used by the deterministic physical-contact probe.",
)
parser.add_argument(
    "--contact-hover-m",
    type=float,
    default=0.015,
    help="World-Z hover offset above the target key-body center.",
)
parser.add_argument(
    "--contact-press-depth-m",
    type=float,
    default=0.005,
    help="Privileged IK endpoint below the target key-body center; collision must move the key.",
)
parser.add_argument(
    "--contact-max-steps-per-waypoint",
    type=int,
    default=240,
    help="Maximum 25 Hz action steps allowed for each privileged joint waypoint.",
)
parser.add_argument(
    "--contact-max-force-n",
    type=float,
    default=10.0,
    help="Maximum robot contact-force norm accepted during the physical key probe.",
)
parser.add_argument(
    "--contact-descent-action-limit",
    type=float,
    default=0.20,
    help="Per-joint absolute action limit for the scripted descent near the key.",
)
parser.add_argument(
    "--contact-max-depression-fraction",
    type=float,
    default=1.30,
    help=(
        "Maximum accepted target-key travel fraction, including Newton's measured compliant "
        "prismatic-limit solve (about 1.24x for this fixed MX Keys contact)."
    ),
)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _max_robot_contact_force_n(base_env) -> float:
    sensor = base_env.scene["robot_contact"]
    force_history = sensor.data.net_forces_w_history
    if force_history is None:
        raise RuntimeError("robot_contact has no net_forces_w_history buffer")
    return float(force_history.torch.norm(dim=-1).max().item())


def _event_snapshot(command) -> dict[str, bool]:
    return {
        "target_down": bool(command.target_down_event.any().item()),
        "target_up": bool(command.target_up_event.any().item()),
        "clearance": bool(command.clearance_event.any().item()),
        "wrong_key": bool(command.wrong_key.any().item()),
        "overlap": bool(command.overlap.any().item()),
        "initially_held": bool(command.initially_held.any().item()),
        "completed": bool(command.completed.any().item()),
        "held": bool(command.held.any().item()),
    }


def _run_zero_action_probe(env: RslRlVecEnvWrapper, zero_steps: int, step_dt: float) -> dict[str, Any]:
    base_env = env.unwrapped
    command = base_env.command_manager.get_term("typing")
    action_term = base_env.action_manager.get_term("action")
    robot = base_env.scene[command.cfg.asset_name]

    observations = env.get_observations()
    actor_obs = observations["policy"]
    expected_shape = (env.num_envs, EXPECTED_OBSERVATION_DIM)
    _assert(
        tuple(actor_obs.shape) == expected_shape,
        f"expected actor observation {expected_shape}, got {tuple(actor_obs.shape)}",
    )
    _assert(bool(torch.isfinite(actor_obs).all().item()), "initial actor observation contains non-finite values")

    q_reset = command.q_reset_ref.clone()
    q_initial = robot.data.joint_pos.torch[:, command._arm_joint_ids].clone()
    q_command_reset_error = float((action_term.joint_command - q_reset).abs().max().item())
    _assert(q_command_reset_error <= 1.0e-7, f"q_command reset error is {q_command_reset_error:.9g} rad")

    zero_action = torch.zeros((env.num_envs, EXPECTED_ACTION_DIM), device=base_env.device)
    max_q_drift = 0.0
    max_q_command_drift = 0.0
    max_contact_force = 0.0
    max_reward_abs = 0.0
    observed_event = {key: False for key in _event_snapshot(command)}
    done_step: int | None = None

    with torch.inference_mode():
        for step in range(zero_steps):
            observations, reward, done, _ = env.step(zero_action)
            actor_obs = observations["policy"]
            _assert(
                tuple(actor_obs.shape) == expected_shape,
                f"step {step}: actor observation shape changed to {tuple(actor_obs.shape)}",
            )
            _assert(
                bool(torch.isfinite(actor_obs).all().item()),
                f"step {step}: actor observation contains non-finite values",
            )
            _assert(bool(torch.isfinite(reward).all().item()), f"step {step}: reward contains non-finite values")

            q_now = robot.data.joint_pos.torch[:, command._arm_joint_ids]
            max_q_drift = max(max_q_drift, float((q_now - q_reset).abs().max().item()))
            max_q_command_drift = max(
                max_q_command_drift,
                float((action_term.joint_command - q_reset).abs().max().item()),
            )
            max_contact_force = max(max_contact_force, _max_robot_contact_force_n(base_env))
            max_reward_abs = max(max_reward_abs, float(reward.abs().max().item()))
            for key, value in _event_snapshot(command).items():
                observed_event[key] |= value
            if bool(done.any().item()):
                done_step = step
                break

    _assert(done_step is None, f"zero-action rollout reset at step {done_step}")
    _assert(not any(observed_event.values()), f"zero-action rollout emitted a key/state event: {observed_event}")
    _assert(max_q_command_drift <= 1.0e-7, f"zero action changed q_command by {max_q_command_drift:.9g} rad")
    _assert(
        max_q_drift <= args_cli.max_zero_drift_rad,
        f"zero-action physical drift {max_q_drift:.9g} rad exceeds {args_cli.max_zero_drift_rad:.9g} rad",
    )
    _assert(
        max_contact_force <= args_cli.max_zero_contact_force_n,
        f"zero-action robot contact {max_contact_force:.9g} N exceeds {args_cli.max_zero_contact_force_n:.9g} N",
    )
    _assert(command._map_validation_complete, "frozen A-Z map validation did not complete")
    _assert(not bool(command.map_invalid.any().item()), "frozen A-Z map failed live-body validation")
    map_max_error_m = float(command.map_max_error_m.max().item())
    _assert(map_max_error_m <= command.cfg.map_tolerance_m, f"map error {map_max_error_m:.9g} m exceeds tolerance")

    q_final = robot.data.joint_pos.torch[:, command._arm_joint_ids]
    return {
        "kind": "real_zero_action_simulation",
        "duration_s": zero_steps * step_dt,
        "steps": zero_steps,
        "actor_observation_shape": list(actor_obs.shape),
        "actor_observation_finite": True,
        "q_command_reset_error_rad": q_command_reset_error,
        "max_q_command_drift_rad": max_q_command_drift,
        "max_physical_joint_drift_from_reset_rad": max_q_drift,
        "final_physical_joint_delta_from_initial_rad": float((q_final - q_initial).abs().max().item()),
        "max_robot_contact_force_n": max_contact_force,
        "map_max_error_m": map_max_error_m,
        "key_or_state_events": observed_event,
        "max_abs_reward": max_reward_abs,
        "status": "passed",
    }


@torch.inference_mode()
def _run_bounded_action_contract_probe(env: RslRlVecEnvWrapper, step_dt: float) -> dict[str, Any]:
    """Send a bounded nonzero action through the live action manager once."""

    env.reset()
    base_env = env.unwrapped
    command = base_env.command_manager.get_term("typing")
    action_term = base_env.action_manager.get_term("action")
    q_before = action_term.joint_command.clone()
    action = torch.tensor(
        ((0.25, -0.50, 0.95, -1.00, 0.50),),
        device=base_env.device,
        dtype=q_before.dtype,
    )
    limits = action_term._asset.data.soft_joint_pos_limits.torch[:, action_term._joint_ids]
    expected = action_term.integrate_command(
        q_before,
        action,
        action_term.velocity_limits_rad_s,
        step_dt,
        limits,
    )
    observations, reward, done, _ = env.step(action)
    q_after = action_term.joint_command.clone()
    expected_near_saturation = torch.abs(action) >= action_term.cfg.saturation_threshold

    _assert(not bool(done.any().item()), "bounded action-contract probe caused an episode reset")
    _assert(bool(torch.isfinite(observations["policy"]).all().item()), "nonzero action produced non-finite obs")
    _assert(bool(torch.isfinite(reward).all().item()), "nonzero action produced non-finite reward")
    _assert(torch.equal(action_term.raw_actions, action), "raw-action telemetry changed the bounded test action")
    _assert(torch.equal(action_term.bounded_actions, action), "in-range action was clipped or remapped")
    _assert(
        torch.equal(action_term.near_saturation, expected_near_saturation),
        "near-saturation mask does not match the configured threshold",
    )
    _assert(not bool(action_term.out_of_bounds.any().item()), "in-range action raised out_of_bounds")
    integration_error = float((q_after - expected).abs().max().item())
    _assert(integration_error <= 1.0e-7, f"live joint-rate integration error is {integration_error:.9g} rad")
    events = _event_snapshot(command)
    _assert(
        not any(events[key] for key in ("target_down", "target_up", "clearance", "wrong_key", "overlap")),
        f"one safe nonzero action emitted a key event: {events}",
    )

    return {
        "kind": "real_bounded_joint_rate_action",
        "action": action[0].tolist(),
        "bounded_action": action_term.bounded_actions[0].tolist(),
        "near_saturation": action_term.near_saturation[0].tolist(),
        "out_of_bounds": action_term.out_of_bounds[0].tolist(),
        "velocity_limits_rad_s": action_term.velocity_limits_rad_s[0].tolist(),
        "expected_q_command_delta_rad": (expected - q_before)[0].tolist(),
        "actual_q_command_delta_rad": (q_after - q_before)[0].tolist(),
        "max_integration_error_rad": integration_error,
        "key_or_state_events": events,
        "status": "passed",
    }


def _fixed_tip_position_jacobian(command) -> torch.Tensor:
    """Return the privileged world-frame fixed-tip position Jacobian."""

    robot = command.robot
    jacobian_body_index = command._tip_body_idx - 1 if robot.is_fixed_base else command._tip_body_idx
    jacobian_joint_ids = [joint_id + robot.num_base_dofs for joint_id in command._arm_joint_ids]
    body_quat = robot.data.body_quat_w.torch[:, command._tip_body_idx]
    lever_w = quat_apply(body_quat, command._tip_offset)
    jacobian = robot.data.body_link_jacobian_w.torch[:, jacobian_body_index, :, jacobian_joint_ids].clone()
    return jacobian[:, :3, :] + torch.bmm(-skew_symmetric_matrix(lever_w), jacobian[:, 3:, :])


@torch.inference_mode()
def _plan_privileged_position_waypoints(
    command,
    target_positions_w: list[torch.Tensor],
    *,
    iterations: int = 96,
    damping: float = 0.03,
    max_joint_step_rad: float = 0.10,
) -> tuple[list[torch.Tensor], list[float]]:
    """Offline position-only DLS; robot writes are restored before dynamics resume.

    This function may pose the *robot* while computing waypoints.  It never
    writes a keyboard position, velocity, target, or state-machine buffer.
    Physical acceptance is based only on the later action-driven rollout.
    """

    robot = command.robot
    sim = command._env.sim
    env_ids = torch.arange(command.num_envs, device=command.device)
    saved_q = robot.data.joint_pos.torch.clone()
    saved_qd = robot.data.joint_vel.torch.clone()
    goals: list[torch.Tensor] = []
    residuals: list[float] = []
    eye = torch.eye(3, device=command.device).expand(command.num_envs, -1, -1)

    try:
        for target_w in target_positions_w:
            for _ in range(iterations):
                error = target_w - command.fixed_tip_pos_w()
                jacobian = _fixed_tip_position_jacobian(command)
                jacobian_t = jacobian.transpose(1, 2)
                delta = jacobian_t @ torch.linalg.solve(
                    jacobian @ jacobian_t + damping**2 * eye,
                    error.unsqueeze(-1),
                )
                q_arm = robot.data.joint_pos.torch[:, command._arm_joint_ids]
                q_next = q_arm + torch.clamp(delta.squeeze(-1), -max_joint_step_rad, max_joint_step_rad)
                limits = robot.data.soft_joint_pos_limits.torch[:, command._arm_joint_ids]
                q_next = torch.clamp(q_next, limits[..., 0], limits[..., 1])
                robot.write_joint_position_to_sim_index(
                    position=q_next,
                    joint_ids=command._arm_joint_ids,
                    env_ids=env_ids,
                )
                sim.forward()
            residual = float(torch.linalg.vector_norm(target_w - command.fixed_tip_pos_w(), dim=-1).max().item())
            goals.append(robot.data.joint_pos.torch[:, command._arm_joint_ids].clone())
            residuals.append(residual)
    finally:
        robot.write_joint_position_to_sim_index(position=saved_q, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=saved_qd, env_ids=env_ids)
        robot.set_joint_position_target_index(target=saved_q, env_ids=env_ids)
        sim.forward()

    return goals, residuals


def _all_key_joint_pos(command) -> torch.Tensor:
    return command.keyboard.data.joint_pos.torch[command._inst_idx, command._col_idx]


def _record_physical_step(command, action_term, base_env, trace: dict[str, Any]) -> dict[str, bool]:
    events = _event_snapshot(command)
    key_pos = _all_key_joint_pos(command)
    lower = command.keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 0]
    upper = command.keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 1]
    depression = (upper - key_pos) / (upper - lower).clamp_min(1.0e-9)
    target_slot = trace["target_slot"]
    non_target = depression.clone()
    non_target[:, target_slot] = -torch.inf

    trace["action_steps"] += 1
    trace["near_saturation_elements"] += int(action_term.near_saturation.sum().item())
    trace["action_elements"] += int(action_term.near_saturation.numel())
    trace["max_abs_bounded_action"] = max(
        trace["max_abs_bounded_action"], float(action_term.bounded_actions.abs().max().item())
    )
    trace["max_robot_contact_force_n"] = max(trace["max_robot_contact_force_n"], _max_robot_contact_force_n(base_env))
    trace["max_target_depression_fraction"] = max(
        trace["max_target_depression_fraction"], float(depression[0, target_slot].item())
    )
    trace["max_non_target_depression_fraction"] = max(
        trace["max_non_target_depression_fraction"], float(non_target.max().item())
    )
    trace["minimum_target_joint_position_m"] = min(
        trace["minimum_target_joint_position_m"], float(key_pos[0, target_slot].item())
    )
    tip_delta = command.fixed_tip_pos_w() - command.target_key_pos_w()
    trace["minimum_tip_key_xy_error_m"] = min(
        trace["minimum_tip_key_xy_error_m"], float(torch.linalg.vector_norm(tip_delta[:, :2], dim=-1).item())
    )
    for name in ("target_down", "target_up", "clearance"):
        if events[name] and name not in trace["event_sequence"]:
            trace["event_sequence"].append(name)
            trace[f"{name}_step"] = trace["action_steps"]
    if events["target_down"]:
        tip_delta_w = command.fixed_tip_pos_w() - command.pressed_key_pos_w()
        trace["target_down_tip_delta_key_frame_m"] = quat_apply_inverse(command.pressed_key_quat_w(), tip_delta_w)[
            0
        ].tolist()
    trace["wrong_key_observed"] |= events["wrong_key"]
    trace["overlap_observed"] |= events["overlap"]
    trace["initially_held_observed"] |= events["initially_held"]
    return events


@torch.inference_mode()
def _run_physical_contact_probe(env: RslRlVecEnvWrapper, step_dt: float) -> dict[str, Any]:
    """Drive an actual jaw/key collision using only bounded runtime actions."""

    env.reset()
    base_env = env.unwrapped
    command = base_env.command_manager.get_term("typing")
    action_term = base_env.action_manager.get_term("action")
    _assert(env.num_envs == 1, "physical contact probe currently requires --num-envs 1")

    target_label = args_cli.contact_target_label.strip().upper()
    _assert(len(target_label) == 1 and target_label.isalpha(), "--contact-target-label must be one A-Z letter")
    labels = tuple(str(label).upper() for label in command.cfg.slot_labels)
    _assert(target_label in labels, f"target label {target_label!r} is not in the keyboard slot map")
    target_slot = labels.index(target_label)
    _assert(target_slot in command.cfg.letter_slots, f"target label {target_label!r} is not in frozen A-Z")

    # This is privileged deterministic target selection, not a key-state
    # mutation.  All keyboard joints remain owned by the physics engine.
    command.target[0].fill_(-1)
    command.target[0, :2] = target_slot
    command.target_len[0] = 2
    command.distance[0] = 2.0

    zero = torch.zeros((1, EXPECTED_ACTION_DIM), device=base_env.device)
    for init_step in range(int(command.cfg.stable_samples)):
        _, _, done, _ = env.step(zero)
        _assert(not bool(done.any().item()), f"key-state initialization reset at step {init_step}")
    _assert(bool(command._initialized[0].item()), "physical probe key state did not initialize")
    _assert(not bool(command.held.any().item()), "physical probe started with a held key")
    _assert(not bool(command.initially_held.any().item()), "physical probe detected an initially held key")

    target_center_w = command.target_key_pos_w().clone()
    hover = float(args_cli.contact_hover_m)
    press_depth = float(args_cli.contact_press_depth_m)
    vertical_offsets = (hover, 0.010, 0.006, 0.003, 0.001, -0.001, -0.003, -press_depth)
    target_positions = [
        target_center_w + torch.tensor(((0.0, 0.0, offset),), device=base_env.device) for offset in vertical_offsets
    ]
    joint_goals, ik_residuals = _plan_privileged_position_waypoints(command, target_positions)
    _assert(max(ik_residuals) <= 0.001, f"privileged contact IK residual is {max(ik_residuals):.9g} m")

    # Let the restored robot/contact caches settle before the trace begins.
    for settle_step in range(2):
        _, _, done, _ = env.step(zero)
        _assert(not bool(done.any().item()), f"post-IK restore reset at step {settle_step}")
    _assert(not any(_event_snapshot(command).values()), "offline IK restore changed live key state")

    initial_key_pos = float(_all_key_joint_pos(command)[0, target_slot].item())
    trace: dict[str, Any] = {
        "target_slot": target_slot,
        "action_steps": 0,
        "action_elements": 0,
        "near_saturation_elements": 0,
        "max_abs_bounded_action": 0.0,
        "max_robot_contact_force_n": 0.0,
        "max_target_depression_fraction": 0.0,
        "max_non_target_depression_fraction": 0.0,
        "minimum_target_joint_position_m": initial_key_pos,
        "minimum_tip_key_xy_error_m": float("inf"),
        "event_sequence": [],
        "wrong_key_observed": False,
        "overlap_observed": False,
        "initially_held_observed": False,
    }

    def drive_goal(
        q_goal: torch.Tensor,
        *,
        stop_on: tuple[str, ...] = (),
        max_abs_action: float = 1.0,
    ) -> str:
        settled = 0
        for _ in range(args_cli.contact_max_steps_per_waypoint):
            q_error = q_goal - action_term.joint_command
            action = torch.clamp(
                q_error / (action_term.velocity_limits_rad_s * step_dt),
                -max_abs_action,
                max_abs_action,
            )
            observations, reward, done, _ = env.step(action)
            _assert(bool(torch.isfinite(observations["policy"]).all().item()), "contact trace produced non-finite obs")
            _assert(bool(torch.isfinite(reward).all().item()), "contact trace produced non-finite reward")
            events = _record_physical_step(command, action_term, base_env, trace)
            _assert(not trace["wrong_key_observed"], "physical contact trace emitted wrong_key")
            _assert(not trace["overlap_observed"], "physical contact trace emitted overlap")
            _assert(not trace["initially_held_observed"], "physical contact trace emitted initially_held")
            _assert(
                trace["max_robot_contact_force_n"] <= args_cli.contact_max_force_n,
                f"physical contact force reached {trace['max_robot_contact_force_n']:.9g} N",
            )
            _assert(not bool(done.any().item()), "physical contact trace terminated before probe inspection")
            for event_name in stop_on:
                if events[event_name]:
                    return event_name
            measured_q = command.robot.data.joint_pos.torch[:, command._arm_joint_ids]
            command_close = float((q_goal - action_term.joint_command).abs().max().item()) <= 1.0e-6
            measured_close = float((q_goal - measured_q).abs().max().item()) <= 0.02
            settled = settled + 1 if command_close and measured_close else 0
            if settled >= 3:
                return "waypoint"
        return "timeout"

    # First reach the collision-free hover.  This is actual bounded action
    # stepping, not the direct robot writes used by the offline planner.
    _assert(drive_goal(joint_goals[0]) == "waypoint", "bounded action did not converge to contact hover")

    down_observed = False
    for index, q_goal in enumerate(joint_goals[1:], start=1):
        outcome = drive_goal(
            q_goal,
            stop_on=("target_down",),
            max_abs_action=args_cli.contact_descent_action_limit,
        )
        if outcome == "target_down":
            down_observed = True
            break
        _assert(outcome == "waypoint", f"descent waypoint {index} did not converge: {outcome}")
    _assert(down_observed, "actual jaw/key collision did not produce target_down")

    up_observed = False
    clearance_observed = False
    for q_goal in reversed(joint_goals[:-1]):
        outcome = drive_goal(q_goal, stop_on=("target_up", "clearance"))
        _assert(outcome in ("waypoint", "target_up", "clearance"), f"lift waypoint did not converge: {outcome}")
        up_observed |= "target_up" in trace["event_sequence"]
        clearance_observed |= "clearance" in trace["event_sequence"]
        if clearance_observed:
            break
    if up_observed and not clearance_observed:
        # Hold the collision-free hover long enough for the two-tick clearance dwell.
        for _ in range(command.cfg.clearance_control_ticks + 2):
            env.step(zero)
            events = _record_physical_step(command, action_term, base_env, trace)
            if events["clearance"]:
                clearance_observed = True
                break

    final_key_pos = float(_all_key_joint_pos(command)[0, target_slot].item())
    key_face_xy_m = tuple(float(value) for value in base_env.cfg.physical_alphabet_key_face_xy_m)
    down_tip_delta_key = trace["target_down_tip_delta_key_frame_m"]
    _assert(up_observed, "physical target key did not emit matching target_up")
    _assert(clearance_observed, "physical trace did not satisfy the clearance dwell")
    _assert(
        trace["event_sequence"] == ["target_down", "target_up", "clearance"],
        f"bad event order: {trace['event_sequence']}",
    )
    _assert(int(command.character_index[0].item()) == 1, "first press/release/clearance did not advance one character")
    _assert(not bool(command.completed[0].item()), "probe unexpectedly reached the terminal second character")
    _assert(
        trace["max_target_depression_fraction"] >= command.cfg.key_down_fraction,
        "target never crossed down threshold",
    )
    _assert(
        trace["max_target_depression_fraction"] <= args_cli.contact_max_depression_fraction,
        "target key exceeded the accepted travel-compliance bound: "
        f"{trace['max_target_depression_fraction']:.6f} > {args_cli.contact_max_depression_fraction:.6f}",
    )
    _assert(
        trace["max_non_target_depression_fraction"] < command.cfg.key_down_fraction,
        "a non-target key crossed the down threshold",
    )
    _assert(trace["max_robot_contact_force_n"] > 1.0e-5, "no robot contact force was observed")
    _assert(
        abs(down_tip_delta_key[0]) <= 0.5 * key_face_xy_m[0] and abs(down_tip_delta_key[1]) <= 0.5 * key_face_xy_m[1],
        f"target-down tip was outside the measured key face: local delta={down_tip_delta_key}",
    )

    return {
        "kind": "privileged_ik_planned_real_jaw_key_collision",
        "physical_contact_claim": True,
        "keyboard_joint_buffer_mutated_by_probe": False,
        "target_label": target_label,
        "target_slot": target_slot,
        "key_face_xy_m": list(key_face_xy_m),
        "target_down_tip_delta_key_frame_m": down_tip_delta_key,
        "control_dt_s": step_dt,
        "descent_action_limit": args_cli.contact_descent_action_limit,
        "max_accepted_depression_fraction": args_cli.contact_max_depression_fraction,
        "vertical_waypoint_offsets_m": list(vertical_offsets),
        "privileged_ik_residuals_m": ik_residuals,
        "action_steps": trace["action_steps"],
        "near_saturation_fraction": trace["near_saturation_elements"] / max(trace["action_elements"], 1),
        "max_abs_bounded_action": trace["max_abs_bounded_action"],
        "initial_target_joint_position_m": initial_key_pos,
        "minimum_target_joint_position_m": trace["minimum_target_joint_position_m"],
        "final_target_joint_position_m": final_key_pos,
        "max_target_depression_fraction": trace["max_target_depression_fraction"],
        "max_non_target_depression_fraction": trace["max_non_target_depression_fraction"],
        "minimum_tip_key_xy_error_m": trace["minimum_tip_key_xy_error_m"],
        "max_robot_contact_force_n": trace["max_robot_contact_force_n"],
        "event_sequence": trace["event_sequence"],
        "event_steps": {name: trace[f"{name}_step"] for name in trace["event_sequence"]},
        "wrong_key_observed": trace["wrong_key_observed"],
        "overlap_observed": trace["overlap_observed"],
        "status": "passed",
    }


@torch.inference_mode()
def _run_logical_switch_probe(env: RslRlVecEnvWrapper) -> dict[str, Any]:
    """Exercise command state transitions without claiming physical contact."""

    base_env = env.unwrapped
    command = base_env.command_manager.get_term("typing")
    _assert(env.num_envs == 1, "logical switch probe currently requires --num-envs 1")
    _assert(bool(command._initialized.all().item()), "key state was not initialized by zero-action probe")
    _assert(not any(_event_snapshot(command).values()), "logical switch probe did not start from a clean state")

    target_slot = int(command.target_key_slot()[0].item())
    target_label = str(command.cfg.slot_labels[target_slot])
    instance = int(command._inst_idx[0, target_slot].item())
    column = int(command._col_idx[0, target_slot].item())
    limits = command.keyboard.data.joint_pos_limits.torch[instance, column]
    down_position = limits[0].clone()
    up_position = limits[1].clone()
    live_joint_pos = command.keyboard.data.joint_pos.torch
    original_position = live_joint_pos[instance, column].clone()

    try:
        command.begin_control_transition()
        live_joint_pos[instance, column] = down_position
        command.sync_key_state()
        _assert(int(command.phase[0].item()) == 0, "single down sample bypassed the two-sample dwell")
        command.sync_key_state()
        _assert(bool(command.target_down_event[0].item()), "target-down event did not fire after two samples")
        _assert(int(command.phase[0].item()) == 1, "target-down did not enter release phase")
        _assert(int(command.pressed_slot[0].item()) == target_slot, "pressed slot did not latch the target")

        command.begin_control_transition()
        live_joint_pos[instance, column] = up_position
        command.sync_key_state()
        _assert(int(command.phase[0].item()) == 1, "single up sample bypassed the two-sample dwell")
        command.sync_key_state()
        _assert(bool(command.target_up_event[0].item()), "target-up event did not fire after two samples")
        _assert(int(command.phase[0].item()) == 2, "target-up did not enter clearance phase")

        clearance_margin_m = float(
            (command.fixed_tip_pos_w()[0, 2] - command.target_key_pos_w()[0, 2] - command.cfg.clearance_m).item()
        )
        _assert(
            clearance_margin_m >= 0.0,
            f"reset tip is {clearance_margin_m:.9g} m below the required logical clearance height",
        )

        command.begin_control_transition()
        base_env.common_step_counter += 1
        command.finalize_control_transition()
        _assert(int(command.phase[0].item()) == 2, "single clearance tick bypassed the dwell")
        _assert(not bool(command.completed[0].item()), "single clearance tick completed the character")

        command.begin_control_transition()
        base_env.common_step_counter += 1
        command.finalize_control_transition()
        _assert(bool(command.clearance_event[0].item()), "clearance event did not fire after two ticks")
        _assert(bool(command.completed[0].item()), "press-release-clearance did not complete the character")
        _assert(not bool(command.wrong_key[0].item()), "logical target sequence raised wrong_key")
        _assert(not bool(command.overlap[0].item()), "logical target sequence raised overlap")
    finally:
        live_joint_pos[instance, column] = original_position

    return {
        "kind": "privileged_key_joint_state_injection",
        "physical_contact_claim": False,
        "target_label": target_label,
        "target_slot": target_slot,
        "stable_samples": int(command.cfg.stable_samples),
        "clearance_control_ticks": int(command.cfg.clearance_control_ticks),
        "clearance_margin_m": clearance_margin_m,
        "sequence": ["seek_press", "release_key", "clearance", "complete"],
        "status": "passed",
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    del agent_cfg
    requires_single_env = not (
        args_cli.skip_logical_switch_probe
        and args_cli.skip_action_contract_probe
        and args_cli.skip_physical_contact_probe
    )
    if args_cli.num_envs != 1 and requires_single_env:
        raise ValueError("logical, action-contract, and physical-contact probes require --num-envs 1")
    if args_cli.zero_seconds <= 0.0:
        raise ValueError("--zero-seconds must be positive")
    if args_cli.max_zero_drift_rad < 0.0 or args_cli.max_zero_contact_force_n < 0.0:
        raise ValueError("probe tolerances must be non-negative")
    if args_cli.contact_hover_m <= 0.0 or args_cli.contact_press_depth_m <= 0.0:
        raise ValueError("contact hover and press depth must be positive")
    if args_cli.contact_max_steps_per_waypoint < 1 or args_cli.contact_max_force_n <= 0.0:
        raise ValueError("contact step and force limits must be positive")
    if not 0.0 < args_cli.contact_descent_action_limit <= 1.0:
        raise ValueError("--contact-descent-action-limit must be in (0, 1]")
    if args_cli.contact_max_depression_fraction < 0.5:
        raise ValueError("--contact-max-depression-fraction must be at least the 0.5 down threshold")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.task_config:
        from public_task import apply_config, load_config
        apply_config(env_cfg, load_config(args_cli.task_config))
    with launch_simulation(env_cfg, args_cli):
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env = RslRlVecEnvWrapper(gym_env, clip_actions=None)
            env.reset()
            step_dt = float(env.unwrapped.step_dt)
            zero_steps = round(args_cli.zero_seconds / step_dt)
            _assert(zero_steps >= 1, "zero-action duration is shorter than one control step")
            _assert(
                abs(zero_steps * step_dt - args_cli.zero_seconds) <= 1.0e-9,
                f"--zero-seconds must be an exact multiple of control dt {step_dt:.9g}s",
            )

            result: dict[str, Any] = {
                "task": args_cli.task,
                "seed": args_cli.seed,
                "num_envs": args_cli.num_envs,
                "control_dt_s": step_dt,
                "zero_action": _run_zero_action_probe(env, zero_steps, step_dt),
            }
            if not args_cli.skip_logical_switch_probe:
                result["scripted_switch"] = _run_logical_switch_probe(env)
            if not args_cli.skip_action_contract_probe:
                result["bounded_action_contract"] = _run_bounded_action_contract_probe(env, step_dt)
            if not args_cli.skip_physical_contact_probe:
                result["physical_contact"] = _run_physical_contact_probe(env, step_dt)
            if args_cli.reset_audit_count:
                command = env.unwrapped.command_manager.get_term("typing")
                nominal = command._letter_xyz_b.clone()
                audits = []
                for _ in range(args_cli.reset_audit_count):
                    with torch.inference_mode():
                        env.reset()
                    reset_result = _run_zero_action_probe(env, zero_steps, step_dt)
                    _assert(torch.equal(command._letter_xyz_b, nominal), "randomized pose leaked into actor target map")
                    audits.append({"sample_xy_yaw_rad": env.unwrapped._public_keyboard_pose_samples.tolist(),
                                   "zero_action": reset_result})
                result["reset_audit"] = audits
            result["status"] = "passed"
            print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            gym_env.close()


if __name__ == "__main__":
    main()
