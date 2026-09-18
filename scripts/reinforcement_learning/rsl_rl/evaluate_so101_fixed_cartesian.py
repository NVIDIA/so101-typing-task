#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict evaluation for the dated fixed-Cartesian SO-101 typing policies."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks.core.dexsuite.config.so101  # noqa: F401 -- register the typing tasks
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


P0_TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P0"
P1A_TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1A"
BASELINE_P1A_TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-Baseline-P1A"
USD_DRIVE_P1A_TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-USDDrive-P1A"
P1B_TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1B"
P1C_TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1C"
P1D_TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1D"
P1D_TRANSIT15_TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1D-Transit15"
BASELINE_P1D_TRANSIT15_TASK = (
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-Baseline-P1D-Transit15"
)
USD_DRIVE_P1D_TRANSIT15_TASK = (
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-USDDrive-P1D-Transit15"
)


@dataclass(frozen=True)
class TaskContract:
    target_length: int
    task_contract: str
    actuator_profile: str = "anchorbench"


TASK_CONTRACTS = {
    P0_TASK: TaskContract(1, "physical_calibrated_20260718_fixed_cartesian_anchorbench_p0_v0"),
    P1A_TASK: TaskContract(2, "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1a_letters2_v0"),
    BASELINE_P1A_TASK: TaskContract(
        2,
        "physical_calibrated_20260718_fixed_cartesian_baseline_p1a_letters2_v0",
        actuator_profile="baseline",
    ),
    USD_DRIVE_P1A_TASK: TaskContract(
        2,
        "physical_calibrated_20260718_fixed_cartesian_usd_drive_p1a_letters2_v0",
        actuator_profile="usd_drive",
    ),
    P1B_TASK: TaskContract(3, "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1b_letters3_v0"),
    P1C_TASK: TaskContract(4, "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1c_letters4_v0"),
    P1D_TASK: TaskContract(6, "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1d_letters6_v0"),
    P1D_TRANSIT15_TASK: TaskContract(
        6, "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1d_transit15_letters6_v0"
    ),
    BASELINE_P1D_TRANSIT15_TASK: TaskContract(
        6,
        "physical_calibrated_20260718_fixed_cartesian_baseline_p1d_transit15_letters6_v0",
        actuator_profile="baseline",
    ),
    USD_DRIVE_P1D_TRANSIT15_TASK: TaskContract(
        6,
        "physical_calibrated_20260718_fixed_cartesian_usd_drive_p1d_transit15_letters6_v0",
        actuator_profile="usd_drive",
    ),
}
FAILURE_TERMS = (
    "time_out",
    "abnormal_robot",
    "excessive_contact",
    "invalid_key",
    "map_invalid",
    "action_contract",
    "phase_timeout",
    "scrape",
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics-config", type=Path, help="Matching public physics JSON from training params")
parser.add_argument("--task-config", type=Path, help="Matching public task JSON")
parser.add_argument("--perturbation-config", type=Path, help="Explicit fixed-policy reset robustness experiment")
parser.add_argument("--task", choices=tuple(TASK_CONTRACTS), required=True)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=0, help="Control-step budget; zero selects the native episode budget.")
parser.add_argument("--seed", type=int, default=1307)
parser.add_argument(
    "--target",
    default=None,
    help="Repeat one exact A-Z target sequence instead of using a seeded bank.",
)
parser.add_argument("--env-config", type=Path, required=True, help="The params/env.yaml exported with the checkpoint.")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--trace-policy-actions", type=int, default=0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-length", type=int, default=0)
parser.add_argument("--step-sleep", type=float, default=0.0)
parser.add_argument("--viewer-hold-seconds", type=float, default=0.0)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + remaining_args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slot_for_label(labels: tuple[str, ...], label: str) -> int:
    matches = [index for index, candidate in enumerate(labels) if candidate.casefold() == label.casefold()]
    if len(matches) != 1:
        raise ValueError(f"key label {label!r} maps to {matches}, expected exactly one slot")
    return matches[0]


def _validate_artifact_contract(env_cfg: ManagerBasedRLEnvCfg) -> dict[str, object]:
    env_config = args_cli.env_config.expanduser().resolve()
    if not env_config.is_file():
        raise FileNotFoundError(f"environment config not found: {env_config}")
    contract = TASK_CONTRACTS[args_cli.task]
    # Isaac Lab config snapshots contain Python-specific YAML tags. BaseLoader
    # treats those values as plain data without constructing Python objects.
    artifact = yaml.load(env_config.read_text(), Loader=yaml.BaseLoader)
    expected = {
        "actuator_profile": contract.actuator_profile,
        "keyboard_profile": "mx_keys_powered_az_20260718",
        "task_contract": env_cfg.task_contract,
        "target_reference_sha256": str(env_cfg.target_reference_sha256),
        "target_manifest_sha256": str(env_cfg.target_manifest_sha256),
    }
    mismatches = {
        key: {"expected": value, "actual": artifact.get(key)}
        for key, value in expected.items()
        if artifact.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{env_config} does not match {args_cli.task}: {mismatches}")
    if env_cfg.commands.typing.letter_length != (contract.target_length, contract.target_length):
        raise ValueError("runtime target length does not match the dated task contract")
    return {
        "path": str(env_config),
        "sha256": _sha256(env_config),
        "task_contract": contract.task_contract,
        "validated": True,
    }


def _configure_protocol(env_cfg: ManagerBasedRLEnvCfg) -> tuple[int, int]:
    target_length = TASK_CONTRACTS[args_cli.task].target_length
    if args_cli.num_envs <= 0 or args_cli.steps < 0:
        raise ValueError("--num-envs must be positive and --steps must be nonnegative")
    if args_cli.trace_policy_actions < 0 or args_cli.step_sleep < 0.0 or args_cli.viewer_hold_seconds < 0.0:
        raise ValueError("trace length and wall-clock delays must be nonnegative")
    if args_cli.video and args_cli.num_envs != 1:
        raise ValueError("--video requires --num-envs=1")

    command = env_cfg.commands.typing
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.reset.enabled = False
    command.reset.ik = None
    command.reset.pre_solve_reset = None
    if not args_cli.task_config:
        env_cfg.events.reset_robot_rest.params["position_range"] = (0.0, 0.0)
        env_cfg.events.reset_robot_rest.params["velocity_range"] = (0.0, 0.0)
        env_cfg.events.reset_keyboard.params["pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
    native_steps = round(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    return target_length, args_cli.steps or native_steps


def _build_target_bank(command, target_length: int) -> tuple[torch.Tensor, str, dict[str, int], dict[str, object]]:
    labels = tuple(str(label) for label in command.cfg.slot_labels)
    candidates = tuple(int(slot) for slot in command.cfg.target_slots)
    if len(candidates) != 26:
        raise ValueError(f"expected 26 A-Z candidates, got {len(candidates)}")

    if args_cli.target is not None:
        target = args_cli.target.strip().upper()
        if len(target) != target_length or not target.isascii() or not target.isalpha():
            raise ValueError(f"{args_cli.task} requires exactly {target_length} A-Z letters, got {target!r}")
        slots = [_slot_for_label(labels, letter) for letter in target]
        if any(slot not in candidates for slot in slots):
            raise ValueError(f"target {target!r} contains a key outside the configured A-Z set")
        bank = torch.tensor(slots, dtype=torch.long).expand(command.num_envs, -1).clone()
        descriptor: dict[str, object] = {"mode": "repeated_sequence", "target": target}
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args_cli.seed)
        choices = torch.randint(len(candidates), (command.num_envs, target_length), generator=generator)
        bank = torch.tensor(candidates, dtype=torch.long)[choices]
        descriptor = {"mode": "seeded_uniform_AZ", "target": None}

    sequences = ["".join(labels[int(slot)] for slot in row.tolist()) for row in bank]
    bank_bytes = bank.contiguous().numpy().tobytes()
    return bank.to(command.device), hashlib.sha256(bank_bytes).hexdigest(), dict(Counter(sequences)), descriptor


def _install_target_bank(command, bank: torch.Tensor, target_length: int) -> None:
    ids = torch.arange(command.num_envs, device=command.device)
    command.target[ids] = -1
    command.target[ids, :target_length] = bank
    command.target_len[ids] = target_length
    command.typed[ids] = -1
    command.typed_len[ids] = 0
    command.character_index[ids] = 0
    command.prefix_len[ids] = 0
    command.distance[ids] = float(target_length)
    command.max_prefix[ids] = 0
    command.min_prefix[ids] = 0
    command._start_distance[ids] = target_length
    command._prev_pressed[ids] = False
    command._just_reset[ids] = True
    command.phase[ids] = 0
    command.pressed_slot[ids] = -1
    command.held[ids] = False
    command._down_count[ids] = 0
    command._up_count[ids] = 0
    command._initial_sample_count[ids] = 0
    command._initialized[ids] = False
    command._clear_ticks[ids] = 0
    command._phase_start_sim_step[ids] = int(getattr(command._env, "_sim_step_counter", 0))
    command.target_down_event[ids] = False
    command.target_up_event[ids] = False
    command.clearance_event[ids] = False
    command.wrong_key[ids] = False
    command.overlap[ids] = False
    command.initially_held[ids] = False
    command.completed[ids] = False
    command._episode_max_target_depression_fraction[ids] = 0.0


def _typed_exact(command, ids: torch.Tensor) -> torch.Tensor:
    columns = torch.arange(command.max_len, device=command.device).unsqueeze(0)
    valid = columns < command.target_len[ids].unsqueeze(1)
    return ((command.typed[ids] == command.target[ids]) | ~valid).all(dim=1)


def _new_stats(command) -> dict[str, object]:
    n = command.num_envs

    def bool_tensor() -> torch.Tensor:
        return torch.zeros(n, dtype=torch.bool, device=command.device)

    return {
        "pending": torch.ones(n, dtype=torch.bool, device=command.device),
        "terminal_completed": bool_tensor(),
        "terminal_typed_exact": bool_tensor(),
        "terminal_wrong_key": bool_tensor(),
        "terminal_overlap": bool_tensor(),
        "terminal_initially_held": bool_tensor(),
        "termination": {name: bool_tensor() for name in FAILURE_TERMS},
        "target_down_count": torch.zeros(n, dtype=torch.long, device=command.device),
        "target_up_count": torch.zeros(n, dtype=torch.long, device=command.device),
        "clearance_count": torch.zeros(n, dtype=torch.long, device=command.device),
        "character_index": torch.zeros(n, dtype=torch.long, device=command.device),
        "action_mean_abs": torch.zeros(n, device=command.device),
        "action_max_abs": torch.zeros(n, device=command.device),
        "action_near_saturation_fraction": torch.zeros(n, device=command.device),
        "action_near_saturation_any": bool_tensor(),
        "action_out_of_bounds_any": bool_tensor(),
        "max_target_depression_fraction": torch.zeros(n, device=command.device),
    }


def _accumulate_events(command, stats: dict[str, object], ids: torch.Tensor) -> None:
    stats["target_down_count"][ids] += command.target_down_event[ids].long()
    stats["target_up_count"][ids] += command.target_up_event[ids].long()
    stats["clearance_count"][ids] += command.clearance_event[ids].long()


def _capture(command, env, stats: dict[str, object], env_ids, *, terminating: bool) -> None:
    if env_ids is None or isinstance(env_ids, slice):
        ids = torch.arange(command.num_envs, device=command.device)
    else:
        ids = torch.as_tensor(env_ids, device=command.device)
    ids = ids[stats["pending"][ids]]
    if ids.numel() == 0:
        return
    if terminating:
        _accumulate_events(command, stats, ids)
    stats["terminal_completed"][ids] = command.completed[ids]
    stats["terminal_typed_exact"][ids] = _typed_exact(command, ids)
    stats["terminal_wrong_key"][ids] = command.wrong_key[ids]
    stats["terminal_overlap"][ids] = command.overlap[ids]
    stats["terminal_initially_held"][ids] = command.initially_held[ids]
    stats["character_index"][ids] = command.character_index[ids]
    manager = env.unwrapped.termination_manager
    for name in FAILURE_TERMS:
        stats["termination"][name][ids] = manager.get_term(name)[ids]

    if terminating:
        stats["action_mean_abs"][ids] = command.metrics["action/mean_abs"][ids]
        stats["action_max_abs"][ids] = command.metrics["action/max_abs"][ids]
        stats["action_near_saturation_fraction"][ids] = command.metrics["action/near_saturation_fraction"][ids]
        stats["action_near_saturation_any"][ids] = command.metrics["action/near_saturation_any"][ids].bool()
        stats["action_out_of_bounds_any"][ids] = command.metrics["action/out_of_bounds_any"][ids].bool()
        stats["max_target_depression_fraction"][ids] = command.metrics["key/max_target_depression_fraction"][ids]
    else:
        action = env.unwrapped.action_manager.get_term("action")
        stats["action_mean_abs"][ids] = action.episode_action_mean_abs[ids]
        stats["action_max_abs"][ids] = action.episode_action_max_abs[ids]
        stats["action_near_saturation_fraction"][ids] = action.episode_near_saturation_fraction[ids]
        stats["action_near_saturation_any"][ids] = action.episode_near_saturation_any[ids]
        stats["action_out_of_bounds_any"][ids] = action.episode_out_of_bounds_any[ids]
        stats["max_target_depression_fraction"][ids] = command._episode_max_target_depression_fraction[ids]
    stats["pending"][ids] = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if args_cli.task_config:
        from public_task import apply_config as apply_task, load_config as load_task
        apply_task(env_cfg, load_task(args_cli.task_config))
    if args_cli.physics_config:
        from public_physics import apply_config, load_config
        apply_config(env_cfg, load_config(args_cli.physics_config))
    target_length, step_budget = _configure_protocol(env_cfg)
    artifact_contract = _validate_artifact_contract(env_cfg)
    perturbation = None
    if args_cli.perturbation_config:
        from public_task import apply_perturbation, load_perturbation
        perturbation = load_perturbation(args_cli.perturbation_config)
        apply_perturbation(env_cfg, perturbation)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    if args_cli.video:
        # Frame the whole keyboard and arm for the public quickstart video.
        env_cfg.video_recorder.eye = (0.85, -0.35, 0.60)
        env_cfg.video_recorder.lookat = (0.16, 0.04, 0.08)
    video_length = args_cli.video_length or step_budget
    with launch_simulation(env_cfg, args_cli):
        base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        try:
            if args_cli.video:
                base_env = gym.wrappers.RecordVideo(
                    base_env,
                    video_folder=str(args_cli.out.parent / "videos"),
                    step_trigger=lambda step: step == 0,
                    video_length=video_length,
                    disable_logger=True,
                    name_prefix=args_cli.out.stem,
                )
            env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            command = env.unwrapped.command_manager.get_term("typing")
            bank, bank_sha256, target_histogram, target_descriptor = _build_target_bank(command, target_length)
            _install_target_bank(command, bank, target_length)
            stats = _new_stats(command)
            original_reset = command.reset
            action_trace: list[dict[str, object]] = []
            event_trace: list[dict[str, object]] = []
            executed_steps = 0

            def reset_with_capture(env_ids=None):
                # Preserve the final transition before the environment clears it.
                if args_cli.num_envs == 1 and bool(stats["pending"][0]):
                    event_trace.append({
                        "step": executed_steps,
                        "phase": int(command.phase[0].item()),
                        "character_index": int(command.character_index[0].item()),
                        **{name: bool(getattr(command, attr)[0].item()) for name, attr in (
                            ("target_down", "target_down_event"), ("target_up", "target_up_event"),
                            ("clearance", "clearance_event"), ("wrong_key", "wrong_key"),
                            ("overlap", "overlap"), ("completed", "completed"))},
                        "terminal": True,
                    })
                _capture(command, env, stats, env_ids, terminating=True)
                return original_reset(env_ids)

            command.reset = reset_with_capture
            observations = env.get_observations()
            initial_observation = observations["policy"][0].detach().cpu().tolist()
            if len(initial_observation) != 22:
                raise RuntimeError(f"expected the fixed-Cartesian 22-D observation, got {len(initial_observation)}")

            executed_steps = 0
            with torch.inference_mode():
                for step in range(step_budget):
                    actions = policy(observations)
                    if step < args_cli.trace_policy_actions:
                        action_trace.append(
                            {
                                "step": step,
                                "observation_env0": observations["policy"][0].detach().cpu().tolist(),
                                "action_env0": actions[0].detach().cpu().tolist(),
                            }
                        )
                    observations, _, _, _ = env.step(actions)
                    executed_steps = step + 1
                    pending_ids = torch.where(stats["pending"])[0]
                    _accumulate_events(command, stats, pending_ids)
                    if args_cli.num_envs == 1 and bool(stats["pending"][0]):
                        event_trace.append(
                            {
                                "step": step,
                                "phase": int(command.phase[0].item()),
                                "character_index": int(command.character_index[0].item()),
                                "target_down": bool(command.target_down_event[0].item()),
                                "target_up": bool(command.target_up_event[0].item()),
                                "clearance": bool(command.clearance_event[0].item()),
                                "wrong_key": bool(command.wrong_key[0].item()),
                                "overlap": bool(command.overlap[0].item()),
                                "completed": bool(command.completed[0].item()),
                            }
                        )
                    if args_cli.step_sleep:
                        time.sleep(args_cli.step_sleep)
                    if not bool(stats["pending"].any()):
                        break

            incomplete_ids = torch.where(stats["pending"])[0]
            _capture(command, env, stats, incomplete_ids, terminating=False)
            failed = stats["terminal_wrong_key"] | stats["terminal_overlap"] | stats["terminal_initially_held"]
            for term in stats["termination"].values():
                failed |= term
            exact_event_counts = (
                (stats["target_down_count"] == target_length)
                & (stats["target_up_count"] == target_length)
                & (stats["clearance_count"] == target_length)
            )
            strict_success = stats["terminal_completed"] & stats["terminal_typed_exact"] & exact_event_counts & ~failed
            from public_task import evaluation_reset_metadata
            report = {
                "kind": "so101_physical_calibrated_fixed_cartesian_strict_evaluation",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "evaluator_sha256": _sha256(Path(__file__).resolve()),
                "artifact_contract": artifact_contract,
                "evaluation_mode": "fixed_policy_reset_robustness" if perturbation else "matched_training_contract",
                "perturbation_config": perturbation,
                "task": args_cli.task,
                "actuator_profile": env_cfg.actuator_profile,
                "keyboard_profile": env_cfg.keyboard_profile,
                "seed": args_cli.seed,
                "num_envs": env.num_envs,
                "target_length": target_length,
                "steps_budget": step_budget,
                "steps_executed": executed_steps,
                "successes": int(strict_success.sum().item()),
                "overall_success_rate": float(strict_success.float().mean().item()),
                "raw_completion_rate": float(stats["terminal_completed"].float().mean().item()),
                "typed_exact_rate": float(stats["terminal_typed_exact"].float().mean().item()),
                "incomplete_rate": float((stats["character_index"] < target_length).float().mean().item()),
                "wrong_key_rate": float(stats["terminal_wrong_key"].float().mean().item()),
                "overlap_rate": float(stats["terminal_overlap"].float().mean().item()),
                "initially_held_rate": float(stats["terminal_initially_held"].float().mean().item()),
                "event_count_mismatch_rate": float((~exact_event_counts).float().mean().item()),
                "termination_rates": {
                    name: float(values.float().mean().item()) for name, values in stats["termination"].items()
                },
                "action": {
                    "mean_abs": float(stats["action_mean_abs"].mean().item()),
                    "max_abs": float(stats["action_max_abs"].max().item()),
                    "mean_near_saturation_fraction": float(stats["action_near_saturation_fraction"].mean().item()),
                    "near_saturation_episode_rate": float(stats["action_near_saturation_any"].float().mean().item()),
                    "out_of_bounds_episode_rate": float(stats["action_out_of_bounds_any"].float().mean().item()),
                },
                "max_target_depression_fraction": float(stats["max_target_depression_fraction"].max().item()),
                "target_bank_sha256": bank_sha256,
                "target_histogram": dict(sorted(target_histogram.items())),
                "target_descriptor": target_descriptor,
                "initial_observation_env0": initial_observation,
                "raw_policy_action_trace": action_trace,
                "event_trace_env0": event_trace,
                "protocol": {
                    "fresh_empty_typed_buffer": True,
                    **evaluation_reset_metadata(env_cfg),
                    "clearance_m": float(command.cfg.clearance_m),
                    "clearance_control_ticks": int(command.cfg.clearance_control_ticks),
                    "required_sequence": (
                        "target down -> matching up -> "
                        f"{int(command.cfg.clearance_control_ticks)} "
                        f"{float(command.cfg.clearance_m) * 1000.0:g} mm clearance ticks"
                    ),
                    "physics": "newton_mjwarp",
                },
            }
            if args_cli.viewer_hold_seconds:
                deadline = time.monotonic() + args_cli.viewer_hold_seconds
                while time.monotonic() < deadline:
                    base_env.render()
                    time.sleep(1.0 / 60.0)
        finally:
            base_env.close()

    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    args_cli.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {args_cli.out}")


if __name__ == "__main__":
    main()
