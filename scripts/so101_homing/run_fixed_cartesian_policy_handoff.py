# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a fixed-layout Cartesian checkpoint on the marked keyboard fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .constants import ARM_JOINT_NAMES, DEFAULT_KEYBOARD_DEVICE, DEFAULT_ROBOT_ID
from .fixed_cartesian_policy import (
    CLEARANCE_CONTROL_TICKS,
    CLEARANCE_M,
    CONTROL_HZ,
    FROZEN_LETTERS,
    JOINT_RATE_LIMITS_RAD_S,
    PHASE_TIMEOUTS_S,
    PHYSICAL_CALIBRATED_20260718_PROFILE,
    STAGE_TARGET_LENGTHS,
    FixedCartesianDeploymentProfile,
    FixedCartesianPolicy,
    FixedCartesianTypingState,
    build_observation,
    deployment_profile,
    frozen_target_xyz,
    integrate_command,
    lerobot_position_to_profile_model_rad,
    profile_model_rad_to_lerobot_position,
)
from .keyboard_events import _suppress_terminal_echo
from .policy_runtime import (
    DEFAULT_RELEASE_SETTLE_S,
    DEFAULT_RETURN_TO_REST_S,
    KeyboardMonitor,
    PolicyContractError,
    PressGate,
    append_jsonl,
    load_rest_pose,
)
from .robot import (
    connect_robot,
    prepare_closed_typing_jaw,
    read_arm_q_deg,
    resolve_robot_port,
    send_typing_q_deg,
    smooth_move_arm,
)
from .so101_kinematics import (
    ARM_JOINT_LIMITS_RAD,
    FIXED_JAW_TYPING_TIP_SITE_POS,
    GRIPPER_MODEL_MAX_RAD,
    GRIPPER_MODEL_MIN_RAD,
    site_position,
)

DEFAULT_OUT_DIR = Path("output/so101_homing/policy_runs")
DEFAULT_FIXED_START_EPSILON_DEG = 0.5
DEFAULT_RESET_TRANSITION_S = 3.0
DEFAULT_RESET_SETTLE_TIMEOUT_S = 3.0
DEFAULT_RESET_MAX_ATTEMPTS = 3
RESET_TRACKING_GAIN = 0.5
RESET_TRACKING_MAX_STEP_DEG = 1.0
RESET_TRACKING_HZ = 4.0
STAGE_TARGET_LENGTH = dict(STAGE_TARGET_LENGTHS)
STAGE_MAX_DURATION_S = {
    "p0": 10.0,
    "p1a": 21.0,
    "p1b": 31.0,
    "p1c": 41.0,
    "p1d": 61.0,
}
PHYSICAL_CALIBRATED_TASK_CONTRACTS = {
    "p0": "physical_calibrated_20260718_fixed_cartesian_anchorbench_p0_v0",
    "p1a": "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1a_letters2_v0",
    "p1b": "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1b_letters3_v0",
    "p1c": "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1c_letters4_v0",
    "p1d": "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1d_letters6_v0",
}
PHYSICAL_CALIBRATED_ADDITIONAL_TASK_CONTRACTS = {
    "p1d": ("physical_calibrated_20260718_fixed_cartesian_anchorbench_p1d_transit15_letters6_v0",),
}
_MIN_SAMPLE_DT_S = 1.0e-6


def _emit_deployment_contract(
    *,
    out: Path,
    summary: dict[str, Any],
    deployment_contract: dict[str, Any],
) -> None:
    summary["deployment_contract"] = deployment_contract
    append_jsonl(out, {"kind": "deployment_contract", **deployment_contract})
    checkpoint = deployment_contract['checkpoint']
    environment = checkpoint['training_environment'] or {}
    adapter = deployment_contract['model_to_hardware']['joint_coordinate_adapter']
    print(f"Policy: iteration {checkpoint['iteration']} | actuator {environment.get('actuator_profile')} "
          f"| target {deployment_contract['target']}")
    print(f"Checkpoint: {checkpoint['path']}")
    print(f"Training scene: {environment.get('task_contract')}")
    print(f"Joint offsets (degrees): {adapter['joint_zero_offset_deg']}")
    print('On execution, operator confirmation precedes the trained reset and typing.')
    print(f'Full deployment contract recorded in: {out}')


def _model_joint_limit_contract() -> dict[str, list[float]]:
    return {
        **{
            name: [float(lower), float(upper)]
            for name, (lower, upper) in zip(ARM_JOINT_NAMES, ARM_JOINT_LIMITS_RAD, strict=True)
        },
        "gripper": [float(GRIPPER_MODEL_MIN_RAD), float(GRIPPER_MODEL_MAX_RAD)],
    }


def _rest_arm_rad(rest_q_deg: np.ndarray, profile: FixedCartesianDeploymentProfile) -> np.ndarray:
    value = lerobot_position_to_profile_model_rad(np.concatenate((rest_q_deg, np.asarray([0.0]))), profile)[:5]
    if value.shape != (5,) or not np.isfinite(value).all():
        raise ValueError("rest pose must produce five finite model arm joints")
    return value


def _validate_rest_pose(
    rest_q_deg: np.ndarray,
    profile: FixedCartesianDeploymentProfile = PHYSICAL_CALIBRATED_20260718_PROFILE,
    *,
    atol_rad: float = 1.0e-9,
) -> None:
    actual = _rest_arm_rad(rest_q_deg, profile)
    expected = np.asarray(profile.q_reset_rad, dtype=np.float64)
    error = float(np.max(np.abs(actual - expected)))
    if error > atol_rad:
        raise PolicyContractError(
            f"rest pose does not match this checkpoint's fixed simulator reset: max error={np.degrees(error):.6f} deg"
        )


def _read_closed_typing_q_rad(robot: Any, profile: FixedCartesianDeploymentProfile) -> np.ndarray:
    arm_deg = read_arm_q_deg(robot)
    return lerobot_position_to_profile_model_rad(
        np.concatenate((arm_deg, np.asarray([0.0], dtype=np.float64))), profile
    )


def _model_goal_to_arm_q_deg(goal_rad: np.ndarray, profile: FixedCartesianDeploymentProfile) -> np.ndarray:
    q_lerobot = profile_model_rad_to_lerobot_position(goal_rad, profile)
    if q_lerobot.shape != (6,) or not np.isfinite(q_lerobot).all():
        raise ValueError("model goal conversion must produce five arm joints and one gripper")
    return q_lerobot[:5]


def _start_error_deg(q_model_rad: np.ndarray, profile: FixedCartesianDeploymentProfile) -> float:
    actual = np.asarray(q_model_rad[:5], dtype=np.float64)
    expected = np.asarray(profile.q_reset_rad, dtype=np.float64)
    return float(np.degrees(np.max(np.abs(actual - expected))))


def _resolve_startup_mode(args: argparse.Namespace, profile: FixedCartesianDeploymentProfile) -> str:
    if profile.name != "physical-calibrated-20260718":
        raise PolicyContractError("this hardware runner only supports physical-calibrated-20260718 checkpoints")
    if args.startup_mode not in {"auto", "fixed-reset"}:
        raise PolicyContractError(
            "physical-calibrated-20260718 was trained from fixed q_reset and cannot use recorded F/Y/N startup"
        )
    return "fixed-reset"


def _next_reset_tracking_goal(
    *, goal_q_deg: np.ndarray, target_q_deg: np.ndarray, measured_q_deg: np.ndarray
) -> np.ndarray:
    goal = np.asarray(goal_q_deg, dtype=np.float64)
    target = np.asarray(target_q_deg, dtype=np.float64)
    measured = np.asarray(measured_q_deg, dtype=np.float64)
    if any(value.shape != (5,) or not np.isfinite(value).all() for value in (goal, target, measured)):
        raise ValueError("fixed-reset tracking requires five finite arm joint positions")
    correction = np.clip(
        RESET_TRACKING_GAIN * (target - measured),
        -RESET_TRACKING_MAX_STEP_DEG,
        RESET_TRACKING_MAX_STEP_DEG,
    )
    return goal + correction


def _move_to_fixed_reset(
    robot: Any,
    *,
    target_q_deg: np.ndarray,
    transition_s: float,
    epsilon_deg: float,
    settle_timeout_s: float,
) -> dict[str, Any]:
    """Move directly to the simulator reset and require measured convergence."""

    target = np.asarray(target_q_deg, dtype=np.float64)
    if target.shape != (5,) or not np.isfinite(target).all():
        raise ValueError("fixed reset must contain five finite arm joint positions")
    initial = read_arm_q_deg(robot)
    initial_error_deg = float(np.max(np.abs(initial - target)))
    print(
        f"Fixed reset startup: initial max joint error {initial_error_deg:.3f} deg; moving over {transition_s:.2f} s."
    )
    smooth_move_arm(robot, target, duration_s=transition_s, control_hz=CONTROL_HZ)
    deadline = time.monotonic() + settle_timeout_s
    goal = target.copy()
    measured = read_arm_q_deg(robot)
    while float(np.max(np.abs(measured - target))) > epsilon_deg and time.monotonic() < deadline:
        goal = _next_reset_tracking_goal(
            goal_q_deg=goal,
            target_q_deg=target,
            measured_q_deg=measured,
        )
        send_typing_q_deg(robot, goal, gripper_deg=0.0)
        time.sleep(1.0 / RESET_TRACKING_HZ)
        measured = read_arm_q_deg(robot)
    error_deg = float(np.max(np.abs(measured - target)))
    return {
        "mode": "fixed-reset",
        "initial_measured_q_deg": initial.tolist(),
        "initial_joint_error_deg": initial_error_deg,
        "target_q_deg": target.tolist(),
        "measured_q_deg": measured.tolist(),
        "joint_error_deg": error_deg,
        "joint_epsilon_deg": epsilon_deg,
        "converged": error_deg <= epsilon_deg,
        "transition_s": transition_s,
        "settle_timeout_s": settle_timeout_s,
    }


def _move_to_fixed_reset_with_retries(
    robot: Any,
    *,
    target_q_deg: np.ndarray,
    profile: FixedCartesianDeploymentProfile,
    transition_s: float,
    epsilon_deg: float,
    settle_timeout_s: float,
    max_attempts: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Retry fixed reset until both encoder and model-frame handoff gates pass."""

    if max_attempts < 1:
        raise ValueError("fixed-reset max attempts must be positive")
    attempts: list[dict[str, Any]] = []
    policy_start_q_rad: np.ndarray | None = None
    for attempt_index in range(1, max_attempts + 1):
        print(f"Fixed reset attempt {attempt_index}/{max_attempts}.")
        attempt = _move_to_fixed_reset(
            robot,
            target_q_deg=target_q_deg,
            transition_s=transition_s,
            epsilon_deg=epsilon_deg,
            settle_timeout_s=settle_timeout_s,
        )
        attempt["attempt_index"] = attempt_index
        attempt["reset_converged"] = attempt["converged"]
        start_error_deg = None
        if attempt["reset_converged"]:
            candidate_q_rad = _read_closed_typing_q_rad(robot, profile)
            start_error_deg = _start_error_deg(candidate_q_rad, profile)
            if start_error_deg <= epsilon_deg:
                policy_start_q_rad = candidate_q_rad
        attempt["policy_start_joint_error_deg"] = start_error_deg
        attempt["converged"] = bool(attempt["reset_converged"] and policy_start_q_rad is not None)
        attempts.append(attempt)
        if attempt["converged"]:
            break
        if attempt_index < max_attempts:
            print("Fixed reset gate missed; retrying before policy handoff.")

    report = dict(attempts[-1])
    report.update(
        {
            "attempt_count": len(attempts),
            "max_attempts": max_attempts,
            "attempts": attempts,
        }
    )
    return report, policy_start_q_rad


def _joint_velocity_from_samples(
    q_rad: np.ndarray,
    previous_q_rad: np.ndarray,
    *,
    elapsed_s: float,
) -> np.ndarray:
    current = np.asarray(q_rad, dtype=np.float64)
    previous = np.asarray(previous_q_rad, dtype=np.float64)
    if current.shape != previous.shape or not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise ValueError("joint samples must have matching finite shapes")
    if not np.isfinite(elapsed_s) or elapsed_s < _MIN_SAMPLE_DT_S:
        raise ValueError(f"joint sample interval must be at least {_MIN_SAMPLE_DT_S:g} s")
    return (current - previous) / elapsed_s


def _validated_training_environment(
    *,
    checkpoint: Path,
    env_config: Path | None,
    profile: FixedCartesianDeploymentProfile,
    stage: str,
    expected_actuator_profile: str = "anchorbench",
    expected_task_contract: str | None = None,
) -> dict[str, Any] | None:
    """Require calibrated checkpoints to carry their matching IsaacLab environment contract."""

    if profile.name != "physical-calibrated-20260718":
        return None
    candidates = (
        [env_config]
        if env_config is not None
        else [
            checkpoint.parent / "params/env.yaml",
            checkpoint.parent.parent / "params/env.yaml",
        ]
    )
    selected = next((path for path in candidates if path is not None and path.is_file()), None)
    if selected is None:
        raise PolicyContractError(
            "physical-calibrated-20260718 deployment requires its matching params/env.yaml; "
            "pass --env-config or keep params/env.yaml beside the checkpoint"
        )
    try:
        config = yaml.load(selected.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except yaml.YAMLError as exc:
        raise PolicyContractError(f"could not parse training environment {selected}: {exc}") from exc
    if not isinstance(config, dict):
        raise PolicyContractError(f"training environment {selected} is not a mapping")

    actual_contract = str(config.get("task_contract", ""))
    if expected_task_contract is not None:
        allowed_contracts = {expected_task_contract}
    elif expected_actuator_profile == "anchorbench":
        allowed_contracts = {
            PHYSICAL_CALIBRATED_TASK_CONTRACTS[stage],
            *PHYSICAL_CALIBRATED_ADDITIONAL_TASK_CONTRACTS.get(stage, ()),
        }
    else:
        raise PolicyContractError("non-AnchorBench deployment requires an explicit expected task contract")
    expected_values = {
        "keyboard_profile": profile.calibration_id,
        "actuator_profile": expected_actuator_profile,
        "target_reference_sha256": profile.target_reference_sha256,
        "target_manifest_sha256": profile.target_manifest_sha256,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_values.items()
        if config.get(key) != expected
    }
    if actual_contract not in allowed_contracts:
        mismatches["task_contract"] = {
            "expected": sorted(allowed_contracts),
            "actual": actual_contract,
        }
    typing = config.get("commands", {}).get("typing", {})
    try:
        clearance_m = float(typing["clearance_m"])
        clearance_control_ticks = int(typing["clearance_control_ticks"])
        episode_length_s = float(config["episode_length_s"])
        letter_length = tuple(int(value) for value in typing["letter_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyContractError(
            f"training environment {selected} is missing a valid episode/clearance contract"
        ) from exc
    if clearance_m <= 0.0 or clearance_control_ticks < 1 or episode_length_s <= 0.0:
        raise PolicyContractError(f"training environment {selected} contains a non-positive episode/clearance contract")
    expected_length = STAGE_TARGET_LENGTH[stage]
    if letter_length != (expected_length, expected_length):
        mismatches["letter_length"] = {
            "expected": [expected_length, expected_length],
            "actual": list(letter_length),
        }
    if mismatches:
        raise PolicyContractError(
            f"training environment {selected} does not match {profile.name}/{stage}; mismatches={mismatches}"
        )
    return {
        "path": str(selected.resolve()),
        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "task_contract": actual_contract,
        "keyboard_profile": profile.calibration_id,
        "actuator_profile": expected_actuator_profile,
        "episode_length_s": episode_length_s,
        "clearance_m": clearance_m,
        "clearance_control_ticks": clearance_control_ticks,
        "letter_length": list(letter_length),
        "validated": True,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.enable_robot == args.dry_run:
        raise ValueError("choose exactly one of --enable-robot or --dry-run")
    if args.stage not in STAGE_TARGET_LENGTH:
        raise ValueError(f"unknown stage {args.stage!r}")
    expected = STAGE_TARGET_LENGTH[args.stage]
    target = args.target.strip().upper()
    if len(target) != expected or not target.isascii() or not target.isalpha():
        raise ValueError(f"{args.stage.upper()} requires exactly {expected} A-Z letter(s), got {args.target!r}")
    for name in (
        "reset_transition_s",
        "reset_settle_timeout_s",
        "start_joint_epsilon_deg",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_duration is not None and args.max_duration <= 0.0:
        raise ValueError("max-duration must be positive when provided")
    if args.reset_max_attempts < 1:
        raise ValueError("reset-max-attempts must be positive")
    if not 0.0 < args.diagnostic_rate_scale <= 1.0:
        raise ValueError("diagnostic-rate-scale must be in (0, 1]")
    if args.return_to_rest_s < 0.0:
        raise ValueError("return-to-rest-s must be non-negative")


def _deployment_contract(
    *,
    args: argparse.Namespace,
    policy: FixedCartesianPolicy,
    profile: FixedCartesianDeploymentProfile,
    training_environment: dict[str, Any] | None,
    state: FixedCartesianTypingState,
    rest_pose: Path,
    startup_mode: str,
) -> dict[str, Any]:
    target = args.target.upper()
    startup = {
        "mode": startup_mode,
        "rest_pose": str(rest_pose.resolve()),
        "fixed_q_reset_rad": list(profile.q_reset_rad),
        "policy_start_epsilon_deg": args.start_joint_epsilon_deg,
    }
    startup.update(
        {
            "manual_fyn_fixture_proof": False,
            "recorded_homing_used": False,
            "reset_transition_s": args.reset_transition_s,
            "reset_settle_timeout_s": args.reset_settle_timeout_s,
            "reset_max_attempts": args.reset_max_attempts,
            "fixture_contract": "frozen calibrated A-Z map; validate separately if the fixture moves",
        }
    )
    return {
        "schema_version": 1,
        "runner": "fixed_cartesian_frozen_fixture",
        "stage": args.stage,
        "target": target,
        "checkpoint": {
            "path": str(policy.path.resolve()),
            "sha256": policy.sha256,
            "iteration": policy.iteration,
            "contract": policy.contract.as_dict(),
            "training_environment": training_environment,
        },
        "observation": {
            "dimension": 22,
            "order": [
                "measured arm q - fixed q_reset (5)",
                "measured-time arm joint velocity (5)",
                "persistent q_command - fixed q_reset (5)",
                "normalized frozen target XYZ in robot-base frame (3)",
                "seek/release/clearance one-hot phase (3)",
                "phase elapsed seconds (1)",
            ],
            "key_positions_source": "frozen marked physical A-Z fixture map",
            "registration_used_by_actor": False,
            "frozen_letter_count": len(FROZEN_LETTERS),
            "frozen_reference_sha256": profile.target_reference_sha256,
            "frozen_reference_manifest_sha256": profile.target_manifest_sha256,
            "target_xyz_b_m": {letter: frozen_target_xyz(letter, profile).tolist() for letter in dict.fromkeys(target)},
        },
        "policy_inference": {
            "kind": "RSL-RL deterministic native-Beta mean",
            "distribution_support": [-1.0, 1.0],
            "control_hz": CONTROL_HZ,
            "persistent_command_update": "q_cmd = clip(q_cmd + action * joint_rate_limit * 0.04)",
            "joint_rate_limits_rad_s": list(JOINT_RATE_LIMITS_RAD_S),
            "diagnostic_rate_scale": args.diagnostic_rate_scale,
            "effective_joint_rate_limits_rad_s": [
                value * args.diagnostic_rate_scale for value in JOINT_RATE_LIMITS_RAD_S
            ],
            "training_contract_modified": args.diagnostic_rate_scale != 1.0,
            "command_rebuilt_from_measured_q": False,
            "extra_action_hold": False,
            "extra_slew_limit": False,
        },
        "startup": startup,
        "keyboard_feedback": {
            "expected_transition": (
                "target down -> matching up -> "
                f"{state.clearance_control_ticks} sustained "
                f"{state.clearance_m * 1000.0:g} mm-clearance ticks -> advance"
            ),
            "clearance_m": state.clearance_m,
            "clearance_control_ticks": state.clearance_control_ticks,
            "wrong_key": "stop",
            "overlapping_key_down": "stop",
            "press_during_release_or_clearance": "stop",
            "phase_timeouts_s": list(PHASE_TIMEOUTS_S),
        },
        "model_to_hardware": {
            "joint_coordinate_adapter": {
                "profile": profile.name,
                "calibration_id": profile.calibration_id,
                "joint_order": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
                "joint_zero_offset_deg": list(profile.joint_zero_offset_deg),
                "encoder_to_model": "q_model = q_lerobot + joint_zero_offset",
                "model_to_encoder": "q_lerobot_goal = q_model_goal - joint_zero_offset",
                "encoder_to_model_conversion_applied": True,
                "model_to_encoder_conversion_applied": True,
            },
            "joint_limit_clip": {
                "enabled": True,
                "limits_rad": _model_joint_limit_contract(),
                "behavior": "same persistent-command saturation as the simulator action term",
            },
            "moving_jaw": {
                "hardware_command_deg": 0.0,
                "model_goal_rad": GRIPPER_MODEL_MIN_RAD,
                "typing_contact": "fixed jaw",
            },
        },
        "shutdown": {
            "max_policy_duration_s": args.max_duration,
            "max_policy_duration_source": args.max_duration_source,
            "return_to_rest_s": args.return_to_rest_s,
        },
    }


def _operator_confirmed(*, skip_confirmation: bool) -> bool:
    if skip_confirmation:
        return True
    print("Confirm safety: marked keyboard fixed, fixed jaw closed, operator ready to stop power.")
    prompt = "Move directly to the trained fixed reset, then hand control to the frozen-map policy? [y/N] "
    return input(prompt).strip().casefold() == "y"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_TARGET_LENGTH))
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--task-profile",
        choices=("physical-calibrated-20260718",),
        default="physical-calibrated-20260718",
        help="Frozen physical keyboard and model/encoder joint-coordinate contract.",
    )
    parser.add_argument("--deployment-config", type=Path, help="Prepared geometry/identity bundle from this checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--env-config",
        type=Path,
        required=True,
        help="Exact IsaacLab params/env.yaml archived with the checkpoint.",
    )
    parser.add_argument(
        "--expected-actuator-profile",
        choices=("anchorbench", "baseline", "usd_drive"),
        default="anchorbench",
        help="Actuator profile that params/env.yaml must declare.",
    )
    parser.add_argument(
        "--expected-task-contract",
        default=None,
        help="Exact task contract required for a new matched recipe, including Baseline.",
    )
    parser.add_argument(
        "--rest-pose",
        type=Path,
        required=True,
        help="Physical LeRobot encoder rest pose matching the fixed simulator reset.",
    )
    parser.add_argument(
        "--startup-mode",
        choices=("fixed-reset",),
        default="fixed-reset",
        help="The calibrated policy starts from its fixed simulator reset; manual F/Y/N replay is invalid.",
    )
    parser.add_argument("--device", default=DEFAULT_KEYBOARD_DEVICE)
    parser.add_argument("--port", default=None)
    parser.add_argument("--id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument('--experimental', action='store_true', help='supervised candidate-mapping experiment; no alignment qualification claim')
    parser.add_argument('--experimental-clearance-m', type=float, default=None,
                        help='candidate experiment only: require this measured FK clearance before advancing')
    parser.add_argument("--yes", action="store_true", help="Skip only the final operator confirmation.")
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--reset-transition-s", type=float, default=DEFAULT_RESET_TRANSITION_S)
    parser.add_argument("--reset-settle-timeout-s", type=float, default=DEFAULT_RESET_SETTLE_TIMEOUT_S)
    parser.add_argument("--reset-max-attempts", type=int, default=DEFAULT_RESET_MAX_ATTEMPTS)
    parser.add_argument(
        "--start-joint-epsilon-deg",
        type=float,
        default=DEFAULT_FIXED_START_EPSILON_DEG,
    )
    parser.add_argument("--return-to-rest-s", type=float, default=DEFAULT_RETURN_TO_REST_S)
    parser.add_argument(
        "--diagnostic-rate-scale",
        type=float,
        default=1.0,
        help="Scale the trained joint-rate envelope for a labeled physical diagnostic; 1.0 preserves the contract.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    args.max_duration_source = "explicit_cli" if args.max_duration is not None else None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or DEFAULT_OUT_DIR / (
        f"fixed_cartesian_{args.task_profile}_{args.stage}_{args.target.upper()}_{timestamp}.jsonl"
    )
    summary: dict[str, Any] = {
        "kind": "so101_fixed_cartesian_handoff",
        "status": "rejected",
        "stage": args.stage,
        "target": args.target.upper(),
        "checkpoint": str(args.checkpoint.resolve()),
        "task_profile": args.task_profile,
    }
    robot = None
    rest_q_deg: np.ndarray | None = None
    try:
        _validate_args(args)
        if (not args.dry_run or args.experimental) and not args.deployment_config:
            raise PolicyContractError('Physical or experimental policy execution requires a prepared deployment bundle.')
        profile = deployment_profile(args.task_profile)
        if args.deployment_config:
            from .deployment_bundle import validate_manifest, validate_execution_mode
            profile, bundle = validate_manifest(args.deployment_config, args.checkpoint, args.env_config, args.id)
            if not args.dry_run or args.experimental:
                summary['execution_mode'] = validate_execution_mode(bundle, experimental=args.experimental)
            summary['alignment_qualified'] = bool(bundle.get('alignment_qualified', False))
            summary['experimental'] = args.experimental
            if args.stage != bundle['stage']:
                raise PolicyContractError('stage differs from deployment bundle')
            if args.expected_task_contract is not None and args.expected_task_contract != bundle['task_contract']:
                raise PolicyContractError('expected task contract differs from deployment bundle')
            args.expected_task_contract = bundle['task_contract']
            if args.expected_actuator_profile != bundle['actuator_profile']:
                raise PolicyContractError('actuator profile differs from deployment bundle')
            summary['deployment_bundle_sha256'] = hashlib.sha256(args.deployment_config.read_bytes()).hexdigest()
            summary['encoder_convention'] = bundle['encoder_convention']
            summary['lerobot_calibration_sha256'] = bundle.get('lerobot_calibration_sha256')
            summary['joint_calibration_sha256'] = bundle.get('joint_calibration_sha256')
            summary['alignment_sha256'] = bundle.get('alignment_sha256')
            summary['alignment_candidate_sha256'] = bundle.get('alignment_candidate_sha256')
        startup_mode = _resolve_startup_mode(args, profile)
        rest_q_deg = load_rest_pose(args.rest_pose)
        _validate_rest_pose(rest_q_deg, profile)
        policy = FixedCartesianPolicy(args.checkpoint)
        training_environment = _validated_training_environment(
            checkpoint=args.checkpoint,
            env_config=args.env_config,
            profile=profile,
            stage=args.stage,
            expected_actuator_profile=args.expected_actuator_profile,
            expected_task_contract=args.expected_task_contract,
        )
        if args.experimental_clearance_m is not None:
            if not args.experimental:
                raise PolicyContractError('--experimental-clearance-m requires --experimental')
            trained_clearance_m = (training_environment["clearance_m"]
                                   if training_environment is not None else CLEARANCE_M)
            if not np.isfinite(args.experimental_clearance_m) or args.experimental_clearance_m < trained_clearance_m:
                raise PolicyContractError('--experimental-clearance-m must be finite and at least the trained clearance')
        if args.max_duration is None:
            if training_environment is None:
                args.max_duration = STAGE_MAX_DURATION_S[args.stage]
                args.max_duration_source = "native_stage_default"
            else:
                args.max_duration = training_environment["episode_length_s"]
                args.max_duration_source = "archived_training_environment"
        state = FixedCartesianTypingState.from_text(
            args.target,
            stage=args.stage,
            now_s=0.0,
            profile=profile,
            clearance_m=(args.experimental_clearance_m if args.experimental_clearance_m is not None else
                         (training_environment["clearance_m"] if training_environment is not None else CLEARANCE_M)),
            clearance_control_ticks=(
                training_environment["clearance_control_ticks"]
                if training_environment is not None
                else CLEARANCE_CONTROL_TICKS
            ),
        )
        initial_observation = build_observation(
            q_rad=np.asarray(profile.q_reset_rad),
            qd_rad_s=np.zeros(5),
            q_command_rad=np.asarray(profile.q_reset_rad),
            state=state,
            now_s=0.0,
        )
        initial_policy = policy.action(initial_observation)
        deployment_contract = _deployment_contract(
            args=args,
            policy=policy,
            profile=profile,
            training_environment=training_environment,
            state=state,
            rest_pose=args.rest_pose,
            startup_mode=startup_mode,
        )
        summary.update(
            {
                "checkpoint_sha256": policy.sha256,
                "checkpoint_iteration": policy.iteration,
                "checkpoint_contract": policy.contract.as_dict(),
                "initial_observation": initial_observation.tolist(),
                "initial_policy_action": initial_policy["action"].tolist(),
                "experimental_clearance_m": args.experimental_clearance_m,
            }
        )
        _emit_deployment_contract(out=out, summary=summary, deployment_contract=deployment_contract)
        if args.experimental:
            mode = ('direct LeRobot mapping' if np.allclose(profile.joint_zero_offset_deg, 0.0)
                    else 'candidate mapping')
            print(f'Supervised policy experiment: {mode} is not qualified; motion/contact checks remain enabled.')
        if args.dry_run:
            summary["status"] = "dry_run_ready"
            print(f"Software dry run passed ({args.stage.upper()}); no robot command was sent. "
                  "Physical alignment and typing are not tested by this check.")
            return 0

        resolved_port = resolve_robot_port(args.port)
        if not _operator_confirmed(skip_confirmation=args.yes):
            summary["status"] = "operator_cancelled"
            return 1

        robot = connect_robot(port=resolved_port, robot_id=args.id)
        prepare_closed_typing_jaw(robot, gripper_deg=0.0)
        reset, previous_full_q_rad = _move_to_fixed_reset_with_retries(
            robot,
            target_q_deg=rest_q_deg,
            profile=profile,
            transition_s=args.reset_transition_s,
            epsilon_deg=args.start_joint_epsilon_deg,
            settle_timeout_s=args.reset_settle_timeout_s,
            max_attempts=args.reset_max_attempts,
        )
        summary["startup"] = reset
        if not reset["converged"]:
            summary["status"] = "fixed_reset_rejected"
            handoff_error = reset["policy_start_joint_error_deg"]
            handoff_error_text = "not evaluated" if handoff_error is None else f"{handoff_error:.3f} deg"
            print(
                f"Fixed reset rejected after {reset['attempt_count']} attempts: "
                f"encoder error {reset['joint_error_deg']:.3f} deg; "
                f"model handoff error {handoff_error_text}; "
                f"limit {args.start_joint_epsilon_deg:.3f} deg."
            )
            return 2
        start_error_deg = _start_error_deg(previous_full_q_rad, profile)
        summary["policy_start_joint_error_deg"] = start_error_deg
        if start_error_deg > args.start_joint_epsilon_deg:
            raise PolicyContractError(
                f"policy handoff is {start_error_deg:.3f} deg from fixed q_reset, "
                f"limit is {args.start_joint_epsilon_deg:.3f} deg"
            )

        q_command_rad = np.asarray(profile.q_reset_rad, dtype=np.float64).copy()
        previous_sample_at_s = time.perf_counter()
        press_gate = PressGate()
        physical_key_downs: list[str] = []
        stop_reason = "max_duration"
        policy_started_at_unix_s = time.time()
        started_at = time.monotonic()
        state.phase_started_s = started_at
        step = 0
        keyboard_release_confirmed = None
        control_period_s = 1.0 / CONTROL_HZ
        sample_dt_history: list[float] = []
        overrun_count = 0
        policy_terminal_at_s: float | None = None
        policy_terminal_at_unix_s: float | None = None

        with _suppress_terminal_echo(enabled=True):
            with KeyboardMonitor(args.device) as keyboard:
                try:
                    print(
                        f"Fixed-Cartesian {args.stage.upper()} handoff active at "
                        f"{CONTROL_HZ:.0f} Hz; press Esc to stop."
                    )
                    print(
                        "Joint coordinates: "
                        f"{profile.name}; model=LeRobot+offset on read, LeRobot=model-offset on write."
                    )
                    print("Actor geometry: frozen calibrated A-Z XYZ map; no manual F/Y/N input or replay.")
                    if args.diagnostic_rate_scale == 1.0:
                        print("Native command: persistent per-joint rate integration; no hold and no extra slew cap.")
                    else:
                        print(
                            "DIAGNOSTIC RATE SCALE: "
                            f"{args.diagnostic_rate_scale:.2f}x trained joint rates; "
                            "this is not a native policy rollout."
                        )
                    print(f"Target: {state.target}")
                    while time.monotonic() - started_at < args.max_duration:
                        tick_started_at = time.perf_counter()
                        events, stop = keyboard.poll()
                        if stop:
                            stop_reason = "escape"
                            break
                        for event in events:
                            record = {"kind": "keyboard_edge", "step": step, **event}
                            if press_gate.process(event):
                                record["transition"] = "multiple_key_press"
                                append_jsonl(out, record)
                                stop_reason = "multiple_key_press"
                                break
                            char = event["char"]
                            if not isinstance(char, str) or not char.isalpha():
                                record["transition"] = "unmapped_key"
                                append_jsonl(out, record)
                                if event["value"] == 1:
                                    stop_reason = "unmapped_key"
                                    break
                                continue
                            now_s = time.monotonic()
                            if event["value"] == 1:
                                physical_key_downs.append(char.upper())
                                transition = state.key_down(char, now_s=now_s)
                            else:
                                transition = state.key_up(char, now_s=now_s)
                            record.update(
                                {
                                    "transition": transition,
                                    "phase_after": state.phase_name,
                                    "typed_after": state.typed,
                                }
                            )
                            append_jsonl(out, record)
                            if transition == "target_down":
                                print(f"KEY_DOWN_OK: {char.upper()}; waiting for release")
                            elif transition == "target_up":
                                print(
                                    f"KEY_UP_OK: {char.upper()}; waiting for "
                                    f"{state.clearance_m * 1000.0:g} mm clearance"
                                )
                            elif transition in {"wrong_key", "press_before_release", "press_before_clearance"}:
                                print(f"KEY_FAIL: {char.upper()} ({transition})")
                                stop_reason = transition
                                break
                        if stop_reason != "max_duration":
                            break

                        joint_read_started_at_s = time.perf_counter()
                        full_q_rad = _read_closed_typing_q_rad(robot, profile)
                        q_sample_at_s = time.perf_counter()
                        joint_read_s = q_sample_at_s - joint_read_started_at_s
                        sample_dt_s = q_sample_at_s - previous_sample_at_s
                        q_rad = full_q_rad[:5]
                        if step == 0:
                            qd_rad_s = np.zeros(5, dtype=np.float64)
                        else:
                            qd_rad_s = _joint_velocity_from_samples(
                                q_rad,
                                previous_full_q_rad[:5],
                                elapsed_s=sample_dt_s,
                            )

                        now_s = time.monotonic()
                        clearance_transition = state.clearance_tick(
                            tip_z_b_m=float(site_position(full_q_rad, site_pos=FIXED_JAW_TYPING_TIP_SITE_POS)[2]),
                            any_key_held=bool(keyboard.pressed_codes),
                            now_s=now_s,
                        )
                        if clearance_transition == "clearance_complete":
                            print(f"CLEARANCE_OK; advancing to {state.active_letter()}")
                        elif clearance_transition == "target_complete":
                            print("CLEARANCE_OK; target complete")
                            stop_reason = "target_complete"
                        if clearance_transition is not None:
                            append_jsonl(
                                out,
                                {
                                    "kind": "phase_transition",
                                    "step": step,
                                    "transition": clearance_transition,
                                    "policy_elapsed_s": now_s - started_at,
                                    "clearance_m": state.clearance_m,
                                    "clearance_control_ticks": state.clearance_control_ticks,
                                    "typed_after": state.typed,
                                },
                            )
                        if clearance_transition == "target_complete":
                            break
                        if state.phase_timed_out(now_s):
                            stop_reason = f"phase_timeout:{state.phase_name}"
                            print(f"PHASE_TIMEOUT: {state.phase_name}")
                            break

                        observation = build_observation(
                            q_rad=q_rad,
                            qd_rad_s=qd_rad_s,
                            q_command_rad=q_command_rad,
                            state=state,
                            now_s=now_s,
                        )
                        actor_started_at_s = time.perf_counter()
                        policy_output = policy.action(observation)
                        actor_inference_s = time.perf_counter() - actor_started_at_s
                        previous_command_rad = q_command_rad.copy()
                        q_command_rad = integrate_command(
                            q_command_rad,
                            policy_output["action"],
                            rate_scale=args.diagnostic_rate_scale,
                        )
                        full_goal_rad = np.concatenate((q_command_rad, np.asarray([GRIPPER_MODEL_MIN_RAD])))
                        command_started_at_s = time.perf_counter()
                        send_typing_q_deg(
                            robot,
                            _model_goal_to_arm_q_deg(full_goal_rad, profile),
                            gripper_deg=0.0,
                        )
                        command_write_s = time.perf_counter() - command_started_at_s
                        pre_log_s = time.perf_counter() - tick_started_at
                        overrun_s = max(0.0, pre_log_s - control_period_s)
                        overrun_count += int(overrun_s > 0.0)
                        sample_dt_history.append(sample_dt_s)
                        append_jsonl(
                            out,
                            {
                                "kind": "policy_tick",
                                "step": step,
                                "active_letter": state.active_letter(),
                                "phase": state.phase_name,
                                "phase_elapsed_s": state.phase_elapsed_s(now_s),
                                "clearance_ticks": state.clearance_ticks,
                                "typed": state.typed,
                                "observation": observation.tolist(),
                                "q_rad": q_rad.tolist(),
                                "qd_rad_s": qd_rad_s.tolist(),
                                "q_command_before_rad": previous_command_rad.tolist(),
                                "distribution_logits": policy_output["distribution_logits"].tolist(),
                                "alpha": policy_output["alpha"].tolist(),
                                "beta": policy_output["beta"].tolist(),
                                "action": policy_output["action"].tolist(),
                                "diagnostic_rate_scale": args.diagnostic_rate_scale,
                                "q_command_after_rad": q_command_rad.tolist(),
                                "command_increment_deg": np.degrees(q_command_rad - previous_command_rad).tolist(),
                                "command_tracking_error_deg": np.degrees(q_command_rad - q_rad).tolist(),
                                "joint_sample_dt_s": sample_dt_s,
                                "joint_read_s": joint_read_s,
                                "actor_inference_s": actor_inference_s,
                                "command_write_s": command_write_s,
                                "tick_pre_log_s": pre_log_s,
                                "pre_log_overrun_s": overrun_s,
                            },
                        )
                        previous_full_q_rad = full_q_rad
                        previous_sample_at_s = q_sample_at_s
                        step += 1
                        sleep_s = control_period_s - (time.perf_counter() - tick_started_at)
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)
                finally:
                    policy_terminal_at_s = time.monotonic()
                    policy_terminal_at_unix_s = time.time()
                    policy_elapsed_s = policy_terminal_at_s - started_at
                    summary.update(
                        {
                            "policy_started_at_unix_s": policy_started_at_unix_s,
                            "policy_terminal_at_unix_s": policy_terminal_at_unix_s,
                            "policy_elapsed_s": policy_elapsed_s,
                            "completion_time_s": policy_elapsed_s if state.complete else None,
                        }
                    )
                    append_jsonl(
                        out,
                        {
                            "kind": "policy_terminal",
                            "stop_reason": stop_reason,
                            "exact_match": state.complete and state.typed == state.target,
                            "typed": state.typed,
                            "policy_started_at_unix_s": policy_started_at_unix_s,
                            "policy_terminal_at_unix_s": policy_terminal_at_unix_s,
                            "policy_elapsed_s": policy_elapsed_s,
                            "completion_time_s": policy_elapsed_s if state.complete else None,
                        },
                    )
                    cleanup_started_at_s = time.monotonic()
                    try:
                        if args.return_to_rest_s > 0.0:
                            print("Returning to fixed rest before releasing keyboard capture.")
                            smooth_move_arm(
                                robot,
                                rest_q_deg,
                                duration_s=args.return_to_rest_s,
                                control_hz=CONTROL_HZ,
                            )
                            time.sleep(DEFAULT_RELEASE_SETTLE_S)
                        keyboard_release_confirmed = keyboard.drain_until_released(timeout_s=2.0)
                    finally:
                        summary["cleanup_duration_s"] = time.monotonic() - cleanup_started_at_s

        summary.update(
            {
                "status": "completed" if state.complete else "stopped",
                "stop_reason": stop_reason,
                "typed": state.typed,
                "physical_key_downs": "".join(physical_key_downs),
                "exact_match": state.complete and state.typed == state.target,
                "multiple_key_press": press_gate.invalid_overlap,
                "keyboard_release_confirmed": keyboard_release_confirmed,
                "duration_s": time.monotonic() - started_at,
                "wall_duration_s": time.monotonic() - started_at,
                "steps": step,
                "joint_sample_dt_s_mean": float(np.mean(sample_dt_history)) if sample_dt_history else None,
                "joint_sample_dt_s_max": float(np.max(sample_dt_history)) if sample_dt_history else None,
                "pre_log_overrun_count": overrun_count,
            }
        )
        print(f"Physical key-downs: {''.join(physical_key_downs)!r}")
        print(f"Exact match after release and clearance: {summary['exact_match']}")
        return 0 if summary["exact_match"] else 2
    except Exception as exc:
        summary["error"] = str(exc)
        if summary["status"] == "rejected":
            summary["status"] = "error"
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            robot.disconnect()
        summary["timestamp_unix_s"] = time.time()
        append_jsonl(out, {"kind": "summary", **summary})
        print(f"Log: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
