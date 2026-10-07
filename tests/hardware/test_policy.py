# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy observations, command conversion and physical key-event handling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from so101_typing.hardware.fixed_cartesian_policy import (
    ACTION_DIM,
    CLEARANCE_M,
    CONTROL_DT_S,
    FixedCartesianTypingState,
    JOINT_RATE_LIMITS_RAD_S,
    OBSERVATION_DIM,
    PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG,
    PHYSICAL_CALIBRATED_20260718_PROFILE,
    build_observation,
    frozen_target_xyz,
    infer_policy_contract,
    integrate_command,
    lerobot_position_to_profile_model_rad,
    profile_model_rad_to_lerobot_position,
)
from so101_typing.hardware.keyboard_events import (
    CapturedSession,
    KeyEvent,
    _normalize_key_name,
    levenshtein,
    score_typed,
)
from so101_typing.hardware.policy_runtime import (
    KeyboardMonitor,
    PolicyContractError,
    PressGate,
    load_rest_pose,
)
from so101_typing.hardware.run_fixed_cartesian_policy_handoff import (
    _model_goal_to_arm_q_deg,
    _move_to_fixed_reset,
    _move_to_fixed_reset_with_retries,
    _next_reset_tracking_goal,
    _resolve_startup_mode,
    _start_error_deg,
    _validate_rest_pose,
    _validated_training_environment,
)
from so101_typing.hardware.so101_kinematics import GRIPPER_MODEL_MIN_RAD


# Policy observations and joint commands

def _synthetic_state() -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {
        "obs_normalizer._mean": np.empty((1, OBSERVATION_DIM)),
        "obs_normalizer._std": np.empty((1, OBSERVATION_DIM)),
    }
    for prefix, output, input_ in (
        ("mlp.0", 256, OBSERVATION_DIM),
        ("mlp.2", 256, 256),
        ("mlp.4", 128, 256),
        ("mlp.6", 2 * ACTION_DIM, 128),
    ):
        state[f"{prefix}.weight"] = np.empty((output, input_))
        state[f"{prefix}.bias"] = np.empty((output,))
    return state


def test_checkpoint_contract_is_native_beta_22_to_10() -> None:
    contract = infer_policy_contract(_synthetic_state())

    assert contract.observation_dim == 22
    assert contract.action_dim == 5
    assert contract.hidden_dims == (256, 256, 128)
    assert contract.distribution == "BetaDistribution[-1,1]"


def test_observation_order_uses_frozen_xyz_and_persistent_command() -> None:
    state = FixedCartesianTypingState.from_text("H", stage="p0", now_s=10.0)
    profile = PHYSICAL_CALIBRATED_20260718_PROFILE
    q_reset = np.asarray(profile.q_reset_rad)
    q = q_reset + np.arange(5) / 100.0
    qd = np.arange(5) / 10.0
    q_command = q_reset - np.arange(5) / 200.0

    observation = build_observation(
        q_rad=q,
        qd_rad_s=qd,
        q_command_rad=q_command,
        state=state,
        now_s=10.25,
    )

    np.testing.assert_allclose(observation[:5], q - q_reset)
    np.testing.assert_allclose(observation[5:10], qd)
    np.testing.assert_allclose(observation[10:15], q_command - q_reset)
    expected_xyz = (frozen_target_xyz("H") - np.asarray(profile.az_xyz_mean_b_m)) / np.asarray(
        profile.az_xyz_std_b_m
    )
    np.testing.assert_allclose(observation[15:18], expected_xyz)
    np.testing.assert_array_equal(observation[18:21], (1.0, 0.0, 0.0))
    assert observation[21] == pytest.approx(0.25)


def test_persistent_rate_command_accumulates_from_prior_command() -> None:
    q_reset = np.asarray(PHYSICAL_CALIBRATED_20260718_PROFILE.q_reset_rad)
    action = np.ones(5)

    first = integrate_command(q_reset, action)
    second = integrate_command(first, action)
    increment = np.asarray(JOINT_RATE_LIMITS_RAD_S) * CONTROL_DT_S

    np.testing.assert_allclose(first, q_reset + increment)
    np.testing.assert_allclose(second, q_reset + 2.0 * increment)


def test_diagnostic_rate_scale_scales_persistent_increment() -> None:
    q_reset = np.asarray(PHYSICAL_CALIBRATED_20260718_PROFILE.q_reset_rad)
    action = np.ones(5)

    result = integrate_command(q_reset, action, rate_scale=0.5)

    expected_increment = 0.5 * np.asarray(JOINT_RATE_LIMITS_RAD_S) * CONTROL_DT_S
    np.testing.assert_allclose(result, q_reset + expected_increment)
    with pytest.raises(ValueError, match="rate_scale"):
        integrate_command(q_reset, action, rate_scale=0.0)


def test_state_requires_down_up_and_two_clearance_ticks() -> None:
    state = FixedCartesianTypingState.from_text("HE", stage="p1a", now_s=0.0)
    clear_tip_z = float(frozen_target_xyz("H")[2] + CLEARANCE_M)

    assert state.key_down("h", now_s=0.1) == "target_down"
    assert state.active_letter() == "H"
    assert state.key_up("h", now_s=0.2) == "target_up"
    assert state.active_letter() == "H"
    assert state.clearance_tick(tip_z_b_m=clear_tip_z, any_key_held=False, now_s=0.24) is None
    assert state.clearance_tick(tip_z_b_m=clear_tip_z, any_key_held=False, now_s=0.28) == "clearance_complete"
    assert state.active_letter() == "E"
    assert state.complete is False


def test_clearance_counter_resets_and_press_during_clearance_fails() -> None:
    state = FixedCartesianTypingState.from_text("H", stage="p0", now_s=0.0)
    clear_tip_z = float(frozen_target_xyz("H")[2] + CLEARANCE_M)
    state.key_down("h", now_s=0.1)
    state.key_up("h", now_s=0.2)

    assert state.clearance_tick(tip_z_b_m=clear_tip_z, any_key_held=False, now_s=0.24) is None
    assert state.clearance_ticks == 1
    assert state.clearance_tick(tip_z_b_m=clear_tip_z, any_key_held=True, now_s=0.28) is None
    assert state.clearance_ticks == 0
    assert state.key_down("e", now_s=0.29) == "press_before_clearance"


def test_state_uses_checkpoint_specific_clearance_contract() -> None:
    clearance_m = 0.015
    state = FixedCartesianTypingState.from_text(
        "H",
        stage="p0",
        now_s=0.0,
        clearance_m=clearance_m,
        clearance_control_ticks=3,
    )
    target_z = float(frozen_target_xyz("H")[2])
    state.key_down("h", now_s=0.1)
    state.key_up("h", now_s=0.2)

    assert state.clearance_tick(tip_z_b_m=target_z + CLEARANCE_M, any_key_held=False, now_s=0.24) is None
    assert state.clearance_ticks == 0
    for tick in range(2):
        assert (
            state.clearance_tick(
                tip_z_b_m=target_z + clearance_m,
                any_key_held=False,
                now_s=0.28 + tick * CONTROL_DT_S,
            )
            is None
        )
    assert (
        state.clearance_tick(
            tip_z_b_m=target_z + clearance_m,
            any_key_held=False,
            now_s=0.36,
        )
        == "target_complete"
    )


@pytest.mark.parametrize(
    ("clearance_m", "clearance_control_ticks"),
    ((0.0, 2), (float("nan"), 2), (0.004, 0)),
)
def test_state_rejects_invalid_clearance_contract(clearance_m: float, clearance_control_ticks: int) -> None:
    with pytest.raises(ValueError, match="clearance"):
        FixedCartesianTypingState.from_text(
            "H",
            stage="p0",
            clearance_m=clearance_m,
            clearance_control_ticks=clearance_control_ticks,
        )


def test_stage_target_lengths_are_strict() -> None:
    with pytest.raises(ValueError, match="exactly 1"):
        FixedCartesianTypingState.from_text("HE", stage="p0")
    with pytest.raises(ValueError, match="exactly 2"):
        FixedCartesianTypingState.from_text("H", stage="p1a")
    assert FixedCartesianTypingState.from_text("HEY", stage="p1b").target == "HEY"
    assert FixedCartesianTypingState.from_text("NVDA", stage="p1c").target == "NVDA"
    assert FixedCartesianTypingState.from_text("NVIDIA", stage="p1d").target == "NVIDIA"
    with pytest.raises(ValueError, match="exactly 6"):
        FixedCartesianTypingState.from_text("NVIDI", stage="p1d")


def test_physical_calibrated_rest_pose_converts_to_model_reset() -> None:
    profile = PHYSICAL_CALIBRATED_20260718_PROFILE
    encoder_rest_deg = np.rad2deg(np.asarray(profile.q_reset_rad)) - np.asarray(
        PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG
    )

    _validate_rest_pose(encoder_rest_deg, profile)
    model_q = lerobot_position_to_profile_model_rad(np.concatenate((encoder_rest_deg, np.asarray([0.0]))), profile)

    np.testing.assert_allclose(model_q[:5], profile.q_reset_rad, atol=1.0e-12, rtol=0.0)
    assert model_q[5] == pytest.approx(GRIPPER_MODEL_MIN_RAD)
    assert _start_error_deg(model_q, profile) == pytest.approx(0.0, abs=1.0e-12)


def test_physical_calibrated_startup_requires_fixed_reset() -> None:
    profile = PHYSICAL_CALIBRATED_20260718_PROFILE

    assert _resolve_startup_mode(argparse.Namespace(startup_mode="auto"), profile) == "fixed-reset"
    assert _resolve_startup_mode(argparse.Namespace(startup_mode="fixed-reset"), profile) == "fixed-reset"
    with pytest.raises(PolicyContractError, match="cannot use recorded F/Y/N"):
        _resolve_startup_mode(argparse.Namespace(startup_mode="recorded-fyn"), profile)


def test_fixed_reset_tracking_correction_is_bounded() -> None:
    goal = np.zeros(5)
    target = np.asarray([0.0, 2.0, -3.0, 0.5, -0.5])
    measured = np.asarray([0.0, -4.0, 4.0, 0.0, 0.0])

    updated = _next_reset_tracking_goal(
        goal_q_deg=goal,
        target_q_deg=target,
        measured_q_deg=measured,
    )

    np.testing.assert_allclose(updated, np.asarray([0.0, 1.0, -1.0, 0.25, -0.25]))


def test_fixed_reset_records_initial_and_final_error(monkeypatch: pytest.MonkeyPatch) -> None:
    initial = np.asarray([3.0, -2.0, 1.0, 0.0, -0.5])
    target = np.zeros(5)
    readings = iter((initial, target))
    moves: list[np.ndarray] = []

    monkeypatch.setattr(
        "so101_typing.hardware.run_fixed_cartesian_policy_handoff.read_arm_q_deg",
        lambda _robot: next(readings),
    )
    monkeypatch.setattr(
        "so101_typing.hardware.run_fixed_cartesian_policy_handoff.smooth_move_arm",
        lambda _robot, q_deg, **_kwargs: moves.append(np.asarray(q_deg)),
    )

    report = _move_to_fixed_reset(
        object(),
        target_q_deg=target,
        transition_s=3.0,
        epsilon_deg=0.5,
        settle_timeout_s=3.0,
    )

    np.testing.assert_allclose(moves, [target])
    np.testing.assert_allclose(report["initial_measured_q_deg"], initial)
    assert report["initial_joint_error_deg"] == pytest.approx(3.0)
    assert report["joint_error_deg"] == pytest.approx(0.0)
    assert report["converged"] is True


def test_fixed_reset_retries_until_model_handoff_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = iter(
        (
            {"mode": "fixed-reset", "joint_error_deg": 0.8, "converged": False},
            {"mode": "fixed-reset", "joint_error_deg": 0.2, "converged": True},
            {"mode": "fixed-reset", "joint_error_deg": 0.1, "converged": True},
        )
    )
    reset_q = np.asarray((*PHYSICAL_CALIBRATED_20260718_PROFILE.q_reset_rad, GRIPPER_MODEL_MIN_RAD))
    off_reset_q = reset_q.copy()
    off_reset_q[0] += np.deg2rad(1.0)
    policy_start_readings = iter((off_reset_q, reset_q))
    monkeypatch.setattr(
        "so101_typing.hardware.run_fixed_cartesian_policy_handoff._move_to_fixed_reset",
        lambda *_args, **_kwargs: next(attempts),
    )
    monkeypatch.setattr(
        "so101_typing.hardware.run_fixed_cartesian_policy_handoff._read_closed_typing_q_rad",
        lambda *_args, **_kwargs: next(policy_start_readings),
    )

    report, policy_start_q = _move_to_fixed_reset_with_retries(
        object(),
        target_q_deg=np.zeros(5),
        profile=PHYSICAL_CALIBRATED_20260718_PROFILE,
        transition_s=3.0,
        epsilon_deg=0.5,
        settle_timeout_s=3.0,
        max_attempts=3,
    )

    assert report["converged"] is True
    assert report["attempt_count"] == 3
    assert report["max_attempts"] == 3
    assert [attempt["reset_converged"] for attempt in report["attempts"]] == [False, True, True]
    assert [attempt["converged"] for attempt in report["attempts"]] == [False, False, True]
    assert report["attempts"][1]["policy_start_joint_error_deg"] == pytest.approx(1.0)
    np.testing.assert_allclose(policy_start_q, reset_q)


def test_physical_calibrated_model_lerobot_conversion_round_trips() -> None:
    profile = PHYSICAL_CALIBRATED_20260718_PROFILE
    model_goal = np.asarray((*profile.q_reset_rad, GRIPPER_MODEL_MIN_RAD), dtype=np.float64)

    lerobot_goal = profile_model_rad_to_lerobot_position(model_goal, profile)
    reconstructed = lerobot_position_to_profile_model_rad(lerobot_goal, profile)

    np.testing.assert_allclose(reconstructed, model_goal, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(
        _model_goal_to_arm_q_deg(model_goal, profile),
        np.rad2deg(np.asarray(profile.q_reset_rad)) - np.asarray(profile.joint_zero_offset_deg),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_physical_calibrated_observation_uses_model_frame_and_calibrated_xyz() -> None:
    profile = PHYSICAL_CALIBRATED_20260718_PROFILE
    state = FixedCartesianTypingState.from_text("H", stage="p0", profile=profile, now_s=2.0)
    q_reset = np.asarray(profile.q_reset_rad)

    observation = build_observation(
        q_rad=q_reset,
        qd_rad_s=np.zeros(5),
        q_command_rad=q_reset,
        state=state,
        now_s=2.0,
    )

    np.testing.assert_array_equal(observation[:15], np.zeros(15))
    expected_xyz = (frozen_target_xyz("H", profile) - np.asarray(profile.az_xyz_mean_b_m)) / np.asarray(
        profile.az_xyz_std_b_m
    )
    np.testing.assert_allclose(observation[15:18], expected_xyz)


def test_physical_calibrated_checkpoint_requires_matching_training_environment(tmp_path: Path) -> None:
    profile = PHYSICAL_CALIBRATED_20260718_PROFILE
    checkpoint = tmp_path / "model_19999.pt"
    checkpoint.touch()
    env_config = tmp_path / "params/env.yaml"
    env_config.parent.mkdir()
    env_config.write_text(
        "\n".join(
            (
                "episode_length_s: 10.0",
                "commands:",
                "  typing:",
                "    letter_length: [1, 1]",
                "    clearance_m: 0.004",
                "    clearance_control_ticks: 2",
                f"keyboard_profile: {profile.calibration_id}",
                "actuator_profile: anchorbench",
                "task_contract: physical_calibrated_20260718_fixed_cartesian_anchorbench_p0_v0",
                f"target_reference_sha256: {profile.target_reference_sha256}",
                f"target_manifest_sha256: {profile.target_manifest_sha256}",
            )
        )
        + "\n"
    )

    report = _validated_training_environment(
        checkpoint=checkpoint,
        env_config=env_config,
        profile=profile,
        stage="p0",
    )

    assert report is not None
    assert report["validated"] is True
    assert report["task_contract"].endswith("_p0_v0")
    assert report["episode_length_s"] == pytest.approx(10.0)
    assert report["clearance_m"] == pytest.approx(0.004)
    assert report["clearance_control_ticks"] == 2

    env_config.write_text(env_config.read_text().replace(profile.calibration_id, "wrong_keyboard"))
    with pytest.raises(PolicyContractError, match="does not match"):
        _validated_training_environment(
            checkpoint=checkpoint,
            env_config=env_config,
            profile=profile,
            stage="p0",
        )


def test_training_environment_recognizes_transit15_clearance(tmp_path: Path) -> None:
    profile = PHYSICAL_CALIBRATED_20260718_PROFILE
    checkpoint = tmp_path / "model_19999.pt"
    checkpoint.touch()
    env_config = tmp_path / "params/env.yaml"
    env_config.parent.mkdir()
    env_config.write_text(
        "\n".join(
            (
                "episode_length_s: 61.0",
                "commands:",
                "  typing:",
                "    letter_length: [6, 6]",
                "    clearance_m: 0.015",
                "    clearance_control_ticks: 2",
                f"keyboard_profile: {profile.calibration_id}",
                "actuator_profile: anchorbench",
                "task_contract: physical_calibrated_20260718_fixed_cartesian_anchorbench_p1d_transit15_letters6_v0",
                f"target_reference_sha256: {profile.target_reference_sha256}",
                f"target_manifest_sha256: {profile.target_manifest_sha256}",
            )
        )
        + "\n"
    )

    report = _validated_training_environment(
        checkpoint=checkpoint,
        env_config=env_config,
        profile=profile,
        stage="p1d",
    )

    assert report is not None
    assert report["task_contract"].endswith("p1d_transit15_letters6_v0")
    assert report["clearance_m"] == pytest.approx(0.015)
    assert report["episode_length_s"] == pytest.approx(61.0)


@pytest.mark.parametrize(
    ("actuator_profile", "task_contract"),
    (
        (
            "baseline",
            "physical_calibrated_20260718_fixed_cartesian_baseline_p1d_transit15_letters6_v0",
        ),
        (
            "usd_drive",
            "physical_calibrated_20260718_fixed_cartesian_usd_drive_p1d_transit15_letters6_v0",
        ),
    ),
)
def test_training_environment_recognizes_matched_actuator_profile(
    tmp_path: Path, actuator_profile: str, task_contract: str
) -> None:
    profile = PHYSICAL_CALIBRATED_20260718_PROFILE
    checkpoint = tmp_path / "model_19999.pt"
    checkpoint.touch()
    env_config = tmp_path / "params/env.yaml"
    env_config.parent.mkdir()
    env_config.write_text(
        "\n".join(
            (
                "episode_length_s: 61.0",
                "commands:",
                "  typing:",
                "    letter_length: [6, 6]",
                "    clearance_m: 0.015",
                "    clearance_control_ticks: 2",
                f"keyboard_profile: {profile.calibration_id}",
                f"actuator_profile: {actuator_profile}",
                f"task_contract: {task_contract}",
                f"target_reference_sha256: {profile.target_reference_sha256}",
                f"target_manifest_sha256: {profile.target_manifest_sha256}",
            )
        )
        + "\n"
    )

    report = _validated_training_environment(
        checkpoint=checkpoint,
        env_config=env_config,
        profile=profile,
        stage="p1d",
        expected_actuator_profile=actuator_profile,
        expected_task_contract=task_contract,
    )

    assert report is not None
    assert report["actuator_profile"] == actuator_profile
    assert report["task_contract"] == task_contract
    assert report["clearance_m"] == pytest.approx(0.015)


# Keyboard event capture and scoring

def test_score_exact_phrase() -> None:
    score = score_typed("newton is the best", "newton is the best")

    assert score["exact_match"] is True
    assert score["position_accuracy"] == 1.0
    assert score["levenshtein"] == 0


def test_score_empty_capture_is_failure() -> None:
    score = score_typed("", "newton is the best")

    assert score["exact_match"] is False
    assert score["correct_chars"] == 0
    assert score["levenshtein"] == 18


def test_levenshtein() -> None:
    assert levenshtein("newton is best", "newton is the best") == 4


def test_captured_session_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "typed.json"
    session = CapturedSession(
        events=[
            KeyEvent(char="n", code=49, key_name="KEY_N", t_press=1.0, t_release=1.1),
            KeyEvent(char="e", code=18, key_name="KEY_E", t_press=1.2, t_release=1.3),
        ],
        device_path="/dev/input/event99",
        device_name="test keyboard",
        stop_reason="stop_key:KEY_ESC",
        t_start_monotonic=10.0,
        t_end_monotonic=11.0,
    )

    session.write(path)
    loaded = CapturedSession.load(path)

    assert loaded.typed_string() == "ne"
    assert loaded.device_path == "/dev/input/event99"
    assert loaded.device_name == "test keyboard"
    assert loaded.stop_reason == "stop_key:KEY_ESC"


def test_normalize_stop_key_name() -> None:
    assert _normalize_key_name("esc") == "KEY_ESC"
    assert _normalize_key_name("KEY_F12") == "KEY_F12"


# Press, release and clearance handling

def test_press_gate_rejects_overlapping_key_down_edges() -> None:
    gate = PressGate()
    assert gate.process({"code": 30, "value": 1}) is False
    assert gate.process({"code": 30, "value": 0}) is False
    assert gate.process({"code": 31, "value": 1}) is False
    assert gate.process({"code": 32, "value": 1}) is True
    assert gate.invalid_overlap is True


def test_keyboard_monitor_tracks_pressed_key_edges() -> None:
    monitor = KeyboardMonitor.__new__(KeyboardMonitor)
    monitor.pressed_codes = set()

    monitor._record_key_state(code=33, value=1)
    assert monitor.pressed_codes == {33}
    monitor._record_key_state(code=33, value=0)

    assert monitor.pressed_codes == set()


def test_keyboard_monitor_drains_pressed_keys_before_release() -> None:
    monitor = KeyboardMonitor.__new__(KeyboardMonitor)
    monitor.pressed_codes = {33}

    def release_key() -> tuple[list[dict[str, object]], bool]:
        monitor._record_key_state(code=33, value=0)
        return [], False

    monitor.poll = release_key

    assert monitor.drain_until_released(timeout_s=0.01) is True


def test_rest_pose_requires_expected_joint_order_and_no_contact(tmp_path: Path) -> None:
    rest_path = tmp_path / "rest_pose.json"
    rest_path.write_text(
        json.dumps(
            {
                "kind": "so101_policy_rest_pose",
                "schema_version": 1,
                "arm_joint_names": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
                "closed_fixed_jaw_required": True,
                "contact_required": False,
                "pose_deg": {
                    "shoulder_pan": -4.0,
                    "shoulder_lift": -9.0,
                    "elbow_flex": 12.0,
                    "wrist_flex": 65.0,
                    "wrist_roll": 2.0,
                },
            }
        )
    )

    np.testing.assert_allclose(load_rest_pose(rest_path), [-4.0, -9.0, 12.0, 65.0, 2.0])

    invalid = json.loads(rest_path.read_text())
    invalid["contact_required"] = True
    rest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="closed fixed jaw"):
        load_rest_pose(rest_path)
