# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract gates for the 2026-07-18 powered-contact calibrated PPO tasks."""

import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from isaaclab_tasks.core.dexsuite.config.so101.agents.rsl_rl_ppo_cfg import (
    BoundedBetaDistributionCfg,
    SO101PhysicalCalibrated20260718FixedCartesianP0PPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg,
)
from isaaclab_tasks.core.dexsuite.config.so101.fixed_cartesian_targets import (
    FROZEN_LETTER_SLOTS,
    FROZEN_LETTERS,
)
from isaaclab_tasks.core.dexsuite.config.so101.physical_calibrated_20260718_env_cfg import (
    PHYSICAL_CALIBRATED_AZ_XYZ_B_M,
    PHYSICAL_CALIBRATED_AZ_XYZ_MEAN_B_M,
    PHYSICAL_CALIBRATED_AZ_XYZ_STD_B_M,
    PHYSICAL_CALIBRATED_KEYBOARD_POSITION_B_M,
    PHYSICAL_CALIBRATED_KEYBOARD_ROTATION_XYZW,
    PHYSICAL_CALIBRATED_POLICY_Q_RESET_RAD,
    GTC_BERLIN_CAMERA_TOP_WRIST_ROLL_RAD,
    PHYSICAL_CALIBRATED_POLICY_REST_JOINT_POS_RAD,
    PHYSICAL_CALIBRATED_TARGET_MAP_SHA256,
    PHYSICAL_CALIBRATION_DATASET_SHA256,
    PUBLIC_CALIBRATION_DATASET_SHA256,
    PHYSICAL_CALIBRATION_TRACE_ARCHIVE_SHA256,
    PHYSICAL_TO_MODEL_JOINT_ZERO_DEG,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP0EnvCfg,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg,
)

from so101_typing.hardware.fixed_cartesian_policy import PHYSICAL_CALIBRATED_20260718_PROFILE


def test_dated_tasks_are_registered_without_old_namespaces() -> None:
    p0 = gym.spec("Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P0")
    p1 = gym.spec("Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1A")

    assert "PhysicalCalibrated20260718FixedCartesianP0" in p0.kwargs["env_cfg_entry_point"]
    assert "PhysicalCalibrated20260718FixedCartesianP1A" in p1.kwargs["env_cfg_entry_point"]
    assert "PhysicalCalibrated20260718FixedCartesianP0" in p0.kwargs["rsl_rl_cfg_entry_point"]
    assert "PhysicalCalibrated20260718FixedCartesianP1A" in p1.kwargs["rsl_rl_cfg_entry_point"]

    legacy_task_ids = {
        "Isaac-Keyboard-SO101",
        "Isaac-Keyboard-SO101-Play",
        "Isaac-Keyboard-SO101-IK-Abs",
        "Isaac-Keyboard-SO101-IK-Rel",
        "Isaac-Keyboard-SO101-Octi-MX-Registered-AnchorBench",
        "Isaac-Keyboard-SO101-Octi-MX-Registered-Baseline",
        "Isaac-Keyboard-SO101-Octi-MX-Registered-AnchorBench-Eval",
        "Isaac-Keyboard-SO101-Octi-MX-Registered-Baseline-Eval",
    }
    assert legacy_task_ids.isdisjoint(gym.registry)


def test_usd_drive_tasks_use_distinct_registered_contracts() -> None:
    p1a = gym.spec("Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-USDDrive-P1A")
    transit15 = gym.spec("Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-USDDrive-P1D-Transit15")

    assert "USDDriveP1AEnvCfg" in p1a.kwargs["env_cfg_entry_point"]
    assert "USDDriveP1APPORunnerCfg" in p1a.kwargs["rsl_rl_cfg_entry_point"]
    assert "USDDriveP1DTransit15EnvCfg" in transit15.kwargs["env_cfg_entry_point"]
    assert "USDDriveP1DTransit15PPORunnerCfg" in transit15.kwargs["rsl_rl_cfg_entry_point"]


def test_calibrated_p0_freezes_scene_reset_and_target_map() -> None:
    cfg = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP0EnvCfg()

    assert cfg.keyboard_profile == "mx_keys_powered_az_20260718"
    assert cfg.actuator_profile == "anchorbench"
    assert cfg.target_reference_sha256 == PHYSICAL_CALIBRATED_TARGET_MAP_SHA256
    assert cfg.target_manifest_sha256 == PHYSICAL_CALIBRATION_DATASET_SHA256
    assert cfg.scene.keyboard.newton_mjwarp.init_state.pos == PHYSICAL_CALIBRATED_KEYBOARD_POSITION_B_M
    assert cfg.scene.keyboard.newton_mjwarp.init_state.rot == PHYSICAL_CALIBRATED_KEYBOARD_ROTATION_XYZW
    assert cfg.scene.robot.init_state.joint_pos == PHYSICAL_CALIBRATED_POLICY_REST_JOINT_POS_RAD
    assert cfg.actions.action.q_reset_ref == PHYSICAL_CALIBRATED_POLICY_Q_RESET_RAD
    assert cfg.commands.typing.q_reset_ref == PHYSICAL_CALIBRATED_POLICY_Q_RESET_RAD
    assert cfg.commands.typing.letter_slots == FROZEN_LETTER_SLOTS
    assert cfg.commands.typing.letter_xyz_b_m == PHYSICAL_CALIBRATED_AZ_XYZ_B_M
    assert cfg.commands.typing.target_xyz_mean_b_m == PHYSICAL_CALIBRATED_AZ_XYZ_MEAN_B_M
    assert cfg.commands.typing.target_xyz_std_b_m == PHYSICAL_CALIBRATED_AZ_XYZ_STD_B_M
    assert cfg.rest_joint_noise_rad == 0.0
    assert all(bounds == [0.0, 0.0] for bounds in cfg.events.reset_keyboard.params["pose_range"].values())


def test_physical_runner_matches_the_calibrated_simulator_contract() -> None:
    profile = PHYSICAL_CALIBRATED_20260718_PROFILE

    assert profile.calibration_id == "mx_keys_powered_az_20260718"
    assert profile.q_reset_rad == PHYSICAL_CALIBRATED_POLICY_Q_RESET_RAD
    assert profile.az_xyz_b_m == PHYSICAL_CALIBRATED_AZ_XYZ_B_M
    assert profile.az_xyz_mean_b_m == PHYSICAL_CALIBRATED_AZ_XYZ_MEAN_B_M
    assert profile.az_xyz_std_b_m == PHYSICAL_CALIBRATED_AZ_XYZ_STD_B_M
    assert profile.target_reference_sha256 == PHYSICAL_CALIBRATED_TARGET_MAP_SHA256
    assert profile.target_manifest_sha256 == PHYSICAL_CALIBRATION_DATASET_SHA256
    assert profile.joint_zero_offset_deg == tuple(
        PHYSICAL_TO_MODEL_JOINT_ZERO_DEG[name]
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    )


def test_calibrated_target_table_statistics_and_hash_are_frozen() -> None:
    xyz = np.asarray(PHYSICAL_CALIBRATED_AZ_XYZ_B_M, dtype=np.float64)
    payload = {
        "letters": "".join(FROZEN_LETTERS),
        "slots": FROZEN_LETTER_SLOTS,
        "xyz_b_m": xyz.tolist(),
        "keyboard_position_b_m": PHYSICAL_CALIBRATED_KEYBOARD_POSITION_B_M,
        "keyboard_rotation_xyzw": PHYSICAL_CALIBRATED_KEYBOARD_ROTATION_XYZW,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    assert xyz.shape == (26, 3)
    assert np.allclose(xyz.mean(axis=0), PHYSICAL_CALIBRATED_AZ_XYZ_MEAN_B_M, atol=1.0e-15)
    assert np.allclose(xyz.std(axis=0), PHYSICAL_CALIBRATED_AZ_XYZ_STD_B_M, atol=1.0e-15)
    assert digest == PHYSICAL_CALIBRATED_TARGET_MAP_SHA256


def test_powered_contact_source_dataset_is_committed_and_hash_locked() -> None:
    dataset = (
        Path(__file__).parents[2]
        / "source/so101_typing/tasks/keyboards/data/mx_keys"
        / "powered_az_repeated_contact_dataset_20260718.json"
    )

    assert dataset.is_file()
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == PUBLIC_CALIBRATION_DATASET_SHA256
    data = json.loads(dataset.read_text())
    assert data["public_provenance"]["original_dataset_sha256"] == PHYSICAL_CALIBRATION_DATASET_SHA256


def test_powered_contact_trace_archive_is_complete_and_hash_locked() -> None:
    archive_path = (
        Path(__file__).parents[2]
        / "source/so101_typing/tasks/keyboards/data/mx_keys"
        / "powered_az_probe_traces_20260718.npz"
    )
    expected_members = {f"rep_{repetition:02d}_{letter}" for repetition in range(1, 4) for letter in FROZEN_LETTERS}

    assert archive_path.is_file()
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == PHYSICAL_CALIBRATION_TRACE_ARCHIVE_SHA256
    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == expected_members
        assert all(archive[member].ndim == 2 and archive[member].shape[1] == 5 for member in archive.files)
        assert all(np.isfinite(archive[member]).all() for member in archive.files)


def test_p1a_changes_only_sequence_budget_from_calibrated_p0() -> None:
    p0 = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP0EnvCfg()
    p1 = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg()
    p0_runner = SO101PhysicalCalibrated20260718FixedCartesianP0PPORunnerCfg().default
    p1_runner = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default

    assert p0.commands.typing.letter_length == (1, 1)
    assert p0.episode_length_s == 10.0
    assert p1.commands.typing.letter_length == (2, 2)
    assert p1.episode_length_s == 21.0
    assert p1.commands.typing.letter_xyz_b_m == p0.commands.typing.letter_xyz_b_m
    assert p1.actions.action.q_reset_ref == p0.actions.action.q_reset_ref
    assert p1.actions.action.velocity_limits_rad_s == p0.actions.action.velocity_limits_rad_s
    assert p0_runner.experiment_name == "so101_physical_calibrated_20260718_fixed_cartesian_p0"
    assert p1_runner.experiment_name == "so101_physical_calibrated_20260718_fixed_cartesian_p1a_letters2"
    assert p0_runner.max_iterations == p1_runner.max_iterations == 20000
    assert p0_runner.num_steps_per_env == p1_runner.num_steps_per_env == 64
    assert p0_runner.clip_actions is p1_runner.clip_actions is None
    assert isinstance(p0_runner.actor.distribution_cfg, BoundedBetaDistributionCfg)
    assert isinstance(p1_runner.actor.distribution_cfg, BoundedBetaDistributionCfg)


def test_gtc_berlin_tasks_are_registered_with_distinct_contracts() -> None:
    from isaaclab_tasks.core.dexsuite.config.so101.physical_calibrated_20260718_env_cfg import (
        SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianGTCBerlinP1AEnvCfg,
        SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianGTCBerlinP1DTransit15EnvCfg,
    )
    from isaaclab_tasks.core.dexsuite.config.so101.SO101.actuator_profiles import actuator_profile_manifest

    for stage, config_type in (
        ("P1A", SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianGTCBerlinP1AEnvCfg),
        ("P1D-Transit15", SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianGTCBerlinP1DTransit15EnvCfg),
    ):
        spec = gym.spec(f"Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-GTCBerlin-{stage}")
        cfg = config_type()
        profile = actuator_profile_manifest("gtc_berlin")
        assert "GTCBerlin" in spec.kwargs["env_cfg_entry_point"]
        assert "GTCBerlin" in spec.kwargs["rsl_rl_cfg_entry_point"]
        assert cfg.actuator_profile == "gtc_berlin"
        assert cfg.scene.robot.actuators["all"].stiffness == profile["stiffness"]
        assert cfg.scene.robot.actuators["all"].friction == profile["friction"]
        assert "gtc_berlin" in cfg.task_contract
        assert "camera_top_v1" in cfg.task_contract
        expected_roll = PHYSICAL_CALIBRATED_POLICY_REST_JOINT_POS_RAD["wrist_roll"] - math.pi / 2
        assert math.isclose(GTC_BERLIN_CAMERA_TOP_WRIST_ROLL_RAD, expected_roll, abs_tol=1e-12)
        assert cfg.scene.robot.init_state.joint_pos["wrist_roll"] == expected_roll
        roll_index = tuple(PHYSICAL_CALIBRATED_POLICY_REST_JOINT_POS_RAD).index("wrist_roll")
        assert cfg.actions.action.q_reset_ref[roll_index] == expected_roll
        assert cfg.commands.typing.q_reset_ref[roll_index] == expected_roll
