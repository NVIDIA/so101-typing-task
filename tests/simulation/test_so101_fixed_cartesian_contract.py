# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fast contract gates for fixed-layout AnchorBench typing."""

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab_tasks.core.dexsuite.config.so101 import mdp
from isaaclab_tasks.core.dexsuite.config.so101.agents.rsl_rl_ppo_cfg import (
    BoundedBetaDistributionCfg,
    SO101FixedCartesianAnchorBenchP0PPORunnerCfg,
    SO101FixedCartesianAnchorBenchP1APPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianBaselineP1APPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianBaselineP1DTransit15PPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianP1BPPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianP1CPPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianP1DPPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianUSDDriveP1APPORunnerCfg,
    SO101PhysicalCalibrated20260718FixedCartesianUSDDriveP1DTransit15PPORunnerCfg,
)
from isaaclab_tasks.core.dexsuite.config.so101.fixed_cartesian_env_cfg import (
    ARM_JOINT_NAMES,
    FIXED_JAW_TIP_OFFSET_M,
    JOINT_RATE_LIMITS_RAD_S,
    JOINT_RATE_SOURCE_SHA256,
    SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg,
    SO101KeyboardMXFixedCartesianAnchorBenchP1AEnvCfg,
)
from isaaclab_tasks.core.dexsuite.config.so101.fixed_cartesian_targets import (
    FROZEN_AZ_XYZ_B_M,
    FROZEN_LETTER_SLOTS,
)
from isaaclab_tasks.core.dexsuite.config.so101.mdp.commands.fixed_cartesian_typing_commands import (
    CLEARANCE,
    RELEASE_KEY,
    SEEK_PRESS,
    FixedCartesianTypingCommand,
)
from isaaclab_tasks.core.dexsuite.config.so101.mdp.joint_rate_actions import JointRatePositionAction
from isaaclab_tasks.core.dexsuite.config.so101.physical_calibrated_20260718_env_cfg import (
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianBaselineP1AEnvCfg,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianBaselineP1DTransit15EnvCfg,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1BEnvCfg,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1CEnvCfg,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DEnvCfg,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DTransit15EnvCfg,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianUSDDriveP1AEnvCfg,
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianUSDDriveP1DTransit15EnvCfg,
)
from isaaclab_tasks.core.dexsuite.config.so101.registered_mx_env_cfg import REGISTERED_LETTER_SLOTS

_WORKSHOP_STIFFNESS = {
    "shoulder_pan": 55.0,
    "shoulder_lift": 30.0,
    "elbow_flex": 25.0,
    "wrist_flex": 12.0,
    "wrist_roll": 7.0,
    "gripper": 4.0,
}
_WORKSHOP_DAMPING = {
    "shoulder_pan": 0.7,
    "shoulder_lift": 0.8,
    "elbow_flex": 0.7,
    "wrist_flex": 0.5,
    "wrist_roll": 0.5,
    "gripper": 0.3,
}


def _assert_original_workshop_actuator(actuator) -> None:
    assert actuator.effort_limit_sim == 30.0
    assert actuator.velocity_limit_sim is None
    assert actuator.stiffness == _WORKSHOP_STIFFNESS
    assert actuator.damping == _WORKSHOP_DAMPING
    for field in ("armature", "friction", "dynamic_friction", "viscous_friction"):
        assert getattr(actuator, field) is None


def _assert_loaded_usd_actuator(actuator) -> None:
    joint_names = set(_WORKSHOP_STIFFNESS)
    assert actuator.effort_limit_sim == 10.0
    assert actuator.velocity_limit_sim == 10.0
    assert actuator.stiffness == {name: 100.0 for name in joint_names}
    assert actuator.damping == {name: 1.0 for name in joint_names}
    assert 17.8 not in actuator.stiffness.values()
    for field in ("armature", "friction", "dynamic_friction", "viscous_friction"):
        assert getattr(actuator, field) is None


def _assert_matching_runner_recipe(candidate, reference) -> None:
    assert candidate.num_steps_per_env == reference.num_steps_per_env
    assert candidate.max_iterations == reference.max_iterations
    assert candidate.save_interval == reference.save_interval
    assert candidate.obs_groups == reference.obs_groups
    assert candidate.actor.class_name == reference.actor.class_name
    assert candidate.actor.hidden_dims == reference.actor.hidden_dims
    assert candidate.actor.activation == reference.actor.activation
    assert candidate.actor.distribution_cfg.class_name == reference.actor.distribution_cfg.class_name
    assert candidate.actor.distribution_cfg.action_range == reference.actor.distribution_cfg.action_range
    assert candidate.critic.class_name == reference.critic.class_name
    assert candidate.critic.hidden_dims == reference.critic.hidden_dims
    assert candidate.critic.activation == reference.critic.activation
    assert candidate.algorithm.class_name == reference.algorithm.class_name
    assert candidate.algorithm.learning_rate == reference.algorithm.learning_rate
    assert candidate.algorithm.num_learning_epochs == reference.algorithm.num_learning_epochs
    assert candidate.algorithm.num_mini_batches == reference.algorithm.num_mini_batches
    assert candidate.algorithm.gamma == reference.algorithm.gamma
    assert candidate.algorithm.lam == reference.algorithm.lam
    assert candidate.algorithm.clip_param == reference.algorithm.clip_param
    assert candidate.clip_actions == reference.clip_actions


def test_contact_probe_keeps_measured_newton_compliance_bounded() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    probe_source = (repo_root / "scripts/reinforcement_learning/rsl_rl/probe_so101_fixed_cartesian.py").read_text()
    assert '"--contact-max-depression-fraction"' in probe_source
    assert "default=1.30" in probe_source
    command_source = (
        repo_root / "source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/mdp/commands/"
        "fixed_cartesian_typing_commands.py"
    ).read_text()
    assert 'metrics["key/max_target_depression_fraction"]' in command_source


def test_fixed_cartesian_base_contract_uses_anchorbench_profile() -> None:
    cfg = SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg()

    assert cfg.actuator_profile == "anchorbench"
    assert cfg.scene.robot.actuators["all"].stiffness is not None
    assert cfg.rest_joint_noise_rad == 0.0
    assert cfg.events.reset_keyboard.params["pose_range"] == {
        "x": [0.0, 0.0],
        "y": [0.0, 0.0],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [0.0, 0.0],
    }


def test_p1a_is_exactly_two_letters_with_the_frozen_p0_contract() -> None:
    p0 = SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg()
    p1 = SO101KeyboardMXFixedCartesianAnchorBenchP1AEnvCfg()
    p0_runner = SO101FixedCartesianAnchorBenchP0PPORunnerCfg().default
    p1_runner = SO101FixedCartesianAnchorBenchP1APPORunnerCfg().default

    assert p1.actuator_profile == "anchorbench"
    assert p1.task_contract == "fixed_cartesian_anchorbench_p1a_letters2_v0"
    assert p1.commands.typing.letter_length == (2, 2)
    assert p1.commands.typing.max_len == 3
    assert p1.episode_length_s == 21.0
    assert p1.commands.typing.target_slots == p0.commands.typing.target_slots
    assert p1.actions.action.velocity_limits_rad_s == p0.actions.action.velocity_limits_rad_s
    assert p1.actions.action.q_reset_ref == p0.actions.action.q_reset_ref
    assert (
        tuple(p1.observations.policy.__class__.__annotations__)[-6:]
        == tuple(p0.observations.policy.__class__.__annotations__)[-6:]
    )
    assert p1_runner.num_steps_per_env == p0_runner.num_steps_per_env == 64
    assert p1_runner.max_iterations == 20000
    assert p1_runner.actor.hidden_dims == p0_runner.actor.hidden_dims
    assert isinstance(p1_runner.actor.distribution_cfg, BoundedBetaDistributionCfg)
    assert p1_runner.actor.distribution_cfg.action_range == (-1.0, 1.0)
    assert p1_runner.clip_actions is None


@pytest.mark.parametrize(
    ("stage", "cfg_type", "runner_type", "length", "horizon", "contract"),
    (
        (
            "P1B",
            SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1BEnvCfg,
            SO101PhysicalCalibrated20260718FixedCartesianP1BPPORunnerCfg,
            3,
            31.0,
            "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1b_letters3_v0",
        ),
        (
            "P1C",
            SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1CEnvCfg,
            SO101PhysicalCalibrated20260718FixedCartesianP1CPPORunnerCfg,
            4,
            41.0,
            "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1c_letters4_v0",
        ),
        (
            "P1D",
            SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DEnvCfg,
            SO101PhysicalCalibrated20260718FixedCartesianP1DPPORunnerCfg,
            6,
            61.0,
            "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1d_letters6_v0",
        ),
    ),
)
def test_physical_sequence_scaling_changes_only_length_and_horizon(
    stage: str,
    cfg_type: type,
    runner_type: type,
    length: int,
    horizon: float,
    contract: str,
) -> None:
    spec = gym.spec(f"Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-{stage}")
    p1a = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg()
    candidate = cfg_type()
    p1a_runner = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default
    candidate_runner = runner_type().default

    assert cfg_type.__name__ in spec.kwargs["env_cfg_entry_point"]
    assert candidate.task_contract == contract
    assert candidate.commands.typing.letter_length == (length, length)
    assert candidate.commands.typing.max_len == length
    assert candidate.episode_length_s == horizon
    assert candidate.actuator_profile == p1a.actuator_profile == "anchorbench"
    assert candidate.commands.typing.letter_xyz_b_m == p1a.commands.typing.letter_xyz_b_m
    assert candidate.commands.typing.target_slots == p1a.commands.typing.target_slots
    assert candidate.commands.typing.q_reset_ref == p1a.commands.typing.q_reset_ref
    assert candidate.actions.action == p1a.actions.action
    assert candidate.rewards == p1a.rewards
    assert candidate.terminations == p1a.terminations
    assert candidate.observations == p1a.observations
    assert candidate_runner.num_steps_per_env == p1a_runner.num_steps_per_env == 64
    assert candidate_runner.save_interval == p1a_runner.save_interval == 500
    assert candidate_runner.obs_groups == p1a_runner.obs_groups
    assert candidate_runner.actor.class_name == p1a_runner.actor.class_name
    assert candidate_runner.actor.hidden_dims == p1a_runner.actor.hidden_dims
    assert candidate_runner.actor.activation == p1a_runner.actor.activation
    assert candidate_runner.actor.distribution_cfg.class_name == p1a_runner.actor.distribution_cfg.class_name
    assert candidate_runner.actor.distribution_cfg.action_range == p1a_runner.actor.distribution_cfg.action_range
    assert candidate_runner.critic.class_name == p1a_runner.critic.class_name
    assert candidate_runner.critic.hidden_dims == p1a_runner.critic.hidden_dims
    assert candidate_runner.critic.activation == p1a_runner.critic.activation
    assert candidate_runner.algorithm.class_name == p1a_runner.algorithm.class_name
    assert candidate_runner.algorithm.learning_rate == p1a_runner.algorithm.learning_rate
    assert candidate_runner.algorithm.num_learning_epochs == p1a_runner.algorithm.num_learning_epochs
    assert candidate_runner.algorithm.num_mini_batches == p1a_runner.algorithm.num_mini_batches
    assert candidate_runner.algorithm.gamma == p1a_runner.algorithm.gamma
    assert candidate_runner.algorithm.lam == p1a_runner.algorithm.lam
    assert candidate_runner.algorithm.clip_param == p1a_runner.algorithm.clip_param
    assert candidate_runner.clip_actions is None


def test_p1d_transit15_changes_only_the_clearance_contract() -> None:
    task_id = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1D-Transit15"
    spec = gym.spec(task_id)
    base = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DEnvCfg()
    candidate = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DTransit15EnvCfg()
    base_runner = SO101PhysicalCalibrated20260718FixedCartesianP1DPPORunnerCfg().default
    candidate_runner = SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg().default

    assert candidate.__class__.__name__ in spec.kwargs["env_cfg_entry_point"]
    assert candidate.task_contract.endswith("p1d_transit15_letters6_v0")
    assert candidate.commands.typing.letter_length == base.commands.typing.letter_length == (6, 6)
    assert candidate.commands.typing.max_len == base.commands.typing.max_len == 6
    assert candidate.episode_length_s == base.episode_length_s == 61.0
    assert base.commands.typing.clearance_m == pytest.approx(0.004)
    assert candidate.commands.typing.clearance_m == pytest.approx(0.015)
    assert candidate.commands.typing.clearance_control_ticks == base.commands.typing.clearance_control_ticks == 2
    assert base.rewards.low_clearance_lateral_motion.params["clearance_m"] == pytest.approx(0.004)
    assert candidate.rewards.low_clearance_lateral_motion.params["clearance_m"] == pytest.approx(0.015)
    assert base.terminations.scrape.params["clearance_m"] == pytest.approx(0.004)
    assert candidate.terminations.scrape.params["clearance_m"] == pytest.approx(0.015)
    assert candidate.actions.action == base.actions.action
    assert candidate.observations == base.observations
    assert candidate.commands.typing.letter_xyz_b_m == base.commands.typing.letter_xyz_b_m
    assert candidate_runner.actor.class_name == base_runner.actor.class_name
    assert candidate_runner.actor.hidden_dims == base_runner.actor.hidden_dims
    assert candidate_runner.actor.activation == base_runner.actor.activation
    assert candidate_runner.actor.distribution_cfg.class_name == base_runner.actor.distribution_cfg.class_name
    assert candidate_runner.actor.distribution_cfg.action_range == base_runner.actor.distribution_cfg.action_range
    assert candidate_runner.critic.class_name == base_runner.critic.class_name
    assert candidate_runner.critic.hidden_dims == base_runner.critic.hidden_dims
    assert candidate_runner.critic.activation == base_runner.critic.activation
    assert candidate_runner.algorithm.class_name == base_runner.algorithm.class_name
    assert candidate_runner.algorithm.learning_rate == base_runner.algorithm.learning_rate
    assert candidate_runner.algorithm.num_learning_epochs == base_runner.algorithm.num_learning_epochs
    assert candidate_runner.algorithm.num_mini_batches == base_runner.algorithm.num_mini_batches
    assert candidate_runner.algorithm.gamma == base_runner.algorithm.gamma
    assert candidate_runner.algorithm.lam == base_runner.algorithm.lam
    assert candidate_runner.algorithm.clip_param == base_runner.algorithm.clip_param
    assert candidate_runner.clip_actions is None


def test_independent_p1a_workshop_baseline_changes_only_the_actuator_profile() -> None:
    task_id = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-Baseline-P1A"
    spec = gym.spec(task_id)
    anchorbench = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg()
    baseline = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianBaselineP1AEnvCfg()
    anchorbench_runner = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default
    baseline_runner = SO101PhysicalCalibrated20260718FixedCartesianBaselineP1APPORunnerCfg().default

    assert baseline.__class__.__name__ in spec.kwargs["env_cfg_entry_point"]
    assert (
        "SO101PhysicalCalibrated20260718FixedCartesianBaselineP1APPORunnerCfg" in spec.kwargs["rsl_rl_cfg_entry_point"]
    )
    assert anchorbench.actuator_profile == "anchorbench"
    assert baseline.actuator_profile == "baseline"
    assert baseline.task_contract == "physical_calibrated_20260718_fixed_cartesian_baseline_p1a_letters2_v0"
    assert baseline.commands == anchorbench.commands
    assert baseline.actions == anchorbench.actions
    assert baseline.observations == anchorbench.observations
    assert baseline.rewards == anchorbench.rewards
    assert baseline.terminations == anchorbench.terminations
    assert baseline.events == anchorbench.events
    assert baseline.scene.keyboard == anchorbench.scene.keyboard
    _assert_original_workshop_actuator(baseline.scene.robot.actuators["all"])
    for field in (
        "effort_limit_sim",
        "velocity_limit_sim",
        "stiffness",
        "damping",
        "armature",
        "friction",
        "dynamic_friction",
        "viscous_friction",
    ):
        assert getattr(anchorbench.scene.robot.actuators["all"], field) is not None
    assert baseline_runner.experiment_name.endswith("baseline_p1a_letters2")
    assert baseline_runner.num_steps_per_env == anchorbench_runner.num_steps_per_env
    assert baseline_runner.max_iterations == anchorbench_runner.max_iterations
    assert baseline_runner.save_interval == anchorbench_runner.save_interval
    assert baseline_runner.obs_groups == anchorbench_runner.obs_groups
    assert baseline_runner.actor.class_name == anchorbench_runner.actor.class_name
    assert baseline_runner.actor.hidden_dims == anchorbench_runner.actor.hidden_dims
    assert baseline_runner.actor.activation == anchorbench_runner.actor.activation
    assert baseline_runner.actor.distribution_cfg.class_name == anchorbench_runner.actor.distribution_cfg.class_name
    assert baseline_runner.actor.distribution_cfg.action_range == anchorbench_runner.actor.distribution_cfg.action_range
    assert baseline_runner.critic.class_name == anchorbench_runner.critic.class_name
    assert baseline_runner.critic.hidden_dims == anchorbench_runner.critic.hidden_dims
    assert baseline_runner.critic.activation == anchorbench_runner.critic.activation
    assert baseline_runner.algorithm.class_name == anchorbench_runner.algorithm.class_name
    assert baseline_runner.algorithm.learning_rate == anchorbench_runner.algorithm.learning_rate
    assert baseline_runner.algorithm.num_learning_epochs == anchorbench_runner.algorithm.num_learning_epochs
    assert baseline_runner.algorithm.num_mini_batches == anchorbench_runner.algorithm.num_mini_batches
    assert baseline_runner.algorithm.gamma == anchorbench_runner.algorithm.gamma
    assert baseline_runner.algorithm.lam == anchorbench_runner.algorithm.lam
    assert baseline_runner.algorithm.clip_param == anchorbench_runner.algorithm.clip_param
    assert baseline_runner.clip_actions == anchorbench_runner.clip_actions


def test_transit15_workshop_baseline_changes_only_the_actuator_profile() -> None:
    task_id = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-Baseline-P1D-Transit15"
    spec = gym.spec(task_id)
    anchorbench = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DTransit15EnvCfg()
    baseline = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianBaselineP1DTransit15EnvCfg()
    anchorbench_runner = SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg().default
    baseline_runner = SO101PhysicalCalibrated20260718FixedCartesianBaselineP1DTransit15PPORunnerCfg().default

    assert baseline.__class__.__name__ in spec.kwargs["env_cfg_entry_point"]
    assert (
        "SO101PhysicalCalibrated20260718FixedCartesianBaselineP1DTransit15PPORunnerCfg"
        in spec.kwargs["rsl_rl_cfg_entry_point"]
    )
    assert anchorbench.actuator_profile == "anchorbench"
    assert baseline.actuator_profile == "baseline"
    assert baseline.task_contract == ("physical_calibrated_20260718_fixed_cartesian_baseline_p1d_transit15_letters6_v0")
    assert baseline.commands == anchorbench.commands
    assert baseline.actions == anchorbench.actions
    assert baseline.observations == anchorbench.observations
    assert baseline.rewards == anchorbench.rewards
    assert baseline.terminations == anchorbench.terminations
    assert baseline.events == anchorbench.events
    assert baseline.scene.keyboard == anchorbench.scene.keyboard
    baseline_actuator = baseline.scene.robot.actuators["all"]
    anchorbench_actuator = anchorbench.scene.robot.actuators["all"]
    _assert_original_workshop_actuator(baseline_actuator)
    for field in (
        "effort_limit_sim",
        "velocity_limit_sim",
        "stiffness",
        "damping",
        "armature",
        "friction",
        "dynamic_friction",
        "viscous_friction",
    ):
        assert getattr(anchorbench_actuator, field) is not None
    assert baseline_runner.experiment_name.endswith("baseline_p1d_transit15_letters6")
    assert baseline_runner.num_steps_per_env == anchorbench_runner.num_steps_per_env
    assert baseline_runner.max_iterations == anchorbench_runner.max_iterations
    assert baseline_runner.save_interval == anchorbench_runner.save_interval
    assert baseline_runner.obs_groups == anchorbench_runner.obs_groups
    assert baseline_runner.actor.class_name == anchorbench_runner.actor.class_name
    assert baseline_runner.actor.hidden_dims == anchorbench_runner.actor.hidden_dims
    assert baseline_runner.actor.activation == anchorbench_runner.actor.activation
    assert baseline_runner.actor.distribution_cfg.class_name == anchorbench_runner.actor.distribution_cfg.class_name
    assert baseline_runner.actor.distribution_cfg.action_range == anchorbench_runner.actor.distribution_cfg.action_range
    assert baseline_runner.critic.class_name == anchorbench_runner.critic.class_name
    assert baseline_runner.critic.hidden_dims == anchorbench_runner.critic.hidden_dims
    assert baseline_runner.critic.activation == anchorbench_runner.critic.activation
    assert baseline_runner.algorithm.class_name == anchorbench_runner.algorithm.class_name
    assert baseline_runner.algorithm.learning_rate == anchorbench_runner.algorithm.learning_rate
    assert baseline_runner.algorithm.num_learning_epochs == anchorbench_runner.algorithm.num_learning_epochs
    assert baseline_runner.algorithm.num_mini_batches == anchorbench_runner.algorithm.num_mini_batches
    assert baseline_runner.algorithm.gamma == anchorbench_runner.algorithm.gamma
    assert baseline_runner.algorithm.lam == anchorbench_runner.algorithm.lam
    assert baseline_runner.algorithm.clip_param == anchorbench_runner.algorithm.clip_param
    assert baseline_runner.clip_actions == anchorbench_runner.clip_actions


@pytest.mark.parametrize(
    ("task_id", "env_type", "runner_type", "contract", "experiment_suffix"),
    (
        (
            "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-USDDrive-P1A",
            SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianUSDDriveP1AEnvCfg,
            SO101PhysicalCalibrated20260718FixedCartesianUSDDriveP1APPORunnerCfg,
            "physical_calibrated_20260718_fixed_cartesian_usd_drive_p1a_letters2_v0",
            "usd_drive_p1a_letters2",
        ),
        (
            "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-USDDrive-P1D-Transit15",
            SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianUSDDriveP1DTransit15EnvCfg,
            SO101PhysicalCalibrated20260718FixedCartesianUSDDriveP1DTransit15PPORunnerCfg,
            "physical_calibrated_20260718_fixed_cartesian_usd_drive_p1d_transit15_letters6_v0",
            "usd_drive_p1d_transit15_letters6",
        ),
    ),
)
def test_usd_drive_tasks_change_only_the_actuator_profile(
    task_id: str,
    env_type: type,
    runner_type: type,
    contract: str,
    experiment_suffix: str,
) -> None:
    spec = gym.spec(task_id)
    if "P1D-Transit15" in task_id:
        anchorbench = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DTransit15EnvCfg()
        anchorbench_runner = SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg().default
    else:
        anchorbench = SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg()
        anchorbench_runner = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default
    candidate = env_type()
    candidate_runner = runner_type().default

    assert candidate.__class__.__name__ in spec.kwargs["env_cfg_entry_point"]
    assert runner_type.__name__ in spec.kwargs["rsl_rl_cfg_entry_point"]
    assert anchorbench.actuator_profile == "anchorbench"
    assert candidate.actuator_profile == "usd_drive"
    assert candidate.task_contract == contract
    assert candidate.commands == anchorbench.commands
    assert candidate.actions == anchorbench.actions
    assert candidate.observations == anchorbench.observations
    assert candidate.rewards == anchorbench.rewards
    assert candidate.terminations == anchorbench.terminations
    assert candidate.events == anchorbench.events
    assert candidate.scene.keyboard == anchorbench.scene.keyboard
    _assert_loaded_usd_actuator(candidate.scene.robot.actuators["all"])
    assert candidate_runner.experiment_name.endswith(experiment_suffix)
    _assert_matching_runner_recipe(candidate_runner, anchorbench_runner)


def test_exact_22d_observation_order() -> None:
    cfg = SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg()
    policy = cfg.observations.policy
    all_annotations = tuple(policy.__class__.__annotations__)
    assert all_annotations[:5] == (
        "concatenate_terms",
        "concatenate_dim",
        "enable_corruption",
        "history_length",
        "flatten_history_dim",
    )
    observation_terms = all_annotations[5:]

    assert observation_terms == (
        "joint_pos_rel_reset",
        "joint_vel",
        "joint_command_rel_reset",
        "target_xyz_b",
        "phase_onehot",
        "phase_elapsed_s",
    )
    assert [name for name in observation_terms if "onehot" in name] == ["phase_onehot"]
    assert all("letter" not in name and "slot" not in name for name in observation_terms)
    assert [
        policy.joint_pos_rel_reset.func,
        policy.joint_vel.func,
        policy.joint_command_rel_reset.func,
        policy.target_xyz_b.func,
        policy.phase_onehot.func,
        policy.phase_elapsed_s.func,
    ] == [
        mdp.fixed_joint_pos_rel_reset,
        mdp.fixed_joint_vel,
        mdp.fixed_joint_command_rel_reset,
        mdp.fixed_target_xyz_b,
        mdp.fixed_typing_phase_onehot,
        mdp.fixed_phase_elapsed_s,
    ]
    assert 5 + 5 + 5 + 3 + 3 + 1 == 22


def test_all_letters_are_targets_and_no_reset_ik_remains() -> None:
    cfg = SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg()

    assert cfg.commands.typing.target_slots == REGISTERED_LETTER_SLOTS
    assert set(cfg.commands.typing.target_slots) == set(FROZEN_LETTER_SLOTS)
    assert cfg.commands.typing.letter_length == (1, 1)
    assert cfg.commands.typing.max_len == 3
    assert cfg.commands.typing.reset.enabled is False
    assert cfg.commands.typing.reset.ik is None
    assert cfg.commands.typing.reset.pre_solve_reset is None


def test_joint_rate_action_integrates_once_with_physical_limits() -> None:
    q = torch.tensor([[0.0, 0.0, 0.0, 0.99, -0.99]])
    action = torch.tensor([[1.0, -1.0, 0.5, 1.0, -1.0]])
    velocity = torch.tensor([JOINT_RATE_LIMITS_RAD_S])
    limits = torch.tensor([[[-1.0, 1.0]] * 5])

    result = JointRatePositionAction.integrate_command(q, action, velocity, 0.04, limits)

    assert result[0, :3].tolist() == pytest.approx([0.012, -0.044, 0.015])
    assert result[0, 3:].tolist() == pytest.approx([0.998, -0.994])
    assert JOINT_RATE_SOURCE_SHA256 == "15dc8fcf17dfafd274f84de53d75e085240d5e44c5aab2f5b741ee170896fb04"


def _action_stub() -> JointRatePositionAction:
    action = object.__new__(JointRatePositionAction)
    shape = (1, 5)
    action._joint_ids = list(range(5))
    action._env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: SimpleNamespace(begin_control_transition=lambda: None))
    )
    action.cfg = SimpleNamespace(
        command_name="typing",
        action_bound_epsilon=1.0e-6,
        saturation_threshold=0.95,
        control_dt_s=0.04,
    )
    action._asset = SimpleNamespace(
        data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-2.0, 2.0]] * 5])))
    )
    action._raw_actions = torch.zeros(shape)
    action._bounded_actions = torch.zeros(shape)
    action._previous_bounded_actions = torch.zeros(shape)
    action._joint_command = torch.zeros(shape)
    action._velocity_limits = torch.ones(shape)
    action._q_reset_ref = torch.zeros(shape)
    action._near_saturation = torch.zeros(shape, dtype=torch.bool)
    action._out_of_bounds = torch.zeros(shape, dtype=torch.bool)
    action._episode_action_samples = torch.zeros(1, dtype=torch.long)
    action._episode_action_mean_abs = torch.zeros(1)
    action._episode_action_max_abs = torch.zeros(1)
    action._episode_near_saturation_fraction = torch.zeros(1)
    action._episode_near_saturation_any = torch.zeros(1, dtype=torch.bool)
    action._episode_out_of_bounds_any = torch.zeros(1, dtype=torch.bool)
    return action


def test_joint_rate_action_rejects_nonfinite_without_poisoning_command() -> None:
    action = _action_stub()

    action.process_actions(torch.tensor([[float("nan"), float("inf"), -float("inf"), 0.5, 1.01]]))

    assert torch.isfinite(action.raw_actions).all()
    assert torch.isfinite(action.bounded_actions).all()
    assert torch.isfinite(action.joint_command).all()
    assert action.out_of_bounds.tolist() == [[True, True, True, False, True]]
    assert action.episode_out_of_bounds_any.item() is True
    assert action.episode_action_mean_abs.item() == pytest.approx(0.3)
    assert action.episode_action_max_abs.item() == pytest.approx(1.0)
    assert action.episode_near_saturation_fraction.item() == pytest.approx(0.2)
    assert action.episode_near_saturation_any.item() is True


def test_joint_rate_action_online_telemetry_and_boundaries() -> None:
    action = _action_stub()
    action.process_actions(torch.tensor([[1.0, -0.95, 0.5, 0.0, 0.0]]))
    action.process_actions(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]))

    assert not action.out_of_bounds.any()
    assert action.episode_action_mean_abs.item() == pytest.approx(0.245)
    assert action.episode_action_max_abs.item() == pytest.approx(1.0)
    assert action.episode_near_saturation_fraction.item() == pytest.approx(0.2)
    assert action.episode_near_saturation_any.item() is True
    assert action.episode_out_of_bounds_any.item() is False

    action.reset(torch.tensor([0]))
    assert action.episode_action_mean_abs.item() == 0.0
    assert action.episode_action_max_abs.item() == 0.0
    assert action.episode_near_saturation_fraction.item() == 0.0
    assert action.episode_near_saturation_any.item() is False
    assert action.episode_out_of_bounds_any.item() is False


def test_policy_distribution_is_natively_bounded_without_wrapper_clip() -> None:
    runner = SO101FixedCartesianAnchorBenchP0PPORunnerCfg().default

    assert isinstance(runner.actor.distribution_cfg, BoundedBetaDistributionCfg)
    assert runner.actor.distribution_cfg.class_name == "BetaDistribution"
    assert runner.actor.distribution_cfg.action_range == (-1.0, 1.0)
    assert runner.clip_actions is None
    assert runner.num_steps_per_env == 64
    assert runner.max_iterations == 20000


def _switch_stub() -> FixedCartesianTypingCommand:
    command = object.__new__(FixedCartesianTypingCommand)
    command.num_keys = 3
    command._env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        _sim_step_counter=0,
        common_step_counter=0,
        physics_dt=0.01,
    )
    command.cfg = SimpleNamespace(
        asset_name="robot",
        key_down_fraction=0.50,
        key_up_fraction=0.25,
        stable_samples=2,
        clearance_m=0.004,
        clearance_control_ticks=2,
        map_tolerance_m=0.001,
        phase_timeouts_s=(6.0, 2.0, 2.0),
        action_name="action",
    )
    command._inst_idx = torch.zeros((1, 3), dtype=torch.long)
    command._col_idx = torch.tensor([[0, 1, 2]], dtype=torch.long)
    command.keyboard = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])),
            joint_pos=SimpleNamespace(torch=torch.zeros((1, 3))),
        )
    )
    command._press_threshold = None
    command._release_threshold = None
    command._key_upper_limit = None
    command._key_travel = None
    command._episode_max_target_depression_fraction = torch.zeros(1)
    command.held = torch.zeros((1, 3), dtype=torch.bool)
    command._down_count = torch.zeros((1, 3), dtype=torch.int16)
    command._up_count = torch.zeros((1, 3), dtype=torch.int16)
    command._initial_sample_count = torch.zeros(1, dtype=torch.int16)
    command._initialized = torch.zeros(1, dtype=torch.bool)
    command.phase = torch.tensor([SEEK_PRESS])
    command.character_index = torch.zeros(1, dtype=torch.long)
    command.pressed_slot = torch.full((1,), -1, dtype=torch.long)
    command.target = torch.tensor([[0, 1, -1]], dtype=torch.long)
    command.target_len = torch.tensor([2], dtype=torch.long)
    command.max_len = 3
    command.target_down_event = torch.zeros(1, dtype=torch.bool)
    command.target_up_event = torch.zeros(1, dtype=torch.bool)
    command.clearance_event = torch.zeros(1, dtype=torch.bool)
    command.wrong_key = torch.zeros(1, dtype=torch.bool)
    command.overlap = torch.zeros(1, dtype=torch.bool)
    command.map_max_error_m = torch.zeros(1)
    command.initially_held = torch.zeros(1, dtype=torch.bool)
    command.completed = torch.zeros(1, dtype=torch.bool)
    command._clear_ticks = torch.zeros(1, dtype=torch.int16)
    command._phase_start_sim_step = torch.zeros(1, dtype=torch.long)
    command._last_global_finalized_control_step = -1
    command.typed = torch.full((1, 3), -1, dtype=torch.long)
    command.typed_len = torch.zeros(1, dtype=torch.long)
    command.distance = torch.tensor([2.0])
    command._validate_map_once = lambda: None
    command.target_key_pos_w = lambda: torch.tensor([[0.0, 0.0, 0.0]])
    command.fixed_tip_pos_w = lambda: torch.tensor([[0.0, 0.0, 0.005]])
    command.metrics = {
        "phase": torch.zeros(1),
        "map_max_error_m": torch.zeros(1),
        "held_key_count": torch.zeros(1),
        "wrong_key": torch.zeros(1),
        "overlap": torch.zeros(1),
        "sequence/first_character_complete": torch.zeros(1),
        "sequence/second_character_reached": torch.zeros(1),
        "sequence/second_character_down": torch.zeros(1),
        "sequence/second_character_complete": torch.zeros(1),
        "sequence/full_complete": torch.zeros(1),
        "key/max_target_depression_fraction": torch.zeros(1),
        "action/mean_abs": torch.zeros(1),
        "action/max_abs": torch.zeros(1),
        "action/near_saturation_fraction": torch.zeros(1),
        "action/near_saturation_any": torch.zeros(1),
        "action/out_of_bounds_any": torch.zeros(1),
    }
    return command


def _sample(command: FixedCartesianTypingCommand, positions: tuple[float, float, float]) -> None:
    command.keyboard.data.joint_pos.torch[:] = torch.tensor([positions])
    FixedCartesianTypingCommand.sync_key_state(command)


def test_hysteresis_requires_two_samples_and_holds_target_through_release() -> None:
    command = _switch_stub()
    _sample(command, (0.0, 0.0, 0.0))
    _sample(command, (0.0, 0.0, 0.0))  # released-state initialization

    _sample(command, (-0.6, 0.0, 0.0))
    assert command.phase.item() == SEEK_PRESS
    _sample(command, (-0.6, 0.0, 0.0))
    assert command.phase.item() == RELEASE_KEY
    assert command.target_down_event.item() is True
    assert command.target_key_slot().item() == 0

    _sample(command, (0.0, 0.0, 0.0))
    assert command.phase.item() == RELEASE_KEY
    _sample(command, (0.0, 0.0, 0.0))
    assert command.phase.item() == CLEARANCE
    assert command.target_up_event.item() is True
    assert command.target_key_slot().item() == 0


def test_clearance_requires_two_control_ticks_before_next_target() -> None:
    command = _switch_stub()
    command._initialized[:] = True
    command.phase[:] = CLEARANCE
    command.pressed_slot[:] = 0

    command._env.common_step_counter = 1
    command.finalize_control_transition()
    assert command.character_index.item() == 0
    assert command.target_key_slot().item() == 0

    command._env.common_step_counter = 2
    command.finalize_control_transition()
    assert command.clearance_event.item() is True
    assert command.character_index.item() == 1
    assert command.phase.item() == SEEK_PRESS
    assert command.target_key_slot().item() == 1


def test_transit15_keeps_next_target_hidden_until_two_15mm_ticks() -> None:
    command = _switch_stub()
    command.cfg.clearance_m = 0.015
    command._initialized[:] = True
    command.phase[:] = CLEARANCE
    command.pressed_slot[:] = 0

    command.fixed_tip_pos_w = lambda: torch.tensor([[0.0, 0.0, 0.0149]])
    for control_step in (1, 2, 3):
        command._env.common_step_counter = control_step
        command.finalize_control_transition()
    assert command.character_index.item() == 0
    assert command.target_key_slot().item() == 0

    command.fixed_tip_pos_w = lambda: torch.tensor([[0.0, 0.0, 0.0151]])
    command._env.common_step_counter = 4
    command.finalize_control_transition()
    assert command.character_index.item() == 0
    command._env.common_step_counter = 5
    command.finalize_control_transition()
    assert command.clearance_event.item() is True
    assert command.character_index.item() == 1
    assert command.phase.item() == SEEK_PRESS
    assert command.target_key_slot().item() == 1


def test_two_letter_command_completes_only_after_second_release_and_clearance() -> None:
    command = _switch_stub()
    command._initialized[:] = True

    # First key down, up, and two-tick clearance reveal the second key.
    _sample(command, (-0.6, 0.0, 0.0))
    _sample(command, (-0.6, 0.0, 0.0))
    _sample(command, (0.0, 0.0, 0.0))
    _sample(command, (0.0, 0.0, 0.0))
    command._env.common_step_counter = 1
    command.finalize_control_transition()
    command._env.common_step_counter = 2
    command.finalize_control_transition()
    assert command.character_index.item() == 1
    assert command.target_key_slot().item() == 1
    assert command.completed.item() is False

    # The second character must independently produce down, up, and clearance.
    _sample(command, (0.0, -0.6, 0.0))
    _sample(command, (0.0, -0.6, 0.0))
    assert command.phase.item() == RELEASE_KEY
    _sample(command, (0.0, 0.0, 0.0))
    _sample(command, (0.0, 0.0, 0.0))
    assert command.phase.item() == CLEARANCE
    command._env.common_step_counter = 3
    command.finalize_control_transition()
    assert command.completed.item() is False
    command._env.common_step_counter = 4
    command.finalize_control_transition()

    assert command.character_index.item() == 2
    assert command.completed.item() is True
    assert command.typed[0, :2].tolist() == [0, 1]
    action = SimpleNamespace(
        episode_action_mean_abs=torch.zeros(1),
        episode_action_max_abs=torch.zeros(1),
        episode_near_saturation_fraction=torch.zeros(1),
        episode_near_saturation_any=torch.zeros(1, dtype=torch.bool),
        episode_out_of_bounds_any=torch.zeros(1, dtype=torch.bool),
    )
    command._env.action_manager = SimpleNamespace(get_term=lambda _name: action)
    FixedCartesianTypingCommand._update_metrics(command)
    assert command.metrics["sequence/first_character_complete"].item() == 1.0
    assert command.metrics["sequence/second_character_reached"].item() == 1.0
    assert command.metrics["sequence/second_character_down"].item() == 1.0
    assert command.metrics["sequence/second_character_complete"].item() == 1.0
    assert command.metrics["sequence/full_complete"].item() == 1.0


def test_neighbor_down_terminates_without_advancing() -> None:
    command = _switch_stub()
    _sample(command, (0.0, 0.0, 0.0))
    _sample(command, (0.0, 0.0, 0.0))
    _sample(command, (0.0, -0.6, 0.0))
    _sample(command, (0.0, -0.6, 0.0))

    assert command.wrong_key.item() is True
    assert command.phase.item() == SEEK_PRESS
    assert command.character_index.item() == 0


@pytest.mark.parametrize("public_pose_randomization", [False, True])
def test_live_map_mismatch_fails_fast_after_key_state_initialization(public_pose_randomization) -> None:
    command = object.__new__(FixedCartesianTypingCommand)
    command._env = SimpleNamespace(num_envs=1, device=torch.device("cpu"))
    command.cfg = SimpleNamespace(map_tolerance_m=0.001, public_pose_randomization=public_pose_randomization)
    command._map_validation_complete = False
    command._initialized = torch.tensor([True])
    command._map_checked = torch.tensor([False])
    command.map_invalid = torch.tensor([False])
    command.map_max_error_m = torch.zeros(1)
    command._letter_slots = torch.arange(26, dtype=torch.long)
    command._letter_xyz_b = torch.tensor(FROZEN_AZ_XYZ_B_M)
    command._env._public_expected_key_xyz_b = command._letter_xyz_b.unsqueeze(0).clone()
    live_xyz_w = command._letter_xyz_b.clone()
    live_xyz_w[0, 0] += 0.002
    command.key_pos_w = lambda: live_xyz_w.unsqueeze(0)
    command.robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.zeros((1, 3))),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
        )
    )

    with pytest.raises(RuntimeError, match=r"live key centers exceed map_tolerance_m=0\.001000"):
        FixedCartesianTypingCommand._validate_map_once(command)

    assert command.map_invalid.item() is True
    assert command.map_max_error_m.item() == pytest.approx(0.002, abs=1.0e-8)


def test_fixed_command_exports_action_telemetry_without_observation_changes() -> None:
    command = _switch_stub()
    action = SimpleNamespace(
        episode_action_mean_abs=torch.tensor([0.25]),
        episode_action_max_abs=torch.tensor([1.0]),
        episode_near_saturation_fraction=torch.tensor([0.125]),
        episode_near_saturation_any=torch.tensor([True]),
        episode_out_of_bounds_any=torch.tensor([True]),
    )
    command._env.action_manager = SimpleNamespace(get_term=lambda name: action)

    FixedCartesianTypingCommand._update_metrics(command)

    assert command.cfg.action_name == "action"
    assert command.metrics["action/mean_abs"].item() == pytest.approx(0.25)
    assert command.metrics["action/max_abs"].item() == pytest.approx(1.0)
    assert command.metrics["action/near_saturation_fraction"].item() == pytest.approx(0.125)
    assert command.metrics["action/near_saturation_any"].item() == 1.0
    assert command.metrics["action/out_of_bounds_any"].item() == 1.0
    action.episode_action_mean_abs.zero_()
    action.episode_action_max_abs.zero_()
    action.episode_near_saturation_fraction.zero_()
    action.episode_near_saturation_any.zero_()
    action.episode_out_of_bounds_any.zero_()
    assert command.metrics["action/mean_abs"].item() == pytest.approx(0.25)
    assert command.metrics["action/max_abs"].item() == pytest.approx(1.0)
    assert command.metrics["action/near_saturation_fraction"].item() == pytest.approx(0.125)
    assert command.metrics["action/near_saturation_any"].item() == 1.0
    assert command.metrics["action/out_of_bounds_any"].item() == 1.0
    assert 5 + 5 + 5 + 3 + 3 + 1 == 22


def test_reward_stack_penalizes_motion_change_not_action_magnitude() -> None:
    cfg = SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg()

    assert cfg.rewards.action_change.func is mdp.fixed_cartesian_action_change_l2
    assert cfg.rewards.measured_joint_velocity.func is mdp.fixed_cartesian_joint_velocity_l2
    assert not hasattr(cfg.rewards, "action_l2")
    assert not hasattr(cfg.rewards, "mechanical_power")
    assert cfg.rewards.low_clearance_lateral_motion.params["tip_offset"] == FIXED_JAW_TIP_OFFSET_M
    assert cfg.terminations.scrape.params["persistence_steps"] == 3
    assert cfg.actions.action.joint_names == ARM_JOINT_NAMES
