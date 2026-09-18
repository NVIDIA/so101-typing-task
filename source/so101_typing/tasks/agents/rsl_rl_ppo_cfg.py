# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab_tasks.utils import PresetCfg


@configclass
class BoundedBetaDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    """RSL-RL 5.4.1 native bounded distribution with corrected log-probability."""

    class_name: str = "BetaDistribution"
    action_range: tuple[float, float] = (-1.0, 1.0)


##
# Model and algorithm presets.
##

FIXED_CARTESIAN_POLICY_CFG = RslRlMLPModelCfg(
    distribution_cfg=BoundedBetaDistributionCfg(),
    obs_normalization=True,
    hidden_dims=[256, 256, 128],
    activation="elu",
)

FIXED_CARTESIAN_CRITIC_CFG = RslRlMLPModelCfg(
    obs_normalization=True,
    hidden_dims=[256, 256, 128],
    activation="elu",
)

_PPO_HYPERPARAMS = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-4,
    schedule="adaptive",
    gamma=0.995,
    lam=0.90,
    desired_kl=0.01,
    max_grad_norm=1.0,
)

ALGO_CFG = RslRlPpoAlgorithmCfg(**_PPO_HYPERPARAMS)

##
# Runner configurations.
##


@configclass
class SO101PPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 250
    experiment_name = (MISSING,)  # type: ignore
    obs_groups = (MISSING,)  # type: ignore
    actor = (MISSING,)  # type: ignore
    critic = (MISSING,)  # type: ignore
    algorithm = MISSING  # type: ignore


@configclass
class SO101FixedCartesianAnchorBenchP0PPORunnerCfg(PresetCfg):
    """Bounded, physically timed PPO for the full A-Z press/release/clear P0."""

    default = SO101PPOBaseRunnerCfg().replace(
        num_steps_per_env=64,
        max_iterations=20000,
        save_interval=500,
        experiment_name="so101_fixed_cartesian_anchorbench_p0c",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=FIXED_CARTESIAN_POLICY_CFG,
        critic=FIXED_CARTESIAN_CRITIC_CFG,
        algorithm=ALGO_CFG,
        # The Beta distribution already has the exact [-1, 1] support.  A
        # wrapper clip would recreate the old off-contract Gaussian behavior.
        clip_actions=None,
    )


@configclass
class SO101FixedCartesianAnchorBenchP1APPORunnerCfg(PresetCfg):
    """Direct two-letter P1A hedge using the frozen P0 actor contract."""

    default = SO101PPOBaseRunnerCfg().replace(
        num_steps_per_env=64,
        max_iterations=20000,
        save_interval=500,
        experiment_name="so101_fixed_cartesian_anchorbench_p1a_letters2",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=FIXED_CARTESIAN_POLICY_CFG,
        critic=FIXED_CARTESIAN_CRITIC_CFG,
        algorithm=ALGO_CFG,
        clip_actions=None,
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianP0PPORunnerCfg(PresetCfg):
    """Run-76 PPO contract with an unambiguous calibrated output namespace."""

    default = SO101FixedCartesianAnchorBenchP0PPORunnerCfg().default.replace(
        experiment_name="so101_physical_calibrated_20260718_fixed_cartesian_p0"
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg(PresetCfg):
    """Run-77 PPO contract with an unambiguous calibrated output namespace."""

    default = SO101FixedCartesianAnchorBenchP1APPORunnerCfg().default.replace(
        experiment_name="so101_physical_calibrated_20260718_fixed_cartesian_p1a_letters2"
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianBaselineP1APPORunnerCfg(PresetCfg):
    """Independent original-Workshop P1A output namespace."""

    default = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default.replace(
        experiment_name="so101_physical_calibrated_20260718_fixed_cartesian_baseline_p1a_letters2"
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianUSDDriveP1APPORunnerCfg(PresetCfg):
    """Independent loaded-USD-drive P1A output namespace."""

    default = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default.replace(
        experiment_name="so101_physical_calibrated_20260718_fixed_cartesian_usd_drive_p1a_letters2"
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianP1BPPORunnerCfg(PresetCfg):
    """Three-letter continuation of the physically qualified P1A policy."""

    default = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default.replace(
        experiment_name="so101_physical_calibrated_20260718_fixed_cartesian_p1b_letters3"
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianP1CPPORunnerCfg(PresetCfg):
    """Four-letter continuation of the physically qualified P1A policy."""

    default = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default.replace(
        experiment_name="so101_physical_calibrated_20260718_fixed_cartesian_p1c_letters4"
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianP1DPPORunnerCfg(PresetCfg):
    """Six-letter continuation of the physically qualified P1A policy."""

    default = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default.replace(
        experiment_name="so101_physical_calibrated_20260718_fixed_cartesian_p1d_letters6"
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg(PresetCfg):
    """Six-letter P1D with a 15 mm post-release transit clearance."""

    default = SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg().default.replace(
        experiment_name=("so101_physical_calibrated_20260718_fixed_cartesian_p1d_transit15_letters6")
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianBaselineP1DTransit15PPORunnerCfg(PresetCfg):
    """Matched Transit15 continuation using the SO-101 workshop baseline drives."""

    default = SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg().default.replace(
        experiment_name=("so101_physical_calibrated_20260718_fixed_cartesian_baseline_p1d_transit15_letters6")
    )


@configclass
class SO101PhysicalCalibrated20260718FixedCartesianUSDDriveP1DTransit15PPORunnerCfg(PresetCfg):
    """Loaded-USD-drive Transit15 continuation output namespace."""

    default = SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg().default.replace(
        experiment_name=("so101_physical_calibrated_20260718_fixed_cartesian_usd_drive_p1d_transit15_letters6")
    )
