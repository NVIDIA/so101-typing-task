# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Production SO-101 physical-calibrated keyboard environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

def _register_physical_fixed_cartesian_task(
    task_id: str,
    env_cfg_name: str,
    runner_cfg_name: str,
) -> None:
    """Register one checkpoint-compatible physical-calibration task."""

    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (f"{__name__}.physical_calibrated_20260718_env_cfg:{env_cfg_name}"),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg_name}",
        },
    )


_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P0",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP0EnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianP0PPORunnerCfg",
)
_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1A",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianP1APPORunnerCfg",
)
_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-Baseline-P1A",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianBaselineP1AEnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianBaselineP1APPORunnerCfg",
)
_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-USDDrive-P1A",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianUSDDriveP1AEnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianUSDDriveP1APPORunnerCfg",
)
_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1B",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1BEnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianP1BPPORunnerCfg",
)
_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1C",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1CEnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianP1CPPORunnerCfg",
)
_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1D",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DEnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianP1DPPORunnerCfg",
)
_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-AnchorBench-P1D-Transit15",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DTransit15EnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg",
)
_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-Baseline-P1D-Transit15",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianBaselineP1DTransit15EnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianBaselineP1DTransit15PPORunnerCfg",
)
_register_physical_fixed_cartesian_task(
    "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-USDDrive-P1D-Transit15",
    "SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianUSDDriveP1DTransit15EnvCfg",
    "SO101PhysicalCalibrated20260718FixedCartesianUSDDriveP1DTransit15PPORunnerCfg",
)
