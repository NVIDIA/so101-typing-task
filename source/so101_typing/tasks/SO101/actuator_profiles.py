# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen actuator profiles for the SO-101 keyboard sim-to-real comparison."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from isaaclab.actuators import ImplicitActuatorCfg

SO101_ACTUATOR_PROFILES: dict[str, dict[str, Any]] = {
    "usd_drive": {
        # Effective drive values authored by the robot USD loaded by this task.
        # The generated 17.8/0 converter artifact is not the loaded asset and
        # must not be used as a training profile.
        "label": "loaded_so101_usd_drive",
        "source": (
            "https://huggingface.co/datasets/nvidia/Anchor-Lab/resolve/main/robot_assets/so101_no_camera_new_calib.usd"
        ),
        "source_sha256": "c6c82840925ace388b0ff0acb7d8538c2b419d92fbe01dc70fe833b974d6d462",
        "fit_id": None,
        "effort_limit_sim": 10.0,
        "velocity_limit_sim": 10.0,
        "stiffness": {
            name: 100.0
            for name in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )
        },
        "damping": {
            name: 1.0
            for name in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )
        },
        # These properties are unauthored in the loaded USD, so Newton's
        # effective value is zero rather than the AnchorBench fitted values.
        "armature": None,
        "friction": None,
        "dynamic_friction": None,
        "viscous_friction": None,
    },
    "baseline": {
        # Exact SO-101 Workshop profile used as the baseline arm in the A/B.
        "label": "workshop_actuator_profile",
        "source": "Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/so101.py::workshop",
        "fit_id": None,
        "effort_limit_sim": 30.0,
        "velocity_limit_sim": None,
        "stiffness": {
            "shoulder_pan": 55.0,
            "shoulder_lift": 30.0,
            "elbow_flex": 25.0,
            "wrist_flex": 12.0,
            "wrist_roll": 7.0,
            "gripper": 4.0,
        },
        "damping": {
            "shoulder_pan": 0.7,
            "shoulder_lift": 0.8,
            "elbow_flex": 0.7,
            "wrist_flex": 0.5,
            "wrist_roll": 0.5,
            "gripper": 0.3,
        },
        # The workshop profile does not override these physical properties.
        # The shared robot USD leaves them unauthored, so Newton resolves them
        # to zero for the baseline arm.
        "armature": None,
        "friction": None,
        "dynamic_friction": None,
        "viscous_friction": None,
    },
    "anchorbench": {
        # Exact SO-101 AnchorBench table used by the workshop A/B comparison.
        # Keep the public key stable so existing task configs select this profile.
        "label": "anchorbench_workshop_ab_sysid",
        "source": "Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/so101.py::anchorbench_sysid",
        "fit_id": "04_multiseed_seed42_camera_6p53mm_ACTIVE",
        "effort_limit_sim": 3.35,
        "velocity_limit_sim": 30.0,
        "stiffness": {
            "shoulder_pan": 48.419899,
            "shoulder_lift": 47.782635,
            "elbow_flex": 14.715496,
            "wrist_flex": 43.607166,
            "wrist_roll": 54.873917,
            "gripper": 68.250793,
        },
        "damping": {
            "shoulder_pan": 3.334595,
            "shoulder_lift": 2.563165,
            "elbow_flex": 0.343250,
            "wrist_flex": 4.139243,
            "wrist_roll": 2.968392,
            "gripper": 2.818730,
        },
        "armature": {
            "shoulder_pan": 0.067620,
            "shoulder_lift": 0.027645,
            "elbow_flex": 0.037720,
            "wrist_flex": 0.050714,
            "wrist_roll": 0.054898,
            "gripper": 0.077625,
        },
        # The workshop sets static friction equal to the fitted Coulomb term.
        # Isaac Sim requires static friction to be at least dynamic friction.
        "friction": {
            "shoulder_pan": 0.347432,
            "shoulder_lift": 0.344793,
            "elbow_flex": 0.411190,
            "wrist_flex": 0.248233,
            "wrist_roll": 0.221761,
            "gripper": 0.083458,
        },
        "dynamic_friction": {
            "shoulder_pan": 0.347432,
            "shoulder_lift": 0.344793,
            "elbow_flex": 0.411190,
            "wrist_flex": 0.248233,
            "wrist_roll": 0.221761,
            "gripper": 0.083458,
        },
        "viscous_friction": {
            "shoulder_pan": 1.590709,
            "shoulder_lift": 0.561425,
            "elbow_flex": 0.796639,
            "wrist_flex": 1.071871,
            "wrist_roll": 1.637671,
            "gripper": 0.958565,
        },
        "motor_lag_ms": 0.0,
    },
}


def actuator_profile_manifest(profile: str) -> dict[str, Any]:
    """Return a serializable copy of one frozen actuator profile."""

    if profile not in SO101_ACTUATOR_PROFILES:
        choices = ", ".join(sorted(SO101_ACTUATOR_PROFILES))
        raise ValueError(f"unknown SO-101 actuator profile {profile!r}; choose from {choices}")
    return {"name": profile, **deepcopy(SO101_ACTUATOR_PROFILES[profile])}


def make_implicit_actuator_cfg(profile: str) -> ImplicitActuatorCfg:
    """Build an Isaac Lab implicit actuator from a frozen profile."""

    params = actuator_profile_manifest(profile)
    return ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit_sim=params["effort_limit_sim"],
        velocity_limit_sim=params["velocity_limit_sim"],
        stiffness=params["stiffness"],
        damping=params["damping"],
        armature=params["armature"],
        friction=params["friction"],
        dynamic_friction=params["dynamic_friction"],
        viscous_friction=params["viscous_friction"],
    )
