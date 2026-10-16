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
        # Native Workshop USD drives. Newton imports gains, armature and Coulomb friction;
        # the pinned importer needs explicit velocity and viscous-friction bridges.
        "label": "workshop_so101_usd_drive",
        "source": "source/so101_typing/tasks/SO101/data/workshop_asset_sources.json",
        "source_sha256": "893963bb1d4e79bf120885355357a8a0d62186a45cf84a11cd267a8f5db97be3",
        "fit_id": "04_multiseed_seed42_camera_6p53mm_ACTIVE",
        "effort_limit_sim": None,
        "velocity_limit_sim": 30.0,
        "stiffness": None,
        "damping": None,
        "armature": None,
        "friction": None,
        "dynamic_friction": None,
        "viscous_friction": {
            "shoulder_pan": 1.590709,
            "shoulder_lift": 0.561425,
            "elbow_flex": 0.796639,
            "wrist_flex": 1.071871,
            "wrist_roll": 1.637671,
            "gripper": 0.958565,
        },
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
        # The Workshop USD provides the passive terms for this profile.
        "armature": None,
        "friction": None,
        "dynamic_friction": None,
        "viscous_friction": None,
    },
    "gtc_berlin": {
        # 2026-10-12 position-MSE fit; exact source YAML is copied beside this module.
        "label": "gtc_berlin_sysid_oct12",
        "source": "source/so101_typing/tasks/SO101/data/gtc_berlin.yaml",
        "source_sha256": "7231ab90db35368e1506fec94bccad88187e5e2bf5d5304f1a113c24de977b6d",
        "fit_id": "sysid_Oct12",
        "metric": "position_mse",
        "best_mse": 0.0001072153189590853,
        # The fit does not contain motor limits; retain the Workshop limits.
        "effort_limit_sim": 3.35,
        "velocity_limit_sim": 30.0,
        "stiffness": {
            "shoulder_pan": 48.93682087435154,
            "shoulder_lift": 51.336503963086734,
            "elbow_flex": 18.565405780730025,
            "wrist_flex": 43.03380653711947,
            "wrist_roll": 47.135886076886166,
            "gripper": 50.83054002253434,
        },
        "damping": {
            "shoulder_pan": 2.309705389862818,
            "shoulder_lift": 2.7829628172221677,
            "elbow_flex": 0.534694353676958,
            "wrist_flex": 2.7191323580266284,
            "wrist_roll": 1.7895194158608136,
            "gripper": 2.345108101123728,
        },
        "armature": {
            "shoulder_pan": 0.058228342618297214,
            "shoulder_lift": 0.04425337526513037,
            "elbow_flex": 0.054507128694178224,
            "wrist_flex": 0.05110346770103517,
            "wrist_roll": 0.05138195786130317,
            "gripper": 0.04338043435613245,
        },
        # Map fitted dry friction to Newton friction and PhysX dynamic friction.
        "friction": {
            "shoulder_pan": 0.2154699280851669,
            "shoulder_lift": 0.3181311647624241,
            "elbow_flex": 0.4018746873200306,
            "wrist_flex": 0.29646325177526045,
            "wrist_roll": 0.2827718443928624,
            "gripper": 0.31151570694796027,
        },
        "dynamic_friction": {
            "shoulder_pan": 0.2154699280851669,
            "shoulder_lift": 0.3181311647624241,
            "elbow_flex": 0.4018746873200306,
            "wrist_flex": 0.29646325177526045,
            "wrist_roll": 0.2827718443928624,
            "gripper": 0.31151570694796027,
        },
        "viscous_friction": {
            "shoulder_pan": 1.0732503444852939,
            "shoulder_lift": 0.8773707044105059,
            "elbow_flex": 0.4925605028705261,
            "wrist_flex": 0.51747384516758,
            "wrist_roll": 0.4502010515484256,
            "gripper": 1.1389592259244838,
        },
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
