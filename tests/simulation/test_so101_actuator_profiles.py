# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression checks for SO-101 actuator profiles and legacy compatibility."""

import hashlib
from pathlib import Path

import yaml

from isaaclab_tasks.core.dexsuite.config.so101.SO101.actuator_profiles import actuator_profile_manifest
from isaaclab_tasks.core.dexsuite.config.so101.SO101.so101_cfg import SO101_CFG


def test_anchorbench_profile_matches_published_fit() -> None:
    profile = actuator_profile_manifest("anchorbench")

    assert profile["fit_id"] == "04_multiseed_seed42_camera_6p53mm_ACTIVE"
    assert profile["stiffness"] == {
        "shoulder_pan": 48.419899,
        "shoulder_lift": 47.782635,
        "elbow_flex": 14.715496,
        "wrist_flex": 43.607166,
        "wrist_roll": 54.873917,
        "gripper": 68.250793,
    }
    assert profile["damping"] == {
        "shoulder_pan": 3.334595,
        "shoulder_lift": 2.563165,
        "elbow_flex": 0.343250,
        "wrist_flex": 4.139243,
        "wrist_roll": 2.968392,
        "gripper": 2.818730,
    }
    assert profile["armature"] == {
        "shoulder_pan": 0.067620,
        "shoulder_lift": 0.027645,
        "elbow_flex": 0.037720,
        "wrist_flex": 0.050714,
        "wrist_roll": 0.054898,
        "gripper": 0.077625,
    }
    assert profile["dynamic_friction"] == {
        "shoulder_pan": 0.347432,
        "shoulder_lift": 0.344793,
        "elbow_flex": 0.411190,
        "wrist_flex": 0.248233,
        "wrist_roll": 0.221761,
        "gripper": 0.083458,
    }
    assert profile["friction"] == profile["dynamic_friction"]
    assert profile["viscous_friction"] == {
        "shoulder_pan": 1.590709,
        "shoulder_lift": 0.561425,
        "elbow_flex": 0.796639,
        "wrist_flex": 1.071871,
        "wrist_roll": 1.637671,
        "gripper": 0.958565,
    }


def test_baseline_profile_matches_original_workshop_table() -> None:
    profile = actuator_profile_manifest("baseline")

    assert profile["label"] == "workshop_actuator_profile"
    assert profile["source"].endswith("assets/so101.py::workshop")
    assert profile["fit_id"] is None
    assert profile["effort_limit_sim"] == 30.0
    assert profile["velocity_limit_sim"] is None
    assert profile["stiffness"] == {
        "shoulder_pan": 55.0,
        "shoulder_lift": 30.0,
        "elbow_flex": 25.0,
        "wrist_flex": 12.0,
        "wrist_roll": 7.0,
        "gripper": 4.0,
    }
    assert profile["damping"] == {
        "shoulder_pan": 0.7,
        "shoulder_lift": 0.8,
        "elbow_flex": 0.7,
        "wrist_flex": 0.5,
        "wrist_roll": 0.5,
        "gripper": 0.3,
    }
    for field in ("armature", "friction", "dynamic_friction", "viscous_friction"):
        assert profile[field] is None


def test_usd_drive_profile_matches_the_asset_loaded_by_the_task() -> None:
    profile = actuator_profile_manifest("usd_drive")
    assert profile["label"] == "workshop_so101_usd_drive"
    assert profile["source"].endswith("workshop_asset_sources.json")
    assert profile["source_sha256"] == "893963bb1d4e79bf120885355357a8a0d62186a45cf84a11cd267a8f5db97be3"
    assert profile["fit_id"] == "04_multiseed_seed42_camera_6p53mm_ACTIVE"
    assert profile["effort_limit_sim"] is None
    assert profile["velocity_limit_sim"] == 30.0
    for field in ("stiffness", "damping", "armature", "friction", "dynamic_friction"):
        assert profile[field] is None
    assert profile["viscous_friction"] == actuator_profile_manifest("anchorbench")["viscous_friction"]


def test_gtc_berlin_profile_matches_copied_sysid_result() -> None:
    profile = actuator_profile_manifest("gtc_berlin")
    source = Path(__file__).resolve().parents[2] / profile["source"]
    raw = source.read_bytes()
    fit = yaml.safe_load(raw)

    assert hashlib.sha256(raw).hexdigest() == profile["source_sha256"]
    assert fit["schema_version"] == 1
    assert fit["type"] == "actuator_sys_id_result"
    assert fit["metric"] == profile["metric"]
    assert fit["best_mse"] == profile["best_mse"]
    assert profile["effort_limit_sim"] == 3.35
    assert profile["velocity_limit_sim"] == 30.0
    assert profile["stiffness"] == {name: values["stiffness"] for name, values in fit["parameters"].items()}
    assert profile["damping"] == {name: values["damping"] for name, values in fit["parameters"].items()}
    assert profile["armature"] == {name: values["armature"] for name, values in fit["parameters"].items()}
    assert profile["friction"] == profile["dynamic_friction"] == {
        name: values["dry_friction"] for name, values in fit["parameters"].items()
    }
    assert profile["viscous_friction"] == {
        name: values["viscous_friction"] for name, values in fit["parameters"].items()
    }
    assert SO101_CFG.actuators["all"].stiffness == profile["stiffness"]
