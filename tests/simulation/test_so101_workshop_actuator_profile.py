# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression lock for the SO-101 workshop AnchorBench A/B actuator fit."""

from isaaclab_tasks.core.dexsuite.config.so101.SO101.actuator_profiles import actuator_profile_manifest


def test_anchorbench_profile_matches_workshop_ab_fit() -> None:
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
    joint_names = {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    }

    assert profile["label"] == "loaded_so101_usd_drive"
    assert profile["source"].endswith("robot_assets/so101_no_camera_new_calib.usd")
    assert profile["source_sha256"] == "c6c82840925ace388b0ff0acb7d8538c2b419d92fbe01dc70fe833b974d6d462"
    assert profile["fit_id"] is None
    assert profile["effort_limit_sim"] == 10.0
    assert profile["velocity_limit_sim"] == 10.0
    assert profile["stiffness"] == {name: 100.0 for name in joint_names}
    assert profile["damping"] == {name: 1.0 for name in joint_names}
    assert 17.8 not in profile["stiffness"].values()
    for field in ("armature", "friction", "dynamic_friction", "viscous_friction"):
        assert profile[field] is None
