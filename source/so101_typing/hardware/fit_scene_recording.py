"""Diagnostic XY/yaw keyboard fit from encoder poses, with held-out letters.

Recorded height and tilt stay fixed. Optional historical offsets are fixed,
never fitted. No keyboard scaling.
This is not independent tip metrology.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.fixture_layout import load_geometry, transform, quaternion
from scripts.reinforcement_learning.rsl_rl.public_task import rotate_xyzw
from .fk_probe_trajectory import ROOT, digest
from .scene_check import load_recording, replay_joint_positions
from .fixed_cartesian_policy import PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG
from .so101_kinematics import FIXED_JAW_TYPING_TIP_SITE_POS, site_position

HELD_OUT = tuple("WTISHLCM")  # Fixed spatial coverage; selected before inspecting residuals.
PRESS_TRAVEL_M = .0018  # Model full stroke; actual per-press depression is not measured.


def planar_fit(source, target):
    """Least-squares XY/yaw only; target Z cannot influence placement."""
    source, target = np.asarray(source), np.asarray(target)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("fit needs at least three paired 3D points")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("fit points must be finite")
    source, target = source[:, :2], target[:, :2]
    a, b = source.mean(0), target.mean(0)
    if min(np.linalg.matrix_rank(source-a), np.linalg.matrix_rank(target-b)) < 2:
        raise ValueError("fit points must not be collinear")
    u, _, vt = np.linalg.svd((source-a).T@(target-b))
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(u@vt)
    rotation = np.eye(3)
    rotation[:2, :2] = (u@correction@vt).T
    translation = np.zeros(3)
    translation[:2] = b-rotation[:2, :2]@a
    return rotation, translation


def contacts(recording, geometry, benchmark_offsets=False):
    times = np.array([s["time_s"] for s in recording["samples"]])
    q, _ = replay_joint_positions(recording, benchmark_offsets)
    key_geometry = {k["label"].upper(): k for k in geometry["keys"]
                    if k["slot"].startswith("main_") and len(k["label"]) == 1}
    asset = ROOT/"source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/keyboards/data/mx_keys/mx_keys_asset.npz"
    with np.load(asset) as data:
        deck = data["deck_quat_xyzw"].tolist()
    # Keyboard rotation in this saved task, including legacy relative-yaw input.
    from scripts.reinforcement_learning.rsl_rl.public_task import nominal_geometry
    _, rotation, _, _ = nominal_geometry(geometry["reference_position_m"], geometry["reference_rotation_xyzw"], [],
                                          recording["task_config"].get("keyboard", {}))
    normal = np.array(rotate_xyzw(rotation, rotate_xyzw(deck, [0,0,1])))
    active = {}
    result = []
    for e in recording["events"]:
        char = (e.get("char") or "").upper()
        if char not in key_geometry:
            continue
        if e["edge"] == "down":
            if active:
                raise ValueError("overlapping letter presses cannot identify one tip contact")
            active[e["code"]] = e
        elif e["edge"] == "up":
            if e["code"] not in active:
                raise ValueError("letter release has no matching down event")
            start = active.pop(e["code"])
            ids = np.where((times >= start["time_s"]) & (times <= e["time_s"]))[0]
            if len(ids) < 3:
                raise ValueError(f"{char}: fewer than three encoder samples during key press")
            tips = np.array([site_position(q[i], site_pos=FIXED_JAW_TYPING_TIP_SITE_POS) for i in ids])
            rim = np.array(transform(key_geometry[char]["rim_center_m"], geometry, recording["task_config"]))
            result.append({"key": char, "down_s": start["time_s"], "up_s": e["time_s"],
                           "samples": len(ids), "tip_xyz_m": tips.mean(0).tolist(),
                           "tip_span_mm": (np.ptp(tips, axis=0)*1000).tolist(),
                           "pressed_key_xyz_m": (rim-PRESS_TRAVEL_M*normal).tolist()})
    if active:
        raise ValueError("recording ends with a letter still held")
    return result


def transform_task(recording, geometry, rotation, translation):
    from scripts.reinforcement_learning.rsl_rl.public_task import nominal_geometry
    p, q, _, _ = nominal_geometry(geometry["reference_position_m"], geometry["reference_rotation_xyzw"], [],
                                  recording["task_config"].get("keyboard", {}))
    original = np.array([rotate_xyzw(q, axis) for axis in np.eye(3)]).T
    orientation = rotation@original
    return {"schema_version": 1, "keyboard": {
        "position_m": (rotation@p+translation).tolist(), "rotation_xyzw": quaternion(orientation.T.tolist())}}


def score(rows, rotation, translation):
    errors = []
    for row in rows:
        delta = (rotation@row["pressed_key_xyz_m"]+translation-row["tip_xyz_m"])*1000
        errors.append({**row, "residual_xyz_mm": delta.tolist(), "residual_mm": float(np.linalg.norm(delta))})
    norms = np.array([e["residual_mm"] for e in errors])
    deltas = np.array([e["residual_xyz_mm"] for e in errors])
    return {"contacts": errors, "count": len(errors), "mean_mm": float(norms.mean()),
            "rms_mm": float(np.sqrt(np.mean(norms**2))), "max_mm": float(norms.max()),
            "mean_xy_mm": float(np.linalg.norm(deltas[:, :2], axis=1).mean()),
            "mean_signed_z_mm": float(deltas[:, 2].mean())}


def fit_recording(path, out_dir, validation_path=None, *, benchmark_offsets=False):
    recording = load_recording(path)
    geometry = load_geometry()
    rows = contacts(recording, geometry, benchmark_offsets=benchmark_offsets)
    if {r["key"] for r in rows} != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") or len(rows) != 26:
        raise ValueError("this fit protocol requires exactly one complete A-Z pass")
    train = [r for r in rows if r["key"] not in HELD_OUT]
    held_out = [r for r in rows if r["key"] in HELD_OUT]
    rotation, translation = planar_fit([r["pressed_key_xyz_m"] for r in train], [r["tip_xyz_m"] for r in train])
    task = transform_task(recording, geometry, rotation, translation)
    all_rotation, all_translation = planar_fit([r["pressed_key_xyz_m"] for r in rows], [r["tip_xyz_m"] for r in rows])
    all_task = transform_task(recording, geometry, all_rotation, all_translation)
    report = {"recording_sha256": digest(path), "robot_id": recording["robot_id"],
              "calibration_sha256": recording["calibration_sha256"],
              "joint_zero_offset_deg": list(PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG) if benchmark_offsets else [0]*5,
              "encoder_convention": "benchmark" if benchmark_offsets else "direct",
              "replay_requires_benchmark_offsets": benchmark_offsets,
              "method": "Unweighted XY/yaw least squares. Recorded height and tilt fixed; no scale or joint fit. One mean point per press.",
              "adjustable_parameters": ["x", "y", "yaw"],
              "height_and_tilt_preserved": True,
              "press_depth_assumption_m": PRESS_TRAVEL_M,
              "limits": "Encoder-derived fixed-tip positions; assumes centre contacts and full model key travel. Held-out letters are from the same recording, not an independent repeat.",
              "sample_count": len(recording["samples"]), "letter_sequence": "".join(r["key"] for r in rows),
              "fit_keys": [r["key"] for r in train], "held_out_keys": [r["key"] for r in held_out],
              "fit": score(train, rotation, translation), "held_out": score(held_out, rotation, translation),
              "all_under_fit18": score(rows, rotation, translation), "original_scene": score(rows, np.eye(3), np.zeros(3)),
              "task_fit18": task, "task_all26": all_task,
              "all26_training_only": score(rows, all_rotation, all_translation)}
    floor = min(transform(p, geometry, task)[2] for p in geometry["case_hull_m"])
    report["lowest_case_relative_to_robot_underside_mm"] = (floor-geometry["datum_m"][2])*1000
    if validation_path:
        old = load_recording(validation_path)
        for field in ("robot_id", "calibration_sha256", "task_config", "source_sha256", "robot_usd_sha256"):
            if old[field] != recording[field]:
                raise ValueError(f"validation recording has different {field}")
        other = contacts(old, geometry, benchmark_offsets=benchmark_offsets)
        report["earlier_recording_validation"] = {"recording_sha256": digest(validation_path),
                                                    **score(other, rotation, translation)}
        report["earlier_recording_assumption"] = "Physical fixture remained fixed between recordings; metadata alone cannot verify that."
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (("task-fit18.json", task), ("task-all26.json", all_task), ("fit-report.json", report)):
        (out_dir/name).write_text(json.dumps(data, indent=2)+"\n")
    print(f"Fit 18 letters: mean {report['fit']['mean_mm']:.2f} mm, max {report['fit']['max_mm']:.2f} mm")
    print(f"Held-out 8 letters: mean {report['held_out']['mean_mm']:.2f} mm, max {report['held_out']['max_mm']:.2f} mm")
    if "earlier_recording_validation" in report:
        print(f"Earlier recording: mean {report['earlier_recording_validation']['mean_mm']:.2f} mm, max {report['earlier_recording_validation']['max_mm']:.2f} mm")
    print(f"Task for held-out replay: {(out_dir/'task-fit18.json').resolve()}")
    if benchmark_offsets:
        print("Replay this fit with --benchmark-offsets; the task JSON contains keyboard placement only.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--validation-recording", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--benchmark-offsets", action="store_true",
                        help="Use the fixed historical joint offsets for FK fitting and validation; replay must use the same flag.")
    args = parser.parse_args()
    try:
        fit_recording(args.recording, args.out_dir, args.validation_recording,
                      benchmark_offsets=args.benchmark_offsets)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
