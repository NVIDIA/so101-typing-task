"""CPU-only diagnostic of joint zero offsets with the keyboard and jaw site frozen.

Fit one recording; score separate recordings without refitting or candidate selection.
Outputs are evidence only, never a deployment calibration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from scripts.fixture_layout import load_geometry
from scripts.reinforcement_learning.rsl_rl.public_task import load_config
from .fit_scene_recording import PRESS_TRAVEL_M, contacts
from .fixed_cartesian_policy import PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG
from .fk_probe_trajectory import ROOT, digest
from .scene_check import load_recording
from .so101_kinematics import (
    ARM_JOINT_NAMES, FIXED_JAW_TYPING_TIP_SITE_POS, _BODY_CHAIN, _quat_to_matrix,
)

HISTORICAL = np.array(PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG)
FAMILIES = {"fit3": (1, 2, 3), "fit4": (0, 1, 2, 3), "fit5_diagnostic": (0, 1, 2, 3, 4)}
IDENTITY_FIELDS = ("robot_id", "calibration_sha256", "robot_usd_sha256", "source_sha256")


def batch_fk(q):
    """Vectorized canonical FK; the moving gripper cannot change this fixed-jaw site."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or not np.isfinite(q).all():
        raise ValueError("FK requires finite N-by-6 joint positions")
    position = np.zeros((len(q), 3))
    rotation = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    for _, origin, quat, index in _BODY_CHAIN:
        position += rotation @ np.asarray(origin)
        rotation = rotation @ _quat_to_matrix(np.asarray(quat))
        c, s = np.cos(q[:, index]), np.sin(q[:, index])
        turn = np.zeros_like(rotation)
        turn[:, 0, 0], turn[:, 1, 1], turn[:, 2, 2] = c, c, 1
        turn[:, 0, 1], turn[:, 1, 0] = -s, s
        rotation = rotation @ turn
    return position + rotation @ np.asarray(FIXED_JAW_TYPING_TIP_SITE_POS)


class Presses:
    def __init__(self, rows, joint_samples):
        if not rows or len(rows) != len(joint_samples) or any(len(q) == 0 for q in joint_samples):
            raise ValueError("recording needs complete letter presses with encoder samples")
        self.rows = rows
        self.lengths = np.array([len(q) for q in joint_samples])
        self.starts = np.r_[0, np.cumsum(self.lengths)[:-1]]
        self.q = np.concatenate(joint_samples)
        self.targets = np.array([r["pressed_key_xyz_m"] for r in rows])
        if self.targets.shape != (len(rows), 3) or not np.isfinite(self.targets).all():
            raise ValueError("contact targets must be finite XYZ points")

    def predict(self, offsets_deg):
        offsets = np.r_[np.deg2rad(offsets_deg), 0.0]
        tips = batch_fk(self.q + offsets)
        return np.add.reduceat(tips, self.starts, axis=0) / self.lengths[:, None]

    def residual(self, offsets_deg):
        return 1000 * (self.predict(offsets_deg) - self.targets)


def recording_presses(recording, geometry, task, sampling="interval_mean"):
    if sampling not in ("interval_mean", "pre_down"):
        raise ValueError("unknown sampling method")
    # Override geometry only in a shallow copy. Original recording remains untouched.
    rows = contacts({**recording, "task_config": task}, geometry)
    times = np.array([s["time_s"] for s in recording["samples"]])
    q = np.array([s["q_model_rad"] for s in recording["samples"]])
    selected = []
    for row in rows:
        if sampling == "interval_mean":
            ids = np.flatnonzero((times >= row["down_s"]) & (times <= row["up_s"]))
        else:
            index = np.searchsorted(times, row["down_s"], side="right") - 1
            if index < 0 or row["down_s"] - times[index] > .05:
                raise ValueError(f"{row['key']}: no encoder sample within 50 ms before down receipt")
            ids = np.array([index])
        row["selected_samples"] = len(ids)
        # Remove direct-FK summary from contacts; each candidate is evaluated independently.
        row.pop("tip_xyz_m")
        row.pop("tip_span_mm")
        selected.append(q[ids])
    return Presses(rows, selected)


def validate_identities(recordings, geometry):
    first = recordings[0]
    for recording in recordings:
        for field in IDENTITY_FIELDS:
            if not recording.get(field) or recording[field] != first.get(field):
                raise ValueError(f"recordings have missing or different {field}")
        if recording["robot_usd_sha256"] != geometry["robot_usd_sha256"]:
            raise ValueError("recording robot USD differs from fixed geometry")


def metrics(residual):
    norms = np.linalg.norm(residual, axis=1)
    return {"count": len(norms), "mean_mm": float(norms.mean()),
            "rms_mm": float(np.sqrt(np.mean(norms**2))),
            "p95_mm": float(np.percentile(norms, 95)), "max_mm": float(norms.max()),
            "mean_xyz_mm": residual.mean(axis=0).tolist()}


def score(presses, offsets):
    residual = presses.residual(offsets)
    keys = np.array([r["key"] for r in presses.rows])
    return {**metrics(residual),
            "per_key": {key: metrics(residual[keys == key]) for key in sorted(set(keys))},
            "contacts": [{**row, "predicted_tip_xyz_m": (target + error / 1000).tolist(),
                          "residual_xyz_mm": error.tolist(), "residual_mm": float(np.linalg.norm(error))}
                         for row, target, error in zip(presses.rows, presses.targets, residual)]}


def fit_candidates(presses):
    """No validation data enters optimization or multistart selection."""
    candidates = {"zero": {"offset_deg": [0.0]*5, "fitted": False},
                  "historical": {"offset_deg": HISTORICAL.tolist(), "fitted": False}}
    for name, indices in FAMILIES.items():
        def unpack(values):
            offset = np.zeros(5)
            offset[list(indices)] = values
            return offset

        def residual(values):
            return presses.residual(unpack(values)).ravel()

        starts = (np.zeros(len(indices)), HISTORICAL[list(indices)])
        fits = [least_squares(residual, start, bounds=(-15., 15.), loss="soft_l1", f_scale=2.,
                              max_nfev=1000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
                for start in starts]
        best_index = min(range(len(fits)), key=lambda i: fits[i].cost)
        best = fits[best_index]
        # Raw geometric sensitivity, not the loss-reweighted optimizer Jacobian.
        epsilon = 1e-4
        jacobian = np.column_stack([(residual(best.x + axis*epsilon) - residual(best.x - axis*epsilon))
                                    / (2*epsilon) for axis in np.eye(len(indices))])
        singular = np.linalg.svd(jacobian, compute_uv=False)
        condition = float(singular[0]/singular[-1]) if singular[-1] > 1e-12 else None
        candidates[name] = {"offset_deg": unpack(best.x).tolist(), "fitted": True,
                            "free_joints": [ARM_JOINT_NAMES[i] for i in indices],
                            "boundary_hits": [ARM_JOINT_NAMES[i] for i in indices
                                              if abs(unpack(best.x)[i]) >= 15.-1e-5],
                            "jacobian_singular_values_mm_per_deg": singular.tolist(),
                            "jacobian_condition_number": condition,
                            "selected_start": ("zero", "historical")[best_index],
                            "multistarts": [{"start": label, "converged": bool(f.success),
                                             "message": str(f.message), "cost": float(f.cost),
                                             "nfev": f.nfev, "offset_deg": unpack(f.x).tolist()}
                                            for label, f in zip(("zero", "historical"), fits)]}
    return candidates


def audit(task_path, fit_path, validation_paths, out_dir, *, sampling="interval_mean"):
    if out_dir.exists():
        raise ValueError(f"refusing to overwrite diagnostic output: {out_dir}")
    if not validation_paths:
        raise ValueError("at least one separate validation recording is required")
    paths = [fit_path, *validation_paths]
    hashes = [digest(path) for path in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError("fit and validation recordings must be distinct (duplicate content hash)")
    task_hash = digest(task_path)
    task, geometry = load_config(task_path), load_geometry()
    recordings = [load_recording(path) for path in paths]
    validate_identities(recordings, geometry)
    presses = [recording_presses(r, geometry, task, sampling) for r in recordings]
    candidates = fit_candidates(presses[0])
    for candidate in candidates.values():
        candidate["fit"] = score(presses[0], candidate["offset_deg"])
        candidate["validation"] = [score(p, candidate["offset_deg"]) for p in presses[1:]]
    dependencies = (Path(__file__), ROOT/"scripts/so101_homing/so101_kinematics.py",
                    ROOT/"scripts/so101_homing/fit_scene_recording.py", ROOT/"scripts/fixture_layout.py",
                    ROOT/"scripts/reinforcement_learning/rsl_rl/public_task.py")
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "diagnostic_only_not_approved_for_deployment", "selected_candidate": None,
              "task_config": task, "task_config_path": str(task_path.resolve()),
              "task_config_sha256": task_hash, "keyboard_placement_frozen": True,
              "fixed_jaw_site_m": list(FIXED_JAW_TYPING_TIP_SITE_POS),
              "mapping": "q_model_rad = q_lerobot_arm_rad + radians(offset_deg)",
              "joint_order": list(ARM_JOINT_NAMES), "sampling": sampling,
              "press_depth_assumption_m": PRESS_TRAVEL_M,
              "method": "One equally weighted mean FK point per press; robust XYZ soft_l1 residuals in mm, f_scale=2 mm. Bounds +/-15 deg. Fit only first recording; no candidate chosen using validation.",
              "limitations": "Assumes fixed physical fixture, key-centre contact and full 1.8 mm depression. USB receipt times are not synchronized physical contact times. Pre-down still uses the same full-depression target for sensitivity comparison. These assumptions cannot identify a unique physical cause or validate physical deployment.",
              "identities": {key: recordings[0][key] for key in IDENTITY_FIELDS},
              "diagnostic_source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in dependencies},
              "recordings": [{"role": "fit" if i == 0 else "validation", "path": str(path.resolve()),
                              "sha256": hashes[i], "recorded_task_config_sha256": r["task_config_sha256"],
                              "recorded_task_overridden": r["task_config"] != task,
                              "press_count": len(p.rows), "sequence": "".join(row["key"] for row in p.rows)}
                             for i, (path, r, p) in enumerate(zip(paths, recordings, presses))],
              "candidates": candidates}
    if digest(task_path) != task_hash:
        raise ValueError("frozen task changed while audit was running")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir/"mapping-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--fit-recording", type=Path, required=True)
    parser.add_argument("--validation-recording", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sampling", choices=("interval_mean", "pre_down"), default="interval_mean")
    args = parser.parse_args()
    try:
        report = audit(args.task_config, args.fit_recording, args.validation_recording,
                       args.out_dir, sampling=args.sampling)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for name, candidate in report["candidates"].items():
        validation = ", ".join(f"{s['mean_mm']:.2f}" for s in candidate["validation"])
        print(f"{name}: fit {candidate['fit']['mean_mm']:.2f} mm; validation means [{validation}] mm")
    print(f"Diagnostic only; no calibration installed. Report: {(args.out_dir/'mapping-report.json').resolve()}")


if __name__ == "__main__":
    main()
