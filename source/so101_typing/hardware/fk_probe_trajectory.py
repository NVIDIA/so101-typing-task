"""One offline joint trajectory, shared by the Newton and physical F probes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.robot_model_identity import WORKSHOP_USD_SHA256

from .source_identity import source_hash_matches
from .so101_kinematics import (
    ARM_JOINT_NAMES, ARM_JOINT_LIMITS_RAD, FIXED_JAW_TYPING_TIP_SITE_POS,
    GRIPPER_MODEL_MIN_RAD, site_position, site_position_jacobian,
)

ROOT = Path(__file__).resolve().parents[3]
DT = 0.04
RESET = np.array([-.08285519086390664, -.13809964128766555, .020637904096827316,
                  1.1561739160158462, .03759170696603171])
RATES = np.array([.3, 1.1, .75, .2, .1])
TASK = "Isaac-Keyboard-SO101-MX-Physical-Calibrated-20260718-FixedCartesian-GTCBerlin-P1D-Transit15"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tip(q):
    return site_position(np.r_[q, GRIPPER_MODEL_MIN_RAD], site_pos=FIXED_JAW_TYPING_TIP_SITE_POS)


def solve(goal, seed):
    q = np.asarray(seed).copy()
    limits = np.asarray(ARM_JOINT_LIMITS_RAD)
    for _ in range(300):
        error = np.asarray(goal) - tip(q)
        if np.linalg.norm(error) < 0.00002:
            return q
        jac = site_position_jacobian(np.r_[q, GRIPPER_MODEL_MIN_RAD],
                                     site_pos=FIXED_JAW_TYPING_TIP_SITE_POS)
        jac[:, 4] = 0.0  # Keep the typing jaw's wrist roll unchanged.
        delta = jac.T @ np.linalg.solve(jac @ jac.T + .005**2 * np.eye(3), error)
        q = np.clip(q + np.clip(delta, -.025, .025), limits[:, 0], limits[:, 1])
    raise ValueError(f"unreachable probe waypoint; IK residual {np.linalg.norm(goal-tip(q))*1000:.3f} mm")


def generate(task_path: Path, destination: Path, *, letter="F", hover_mm=15.0, press_mm=2.0):
    from scripts.fixture_layout import load_geometry, transform
    from scripts.reinforcement_learning.rsl_rl.public_task import load_config
    if letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(letter) != 1:
        raise ValueError("probe letter must be one A-Z letter")
    if not 8 <= hover_mm <= 40 or not .1 <= press_mm <= 3:
        raise ValueError("hover must be 8..40 mm and press depth 0.1..3 mm below the released rim")
    task = load_config(task_path)
    geometry = load_geometry()
    key = next(k for k in geometry["keys"] if k["slot"] == f"main_{letter.lower()}")
    rim = np.array(transform(key["rim_center_m"], geometry, task))
    hover = rim + [0, 0, hover_mm/1000]
    press = rim - [0, 0, press_mm/1000]
    rows = []
    q = RESET.copy()

    def hold(phase, seconds):
        rows.extend({"phase": phase, "q_rad": q.tolist()} for _ in range(round(seconds/DT)))

    def move(goal, phase, minimum_seconds):
        nonlocal q
        start = q.copy()
        steps = max(round(minimum_seconds/DT), int(np.ceil(np.max(np.abs(goal-start)/(RATES*.45*DT))*1.5)))
        for a in np.linspace(0, 1, steps+1)[1:]:
            q = start + (goal-start)*(a*a*(3-2*a))
            rows.append({"phase": phase, "q_rad": q.tolist()})

    hold("start", 1)
    # Approach at the reset tip's height, then follow a near-vertical Cartesian path.
    high = hover.copy()
    high[2] = max(tip(RESET)[2], hover[2]+.03)
    move(solve(high, q), "approach", 4)
    for xyz in np.linspace(high, hover, 17)[1:]:
        move(solve(xyz, q), "approach", .24)
    hover_q = q.copy()
    hold("hover", 1)
    for xyz in np.linspace(hover, press, 33)[1:]:
        move(solve(xyz, q), "descend", .12)
    hold("press", .4)
    for xyz in np.linspace(press, hover, 33)[1:]:
        move(solve(xyz, q), "lift", .12)
    move(hover_q, "lift", .2)
    hold("released", 1)
    for xyz in np.linspace(hover, high, 17)[1:]:
        move(solve(xyz, q), "retreat", .24)
    move(RESET, "return", 4)
    hold("finished", 1)
    source_files = ["scripts/so101_homing/so101_kinematics.py", *geometry["source_sha256"]]
    plan = {"schema_version": 1, "kind": "so101_fk_joint_replay", "task": TASK,
            "task_config": task, "task_config_sha256": digest(task_path),
            "source_sha256": {name: digest(ROOT/name) for name in source_files},
            "robot_usd_sha256": WORKSHOP_USD_SHA256,
            "letter": letter, "joint_names": list(ARM_JOINT_NAMES), "control_dt_s": DT,
            "joint_zero_offset_deg": [0.0]*5, "gripper_percent": 0.0,
            "initial_q_rad": RESET.tolist(), "rim_xyz_b_m": rim.tolist(),
            "hover_xyz_b_m": hover.tolist(), "press_xyz_b_m": press.tolist(),
            "fixed_tip_site_pos_m": list(FIXED_JAW_TYPING_TIP_SITE_POS), "rows": rows}
    validate(plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return plan


def validate(plan):
    if plan.get("schema_version") != 1 or plan.get("kind") != "so101_fk_joint_replay":
        raise ValueError("not a shared SO-101 FK trajectory")
    for key, expected in (("task", TASK), ("joint_names", list(ARM_JOINT_NAMES)),
                          ("control_dt_s", DT), ("gripper_percent", 0),
                          ("joint_zero_offset_deg", [0]*5),
                          ("fixed_tip_site_pos_m", list(FIXED_JAW_TYPING_TIP_SITE_POS))):
        if plan.get(key) != expected:
            raise ValueError(f"unsupported trajectory {key}")
    q = np.asarray([plan["initial_q_rad"]] + [r["q_rad"] for r in plan["rows"]])
    limits = np.asarray(ARM_JOINT_LIMITS_RAD)
    if q.ndim != 2 or q.shape[1] != 5 or not 2 <= len(q) <= 3000 or not np.isfinite(q).all():
        raise ValueError("trajectory must contain finite five-joint rows, at most 120 seconds")
    if (q < limits[:, 0]).any() or (q > limits[:, 1]).any():
        raise ValueError("trajectory exceeds model joint limits")
    if (np.abs(np.diff(q, axis=0)) > RATES*DT + 1e-9).any():
        raise ValueError("trajectory exceeds simulator joint command rate limits")
    if not np.allclose(q[0], RESET, atol=1e-8, rtol=0) or not np.allclose(q[-1], q[0], atol=1e-8, rtol=0):
        raise ValueError("trajectory must start and finish at its fixed reset")
    for name, expected in plan["source_sha256"].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not source_hash_matches(name, digest(path), expected):
            raise ValueError(f"trajectory model source changed: {name}; regenerate and verify in sim")


def load(path: Path):
    plan = json.loads(path.read_text())
    validate(plan)
    return plan
