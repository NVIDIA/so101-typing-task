"""Shared trajectory and supervised hardware replay regressions; no real devices."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.so101_homing import fk_probe_trajectory as trajectory
from scripts.so101_homing import run_fk_probe as runner


@pytest.fixture
def artifact(tmp_path):
    task = tmp_path/"task.json"
    task.write_text(json.dumps({"schema_version": 1, "keyboard": {
        "position_m": [.2572, .0099, .0076366833485381445],
        "rotation_xyzw": [0, 0, -.7079413179513058, .7062712583259836]}}))
    path = tmp_path/"trajectory.json"
    plan = trajectory.generate(task, path)
    return plan, path


def test_cartesian_press_hover_and_return(artifact):
    plan, path = artifact
    np.testing.assert_allclose(trajectory.tip(next(r["q_rad"] for r in plan["rows"] if r["phase"] == "press")),
                               plan["press_xyz_b_m"], atol=2e-5)
    np.testing.assert_allclose(trajectory.tip(next(r["q_rad"] for r in plan["rows"] if r["phase"] == "hover")),
                               plan["hover_xyz_b_m"], atol=2e-5)
    assert trajectory.load(path) == plan
    with pytest.raises(FileExistsError):
        trajectory.generate(path.with_name("task.json"), path)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["rows"][40]["q_rad"].__setitem__(0, float("nan")), "finite"),
    (lambda p: p["rows"][40]["q_rad"].__setitem__(0, .5), "rate limits"),
    (lambda p: p["joint_zero_offset_deg"].__setitem__(2, -10), "joint_zero_offset"),
    (lambda p: p["source_sha256"].update({"scripts/so101_homing/so101_kinematics.py": "changed"}), "source changed"),
])
def test_rejects_invalid_or_changed_commands(artifact, mutation, match):
    plan, _ = artifact
    mutation(plan)
    with pytest.raises(ValueError, match=match):
        trajectory.validate(plan)


def test_hardware_requires_pass_for_exact_file(artifact):
    plan, path = artifact
    report = path.with_name("sim-report.json")
    report.write_text(json.dumps({"status": "failed", "trajectory_sha256": trajectory.digest(path)}))
    with pytest.raises(ValueError, match="not passed"):
        runner.verified_plan(path)
    report.write_text(json.dumps({"status": "passed", "trajectory_sha256": trajectory.digest(path)}))
    assert runner.verified_plan(path) == plan
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="not passed"):
        runner.verified_plan(path)


@pytest.mark.parametrize("wrong_key", [False, True])
@pytest.mark.parametrize('offsets', [None, [3., .7, -9.6, .5, 0.]])
def test_hardware_replays_saved_rows_and_detects_keys(artifact, monkeypatch, wrong_key, offsets):
    from scripts.so101_homing import robot as robot_module, policy_runtime
    plan, path = artifact
    plan = copy.deepcopy(plan)
    # Test transport with a compact sequence; geometry is checked separately.
    plan["rows"] = [{"phase": phase, "q_rad": (trajectory.RESET+[delta,0,0,0,0]).tolist()}
                    for phase, delta in [("descend", .001), ("lift", .002), ("finished", 0.), ("finished", 0.)]]
    sent = []
    fake = SimpleNamespace(disconnect=lambda: None, bus=SimpleNamespace(sync_read=lambda *a, **kw: {'gripper': 0.}))
    monkeypatch.setattr(robot_module, "connect_robot", lambda **kw: fake)
    monkeypatch.setattr(robot_module, "prepare_closed_typing_jaw", lambda r: None)
    offset = np.zeros(5) if offsets is None else np.array(offsets)
    monkeypatch.setattr(robot_module, "read_arm_q_deg", lambda r: np.rad2deg(trajectory.RESET)-offset if not sent else sent[-1])
    monkeypatch.setattr(robot_module, "send_typing_q_deg", lambda r, q: sent.append(q.copy()))
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    class Keyboard:
        pressed_codes = set()
        device = SimpleNamespace(active_keys=lambda: [])
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *exc): pass
        def poll(self):
            if len(sent) in (126, 127):
                char = "d" if wrong_key else "f"
                return [{"char": char, "key_name": "KEY_"+char.upper(),
                         "edge": "down" if len(sent) == 126 else "up"}], False
            return [], False

    monkeypatch.setattr(policy_runtime, "KeyboardMonitor", Keyboard)
    args = SimpleNamespace(robot_id="test", port="fake", keyboard_device="fake")
    if offsets is not None:
        args.alignment_data = {'contract': {'joint_zero_offset_deg': offsets}, 'contract_sha256': 'synthetic-test-only'}
    if wrong_key:
        with pytest.raises(RuntimeError, match="unexpected key"):
            runner.execute(plan, path, args, path)
        assert len(sent) == 126
    else:
        assert runner.execute(plan, path, args, path) == 0
        np.testing.assert_allclose(np.deg2rad(np.array(sent[-len(plan['rows']):])+offset), [r["q_rad"] for r in plan["rows"]], atol=1e-15)
