from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from so101_typing.hardware.policy_runtime import KeyboardMonitor, PressGate, load_rest_pose


def test_press_gate_rejects_overlapping_key_down_edges() -> None:
    gate = PressGate()
    assert gate.process({"code": 30, "value": 1}) is False
    assert gate.process({"code": 30, "value": 0}) is False
    assert gate.process({"code": 31, "value": 1}) is False
    assert gate.process({"code": 32, "value": 1}) is True
    assert gate.invalid_overlap is True


def test_keyboard_monitor_tracks_pressed_key_edges() -> None:
    monitor = KeyboardMonitor.__new__(KeyboardMonitor)
    monitor.pressed_codes = set()

    monitor._record_key_state(code=33, value=1)
    assert monitor.pressed_codes == {33}
    monitor._record_key_state(code=33, value=0)

    assert monitor.pressed_codes == set()


def test_keyboard_monitor_drains_pressed_keys_before_release() -> None:
    monitor = KeyboardMonitor.__new__(KeyboardMonitor)
    monitor.pressed_codes = {33}

    def release_key() -> tuple[list[dict[str, object]], bool]:
        monitor._record_key_state(code=33, value=0)
        return [], False

    monitor.poll = release_key

    assert monitor.drain_until_released(timeout_s=0.01) is True


def test_rest_pose_requires_expected_joint_order_and_no_contact(tmp_path: Path) -> None:
    rest_path = tmp_path / "rest_pose.json"
    rest_path.write_text(
        json.dumps(
            {
                "kind": "so101_policy_rest_pose",
                "schema_version": 1,
                "arm_joint_names": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
                "closed_fixed_jaw_required": True,
                "contact_required": False,
                "pose_deg": {
                    "shoulder_pan": -4.0,
                    "shoulder_lift": -9.0,
                    "elbow_flex": 12.0,
                    "wrist_flex": 65.0,
                    "wrist_roll": 2.0,
                },
            }
        )
    )

    np.testing.assert_allclose(load_rest_pose(rest_path), [-4.0, -9.0, 12.0, 65.0, 2.0])

    invalid = json.loads(rest_path.read_text())
    invalid["contact_required"] = True
    rest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="closed fixed jaw"):
        load_rest_pose(rest_path)
