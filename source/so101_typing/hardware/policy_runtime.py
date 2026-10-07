# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared runtime primitives for production SO-101 policy deployment."""

from __future__ import annotations

import contextlib
import hashlib
import json
import select
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import ARM_JOINT_NAMES
from .keyboard_events import _key_name_to_code, _keycode_to_char, _open_device

DEFAULT_RETURN_TO_REST_S = 1.0
DEFAULT_RELEASE_SETTLE_S = 0.25


class PolicyContractError(ValueError):
    """Raised when a deployment artifact differs from its training contract."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one deterministic JSON object to a JSONL artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def load_rest_pose(rest_pose_path: Path) -> np.ndarray:
    """Load the machine-local LeRobot rest pose used at policy handoff."""

    rest_pose = json.loads(rest_pose_path.read_text())
    if rest_pose.get("kind") != "so101_policy_rest_pose" or rest_pose.get("schema_version") != 1:
        raise ValueError(f"not a policy rest-pose artifact: {rest_pose_path}")
    if rest_pose.get("arm_joint_names") != list(ARM_JOINT_NAMES):
        raise ValueError("rest pose arm joint order does not match the SO-101 deployment contract")
    if rest_pose.get("closed_fixed_jaw_required") is not True or rest_pose.get("contact_required") is not False:
        raise ValueError("rest pose must require the closed fixed jaw and no keyboard contact")
    pose = rest_pose.get("pose_deg")
    if not isinstance(pose, dict):
        raise ValueError("rest pose has no joint-position mapping")
    try:
        arm_q_deg = np.asarray([float(pose[joint]) for joint in ARM_JOINT_NAMES], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("rest pose is missing a finite arm joint position") from exc
    if not np.isfinite(arm_q_deg).all():
        raise ValueError("rest pose contains a non-finite arm joint position")
    return arm_q_deg


@dataclass
class PressGate:
    """Reject simultaneous physical key presses."""

    pressed_codes: set[int] = field(default_factory=set)
    invalid_overlap: bool = False

    def process(self, event: dict[str, Any]) -> bool:
        code = int(event["code"])
        value = int(event["value"])
        if value == 0:
            self.pressed_codes.discard(code)
            return False
        if value != 1:
            raise ValueError(f"unexpected evdev key value {value}")
        invalid = bool(self.pressed_codes - {code})
        self.pressed_codes.add(code)
        self.invalid_overlap |= invalid
        return invalid


class KeyboardMonitor:
    """Exclusive nonblocking evdev reader with Esc reserved for stopping control."""

    def __init__(self, device_path: str) -> None:
        self.device_path = device_path
        self.device = None
        self.grabbed = False
        self.mapping = _keycode_to_char()
        _, self.stop_code = _key_name_to_code("esc")
        self.pressed_codes: set[int] = set()

    def __enter__(self) -> KeyboardMonitor:
        self.device = _open_device(self.device_path)
        self.device.grab()
        self.grabbed = True
        return self

    def _record_key_state(self, *, code: int, value: int) -> None:
        if value == 0:
            self.pressed_codes.discard(code)
        elif value == 1:
            self.pressed_codes.add(code)
        else:
            raise ValueError(f"unexpected evdev key value {value}")

    def poll(self) -> tuple[list[dict[str, Any]], bool]:
        if self.device is None:
            raise RuntimeError("keyboard monitor is not open")
        from evdev import ecodes

        readable, _, _ = select.select([self.device.fd], [], [], 0.0)
        if not readable:
            return [], False
        try:
            raw_events = self.device.read()
        except BlockingIOError:
            return [], False
        events: list[dict[str, Any]] = []
        stop = False
        for event in raw_events:
            if event.type != ecodes.EV_KEY or event.value not in (0, 1):
                continue
            self._record_key_state(code=int(event.code), value=int(event.value))
            if event.value == 1 and event.code == self.stop_code:
                stop = True
                continue
            mapped = self.mapping.get(event.code)
            key_name = mapped[0] if mapped is not None else str(ecodes.KEY.get(event.code, f"KEY_{event.code}"))
            char = mapped[1] if mapped is not None else None
            events.append(
                {
                    "char": char,
                    "code": int(event.code),
                    "key_name": key_name,
                    "value": int(event.value),
                    "edge": "down" if event.value == 1 else "up",
                    # Preserve event spacing even when multiple edges arrive in one poll.
                    # This evdev clock is separate from our monotonic receipt clock.
                    "evdev_timestamp_s": event.timestamp(),
                    "received_monotonic_s": time.monotonic(),
                }
            )
        return events, stop

    def drain_until_released(self, *, timeout_s: float) -> bool:
        """Keep the exclusive grab until keys pressed during control are released."""

        deadline = time.monotonic() + timeout_s
        while self.pressed_codes and time.monotonic() < deadline:
            self.poll()
            if self.pressed_codes:
                time.sleep(0.01)
        return not self.pressed_codes

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.device is None:
            return
        if self.grabbed:
            with contextlib.suppress(OSError):
                self.device.ungrab()
            self.grabbed = False
        self.device.close()
