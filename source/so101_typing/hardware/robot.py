# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal LeRobot connection helpers for SO-101 homing."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .constants import ARM_JOINT_NAMES, DEFAULT_ROBOT_ID
from .hardware_lock import RobotBusLock

ROBOT_PORT_PATTERNS = (
    "/dev/serial/by-id/*",
    "/dev/ttyACM*",
    "/dev/ttyUSB*",
)


def calibration_directory() -> Path:
    """Use the installed LeRobot calibration location without constructing a robot."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
    return HF_LEROBOT_CALIBRATION / ROBOTS / 'so_follower'


def require_calibration(robot_id: str) -> Path:
    """Offline preflight only: never opens a serial device or enables torque."""
    if not isinstance(robot_id, str) or not robot_id.strip() or robot_id in ('.', '..') or '/' in robot_id or '\\' in robot_id:
        raise ValueError('robot_id must be a nonempty calibration filename, not a path')
    directory = calibration_directory()
    path = directory / f'{robot_id}.json'
    if not path.is_file():
        available = ', '.join(sorted(p.stem for p in directory.glob('*.json'))) or 'none'
        raise FileNotFoundError(
            f'No LeRobot calibration file found for id {robot_id!r} at {path}. '
            f'Available calibration IDs: {available}. Run lerobot-calibrate '
            f'--robot.type=so101_follower --robot.port=YOUR_PORT --robot.id={robot_id} first.')
    try:
        data = json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f'Invalid LeRobot calibration file {path}: {error}') from error
    if not isinstance(data, dict) or not set((*ARM_JOINT_NAMES, 'gripper')).issubset(data):
        raise ValueError(
            f'Invalid LeRobot calibration file {path}: '
            'expected calibration for all five arm motors and gripper')
    from lerobot.motors import MotorCalibration
    try:
        for motor, values in data.items():
            calibration = MotorCalibration(**values)
            if any(type(value) is not int for value in vars(calibration).values()):
                raise ValueError(f'{motor}: calibration fields must be integers')
            if calibration.range_min >= calibration.range_max:
                raise ValueError(f'{motor}: range_min must be less than range_max')
    except (ValueError, TypeError) as error:
        raise ValueError(f'Invalid LeRobot calibration file {path}: {error}') from error
    return path


def load_lerobot():
    """Import the LeRobot follower lazily so offline tests need no hardware SDK."""

    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise RuntimeError("LeRobot SO101 hardware support is not importable in this Python environment.") from exc
    try:
        from lerobot.utils.robot_utils import precise_sleep
    except ImportError:
        precise_sleep = time.sleep
    return SO101Follower, SO101FollowerConfig, precise_sleep


def discover_robot_ports() -> list[str]:
    """Return unique candidate serial ports for an SO-101 controller."""

    ports: list[str] = []
    seen_real_paths: set[str] = set()
    for pattern in ROBOT_PORT_PATTERNS:
        for port in sorted(glob.glob(pattern)):
            real_path = os.path.realpath(port)
            if real_path not in seen_real_paths:
                seen_real_paths.add(real_path)
                ports.append(port)
    return ports


def resolve_robot_port(port: str | None = None) -> str:
    """Resolve a robot port, requiring an explicit choice when ambiguous."""

    if port:
        if os.path.exists(port):
            return port
        candidates = discover_robot_ports()
        suffix = f" Available candidates: {', '.join(candidates)}" if candidates else " No serial candidates found."
        raise FileNotFoundError(f"robot serial port does not exist: {port}.{suffix}")

    candidates = discover_robot_ports()
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "no robot serial ports found. Connect the SO-101 controller, then run lerobot-find-port."
        )
    raise FileNotFoundError(
        f"multiple robot serial ports found; pass --port explicitly. Candidates: {', '.join(candidates)}"
    )


def connect_robot(
    *,
    port: str | None = None,
    robot_id: str = DEFAULT_ROBOT_ID,
    enable_torque: bool = True,
) -> Any:
    """Connect a calibrated follower and optionally hold its current arm pose."""

    SO101Follower, SO101FollowerConfig, _ = load_lerobot()
    resolved_port = resolve_robot_port(port)
    robot = SO101Follower(
        SO101FollowerConfig(
            port=resolved_port,
            id=robot_id,
            use_degrees=True,
            disable_torque_on_disconnect=False,
        )
    )
    if not robot.calibration:
        available = sorted(path.stem for path in robot.calibration_dir.glob("*.json"))
        suffix = f" Available calibration IDs: {', '.join(available)}" if available else " No calibration files found."
        raise FileNotFoundError(
            f"No LeRobot calibration file found for id {robot_id!r} at {robot.calibration_fpath}.{suffix}"
        )

    bus_lock = RobotBusLock().acquire()
    original_disconnect = robot.disconnect

    def disconnect_with_lock() -> None:
        try:
            original_disconnect()
        finally:
            bus_lock.release()

    robot.disconnect = disconnect_with_lock
    try:
        robot.bus.connect()
        # This direct-bus path bypasses SOFollower.connect(), including its
        # calibration check. Verify stored motor registers before any write so
        # normalized readings really use the calibration selected for this robot.
        if not robot.bus.is_calibrated:
            raise ValueError(
                f'Motor calibration registers do not match the LeRobot file for {robot_id!r}. '
                'Read the calibration/register diagnostic before recalibrating; '
                'no position or torque commands were sent.')
        if enable_torque:
            present = robot.bus.sync_read("Present_Position", list(ARM_JOINT_NAMES), normalize=True, num_retry=3)
            robot.bus.sync_write("Goal_Position", present, num_retry=3)
            robot.bus.enable_torque(list(ARM_JOINT_NAMES), num_retry=5)
    except Exception:
        try:
            robot.bus.disconnect(disable_torque=False)
        except Exception:
            pass
        bus_lock.release()
        raise
    return robot


def read_arm_q_deg(robot: Any) -> np.ndarray:
    """Read the five arm joints in LeRobot's degree convention."""

    bus = getattr(robot, "bus", None)
    if bus is not None:
        present = bus.sync_read("Present_Position", list(ARM_JOINT_NAMES), normalize=True, num_retry=3)
        return np.asarray([float(present[joint]) for joint in ARM_JOINT_NAMES], dtype=np.float64)
    observation = robot.get_observation()
    return np.asarray([float(observation[f"{joint}.pos"]) for joint in ARM_JOINT_NAMES], dtype=np.float64)


def send_arm_q_deg(robot: Any, q_deg: np.ndarray | list[float]) -> None:
    """Command only the arm joints through the direct calibrated bus path."""

    target = np.asarray(q_deg, dtype=np.float64)
    if target.shape != (len(ARM_JOINT_NAMES),) or not np.isfinite(target).all():
        raise ValueError(f"q_deg must contain {len(ARM_JOINT_NAMES)} finite arm values")
    bus = getattr(robot, "bus", None)
    command = {joint: float(target[index]) for index, joint in enumerate(ARM_JOINT_NAMES)}
    if bus is not None:
        bus.sync_write("Goal_Position", command, num_retry=0)
    else:
        robot.send_action({f"{joint}.pos": value for joint, value in command.items()})


def prepare_closed_typing_jaw(robot: Any, *, gripper_deg: float = 0.0) -> None:
    """Enable and command the moving jaw while typing with the fixed jaw closed."""

    bus = getattr(robot, "bus", None)
    if bus is None:
        robot.send_action({"gripper.pos": float(gripper_deg)})
        return
    present = bus.sync_read("Present_Position", ["gripper"], normalize=True, num_retry=3)
    bus.sync_write("Goal_Position", present, num_retry=3)
    bus.enable_torque(["gripper"], num_retry=5)
    bus.sync_write("Goal_Position", {"gripper": float(gripper_deg)}, num_retry=3)


def send_typing_q_deg(
    robot: Any, q_deg: np.ndarray | list[float], *, gripper_deg: float = 0.0
) -> None:
    """Command five arm joints and the closed moving jaw through one bus write."""

    target = np.asarray(q_deg, dtype=np.float64)
    if target.shape != (len(ARM_JOINT_NAMES),) or not np.isfinite(target).all():
        raise ValueError(f"q_deg must contain {len(ARM_JOINT_NAMES)} finite arm values")
    bus = getattr(robot, "bus", None)
    command = {joint: float(target[index]) for index, joint in enumerate(ARM_JOINT_NAMES)}
    command["gripper"] = float(gripper_deg)
    if bus is not None:
        bus.sync_write("Goal_Position", command, num_retry=0)
    else:
        robot.send_action({f"{joint}.pos": value for joint, value in command.items()})


def smooth_move_arm(
    robot: Any,
    target_q_deg: np.ndarray | list[float],
    *,
    duration_s: float,
    control_hz: float,
) -> None:
    """Move to an arm target through position commands at a fixed update rate."""

    if duration_s < 0.0:
        raise ValueError("duration_s must be non-negative")
    if control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    target = np.asarray(target_q_deg, dtype=np.float64)
    if duration_s == 0.0:
        send_arm_q_deg(robot, target)
        return

    start = read_arm_q_deg(robot)
    steps = max(2, int(round(duration_s * control_hz)) + 1)
    alpha = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    _, _, precise_sleep = load_lerobot()
    interval_s = 1.0 / control_hz
    for value in alpha:
        step_start = time.perf_counter()
        send_arm_q_deg(robot, start * (1.0 - value) + target * value)
        remaining_s = interval_s - (time.perf_counter() - step_start)
        if remaining_s > 0.0:
            precise_sleep(remaining_s)
