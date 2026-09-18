"""Pure NumPy SO-101 kinematics used by calibration and live policy observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
JOINT_NAMES = (*ARM_JOINT_NAMES, "gripper")
BODY_NAMES = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "moving_jaw_so101_v1_link",
)
_BODY_NAME_TO_KINEMATIC = {
    "base_link": "base",
    "shoulder_link": "shoulder",
    "upper_arm_link": "upper_arm",
    "lower_arm_link": "lower_arm",
    "wrist_link": "wrist",
    "gripper_link": "gripper",
    "moving_jaw_so101_v1_link": "moving_jaw_so101_v1",
}

# Each entry is child body, parent-to-joint translation, parent-to-joint quaternion (wxyz), and joint index.
_BODY_CHAIN = (
    ("shoulder", (0.0388353, -8.97657e-09, 0.0624), (3.56167e-16, 1.22818e-15, -1.0, -4.14635e-16), 0),
    ("upper_arm", (-0.0303992, -0.0182778, -0.0542), (0.5, -0.5, -0.5, -0.5), 1),
    ("lower_arm", (-0.11257, -0.028, 1.73763e-16), (0.707107, -5.98613e-17, -2.58051e-17, 0.707107), 2),
    ("wrist", (-0.1349, 0.0052, 3.62355e-17), (0.707107, 9.58722e-16, -7.51313e-16, -0.707107), 3),
    ("gripper", (5.55112e-17, -0.0611, 0.0181), (0.0172091, -0.0172091, 0.706897, 0.706897), 4),
)
_MOVING_JAW_POS = (0.0202, 0.0188, -0.0234)
_MOVING_JAW_QUAT_WXYZ = (0.707107, 0.707107, -1.85362e-08, 1.85362e-08)
_BODY_COM_POS = {
    "base": (0.0137179, -5.19711e-05, 0.0334843),
    "shoulder": (-0.0307604, -1.66727e-05, -0.0252713),
    "upper_arm": (-0.0898471, -0.00838224, 0.0184089),
    "lower_arm": (-0.0980701, 0.00324376, 0.0182831),
    "wrist": (-0.000103312, -0.0386143, 0.0281156),
    "gripper": (0.000213627, 0.000245138, -0.025187),
    "moving_jaw_so101_v1": (-0.00157495, -0.0300244, 0.0192755),
}

FIXED_JAW_SITE_POS = (-0.0079, -0.000218121, -0.0981274)
TYPING_TIP_SITE_POS = (0.010028509, -0.000102324, -0.105400003)
FIXED_JAW_TYPING_TIP_SITE_POS = (-0.0106769063594562, -0.0005212273838322845, -0.10413855748835771)
WRIST_ROLL_MOUNT_ALIGNMENT_RAD = 0.0
GRIPPER_MODEL_MIN_RAD = math.radians(-10.0)
GRIPPER_MODEL_MAX_RAD = math.radians(100.0)
ARM_JOINT_LIMITS_RAD = (
    (math.radians(-110.0), math.radians(110.0)),
    (math.radians(-100.0), math.radians(100.0)),
    (-1.69, 1.69),
    (-1.65806, 1.65806),
    (-2.74385, 2.84121),
)


def lerobot_position_to_model_rad(position: np.ndarray | list[float]) -> np.ndarray:
    """Convert LeRobot arm degrees plus gripper percent to model joint radians."""

    value = np.asarray(position, dtype=np.float64)
    if value.shape != (len(JOINT_NAMES),) or not np.isfinite(value).all():
        raise ValueError(f"LeRobot position must have shape ({len(JOINT_NAMES)},) and be finite")
    if not 0.0 <= value[-1] <= 100.0:
        raise ValueError(f"LeRobot gripper position must be within [0, 100] percent, got {value[-1]}")
    result = np.deg2rad(value)
    result[-1] = GRIPPER_MODEL_MIN_RAD + value[-1] / 100.0 * (
        GRIPPER_MODEL_MAX_RAD - GRIPPER_MODEL_MIN_RAD
    )
    return result


def model_rad_to_lerobot_position(position_rad: np.ndarray | list[float]) -> np.ndarray:
    """Convert model joint radians to LeRobot arm degrees plus gripper percent."""

    value = np.asarray(position_rad, dtype=np.float64)
    if value.shape != (len(JOINT_NAMES),) or not np.isfinite(value).all():
        raise ValueError(f"model position must have shape ({len(JOINT_NAMES)},) and be finite")
    result = np.rad2deg(value)
    gripper = np.clip(value[-1], GRIPPER_MODEL_MIN_RAD, GRIPPER_MODEL_MAX_RAD)
    result[-1] = 100.0 * (gripper - GRIPPER_MODEL_MIN_RAD) / (
        GRIPPER_MODEL_MAX_RAD - GRIPPER_MODEL_MIN_RAD
    )
    return result


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    return quat / norm


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.asarray((quat[0], -quat[1], -quat[2], -quat[3]), dtype=np.float64)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(quat)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return _quat_to_matrix(quat) @ vector


def _z_joint_quat(angle: float) -> np.ndarray:
    half = 0.5 * float(angle)
    return np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)), dtype=np.float64)


def body_poses(q_rad: np.ndarray | list[float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return body-origin position and quaternion (wxyz) in the fixed robot-base frame."""

    q = np.asarray(q_rad, dtype=np.float64)
    if q.shape != (len(JOINT_NAMES),):
        raise ValueError(f"expected {len(JOINT_NAMES)} joints in {JOINT_NAMES}, got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("joint positions must be finite")

    position = np.zeros(3, dtype=np.float64)
    orientation = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {"base": (position.copy(), orientation.copy())}
    for body, origin, fixed_quat, joint_index in _BODY_CHAIN:
        position = position + _rotate(orientation, np.asarray(origin, dtype=np.float64))
        orientation = _quat_normalize(_quat_mul(orientation, np.asarray(fixed_quat, dtype=np.float64)))
        orientation = _quat_normalize(_quat_mul(orientation, _z_joint_quat(q[joint_index])))
        poses[body] = (position.copy(), orientation.copy())

    gripper_pos, gripper_quat = poses["gripper"]
    moving_pos = gripper_pos + _rotate(gripper_quat, np.asarray(_MOVING_JAW_POS, dtype=np.float64))
    moving_quat = _quat_normalize(_quat_mul(gripper_quat, np.asarray(_MOVING_JAW_QUAT_WXYZ, dtype=np.float64)))
    moving_quat = _quat_normalize(_quat_mul(moving_quat, _z_joint_quat(q[5])))
    poses["moving_jaw_so101_v1"] = (moving_pos, moving_quat)
    return poses


def site_position(
    q_rad: np.ndarray | list[float], site_pos: tuple[float, float, float] = TYPING_TIP_SITE_POS
) -> np.ndarray:
    """Return a gripper-local site position in the robot-base frame."""

    gripper_pos, gripper_quat = body_poses(q_rad)["gripper"]
    return gripper_pos + _rotate(gripper_quat, np.asarray(site_pos, dtype=np.float64))


def site_direction(
    q_rad: np.ndarray | list[float], site_pos: tuple[float, float, float] = TYPING_TIP_SITE_POS
) -> np.ndarray:
    """Return the base-frame direction from the gripper origin to a site."""

    local_direction = np.asarray(site_pos, dtype=np.float64)
    norm = float(np.linalg.norm(local_direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("site_pos must define a finite non-zero direction")
    _, gripper_quat = body_poses(q_rad)["gripper"]
    return _rotate(gripper_quat, local_direction / norm)


def site_position_jacobian(
    q_rad: np.ndarray | list[float],
    site_pos: tuple[float, float, float] = TYPING_TIP_SITE_POS,
    *,
    epsilon_rad: float = 1.0e-5,
) -> np.ndarray:
    """Return the fixed-site translational Jacobian for the five arm joints."""

    q = np.asarray(q_rad, dtype=np.float64)
    if q.shape != (len(JOINT_NAMES),) or not np.isfinite(q).all():
        raise ValueError(f"q_rad must have shape ({len(JOINT_NAMES)},) and be finite")
    if not np.isfinite(epsilon_rad) or epsilon_rad <= 0.0:
        raise ValueError("epsilon_rad must be positive and finite")
    jacobian = np.empty((3, len(ARM_JOINT_NAMES)), dtype=np.float64)
    for joint_index in range(len(ARM_JOINT_NAMES)):
        delta = np.zeros_like(q)
        delta[joint_index] = epsilon_rad
        jacobian[:, joint_index] = (
            site_position(q + delta, site_pos=site_pos) - site_position(q - delta, site_pos=site_pos)
        ) / (2.0 * epsilon_rad)
    return jacobian


def _angular_velocity_wxyz(previous: np.ndarray, current: np.ndarray, dt_s: float) -> np.ndarray:
    delta = _quat_normalize(_quat_mul(current, _quat_conjugate(previous)))
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(delta[0], -1.0, 1.0)))
    return delta[1:] * (angle / (vector_norm * dt_s))


def _com_position(
    poses: dict[str, tuple[np.ndarray, np.ndarray]], kinematic_name: str
) -> np.ndarray:
    position, quat = poses[kinematic_name]
    return position + _rotate(quat, np.asarray(_BODY_COM_POS[kinematic_name], dtype=np.float64))


def body_state_vector(
    q_rad: np.ndarray | list[float],
    previous_q_rad: np.ndarray | list[float] | None = None,
    *,
    joint_velocity_rad_s: np.ndarray | list[float] | None = None,
    dt_s: float = 0.04,
    body_names: tuple[str, ...] = BODY_NAMES,
) -> np.ndarray:
    """Return IsaacLab's ``[pos, quat(xyzw), linear velocity, angular velocity]`` body layout."""

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    q = np.asarray(q_rad, dtype=np.float64)
    current = body_poses(q)
    if joint_velocity_rad_s is not None:
        qd = np.asarray(joint_velocity_rad_s, dtype=np.float64)
        if qd.shape != (len(JOINT_NAMES),) or not np.isfinite(qd).all():
            raise ValueError(f"joint_velocity_rad_s must have shape ({len(JOINT_NAMES)},) and be finite")
        derivative_dt = 1.0e-5
        derivative_previous = body_poses(q - 0.5 * derivative_dt * qd)
        derivative_current = body_poses(q + 0.5 * derivative_dt * qd)
    else:
        derivative_dt = dt_s
        derivative_current = current
        derivative_previous = current if previous_q_rad is None else body_poses(previous_q_rad)
    rows: list[np.ndarray] = []
    for name in body_names:
        kinematic_name = _BODY_NAME_TO_KINEMATIC.get(name, name)
        if kinematic_name not in current:
            raise ValueError(f"unsupported SO-101 body {name!r}; available: {BODY_NAMES}")
        position, quat_wxyz = current[kinematic_name]
        prev_quat = derivative_previous[kinematic_name][1]
        next_quat = derivative_current[kinematic_name][1]
        linear_velocity = (
            _com_position(derivative_current, kinematic_name)
            - _com_position(derivative_previous, kinematic_name)
        ) / derivative_dt
        angular_velocity = _angular_velocity_wxyz(prev_quat, next_quat, derivative_dt)
        quat_xyzw = np.roll(quat_wxyz, -1)
        rows.append(np.concatenate((position, quat_xyzw, linear_velocity, angular_velocity)))
    return np.concatenate(rows)


@dataclass(frozen=True)
class PositionIkResult:
    q_rad: np.ndarray
    error_m: float
    success: bool
    iterations: int


def solve_site_position_ik(
    target_xyz_m: np.ndarray | list[float],
    seed_q_rad: np.ndarray | list[float],
    *,
    site_pos: tuple[float, float, float] = FIXED_JAW_SITE_POS,
    max_nfev: int = 500,
    joint_reference_q_rad: np.ndarray | list[float] | None = None,
    joint_reference_weight_m_per_rad: float = 0.0,
    target_tip_direction_b: np.ndarray | list[float] | None = None,
    tip_direction_weight_m: float = 0.0,
) -> PositionIkResult:
    """Solve a fixed-jaw target with optional posture and direction preservation."""

    from scipy.optimize import least_squares

    target = np.asarray(target_xyz_m, dtype=np.float64)
    seed = np.asarray(seed_q_rad, dtype=np.float64)
    if target.shape != (3,):
        raise ValueError(f"target_xyz_m must have shape (3,), got {target.shape}")
    if seed.shape != (len(ARM_JOINT_NAMES),):
        raise ValueError(f"seed_q_rad must have shape ({len(ARM_JOINT_NAMES)},), got {seed.shape}")
    if joint_reference_weight_m_per_rad < 0.0 or not np.isfinite(joint_reference_weight_m_per_rad):
        raise ValueError("joint_reference_weight_m_per_rad must be finite and non-negative")
    reference: np.ndarray | None = None
    if joint_reference_q_rad is not None:
        reference = np.asarray(joint_reference_q_rad, dtype=np.float64)
        if reference.shape != (len(ARM_JOINT_NAMES),) or not np.isfinite(reference).all():
            raise ValueError(
                f"joint_reference_q_rad must have shape ({len(ARM_JOINT_NAMES)},) with finite values"
            )
    elif joint_reference_weight_m_per_rad > 0.0:
        raise ValueError("joint_reference_q_rad is required when its weight is non-zero")
    target_direction: np.ndarray | None = None
    if target_tip_direction_b is not None:
        target_direction = np.asarray(target_tip_direction_b, dtype=np.float64)
        direction_norm = float(np.linalg.norm(target_direction))
        if target_direction.shape != (3,) or not np.isfinite(target_direction).all() or direction_norm <= 0.0:
            raise ValueError("target_tip_direction_b must contain three finite non-zero values")
        target_direction = target_direction / direction_norm
    if tip_direction_weight_m < 0.0 or not np.isfinite(tip_direction_weight_m):
        raise ValueError("tip_direction_weight_m must be finite and non-negative")
    if target_direction is None and tip_direction_weight_m > 0.0:
        raise ValueError("target_tip_direction_b is required when its weight is non-zero")
    lower = np.asarray([limit[0] for limit in ARM_JOINT_LIMITS_RAD], dtype=np.float64)
    upper = np.asarray([limit[1] for limit in ARM_JOINT_LIMITS_RAD], dtype=np.float64)

    def position_residual(arm_q: np.ndarray) -> np.ndarray:
        full_q = np.concatenate((arm_q, np.zeros(1, dtype=np.float64)))
        return site_position(full_q, site_pos=site_pos) - target

    def residual(arm_q: np.ndarray) -> np.ndarray:
        position_error = position_residual(arm_q)
        components = [position_error]
        if reference is not None and joint_reference_weight_m_per_rad > 0.0:
            components.append(joint_reference_weight_m_per_rad * (arm_q - reference))
        if target_direction is not None and tip_direction_weight_m > 0.0:
            current_direction = site_direction(
                np.concatenate((arm_q, np.zeros(1, dtype=np.float64))), site_pos=site_pos
            )
            components.append(tip_direction_weight_m * (current_direction - target_direction))
        return np.concatenate(components)

    result = least_squares(
        residual,
        np.clip(seed, lower, upper),
        bounds=(lower, upper),
        max_nfev=max_nfev,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    error = float(np.linalg.norm(position_residual(result.x)))
    return PositionIkResult(
        q_rad=result.x,
        error_m=error,
        success=bool(result.success and error < 0.002),
        iterations=result.nfev,
    )
