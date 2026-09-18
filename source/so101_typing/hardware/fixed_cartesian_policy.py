# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact deployment contract for fixed-layout Cartesian SO-101 policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .policy_runtime import sha256_file
from .so101_kinematics import (
    ARM_JOINT_LIMITS_RAD,
    lerobot_position_to_model_rad,
    model_rad_to_lerobot_position,
)

ACTION_DIM = 5
OBSERVATION_DIM = 22
CONTROL_HZ = 25.0
CONTROL_DT_S = 1.0 / CONTROL_HZ
CLEARANCE_M = 0.004
CLEARANCE_CONTROL_TICKS = 2
PHASE_TIMEOUTS_S = (6.0, 2.0, 2.0)
STAGE_TARGET_LENGTHS = {"p0": 1, "p1a": 2, "p1b": 3, "p1c": 4, "p1d": 6}
JOINT_RATE_LIMITS_RAD_S = (0.30, 1.10, 0.75, 0.20, 0.10)

FROZEN_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

PHYSICAL_CALIBRATED_20260718_Q_RESET_RAD = (
    -0.08285519086390664,
    -0.13809964128766555,
    0.020637904096827316,
    1.1561739160158462,
    0.03759170696603171,
)
PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG = (
    0.0,
    1.9336272481,
    -10.7735791533,
    1.3647648917,
    0.0,
)
PHYSICAL_CALIBRATED_20260718_AZ_XYZ_B_M = (
    (0.24464311501323643, 0.17904083324095146, 0.0042599117494431463),
    (0.22771574743940146, 0.09350215695271337, 0.0036870904084776836),
    (0.22679073451700832, 0.1315108756576093, 0.0037206844201812927),
    (0.24556812802626454, 0.14103211081186945, 0.004226317734447914),
    (0.2646952546429896, 0.14657339002825373, 0.004735047731097078),
    (0.24603063448746113, 0.12202775145942148, 0.004209520728596113),
    (0.24649314099397518, 0.10302339024488046, 0.004192723721098496),
    (0.24695564750048923, 0.08401902903033948, 0.004175926713600884),
    (0.2670077872661949, 0.05155158023136272, 0.0046510626903173875),
    (0.2474181539163683, 0.06501467153998454, 0.004159129709394891),
    (0.24788066033224734, 0.04601031404962956, 0.004142332705188899),
    (0.24834316674812643, 0.027005956559274617, 0.004125535700982907),
    (0.22864076027115957, 0.05549344197200344, 0.0036534964000657002),
    (0.2281782538552805, 0.0744977994623584, 0.003670293404271694),
    (0.267470293500804, 0.03254723018937984, 0.00463426569269464),
    (0.267932799916683, 0.013542872699024881, 0.004617468688488646),
    (0.26377024175458463, 0.1845821073365799, 0.004768641741566328),
    (0.2651577612401387, 0.12756902508952672, 0.004718250720307843),
    (0.2451056215197505, 0.16003647202641047, 0.00424311474194553),
    (0.26562026765601776, 0.10856466759917176, 0.004701453716101849),
    (0.26654528066904587, 0.07055594517008976, 0.004667859701106621),
    (0.2272532410235224, 0.11250651444306832, 0.0037038874126836775),
    (0.26423274822711057, 0.16557774751860868, 0.004751844735303072),
    (0.22632822801049426, 0.15051523687215032, 0.003737481427678906),
    (0.2660827741625318, 0.08956030638463075, 0.004684656708604237),
    (0.22586570379616713, 0.16952032569953857, 0.003754279078277244),
)
PHYSICAL_CALIBRATED_20260718_AZ_XYZ_MEAN_B_M = (
    0.24875869794180977,
    0.104053144318032,
    0.0042535491608431805,
)
PHYSICAL_CALIBRATED_20260718_AZ_XYZ_STD_B_M = (
    0.015494051359203873,
    0.048965322097747226,
    0.00039845633235672894,
)


@dataclass(frozen=True)
class FixedCartesianDeploymentProfile:
    """Geometry and joint-coordinate contract shared by training and deployment."""

    name: str
    calibration_id: str
    q_reset_rad: tuple[float, ...]
    az_xyz_b_m: tuple[tuple[float, float, float], ...]
    az_xyz_mean_b_m: tuple[float, float, float]
    az_xyz_std_b_m: tuple[float, float, float]
    target_reference_sha256: str
    target_manifest_sha256: str
    joint_zero_offset_deg: tuple[float, ...]

    @property
    def joint_zero_offset_rad(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.joint_zero_offset_deg, dtype=np.float64))


PHYSICAL_CALIBRATED_20260718_PROFILE = FixedCartesianDeploymentProfile(
    name="physical-calibrated-20260718",
    calibration_id="mx_keys_powered_az_20260718",
    q_reset_rad=PHYSICAL_CALIBRATED_20260718_Q_RESET_RAD,
    az_xyz_b_m=PHYSICAL_CALIBRATED_20260718_AZ_XYZ_B_M,
    az_xyz_mean_b_m=PHYSICAL_CALIBRATED_20260718_AZ_XYZ_MEAN_B_M,
    az_xyz_std_b_m=PHYSICAL_CALIBRATED_20260718_AZ_XYZ_STD_B_M,
    target_reference_sha256="cd77d70242f5cac8f3dff8378be559c3d8b26cc816d014cfe9dc16dbb522d301",
    target_manifest_sha256="dca2c7c51d4fabf5512cbeac59e617e289ab6deb0bf0855be5cd21c8c616b2d7",
    joint_zero_offset_deg=PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG,
)
DEPLOYMENT_PROFILES = {PHYSICAL_CALIBRATED_20260718_PROFILE.name: PHYSICAL_CALIBRATED_20260718_PROFILE}


def deployment_profile(name: str) -> FixedCartesianDeploymentProfile:
    try:
        return DEPLOYMENT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown fixed-Cartesian deployment profile {name!r}") from exc


def lerobot_position_to_profile_model_rad(
    position: np.ndarray | list[float], profile: FixedCartesianDeploymentProfile
) -> np.ndarray:
    """Convert a six-value LeRobot reading into the checkpoint's model frame."""

    result = lerobot_position_to_model_rad(position)
    result[:ACTION_DIM] += profile.joint_zero_offset_rad
    return result


def profile_model_rad_to_lerobot_position(
    position_rad: np.ndarray | list[float], profile: FixedCartesianDeploymentProfile
) -> np.ndarray:
    """Convert a six-value model-frame goal into LeRobot command coordinates."""

    encoder_model_rad = np.asarray(position_rad, dtype=np.float64).copy()
    if encoder_model_rad.shape != (ACTION_DIM + 1,) or not np.isfinite(encoder_model_rad).all():
        raise ValueError("model goal must contain six finite joint values")
    encoder_model_rad[:ACTION_DIM] -= profile.joint_zero_offset_rad
    return model_rad_to_lerobot_position(encoder_model_rad)


SEEK_PRESS = 0
RELEASE_KEY = 1
CLEARANCE = 2
PHASE_NAMES = ("seek_press", "release_key", "clearance")


@dataclass(frozen=True)
class FixedCartesianPolicyContract:
    observation_dim: int
    action_dim: int
    hidden_dims: tuple[int, int, int]
    distribution: str = "BetaDistribution[-1,1]"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shape(state: Mapping[str, Any], name: str) -> tuple[int, ...]:
    value = state.get(name)
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"fixed-Cartesian checkpoint is missing {name}")
    return tuple(int(dimension) for dimension in shape)


def _linear_shape(state: Mapping[str, Any], prefix: str) -> tuple[int, int]:
    weight = _shape(state, f"{prefix}.weight")
    bias = _shape(state, f"{prefix}.bias")
    if len(weight) != 2 or bias != (weight[0],):
        raise ValueError(f"fixed-Cartesian checkpoint {prefix} is not a valid linear layer")
    return weight


def infer_policy_contract(state: Mapping[str, Any]) -> FixedCartesianPolicyContract:
    """Validate the exact 22-input, five-action native-Beta actor."""

    mean = _shape(state, "obs_normalizer._mean")
    std = _shape(state, "obs_normalizer._std")
    if mean != (1, OBSERVATION_DIM) or std != mean:
        raise ValueError(
            f"fixed-Cartesian normalizer must contain matching (1, 22) mean/std tensors; got mean={mean}, std={std}"
        )
    hidden_0, input_0 = _linear_shape(state, "mlp.0")
    hidden_1, input_1 = _linear_shape(state, "mlp.2")
    hidden_2, input_2 = _linear_shape(state, "mlp.4")
    distribution_width, input_3 = _linear_shape(state, "mlp.6")
    if input_0 != OBSERVATION_DIM or (input_1, input_2, input_3) != (hidden_0, hidden_1, hidden_2):
        raise ValueError("fixed-Cartesian actor MLP dimensions do not chain")
    if distribution_width != 2 * ACTION_DIM:
        raise ValueError(f"fixed-Cartesian Beta head must contain {2 * ACTION_DIM} values, got {distribution_width}")
    return FixedCartesianPolicyContract(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=(hidden_0, hidden_1, hidden_2),
    )


def frozen_target_xyz(
    letter: str,
    profile: FixedCartesianDeploymentProfile = PHYSICAL_CALIBRATED_20260718_PROFILE,
) -> np.ndarray:
    value = letter.strip().upper()
    if len(value) != 1 or value not in FROZEN_LETTERS:
        raise ValueError(f"fixed-Cartesian target must be one A-Z letter, got {letter!r}")
    return np.asarray(profile.az_xyz_b_m[FROZEN_LETTERS.index(value)], dtype=np.float64)


def normalized_frozen_target_xyz(
    letter: str,
    profile: FixedCartesianDeploymentProfile = PHYSICAL_CALIBRATED_20260718_PROFILE,
) -> np.ndarray:
    mean = np.asarray(profile.az_xyz_mean_b_m, dtype=np.float64)
    std = np.asarray(profile.az_xyz_std_b_m, dtype=np.float64)
    return (frozen_target_xyz(letter, profile) - mean) / std


@dataclass
class FixedCartesianTypingState:
    """Physical evdev counterpart of the seek/release/clearance command."""

    target: str
    phase: int = SEEK_PRESS
    character_index: int = 0
    pressed_letter: str | None = None
    typed: str = ""
    clearance_ticks: int = 0
    phase_started_s: float = 0.0
    failure: str | None = None
    profile: FixedCartesianDeploymentProfile = PHYSICAL_CALIBRATED_20260718_PROFILE
    clearance_m: float = CLEARANCE_M
    clearance_control_ticks: int = CLEARANCE_CONTROL_TICKS

    @classmethod
    def from_text(
        cls,
        target: str,
        *,
        stage: str,
        now_s: float = 0.0,
        profile: FixedCartesianDeploymentProfile = PHYSICAL_CALIBRATED_20260718_PROFILE,
        clearance_m: float = CLEARANCE_M,
        clearance_control_ticks: int = CLEARANCE_CONTROL_TICKS,
    ) -> FixedCartesianTypingState:
        text = target.strip().upper()
        expected_length = STAGE_TARGET_LENGTHS.get(stage.casefold())
        if expected_length is None:
            raise ValueError(f"unknown fixed-Cartesian stage {stage!r}")
        if len(text) != expected_length or not text.isascii() or not text.isalpha():
            raise ValueError(f"{stage.upper()} requires exactly {expected_length} A-Z letter(s), got {target!r}")
        if not np.isfinite(clearance_m) or clearance_m <= 0.0:
            raise ValueError("clearance_m must be finite and positive")
        if clearance_control_ticks < 1:
            raise ValueError("clearance_control_ticks must be positive")
        return cls(
            target=text,
            phase_started_s=float(now_s),
            profile=profile,
            clearance_m=float(clearance_m),
            clearance_control_ticks=int(clearance_control_ticks),
        )

    @property
    def complete(self) -> bool:
        return self.character_index >= len(self.target) and self.failure is None

    @property
    def phase_name(self) -> str:
        return PHASE_NAMES[self.phase]

    def active_letter(self) -> str:
        if self.phase != SEEK_PRESS:
            if self.pressed_letter is None:
                raise RuntimeError("release/clearance phase has no pressed letter")
            return self.pressed_letter
        if self.complete:
            raise RuntimeError("typing target is complete")
        return self.target[self.character_index]

    def phase_elapsed_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - self.phase_started_s)

    def phase_timed_out(self, now_s: float) -> bool:
        return self.phase_elapsed_s(now_s) > PHASE_TIMEOUTS_S[self.phase]

    def _set_phase(self, phase: int, now_s: float) -> None:
        self.phase = phase
        self.phase_started_s = float(now_s)

    def key_down(self, letter: str, *, now_s: float) -> str:
        value = letter.upper()
        if self.phase == RELEASE_KEY:
            self.failure = "press_before_release"
            return self.failure
        if self.phase == CLEARANCE:
            self.failure = "press_before_clearance"
            return self.failure
        if value != self.active_letter():
            self.failure = "wrong_key"
            return self.failure
        self.typed += value
        self.pressed_letter = value
        self._set_phase(RELEASE_KEY, now_s)
        return "target_down"

    def key_up(self, letter: str, *, now_s: float) -> str | None:
        if self.phase != RELEASE_KEY or letter.upper() != self.pressed_letter:
            return None
        self.clearance_ticks = 0
        self._set_phase(CLEARANCE, now_s)
        return "target_up"

    def clearance_tick(self, *, tip_z_b_m: float, any_key_held: bool, now_s: float) -> str | None:
        if self.phase != CLEARANCE:
            return None
        target_z = float(frozen_target_xyz(self.active_letter(), self.profile)[2])
        clear = not any_key_held and float(tip_z_b_m) >= target_z + self.clearance_m
        self.clearance_ticks = self.clearance_ticks + 1 if clear else 0
        if self.clearance_ticks < self.clearance_control_ticks:
            return None
        self.character_index += 1
        self.clearance_ticks = 0
        if self.character_index >= len(self.target):
            return "target_complete"
        self.pressed_letter = None
        self._set_phase(SEEK_PRESS, now_s)
        return "clearance_complete"


def build_observation(
    *,
    q_rad: np.ndarray,
    qd_rad_s: np.ndarray,
    q_command_rad: np.ndarray,
    state: FixedCartesianTypingState,
    now_s: float,
) -> np.ndarray:
    """Build the exact 22-D actor input in IsaacLab term order."""

    q = np.asarray(q_rad, dtype=np.float64)
    qd = np.asarray(qd_rad_s, dtype=np.float64)
    command = np.asarray(q_command_rad, dtype=np.float64)
    if q.shape != (ACTION_DIM,) or qd.shape != (ACTION_DIM,) or command.shape != (ACTION_DIM,):
        raise ValueError("q, qd, and q_command must each contain five values")
    if not np.isfinite(q).all() or not np.isfinite(qd).all() or not np.isfinite(command).all():
        raise ValueError("fixed-Cartesian observation joint values must be finite")
    phase_onehot = np.zeros(3, dtype=np.float64)
    phase_onehot[state.phase] = 1.0
    observation = np.concatenate(
        (
            q - np.asarray(state.profile.q_reset_rad),
            qd,
            command - np.asarray(state.profile.q_reset_rad),
            normalized_frozen_target_xyz(state.active_letter(), state.profile),
            phase_onehot,
            np.asarray([state.phase_elapsed_s(now_s)]),
        )
    ).astype(np.float32, copy=False)
    if observation.shape != (OBSERVATION_DIM,):
        raise RuntimeError("fixed-Cartesian observation shape drift")
    return observation


def integrate_command(
    q_command_rad: np.ndarray,
    action: np.ndarray,
    *,
    rate_scale: float = 1.0,
) -> np.ndarray:
    """Integrate the persistent position target exactly once per 25 Hz actor tick."""

    command = np.asarray(q_command_rad, dtype=np.float64)
    bounded = np.asarray(action, dtype=np.float64)
    if command.shape != (ACTION_DIM,) or bounded.shape != (ACTION_DIM,):
        raise ValueError("command and action must each contain five values")
    if not np.isfinite(command).all() or not np.isfinite(bounded).all():
        raise ValueError("command and action must be finite")
    if not np.isfinite(rate_scale) or not 0.0 < rate_scale <= 1.0:
        raise ValueError("rate_scale must be finite and in (0, 1]")
    if np.any(np.abs(bounded) > 1.0 + 1.0e-6):
        raise ValueError("native Beta action exceeded its [-1, 1] support")
    bounded = np.clip(bounded, -1.0, 1.0)
    candidate = command + bounded * np.asarray(JOINT_RATE_LIMITS_RAD_S) * rate_scale * CONTROL_DT_S
    lower = np.asarray([limit[0] for limit in ARM_JOINT_LIMITS_RAD], dtype=np.float64)
    upper = np.asarray([limit[1] for limit in ARM_JOINT_LIMITS_RAD], dtype=np.float64)
    return np.clip(candidate, lower, upper)


class FixedCartesianPolicy:
    """Load the deterministic mean of the native bounded Beta actor."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.path = Path(checkpoint_path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required to run a fixed-Cartesian checkpoint") from exc

        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        state = checkpoint.get("actor_state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint does not contain actor_state_dict")
        self._torch = torch
        self.contract = infer_policy_contract(state)

        hidden_0, hidden_1, hidden_2 = self.contract.hidden_dims
        self.model = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_DIM, hidden_0),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_0, hidden_1),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_1, hidden_2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_2, 2 * ACTION_DIM),
        )
        self.mean = state["obs_normalizer._mean"].detach().clone()
        self.std = state["obs_normalizer._std"].detach().clone()
        with torch.inference_mode():
            for destination, source in (
                (self.model[0], "mlp.0"),
                (self.model[2], "mlp.2"),
                (self.model[4], "mlp.4"),
                (self.model[6], "mlp.6"),
            ):
                destination.weight.copy_(state[f"{source}.weight"])
                destination.bias.copy_(state[f"{source}.bias"])
        self.model.eval()
        self.iteration = int(checkpoint.get("iter", -1))
        self.sha256 = sha256_file(self.path)

    def action(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (OBSERVATION_DIM,) or not np.isfinite(value).all():
            raise ValueError("fixed-Cartesian observation must contain 22 finite values")
        torch = self._torch
        with torch.inference_mode():
            normalized = (torch.from_numpy(value[None, :]) - self.mean) / (self.std + 0.01)
            logits = self.model(normalized).reshape(2, ACTION_DIM)
            alpha = torch.nn.functional.softplus(logits[0]) + 1.0
            beta = torch.nn.functional.softplus(logits[1]) + 1.0
            action = 2.0 * alpha / (alpha + beta) - 1.0
        result = {
            "distribution_logits": logits.detach().cpu().numpy().astype(np.float64),
            "alpha": alpha.detach().cpu().numpy().astype(np.float64),
            "beta": beta.detach().cpu().numpy().astype(np.float64),
            "action": action.detach().cpu().numpy().astype(np.float64),
        }
        if result["action"].shape != (ACTION_DIM,) or not np.isfinite(result["action"]).all():
            raise RuntimeError("fixed-Cartesian policy produced an invalid action")
        return result
