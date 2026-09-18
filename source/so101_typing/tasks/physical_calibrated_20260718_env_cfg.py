# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-101 MX Keys task calibrated from powered physical A-Z contacts on 2026-07-18."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .fixed_cartesian_env_cfg import (
    ARM_JOINT_NAMES,
    SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg,
)
from .fixed_cartesian_targets import FROZEN_ALPHABET_KEY_FACE_XY_M, FROZEN_LETTER_SLOTS
from .registered_mx_env_cfg import (
    REGISTERED_TYPING_GRIPPER_RAD,
    MXKeysKeyboardAssetCfg,
    MXKeysSO101SceneCfg,
)

PHYSICAL_CALIBRATION_ID = "mx_keys_powered_az_20260718"
# Historical dataset identity remains part of the frozen checkpoint contract.
PHYSICAL_CALIBRATION_DATASET_SHA256 = "dca2c7c51d4fabf5512cbeac59e617e289ab6deb0bf0855be5cd21c8c616b2d7"
# Public copy omits machine-specific provenance paths; numerical data is unchanged.
PUBLIC_CALIBRATION_DATASET_SHA256 = "66f0a793bb9953a95b2df6581b2cfb5e048b25c3b19100d2c03c68c963634ff4"
PHYSICAL_CALIBRATION_TRACE_ARCHIVE_SHA256 = "e1bb5b645fddfddca6e02e54c4e0e9c447d3acdaaea406f22b37d51e1d7ea01c"

# q_model = q_lerobot + offset. Pan and wrist-roll remain fixed gauges.
PHYSICAL_TO_MODEL_JOINT_ZERO_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 1.9336272481,
    "elbow_flex": -10.7735791533,
    "wrist_flex": 1.3647648917,
    "wrist_roll": 0.0,
}

# One effective rigid pose in the robot-base frame. It is 1.39 deg from square
# and places the near case edge 125.28 mm beyond the robot base housing.
PHYSICAL_CALIBRATED_KEYBOARD_POSITION_B_M = (0.2625651584, 0.0121353200, 0.0017813943)
PHYSICAL_CALIBRATED_KEYBOARD_RPY_DEG = (-5.0, 0.0506258799, -88.6058757454)
PHYSICAL_CALIBRATED_KEYBOARD_ROTATION_XYZW = (
    -0.030908235122791228,
    0.030781916689009686,
    -0.6977733511584022,
    0.7149891642673463,
)
PHYSICAL_CALIBRATED_KEYBOARD_CASE_MIN_Z_M = -0.014230720631467759

# The physical rest capture converted through the same encoder-to-model map.
PHYSICAL_CALIBRATED_POLICY_REST_JOINT_POS_RAD = {
    "shoulder_pan": -0.08285519086390664,
    "shoulder_lift": -0.13809964128766555,
    "elbow_flex": 0.020637904096827316,
    "wrist_flex": 1.1561739160158462,
    "wrist_roll": 0.03759170696603171,
    "gripper": REGISTERED_TYPING_GRIPPER_RAD,
}
PHYSICAL_CALIBRATED_POLICY_Q_RESET_RAD = tuple(
    PHYSICAL_CALIBRATED_POLICY_REST_JOINT_POS_RAD[name] for name in ARM_JOINT_NAMES
)

# A-Z key-body centers transformed through the same calibrated keyboard pose.
# These are geometric goals; switch state still defines press/release success.
PHYSICAL_CALIBRATED_AZ_XYZ_B_M = (
    (0.24464311501323643, 0.17904083324095146, 0.0042599117494431463),  # A
    (0.22771574743940146, 0.09350215695271337, 0.0036870904084776836),  # B
    (0.22679073451700832, 0.1315108756576093, 0.0037206844201812927),  # C
    (0.24556812802626454, 0.14103211081186945, 0.004226317734447914),  # D
    (0.2646952546429896, 0.14657339002825373, 0.004735047731097078),  # E
    (0.24603063448746113, 0.12202775145942148, 0.004209520728596113),  # F
    (0.24649314099397518, 0.10302339024488046, 0.004192723721098496),  # G
    (0.24695564750048923, 0.08401902903033948, 0.004175926713600884),  # H
    (0.2670077872661949, 0.05155158023136272, 0.0046510626903173875),  # I
    (0.2474181539163683, 0.06501467153998454, 0.004159129709394891),  # J
    (0.24788066033224734, 0.04601031404962956, 0.004142332705188899),  # K
    (0.24834316674812643, 0.027005956559274617, 0.004125535700982907),  # L
    (0.22864076027115957, 0.05549344197200344, 0.0036534964000657002),  # M
    (0.2281782538552805, 0.0744977994623584, 0.003670293404271694),  # N
    (0.267470293500804, 0.03254723018937984, 0.00463426569269464),  # O
    (0.267932799916683, 0.013542872699024881, 0.004617468688488646),  # P
    (0.26377024175458463, 0.1845821073365799, 0.004768641741566328),  # Q
    (0.2651577612401387, 0.12756902508952672, 0.004718250720307843),  # R
    (0.2451056215197505, 0.16003647202641047, 0.00424311474194553),  # S
    (0.26562026765601776, 0.10856466759917176, 0.004701453716101849),  # T
    (0.26654528066904587, 0.07055594517008976, 0.004667859701106621),  # U
    (0.2272532410235224, 0.11250651444306832, 0.0037038874126836775),  # V
    (0.26423274822711057, 0.16557774751860868, 0.004751844735303072),  # W
    (0.22632822801049426, 0.15051523687215032, 0.003737481427678906),  # X
    (0.2660827741625318, 0.08956030638463075, 0.004684656708604237),  # Y
    (0.22586570379616713, 0.16952032569953857, 0.003754279078277244),  # Z
)
PHYSICAL_CALIBRATED_AZ_XYZ_MEAN_B_M = (
    0.24875869794180977,
    0.104053144318032,
    0.0042535491608431805,
)
PHYSICAL_CALIBRATED_AZ_XYZ_STD_B_M = (
    0.015494051359203873,
    0.048965322097747226,
    0.00039845633235672894,
)
PHYSICAL_CALIBRATED_TARGET_MAP_SHA256 = "cd77d70242f5cac8f3dff8378be559c3d8b26cc816d014cfe9dc16dbb522d301"


_REGISTERED_KEYBOARD_ASSET_CFG = MXKeysKeyboardAssetCfg()


@configclass
class PhysicalCalibrated20260718KeyboardAssetCfg(PresetCfg):
    """The existing MX Keys articulation at the powered-contact calibrated pose."""

    physx = _REGISTERED_KEYBOARD_ASSET_CFG.physx.replace(
        init_state=_REGISTERED_KEYBOARD_ASSET_CFG.physx.init_state.replace(
            pos=PHYSICAL_CALIBRATED_KEYBOARD_POSITION_B_M,
            rot=PHYSICAL_CALIBRATED_KEYBOARD_ROTATION_XYZW,
        )
    )
    newton_mjwarp = _REGISTERED_KEYBOARD_ASSET_CFG.newton_mjwarp.replace(
        init_state=_REGISTERED_KEYBOARD_ASSET_CFG.newton_mjwarp.init_state.replace(
            pos=PHYSICAL_CALIBRATED_KEYBOARD_POSITION_B_M,
            rot=PHYSICAL_CALIBRATED_KEYBOARD_ROTATION_XYZW,
        )
    )
    default = newton_mjwarp


@configclass
class PhysicalCalibrated20260718SO101SceneCfg(MXKeysSO101SceneCfg):
    """Registered scene with the 2026-07-18 effective physical calibration."""

    keyboard: ArticulationCfg = PhysicalCalibrated20260718KeyboardAssetCfg()
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, PHYSICAL_CALIBRATED_KEYBOARD_CASE_MIN_Z_M)),
        spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
        collision_group=-1,
    )


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP0EnvCfg(SO101KeyboardMXFixedCartesianAnchorBenchP0EnvCfg):
    """Run-76 P0 contract against the powered-contact calibrated model frame."""

    keyboard_profile: str = PHYSICAL_CALIBRATION_ID
    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_anchorbench_p0_v0"
    target_reference_sha256: str = PHYSICAL_CALIBRATED_TARGET_MAP_SHA256
    target_manifest_sha256: str = PHYSICAL_CALIBRATION_DATASET_SHA256
    physical_alphabet_key_face_xy_m: tuple[float, float] = FROZEN_ALPHABET_KEY_FACE_XY_M
    scene: InteractiveSceneCfg = PhysicalCalibrated20260718SO101SceneCfg(
        num_envs=4096,
        env_spacing=1.0,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = PHYSICAL_CALIBRATED_POLICY_REST_JOINT_POS_RAD.copy()
        self.actions.action.q_reset_ref = PHYSICAL_CALIBRATED_POLICY_Q_RESET_RAD
        self.commands.typing.q_reset_ref = PHYSICAL_CALIBRATED_POLICY_Q_RESET_RAD
        self.commands.typing.letter_slots = FROZEN_LETTER_SLOTS
        self.commands.typing.letter_xyz_b_m = PHYSICAL_CALIBRATED_AZ_XYZ_B_M
        self.commands.typing.target_xyz_mean_b_m = PHYSICAL_CALIBRATED_AZ_XYZ_MEAN_B_M
        self.commands.typing.target_xyz_std_b_m = PHYSICAL_CALIBRATED_AZ_XYZ_STD_B_M


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP0EnvCfg
):
    """Run-77 P1A contract against the powered-contact calibrated model frame."""

    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1a_letters2_v0"

    def __post_init__(self):
        super().__post_init__()
        self.commands.typing.letter_length = (2, 2)
        self.commands.typing.max_len = 3
        self.episode_length_s = 21.0


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianBaselineP1AEnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg
):
    """Independent two-letter P1A using the original Workshop actuator profile."""

    actuator_profile: str = "baseline"
    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_baseline_p1a_letters2_v0"


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianUSDDriveP1AEnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg
):
    """Independent two-letter P1A using the effective loaded-USD drives."""

    actuator_profile: str = "usd_drive"
    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_usd_drive_p1a_letters2_v0"


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1BEnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg
):
    """Exactly three letters under the qualified physical P1A contract."""

    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1b_letters3_v0"

    def __post_init__(self):
        super().__post_init__()
        self.commands.typing.letter_length = (3, 3)
        self.commands.typing.max_len = 3
        self.episode_length_s = 31.0


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1CEnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg
):
    """Exactly four letters under the qualified physical P1A contract."""

    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1c_letters4_v0"

    def __post_init__(self):
        super().__post_init__()
        self.commands.typing.letter_length = (4, 4)
        self.commands.typing.max_len = 4
        self.episode_length_s = 41.0


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DEnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1AEnvCfg
):
    """Exactly six letters under the qualified physical P1A contract."""

    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1d_letters6_v0"

    def __post_init__(self):
        super().__post_init__()
        self.commands.typing.letter_length = (6, 6)
        self.commands.typing.max_len = 6
        self.episode_length_s = 61.0


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DTransit15EnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DEnvCfg
):
    """Six-letter P1D requiring a sustained 15 mm lift before lateral transit."""

    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_anchorbench_p1d_transit15_letters6_v0"

    def __post_init__(self):
        super().__post_init__()
        self.commands.typing.clearance_m = 0.015
        self.rewards.low_clearance_lateral_motion.params["clearance_m"] = 0.015
        self.terminations.scrape.params["clearance_m"] = 0.015


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianBaselineP1DTransit15EnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DTransit15EnvCfg
):
    """Matched Transit15 baseline using the original Workshop actuator profile."""

    actuator_profile: str = "baseline"
    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_baseline_p1d_transit15_letters6_v0"


@configclass
class SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianUSDDriveP1DTransit15EnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianP1DTransit15EnvCfg
):
    """Matched Transit15 continuation using the effective loaded-USD drives."""

    actuator_profile: str = "usd_drive"
    task_contract: str = "physical_calibrated_20260718_fixed_cartesian_usd_drive_p1d_transit15_letters6_v0"
