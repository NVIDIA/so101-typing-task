"""State-based Newton task for the exact 13-key phrase ISAAC LAB 3.0."""
from isaaclab.utils.configclass import configclass

from .physical_calibrated_20260718_env_cfg import (
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianGTCBerlinP1DTransit15EnvCfg,
)
from .phrase_targets import KEY_SLOTS, KEY_XYZ_B_M, TARGET_SLOTS, TARGET_TEXT


@configclass
class SO101KeyboardMXGTCBerlinIsaacLabPhraseEnvCfg(
    SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianGTCBerlinP1DTransit15EnvCfg
):
    task_contract: str = "physical_calibrated_gtc_berlin_state_isaac_lab_3p0_v1"

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.typing
        command.letter_slots = KEY_SLOTS
        command.letter_xyz_b_m = KEY_XYZ_B_M
        command.typeable_slots = KEY_SLOTS
        command.target_slots = KEY_SLOTS
        command.fixed_target_slots = TARGET_SLOTS
        command.letter_length = (len(TARGET_TEXT), len(TARGET_TEXT))
        command.max_len = len(TARGET_TEXT)
        self.episode_length_s = 150.0


def _set_prefix(cfg, length: int) -> None:
    """Shorten the same physical phrase task for staged PPO continuation."""
    command = cfg.commands.typing
    command.fixed_target_slots = TARGET_SLOTS[:length]
    command.letter_length = (length, length)
    command.max_len = length
    cfg.episode_length_s = max(60.0, 12.0 * length)
    cfg.task_contract = f"physical_calibrated_gtc_berlin_state_isaac_lab_3p0_prefix{length}_v1"


@configclass
class SO101KeyboardMXGTCBerlinIsaacLabPhrasePrefix3EnvCfg(SO101KeyboardMXGTCBerlinIsaacLabPhraseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_prefix(self, 3)


@configclass
class SO101KeyboardMXGTCBerlinIsaacLabPhrasePrefix6EnvCfg(SO101KeyboardMXGTCBerlinIsaacLabPhraseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_prefix(self, 6)


@configclass
class SO101KeyboardMXGTCBerlinIsaacLabPhrasePrefix10EnvCfg(SO101KeyboardMXGTCBerlinIsaacLabPhraseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_prefix(self, 10)
