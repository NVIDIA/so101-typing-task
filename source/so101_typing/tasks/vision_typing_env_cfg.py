"""OVRTX wrist-camera typing with a privileged critic."""
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, EventTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_ov.renderers import OVRTXRendererCfg

from .physical_calibrated_20260718_env_cfg import SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianGTCBerlinP1DTransit15EnvCfg
from .fixed_cartesian_env_cfg import FixedCartesianObservationsCfg
from .mdp.vision_observations import VisionTypingCommandCfg, vision_proprio, wrist_rgb_history
from .phrase_targets import KEY_SLOTS, KEY_XYZ_B_M, TARGET_TEXT


@configclass
class VisionObservationsCfg:
    # The exact target geometry is restricted to teacher/critic observations.
    policy: FixedCartesianObservationsCfg.PolicyCfg = FixedCartesianObservationsCfg.PolicyCfg()

    @configclass
    class ProprioCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=vision_proprio)
        concatenate_terms = True
        enable_corruption = False

    @configclass
    class RGBCfg(ObservationGroupCfg):
        image = ObservationTermCfg(func=wrist_rgb_history)
        concatenate_terms = True
        enable_corruption = False

    proprio: ProprioCfg = ProprioCfg()
    rgb: RGBCfg = RGBCfg()


@configclass
class VisionTypingEnvCfg(SO101KeyboardMXPhysicalCalibrated20260718FixedCartesianGTCBerlinP1DTransit15EnvCfg):
    observations: VisionObservationsCfg = VisionObservationsCfg()
    task_contract: str = 'vision_ppo_gtc_berlin_6letters_ovrtx_v1'

    def __post_init__(self):
        super().__post_init__()
        from .vision_renderer import register_renderer
        register_renderer()
        # Copy the state task's switching thresholds, geometry and timing.
        values = vars(self.commands.typing).copy()
        values.pop('class_type', None)
        self.commands.typing = VisionTypingCommandCfg(**values)
        self.scene.camera_wrist = CameraCfg(
            prim_path='{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera',
            update_period=.04, height=120, width=160, data_types=['rgb'],
            update_latest_camera_pose=True, spawn=None,
            renderer_cfg=OVRTXRendererCfg(log_level='warn'))
        self.scene.lazy_sensor_update = True
        self.sim.render_interval = self.decimation


@configclass
class VisionIsaacLabPhraseEnvCfg(VisionTypingEnvCfg):
    """Random 13-character camera-policy training; ISAAC LAB 3.0 at evaluation."""

    task_contract: str = 'vision_ppo_gtc_berlin_random13_38keys_ovrtx_v2'

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.typing
        command.letter_slots = KEY_SLOTS
        command.letter_xyz_b_m = KEY_XYZ_B_M
        command.typeable_slots = KEY_SLOTS
        command.target_slots = KEY_SLOTS
        # Sample each character from the same 38 keys encoded in observations.
        # VisionTypingCommand applies the explicit target only during evaluation.
        command.fixed_target_slots = None
        command.letter_length = (len(TARGET_TEXT), len(TARGET_TEXT))
        command.max_len = len(TARGET_TEXT)
        self.episode_length_s = 150.0
