# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import MultiAssetSpawnerCfg, SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .keyboards import TYPING_KEYBOARD_POOL
from .SO101.so101_cfg import SO101_CFG


@configclass
class KeyboardAssetCfg(PresetCfg):
    """Backend-dependent keyboard articulation, selected by the ``physics=`` preset.

    PhysX flattens all per-env articulation instances onto one axis, so the ``fixed_dof`` partition
    (18 roots/env) exposes every key. IsaacLab's Newton :class:`ArticulationData` reads only the first
    articulation per env (``[:, 0]``), so Newton uses a single 108-DOF articulation instead.
    """

    # PhysX: 18 articulation roots/env under ``parts/part_*`` (its ArticulationView flattens them).
    physx = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Keyboard",
        articulation_root_prim_path="/parts/part_.*",
        spawn=MultiAssetSpawnerCfg(assets_cfg=list(TYPING_KEYBOARD_POOL.spawners_partitioned)),
        init_state=ArticulationCfg.InitialStateCfg(
            # NOTE: placeholder pose; tune so the keyboard sits within the SO101's reach on the table.
            pos=(0.285, 0.0, 0.01),
            rot=(0.0, 0.0, -0.7071068, 0.7071068),  # -90 deg about Z (xyzw)
            joint_pos={"key_.*_joint": 0.0},
            joint_vel={"key_.*_joint": 0.0},
        ),
        # No IsaacLab actuator: the keys are passive springs driven by their USD-authored ``DriveAPI``
        # gains, which both PhysX and Newton import via ``add_usd``; the policy acts on the robot, not
        # the keys, so the imported drive alone provides the key return force.
        actuators={},
    )
    # Newton: one 108-DOF articulation/env, root auto-resolved at ``prim_path`` (Newton's
    # ArticulationData reads only the first articulation per env, so separate parts would be invisible).
    newton_mjwarp = physx.replace(
        articulation_root_prim_path=None,
        # MX Keys only: the converted scan is the last entry of the pool bank.
        spawn=MultiAssetSpawnerCfg(assets_cfg=TYPING_KEYBOARD_POOL.spawners_single),
    )
    default = newton_mjwarp


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """SO101 keyboard-typing scene.

    Inherits :class:`InteractiveSceneCfg` directly (rather than ``dexsuite.SceneCfg``) so there is no
    inherited manipulable ``object`` - this task's manipuland is the keyboard articulation below.
    """

    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # keyboard: one logical keyboard per env from a heterogeneous pool of full 108-key variants.
    # ``KeyboardAssetCfg`` is a physics preset: ``fixed_dof`` parts for PhysX, single 108-DOF for Newton
    # (see the preset's docstring). The ``physics=`` selector resolves it before the scene is built.
    keyboard: ArticulationCfg = KeyboardAssetCfg()

    # Contact sensor on ALL robot bodies (every link ends in ``_link``; keyboard keys are named after
    # characters, so this pattern excludes them). Reports the net contact force per body, used to drive the
    # >10 N over-force termination (see TerminationsCfg). Needs ``history_length >= 1`` because
    # ``mdp.illegal_contact`` reads ``net_forces_w_history``, and the robot spawn must set
    # ``activate_contact_sensors=True`` (done in the env __post_init__).
    robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_link",
        update_period=0.0,
        history_length=1,
        track_pose=False,
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
        collision_group=-1,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# Sampled typing targets: every real key EXCEPT backspace - the whole board, including the numpad. Backspace
# is the one exclusion because it is the dedicated delete key: pressing it never registers as a typed key
# (see ``LetterTypingCommand._update_command``), so a backspace target could never be satisfied.
_TYPEABLE_SLOTS: tuple[int, ...] = tuple(
    slot for slot in TYPING_KEYBOARD_POOL.active_slots if slot != TYPING_KEYBOARD_POOL.backspace_slot
)


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    typing = mdp.LetterTypingCommandCfg(
        asset_name="robot",
        object_name="keyboard",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        letter_length=(1, 5),
        max_len=5,
        # letter_full shows both target and typed letters; letter_left shows only the remaining target
        command_mode="letter_full",
        typeable_slots=_TYPEABLE_SLOTS,
        backspace_slot=TYPING_KEYBOARD_POOL.backspace_slot,
        slot_labels=TYPING_KEYBOARD_POOL.slot_labels,
        # Reset behavior, all under `reset`: (1) IK-snap the arm so the moving-jaw tip hovers above the first
        # key at a downward approach pitch (SO101-specific joints/body/offset); (2) a success-conditioned replay
        # curriculum whose snapshots are each built by running that same snap once per candidate. ``ik=None``
        # disables both. On the first reset the curriculum caches one snapshot per buffer slot; thereafter each
        # env either takes the normal random reset or is restored to a buffered snapshot, biased (Beta at 50%
        # success) toward states the policy half-solves.
        reset=mdp.LetterTypingCommandCfg.ResetCfg(
            enabled=True,
            ik=mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                body_name="gripper_link",
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=(-0.0079, -0.000218121, -0.0981274)
                ),
            ),
            ik_rpy_deg=(0.0, 45.0, 0.0),  # (roll, pitch, yaw) [deg]
            ik_hover_height=0.02,
            ik_iters=(1, 4),
            # Jitter the IK seed like the old reset_joints_by_offset event (which the IK snap overwrote): gives
            # reset start-states the same joint diversity as the no-IK case (mostly on low ik_iters snaps).
            ik_seed_joint_noise=0.25,
            buffer_size=8192,
            normal_weight=0.1,
            pre_solve_reset=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": [-0.0, 0.0],
                        "y": [-0.0, 0.0],
                        "z": [0.015, 0.05],
                        "yaw": [-0.1, 0.1],
                        "roll": [0.0, 0.75],
                    },
                    "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
                    "asset_cfg": SceneEntityCfg("keyboard"),
                },
            ),
        ),
    )


@configclass
class SO101RelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        scale=0.02,
    )


@configclass
class SO101ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Sequenced one-hot typing command: the target word and the typed buffer, each a one-hot sequence
        # over key slots. The compact perception group supplies current and next key positions, so this
        # group preserves symbolic typing progress without requiring the full keyboard position map.
        target_keys_onehot = ObsTerm(func=mdp.target_keys_onehot, params={"command_name": "typing"})
        typed_keys_onehot = ObsTerm(func=mdp.typed_keys_onehot, params={"command_name": "typing"})

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Observations for perception group."""

        # Compact key goals in the robot base frame. ``current_key_xyz`` is the key to press now
        # (or Backspace while correcting a wrong suffix); ``next_key_xyz`` is the key that would matter
        # after one successful current action, giving the policy a short lookahead without the full map.
        current_key_xyz = ObsTerm(
            func=mdp.current_key_position_b,
            clip=(-2.0, 2.0),
            params={
                "command_name": "typing",
                "keyboard_cfg": SceneEntityCfg("keyboard"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        next_key_xyz = ObsTerm(
            func=mdp.next_key_position_b,
            clip=(-2.0, 2.0),
            params={
                "command_name": "typing",
                "keyboard_cfg": SceneEntityCfg("keyboard"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class EventCfg:
    """Reset-mode events (shared by all physics backends)."""

    reset_keyboard = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.0, 0.0],
                "y": [-0.0, 0.0],
                "z": [0.015, 0.05],
                "yaw": [-0.1, 0.1],
                "roll": [0.0, 0.75],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("keyboard"),
        },
    )


@configclass
class SO101ReorientRewardCfg:
    # High-water-mark ratchet: +1 when the correct-prefix length reaches a new episode max, -1 on a new
    # episode min, else 0. Positive reward per episode is bounded, so backspace/retype (or type-wrong/
    # backspace) loops cannot farm it; the new-min penalty discourages deleting already-correct progress.
    typing_progress = RewTerm(func=mdp.letter_typing_progress, weight=2.0, params={"command_name": "typing"})

    success = RewTerm(func=mdp.typing_success, weight=50.0, params={"command_name": "typing"})

    mechanical_power = RewTerm(func=mdp.mechanical_power, weight=-0.0005)

    early_termination = RewTerm(func=mdp.is_terminated_term, weight=-10, params={"term_keys": ["abnormal_robot"]})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=mdp.joint_vel_out_of_limit)

    # Terminate immediately if any robot body's contact force exceeds 10 N (a healthy typing press is
    # ~1-5 N; 10 N rejects hard strikes). Uses the ``robot_contact`` sensor over all links.
    excessive_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot_contact"), "threshold": 10.0},
    )

    # End the episode once the whole target word is typed correctly (goal reached, not a time-out).
    success = DoneTerm(func=mdp.typing_complete, params={"command_name": "typing"})


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_max_rigid_patch_count=16 * 5 * 2**15,
        gpu_found_lost_pairs_capacity=2**27,
        gpu_total_aggregate_pairs_capacity=2**27,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=600,
            nconmax=600,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    default = newton_mjwarp


@configclass
class SO101KeyboardEnvCfg(ManagerBasedRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(eye=(0.75, 0.0, 0.25), lookat=(0.0, 0.0, 0.1), origin_type="env", env_index=0)
    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)
    observations: SO101ObservationsCfg = SO101ObservationsCfg()
    actions: SO101RelJointPosActionCfg = SO101RelJointPosActionCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: SO101ReorientRewardCfg = SO101ReorientRewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    sim: SimulationCfg = SimulationCfg(
        physics=PhysicsCfg(),
        dt=0.01,
    )

    def __post_init__(self):
        self.decimation = 4  # 100 Hz sim -> 25 Hz control
        self.episode_length_s = 6.0
        self.sim.render_interval = 1
        # Enable contact reporting on the robot so the `robot_contact` sensor can read net contact forces
        # (needed by the >10 N `excessive_contact` termination). Uses a spawn copy to avoid mutating the
        # shared SO101_CFG.spawn.
        self.scene.robot.spawn = self.scene.robot.spawn.replace(activate_contact_sensors=True)
        # Single source of truth: the envs the visualizer renders are exactly the envs whose typing debug
        # markers (letter banner + next-key halo) are drawn.
        # viz_env_ids = [i for i in range(1)]
        # self.commands.typing.viz_env_ids = tuple(viz_env_ids)
        # self.sim.visualizer_cfgs = preset(  # type: ignore
        #     default=[
        #         KitVisualizerCfg(
        #             headless=True,
        #             visible_env_indices=viz_env_ids,
        #             eye=(0.75, 0.0, 0.25),
        #             lookat=(0.0, 0.0, 0.1),
        #         )
        #     ],
        #     newton_mjwarp=[
        #         NewtonVisualizerCfg(
        #             headless=True,
        #             eye=(0.75, 0.0, 0.25),
        #             lookat=(0.0, 0.0, 0.1),
        #             visible_env_indices=viz_env_ids,
        #         )
        #     ],
        #     physx=[
        #         KitVisualizerCfg(
        #             headless=True,
        #             visible_env_indices=viz_env_ids,
        #             eye=(0.75, 0.0, 0.25),
        #             lookat=(0.0, 0.0, 0.1),
        #         )
        #     ],
        # )
