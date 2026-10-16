"""Inspect a stationary SO-101 scene in an interactive Newton or Isaac Sim viewport."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import gymnasium as gym
import torch
from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab_tasks.core.dexsuite.config.so101  # noqa: F401 -- register the typing tasks
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config
from public_task import apply_config, load_config
from public_physics import apply_config as apply_physics, load_config as load_physics
from recorded_tabletop import apply_config as apply_tabletop, load_config as load_tabletop

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', required=True)
parser.add_argument('--task-config', type=Path)
parser.add_argument('--physics-config', type=Path)
parser.add_argument('--tabletop-config', type=Path, help='Recorded white table and backdrop dimensions in mm')
parser.add_argument('--export-scene', action='store_true', help='Export posed scene.usdc for OVRTX')
parser.add_argument('--snapshot', action='store_true', help='Save the initial Newton view as scene.png')
parser.add_argument('--seed', type=int, default=1307)
parser.add_argument('--frames', type=int, default=0)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--zero-mapping', type=Path)
parser.add_argument('--zero-pose', choices=['usd', 'lerobot-direct', 'lerobot-fit'], default='usd')
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
if args.frames < 0 or args.headless:
    parser.error('use a visible viewer and --frames >= 0 (0 means unlimited)')
sys.argv = [sys.argv[0]] + remaining
if args.snapshot:
    args.enable_cameras = True


class ZeroComparison:
    """Render-only convention comparison; fitted poses are not physical measurements."""

    joint_names = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
    modes = ('usd', 'lerobot-direct', 'lerobot-fit')
    labels = ('USD arm zero', 'LeRobot zero: direct', 'LeRobot zero: fitted prediction')

    def __init__(self, env, viewers):
        self.env = env
        self.robot = env.scene['robot']
        self.report = json.loads(args.zero_mapping.read_text())
        if self.report.get('joint_order') != list(self.joint_names):
            raise ValueError('zero comparison mapping must use the five named arm joints in order')
        self.offset = self.report.get('offset_deg')
        if (not isinstance(self.offset, list) or len(self.offset) != 5 or
                any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 15 for x in self.offset)):
            raise ValueError('zero comparison requires five finite offsets within +/-15 degrees')
        if not args.task_config or self.report.get('task_config_sha256') != hashlib.sha256(args.task_config.read_bytes()).hexdigest():
            raise ValueError('zero comparison scene differs from the fitted scene')
        if self.report.get('identities', {}).get('robot_usd_sha256') != hashlib.sha256(Path(os.environ['SO101_TYPING_USD']).read_bytes()).hexdigest():
            raise ValueError('zero comparison robot USD differs from the fitted model')
        self.selected = self.modes.index(args.zero_pose)
        self.applied = None
        self.visited = []
        self.joint_ids = [self.robot.joint_names.index(n) for n in (*self.joint_names, 'gripper')]
        self.env_ids = torch.tensor([0], device=env.device)
        for visualizer in viewers:
            visualizer._viewer.register_ui_callback(self.draw, position='side')
        self.update()

    def draw(self, imgui):
        imgui.separator()
        imgui.text('Arm zero conventions')
        for i, label in enumerate(self.labels):
            if imgui.button(('> ' if self.selected == i else '') + label):
                self.selected = i
        imgui.text('Direct zero = USD zero by assumption.')
        imgui.text('Fitted pose is a prediction,')
        imgui.text('not a measured physical zero.')
        imgui.text('q_model = q_LeRobot + offset')
        imgui.text('Displayed model angles (deg):')
        values = self.offset if self.selected == 2 else [0.] * 5
        for name, value in zip(self.joint_names, values):
            imgui.text(f'{name}: {value:+.3f}')
        imgui.text('Closed jaw: gripper = -10 deg')
        imgui.text('Render only; no physics or hardware.')

    def update(self):
        if self.selected == self.applied:
            return
        values = self.offset if self.selected == 2 else [0.] * 5
        q = torch.tensor([[math.radians(x) for x in (*values, -10.)]], device=self.env.device)
        self.robot.write_joint_position_to_sim_index(position=q, joint_ids=self.joint_ids, env_ids=self.env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=self.joint_ids, env_ids=self.env_ids)
        self.env.sim.forward()
        self.expected = self.robot.data.joint_pos.torch.clone()
        self.applied = self.selected
        self.visited.append({'pose': self.modes[self.selected], 'model_arm_deg': list(values)})
        print(f'ZERO POSE: {self.labels[self.selected]}; model arm degrees = {values}', flush=True)

    def evidence(self):
        return {'mapping_sha256': hashlib.sha256(args.zero_mapping.read_bytes()).hexdigest(),
                'mapping_identities': self.report['identities'], 'offset_deg': self.offset,
                'joint_order': list(self.joint_names), 'visited': self.visited,
                'closed_gripper_model_deg': -10., 'physical_zero_independently_measured': False,
                'note': 'The fitted LeRobot-zero pose is an extrapolation of the contact fit, not ground truth.'}


@hydra_task_config(args.task, 'rsl_rl_cfg_entry_point')
def main(env_cfg, agent_cfg):
    if args.task_config:
        apply_config(env_cfg, load_config(args.task_config))
    if args.physics_config:
        apply_physics(env_cfg, load_physics(args.physics_config))
    tabletop = apply_tabletop(env_cfg, load_tabletop(args.tabletop_config)) if args.tabletop_config else None
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    if args.device:
        env_cfg.sim.device = args.device
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg, render_mode='rgb_array' if args.snapshot else None).unwrapped
        frames = 0
        try:
            env.reset()
            env.sim.forward()
            eye, lookat = (env_cfg.viewer.eye, env_cfg.viewer.lookat) if tabletop else ((.62, .48, .42), (.20, .06, .07))
            env.sim.set_camera_view(eye, lookat)
            if tabletop:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                (args.out.parent/'tabletop.json').write_text(json.dumps(tabletop, indent=2)+'\n')
            # Keep references: SimulationContext removes closed viewers from its list.
            viewers = tuple(env.sim._visualizers)
            if not viewers:
                raise RuntimeError('Scene viewer did not initialize')
            assets = {name: env.scene[name] for name in ('robot', 'keyboard')}
            initial = {name: (asset.data.joint_pos.torch.clone(), asset.data.root_pos_w.torch.clone(),
                              asset.data.root_quat_w.torch.clone()) for name, asset in assets.items()}
            if args.export_scene:
                from export_tabletop_usd import export_scene
                export_scene(env, args.out.parent/'scene.usdc', eye, lookat)
            initial_steps = env.sim.get_physics_step_count()
            comparison = ZeroComparison(env, viewers) if args.zero_mapping else None
            print('SCENE READY: stationary robot and keyboard; close the viewer or press Ctrl-C to exit.', flush=True)
            try:
                while any(not v.is_closed and v.is_running() for v in viewers):
                    if comparison:
                        comparison.update()
                    # Rendering pumps camera/window input without advancing physics or task state.
                    env.sim.render()
                    frames += 1
                    if args.snapshot and frames == 5:
                        import imageio.v2 as imageio
                        args.out.parent.mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(args.out.parent/'scene.png', env.render())
                        print(f'SCENE SNAPSHOT: {args.out.parent / "scene.png"}', flush=True)
                    if args.frames and frames >= args.frames:
                        break
                    time.sleep(1 / 30)
            except KeyboardInterrupt:
                pass
            unchanged = all(torch.equal(before, after) for name, asset in assets.items()
                            for index, (before, after) in enumerate(zip(initial[name], (asset.data.joint_pos.torch,
                                asset.data.root_pos_w.torch, asset.data.root_quat_w.torch)))
                            if not (comparison and name == 'robot' and index == 0))
            pose_matches = comparison is None or torch.equal(comparison.expected, assets['robot'].data.joint_pos.torch)
            physics_steps = env.sim.get_physics_step_count() - initial_steps
            report = {'mode': 'zero_convention_comparison' if comparison else 'static_scene_viewer',
                      'visualizers': [type(viewer).__name__ for viewer in viewers],
                      'frames': frames, 'policy_used': False, 'tabletop': tabletop, 'physics_steps_during_view': physics_steps,
                      'task_config': load_config(args.task_config) if args.task_config else None,
                      'task_config_sha256': hashlib.sha256(args.task_config.read_bytes()).hexdigest()
                      if args.task_config else None}
            if comparison:
                report.update(comparison.evidence())
                report.update(keyboard_and_robot_base_unchanged=unchanged, displayed_pose_matches=pose_matches)
            else:
                report['robot_and_keyboard_unchanged'] = unchanged
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + '\n')
            if not unchanged or not pose_matches or physics_steps:
                raise RuntimeError('Scene moved during render-only inspection')
            print('SCENE CLOSED: no physics steps; keyboard and robot base remained fixed.', flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
