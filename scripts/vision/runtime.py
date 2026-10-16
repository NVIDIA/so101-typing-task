"""Shared, Docker-only simulator setup for vision training and evaluation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/reinforcement_learning/rsl_rl'))


def parser():
    from isaaclab.app import add_launcher_args
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--seed', type=int, default=4811)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task-config', type=Path, required=True)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--target', choices=('NEWTON', 'ISAAC LAB 3.0'), default='NEWTON')
    add_launcher_args(p)
    return p


def make_config(args):
    from isaaclab_tasks.core.dexsuite.config.so101.vision_typing_env_cfg import VisionIsaacLabPhraseEnvCfg, VisionTypingEnvCfg
    from isaaclab_tasks.core.dexsuite.config.so101.mdp.vision_observations import reset_vision_domain
    from isaaclab_tasks.core.dexsuite.config.so101.tabletop_scene import apply_config as apply_table
    from public_task import load_config, apply_config
    from isaaclab.managers import EventTermCfg
    from isaaclab_tasks.utils.hydra import resolve_presets
    cfg = resolve_presets(VisionIsaacLabPhraseEnvCfg() if args.target == 'ISAAC LAB 3.0' else VisionTypingEnvCfg())
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = 'cuda:0'
    apply_config(cfg, load_config(args.task_config))
    apply_table(cfg, json.loads((ROOT/'configs/scenes/recorded-white-tabletop.json').read_text()))
    points = cfg.events.reset_keyboard.params['letter_xyz_keyboard_m']
    cfg.events.reset_keyboard = EventTermCfg(func=reset_vision_domain, mode='reset',
                                           params={'letter_xyz_keyboard_m': points})
    return cfg


def create_env(cfg):
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    return RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg), clip_actions=None)
