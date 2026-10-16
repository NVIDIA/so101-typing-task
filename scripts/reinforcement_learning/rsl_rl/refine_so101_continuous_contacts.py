# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Parallel local search of scene/joint parameters using complete recorded replays.

All supplied recordings are optimization data. Never sends hardware commands.
Scores unchanged USB down/up sequences, without removing bounce or extra keys.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab_tasks.core.dexsuite.config.so101  # noqa: F401 -- register the typing tasks
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from so101_typing.hardware.scene_check import load_recording
from so101_typing.hardware.fk_probe_trajectory import TASK, digest
from so101_typing.hardware.so101_kinematics import JOINT_NAMES
from public_task import apply_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trajectory', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--task', default=TASK)
parser.add_argument('--rounds', type=int, default=2)
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]]+remaining


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a):
        current = [i+1]
        for j, y in enumerate(b):
            current.append(min(current[-1]+1, previous[j+1]+1, previous[j]+(x != y)))
        previous = current
    return previous[-1]


@hydra_task_config(args.task, 'rsl_rl_cfg_entry_point')
def main(env_cfg, agent_cfg):
    spec = json.loads(args.trajectory.read_text())
    records = [load_recording(args.trajectory.parent/p) for p in spec['recordings']]
    reports = [json.loads((args.trajectory.parent/p).read_text()) for p in spec['starting_reports']]
    if any(digest(Path(os.environ['SO101_TYPING_USD'])) != r['robot_usd_sha256'] for r in records):
        raise ValueError('runtime USD differs from recording')
    apply_config(env_cfg, reports[0]['task_config'])
    count = len(records)
    candidates_count = 23+len(reports)-1
    env_cfg.scene.num_envs = count*candidates_count
    env_cfg.seed = 1307
    env_cfg.decimation = 1
    env_cfg.actions.action.control_dt_s = float(env_cfg.sim.dt)
    dt = float(env_cfg.sim.dt)
    lengths = [len(np.arange(0, r['samples'][-1]['time_s']-r['samples'][0]['time_s'], dt)) for r in records]
    trajectory = np.empty((max(lengths), count, 6))
    velocity = np.empty_like(trajectory)
    expected = [[(e['char'].upper(), e['edge']) for e in r['events']] for r in records]
    for j, r in enumerate(records):
        times = np.array([s['time_s'] for s in r['samples']]); times -= times[0]
        q = np.array([s['q_model_rad'] for s in r['samples']])
        dq = np.gradient(q, times, axis=0)
        trajectory[:, j] = np.stack([np.interp(np.arange(max(lengths))*dt, times, q[:, k]) for k in range(6)], axis=1)
        velocity[:, j] = np.stack([np.interp(np.arange(max(lengths))*dt, times, dq[:, k]) for k in range(6)], axis=1)
    seeds = []
    for report in reports:
        kb = report['task_config']['keyboard']
        seeds.append(np.r_[next(iter(report['candidates'].values()))['offset_deg'],
                           np.array(kb['position_m'])*1000, Rotation.from_quat(kb['rotation_xyzw']).as_euler('xyz', degrees=True)])
    if args.out.exists():
        raise ValueError('refusing to overwrite search result')
    history = []
    with launch_simulation(env_cfg, args):
        base = gym.make(args.task, cfg=env_cfg)
        try:
            base.reset(); env = base.unwrapped
            robot, keyboard = env.scene['robot'], env.scene['keyboard']
            command = env.command_manager.get_term('typing')
            joint_ids = [robot.joint_names.index(n) for n in JOINT_NAMES]
            labels = [label.upper() for label in command.cfg.slot_labels]
            lower = keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 0]
            upper = keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 1]
            q = torch.tensor(trajectory, device=env.device, dtype=torch.float32).repeat(1, candidates_count, 1)
            dq = torch.tensor(velocity, device=env.device, dtype=torch.float32).repeat(1, candidates_count, 1)
            best = None
            for round_index in range(args.rounds):
                anchor = seeds[0] if best is None else np.array(best['parameters'])
                steps = np.array([.5]*5+[1.]*3+[.5]*3)/(2**round_index)
                values = [anchor.copy()]
                for dim in range(11):
                    for sign in [-1, 1]:
                        x = anchor.copy(); x[dim] += sign*steps[dim]
                        x[:5] = np.clip(x[:5], -15, 15); values.append(x)
                values += seeds[1:]
                base.reset()
                roots = np.array([np.r_[x[5:8]/1000, Rotation.from_euler('xyz', x[8:11], degrees=True).as_quat()] for x in values])
                roots = torch.tensor(np.repeat(roots, count, axis=0), device=env.device, dtype=torch.float32)
                roots[:, :3] += env.scene.env_origins
                keyboard.write_root_pose_to_sim_index(root_pose=roots)
                keyboard.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(roots), 6), device=env.device))
                offsets = torch.tensor(np.repeat(np.array([np.r_[np.deg2rad(x[:5]), 0.] for x in values]), count, axis=0), device=env.device, dtype=torch.float32)
                held = np.zeros((len(roots), len(labels)), bool)
                downs = np.zeros_like(held, dtype=int); ups = downs.copy()
                sequences = [[] for _ in range(len(roots))]
                with torch.inference_mode():
                    for t in range(max(lengths)):
                        robot.write_joint_position_to_sim_index(position=q[t]+offsets, joint_ids=joint_ids)
                        robot.write_joint_velocity_to_sim_index(velocity=dq[t], joint_ids=joint_ids)
                        robot.set_joint_position_target_index(target=q[t]+offsets, joint_ids=joint_ids)
                        env.sim.forward(); env.scene.write_data_to_sim()
                        env.sim.step(render=False); env.scene.update(dt=dt)
                        fraction = ((upper-keyboard.data.joint_pos.torch[command._inst_idx, command._col_idx])/(upper-lower)).cpu().numpy()
                        downs = np.where(fraction >= .5, downs+1, 0); ups = np.where(fraction <= .25, ups+1, 0)
                        down, up = ~held & (downs >= 2), held & (ups >= 2)
                        for env_id, key_id in zip(*np.where(down | up)):
                            if t < lengths[env_id % count]:
                                sequences[env_id].append((labels[key_id], 'down' if down[env_id, key_id] else 'up'))
                        held = (held | down) & ~up
                        if t % 2500 == 0:
                            print(f'CONTINUOUS ROUND {round_index+1}: {t}/{max(lengths)} steps, {len(values)} candidates', flush=True)
                results = []
                for i, x in enumerate(values):
                    actual = sequences[i*count:(i+1)*count]
                    distances = [edit_distance(a, b) for a, b in zip(expected, actual)]
                    result = {'parameters': x.tolist(), 'score': sum(distances), 'edit_distances': distances,
                              'simulated_sequences': actual, 'round': round_index+1}
                    results.append(result); history.append(result)
                best = min(results, key=lambda r: r['score'])
                print('CONTINUOUS BEST:', best['score'], best['edit_distances'], best['parameters'], flush=True)
                args.out.write_text(json.dumps({'status': 'diagnostic_contact_search', 'best': best,
                    'evaluations': history, 'expected_sequences': expected, 'all_recordings_used_for_fitting': True,
                    'specification_sha256': digest(args.trajectory), 'single_environment_replay_required': True}, indent=2)+'\n')
                if best['score'] == 0:
                    break
        finally:
            base.close()


if __name__ == '__main__':
    main()
