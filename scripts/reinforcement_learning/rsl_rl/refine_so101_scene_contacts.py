"""Offline Newton first-contact refinement with a reset between recorded keys.

Each recorded press is replayed in its own reset environment. All five joint
offsets and the keyboard rigid pose are adjustable. Never commands hardware.
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
from so101_typing.hardware.calibrate_scene import key_down_samples
from so101_typing.hardware.scene_check import load_recording
from so101_typing.hardware.fk_probe_trajectory import TASK, digest
from so101_typing.hardware.so101_kinematics import JOINT_NAMES
from public_task import apply_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trajectory', type=Path, required=True, help='JSON specification of recording and starting report paths')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--task', default=TASK)
parser.add_argument('--rounds', type=int, default=2)
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]]+remaining


@hydra_task_config(args.task, 'rsl_rl_cfg_entry_point')
def main(env_cfg, agent_cfg):
    spec = json.loads(args.trajectory.read_text())
    records = [load_recording(args.trajectory.parent/p) for p in spec['recordings']]
    if any(digest(Path(os.environ['SO101_ROBOT_USD'])) != r['robot_usd_sha256'] for r in records):
        raise ValueError('runtime USD differs from the recorded model')
    starts = [json.loads((args.trajectory.parent/p).read_text()) for p in spec['starting_reports']]
    apply_config(env_cfg, starts[0]['task_config'])
    contacts = [(i, row) for i, record in enumerate(records) for row in key_down_samples(record)]
    env_cfg.scene.num_envs = len(contacts)
    env_cfg.seed = 1307
    env_cfg.decimation = 1
    env_cfg.actions.action.control_dt_s = float(env_cfg.sim.dt)
    dt = float(env_cfg.sim.dt)
    # Clip each segment at the physical release; score only its first actuation.
    lengths = [max(1, int(np.ceil((row['up_s']-max(0., row['down_s']-.35))/dt))) for _, row in contacts]
    positions = np.empty((max(lengths), len(contacts), 6))
    for j, (ri, row) in enumerate(contacts):
        source = records[ri]['samples']
        times = np.array([s['time_s'] for s in source])
        q = np.array([s['q_model_rad'] for s in source])
        query = np.minimum(max(0., row['down_s']-.35)+np.arange(max(lengths))*dt, row['up_s'])
        positions[:, j] = np.stack([np.interp(query, times, q[:, k]) for k in range(6)], axis=1)
    velocity = np.gradient(positions, dt, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError('refusing to overwrite contact refinement')
    history = []
    with launch_simulation(env_cfg, args):
        base = gym.make(args.task, cfg=env_cfg)
        try:
            base.reset()
            env = base.unwrapped
            robot, keyboard = env.scene['robot'], env.scene['keyboard']
            command = env.command_manager.get_term('typing')
            joint_ids = [robot.joint_names.index(n) for n in JOINT_NAMES]
            device = env.device
            qs = torch.tensor(positions, device=device, dtype=torch.float32)
            dqs = torch.tensor(velocity, device=device, dtype=torch.float32)
            labels = list(command.cfg.slot_labels)
            expected = np.array([labels.index(row['key']) for _, row in contacts])
            lower = keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 0]
            upper = keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 1]
            def pose(q, dq):
                robot.write_joint_position_to_sim_index(position=q, joint_ids=joint_ids)
                robot.write_joint_velocity_to_sim_index(velocity=dq, joint_ids=joint_ids)
                env.sim.forward()

            def evaluate(x):
                base.reset()
                rotation = Rotation.from_euler('xyz', x[8:11], degrees=True).as_quat()
                root = torch.tensor(np.r_[x[5:8]/1000, rotation], device=device, dtype=torch.float32).repeat(len(contacts), 1)
                root[:, :3] += env.scene.env_origins
                keyboard.write_root_pose_to_sim_index(root_pose=root)
                keyboard.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(contacts), 6), device=device))
                offset = torch.tensor(np.r_[np.deg2rad(x[:5]), 0.], device=device, dtype=torch.float32)
                first = np.full(len(contacts), -1)
                streak = np.zeros((len(contacts), len(labels)), dtype=int)
                peaks = np.zeros(len(contacts))
                with torch.inference_mode():
                    for t in range(max(lengths)):
                        pose(qs[t]+offset, dqs[t])
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(dt=dt)
                        fraction = ((upper-keyboard.data.joint_pos.torch[command._inst_idx, command._col_idx])/(upper-lower)).cpu().numpy()
                        active = (first < 0) & (t < np.array(lengths))
                        peaks[active] = np.maximum(peaks[active], fraction[np.arange(len(contacts)), expected][active])
                        streak = np.where(fraction >= .5, streak+1, 0)
                        triggered = active & np.any(streak >= 2, axis=1)
                        first[triggered] = np.argmax(np.where(streak >= 2, fraction, -1), axis=1)[triggered]
                        first[triggered & ((streak >= 2).sum(axis=1) > 1)] = len(labels)
                        if not ((first < 0) & (t+1 < np.array(lengths))).any():
                            break
                correct = first == expected
                score = float(100*(~correct).sum()+np.maximum(0., .5-peaks).sum())
                result = {'score': score, 'correct_first_contacts': int(correct.sum()), 'total': len(contacts),
                          'parameters': x.tolist(), 'first_keys': [labels[i] if 0 <= i < len(labels) else ('MULTIPLE' if i == len(labels) else None) for i in first]}
                history.append(result)
                with (args.out.parent/'contact-search.jsonl').open('a') as stream:
                    stream.write(json.dumps(result)+'\n')
                print(f"CONTACT SEARCH {len(history)}: {correct.sum()}/{len(contacts)}, score={score:.4f}", flush=True)
                return result

            best = None
            for report in starts:
                candidate = next(iter(report['candidates'].values()))
                keyboard_pose = report['task_config']['keyboard']
                x = np.r_[candidate['offset_deg'], np.array(keyboard_pose['position_m'])*1000,
                          Rotation.from_quat(keyboard_pose['rotation_xyzw']).as_euler('xyz', degrees=True)]
                result = evaluate(x)
                if best is None or result['score'] < best['score']:
                    best = result
            initial = np.array(best['parameters'])
            for round_index in range(args.rounds):
                steps = np.array([.5]*5+[1.]*3+[.5]*3)/(2**round_index)
                for dim in [5, 6, 7, 0, 1, 2, 3, 4, 8, 9, 10]:
                    anchor = np.array(best['parameters'])
                    for sign in [-1, 1]:
                        x = anchor.copy(); x[dim] += sign*steps[dim]
                        if (np.abs(x[:5]) > 15).any():
                            continue
                        result = evaluate(x)
                        if result['score'] < best['score']:
                            best = result
                    if best['correct_first_contacts'] == len(contacts):
                        break
                if best['correct_first_contacts'] == len(contacts):
                    break
            output = {'status': 'first_contact_only_not_continuous_replay', 'best': best,
                      'specification_sha256': digest(args.trajectory), 'evaluations': len(history),
                      'initial_parameters': initial.tolist(), 'all_recordings_used_for_fitting': True,
                      'contacts': [{'recording_index': i, 'key': row['key'], 'down_s': row['down_s'], 'up_s': row['up_s']} for i, row in contacts],
                      'limits': 'Independent key resets and first-actuation scoring; does not establish full down/up sequence, transit clearance, or physical performance.'}
            args.out.write_text(json.dumps(output, indent=2)+'\n')
        finally:
            base.close()


if __name__ == '__main__':
    main()
