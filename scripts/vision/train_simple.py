"""Initialize a wrist-camera actor with a state teacher, then train with PPO."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from rsl_rl.runners import OnPolicyRunner

from agent import load_teacher, runner_config
from runtime import ROOT, create_env, make_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Required empty output directory')
    parser.add_argument('--teacher', type=Path, help='Required evaluated state-teacher checkpoint')
    parser.add_argument('--warmup', type=int, default=300, help='Teacher imitation updates before PPO')
    parser.add_argument('--warmup-learning-rate', type=float, default=3e-4)
    parser.add_argument('--task-config', type=Path,
                        default=ROOT / 'configs/tasks/gtc-berlin-vision.json')
    parser.add_argument('--target', choices=('NEWTON', 'ISAAC LAB 3.0'), default='ISAAC LAB 3.0')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=4811)
    parser.add_argument('--domain-strength', type=float, default=1.,
                        help='Fixed randomization strength, from 0 (nominal) to 1 (full)')
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.output is None or args.teacher is None:
        parser.error('--output and --teacher are required')
    if not args.teacher.is_file():
        parser.error(f'teacher checkpoint does not exist: {args.teacher}')
    if args.warmup < 0 or not 0 < args.warmup_learning_rate < float('inf'):
        parser.error('warmup must be nonnegative and its learning rate finite and positive')
    if min(args.num_envs, args.iterations, args.save_interval) < 1:
        parser.error('environment count, iterations and save interval must be positive')
    if not 0 < args.learning_rate < float('inf') or not 0 <= args.domain_strength <= 1:
        parser.error('learning rate must be finite and positive; domain strength must be in [0,1]')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('use an empty output directory to keep this run separate')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = make_config(args)
    config = runner_config(args.seed, args.learning_rate)
    config['save_interval'] = args.save_interval

    from provenance import save_asset_identity, verify_asset_identity
    verify_asset_identity(args.teacher, cfg)
    save_asset_identity(args.output, cfg)
    with launch_simulation(cfg, args):
        env = create_env(cfg)
        try:
            env.unwrapped._vision_curriculum = args.domain_strength
            env.unwrapped._vision_height_curriculum = args.domain_strength
            env.unwrapped.reset()
            runner = OnPolicyRunner(env, config, log_dir=str(args.output), device='cuda:0')
            teacher, teacher_checkpoint = load_teacher(env.get_observations(), args.teacher, 'cuda:0')
            runner.alg._raw_critic.load_state_dict(teacher_checkpoint['critic_state_dict'])
            for filename, data in (
                ('agent.json', config), ('environment.json', cfg.to_dict()),
                ('run.json', {**vars(args), 'initialization': 'random_actor_teacher_critic',
                              'teacher_sha256': hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                              'checkpoint': None, 'algorithm': 'PPO'}),
            ):
                (args.output / filename).write_text(json.dumps(data, default=str, indent=2) + '\n')
            runner.save(str(args.output / 'model_initial.pt'))
            actor = runner.alg._raw_actor
            optimizer = torch.optim.Adam(actor.parameters(), lr=args.warmup_learning_rate)
            obs = env.get_observations()
            warmup_stats = []
            for update in range(args.warmup):
                loss_sum = 0.
                image_gradient = 0.
                for _ in range(32):
                    actor.update_normalization(obs)
                    with torch.no_grad():
                        targets = teacher(obs)
                    prediction = actor(obs)
                    loss = torch.nn.functional.mse_loss(prediction, targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite teacher imitation loss')
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                    image_gradient += sum(float(p.grad.detach().norm()) for p in actor.cnns.parameters()
                                          if p.grad is not None)
                    optimizer.step()
                    with torch.no_grad():
                        probability = 1. - .5 * update / max(args.warmup - 1, 1)
                        action = torch.where(torch.rand((env.num_envs, 1), device='cuda:0') < probability,
                                             targets, prediction.detach())
                        obs, _, _, _ = env.step(action)
                    loss_sum += float(loss.detach())
                if image_gradient <= 0:
                    raise RuntimeError('No imitation gradients reach the wrist CNN')
                warmup_stats.append({'update': update, 'mse': loss_sum / 32,
                                     'image_gradient_norm': image_gradient / 32})
                print(f'WARMUP {update + 1}/{args.warmup}: mse={loss_sum / 32:.6f}', flush=True)
            (args.output / 'warmup.json').write_text(json.dumps(warmup_stats, indent=2) + '\n')
            runner.save(str(args.output / 'warmup_final.pt'))
            print(f'PPO: {args.iterations} updates; output={args.output}', flush=True)
            runner.learn(num_learning_iterations=args.iterations)
        finally:
            env.close()


if __name__ == '__main__':
    main()
