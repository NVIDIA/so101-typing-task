"""Docker runtime for state teacher training and comparable teacher/vision evaluation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vision'))
from runtime import make_config, create_env
from agent import runner_config, load_teacher
from evaluate_policy import restore_values, merge_config


def write(path, value):
    Path(path).write_text(json.dumps(value, default=str, indent=2) + '\n')


def teacher_config(seed, learning_rate):
    config = runner_config(seed, learning_rate)
    config['obs_groups'] = {'actor': ['policy'], 'critic': ['policy']}
    config['actor'] = copy.deepcopy(config['critic'])
    config['actor']['distribution_cfg'] = {'class_name': 'BetaDistribution', 'action_range': [-1., 1.]}
    return config


def main():
    import torch
    from isaaclab.app import add_launcher_args, launch_simulation
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_tasks.core.dexsuite.config.so101.fixed_cartesian_env_cfg import FixedCartesianObservationsCfg

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('train-teacher', 'evaluate'), required=True)
    p.add_argument('--kind', choices=('teacher', 'vision'), default='teacher')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task-config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--episodes', type=int)
    p.add_argument('--suite', choices=('phrase', 'random'), default='phrase')
    p.add_argument('--max-steps', type=int, default=0, help='Smoke checks only; truncations count as failures')
    add_launcher_args(p)
    args = p.parse_args()
    settings = json.loads(args.config.read_text())
    training = settings['training']
    args.target = settings['evaluation']['target']
    args.num_envs = training['num_envs']
    args.seed = training['seed'] if args.mode == 'train-teacher' else settings['evaluation']['seed']
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    cfg = make_config(args)
    if args.kind == 'teacher':
        cfg.observations = FixedCartesianObservationsCfg()
        cfg.scene.camera_wrist = None
    if args.mode == 'evaluate':
        if args.checkpoint is None:
            p.error('evaluation requires --checkpoint')
        cfg.from_dict(restore_values(json.loads((args.checkpoint.parent / 'environment.json').read_text())))
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = 'cuda:0'
        cfg.seed = args.seed
        # A held-out random suite also works for historical fixed-phrase checkpoints.
        cfg.commands.typing.fixed_target_slots = None
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    from provenance import save_asset_identity, verify_asset_identity
    if args.checkpoint:
        verify_asset_identity(args.checkpoint, cfg)
    else:
        save_asset_identity(args.output, cfg)
    with launch_simulation(cfg, args):
        env = create_env(cfg)
        raw = env.unwrapped
        try:
            strength = training['domain_strength']
            raw._vision_curriculum = strength
            raw._vision_height_curriculum = strength
            raw.reset(seed=args.seed)
            if args.mode == 'train-teacher':
                config = teacher_config(args.seed, training['learning_rate'])
                config['save_interval'] = training['save_interval']
                write(args.output / 'agent.json', config)
                write(args.output / 'environment.json', cfg.to_dict())
                write(args.output / 'run.json', {'kind':'teacher', 'target':args.target,
                      'domain_strength':strength, 'seed':args.seed, 'training':training})
                write(args.output / 'tutorial.json', settings)
                write(args.output / 'task.json', json.loads(args.task_config.read_text()))
                from isaaclab.utils.io import dump_yaml
                (args.output / 'params').mkdir()
                dump_yaml(str(args.output / 'params/env.yaml'), cfg)
                write(args.output / 'params/public_task.json', json.loads(args.task_config.read_text()))
                runner = OnPolicyRunner(env, config, log_dir=str(args.output), device='cuda:0')
                runner.learn(num_learning_iterations=training['teacher_iterations'])
                return
            raw._vision_eval_word = args.target if args.suite == 'phrase' else None
            if args.kind == 'teacher':
                policy, _ = load_teacher(env.get_observations(), args.checkpoint, 'cuda:0')
            else:
                config = merge_config(runner_config(), json.loads((args.checkpoint.parent / 'agent.json').read_text()))
                runner = OnPolicyRunner(env, config, log_dir=None, device='cuda:0')
                runner.load(str(args.checkpoint), load_cfg={'actor':True, 'critic':True, 'optimizer':False})
                policy = runner.get_inference_policy()
            command = raw.command_manager.get_term('typing')
            original_compute = raw.termination_manager.compute
            results, active = [], None
            def capture():
                done = original_compute()
                for i in range(env.num_envs):
                    if active is not None and bool(active[i]) and bool(done[i]):
                        length = int(command.target_len[i])
                        failures = [n for n in raw.termination_manager.active_terms
                                    if n != 'success' and bool(raw.termination_manager.get_term(n)[i])]
                        exact = bool(torch.equal(command.typed[i, :length], command.target[i, :length]))
                        results.append({'success':bool(command.completed[i]) and exact and not failures,
                                        'failures':failures, 'characters':int(command.character_index[i])})
                        active[i] = False
                return done
            raw.termination_manager.compute = capture
            episodes = args.episodes or settings['evaluation']['episodes']
            budget = args.max_steps or int(env.max_episode_length) + 1
            started = time.time()
            with torch.inference_mode():
                while len(results) < episodes:
                    active = None
                    raw.reset(seed=args.seed + len(results))
                    active = torch.arange(env.num_envs, device='cuda:0') < min(env.num_envs, episodes - len(results))
                    obs = env.get_observations()
                    for _ in range(budget):
                        obs, _, _, _ = env.step(policy(obs))
                        if not bool(active.any()):
                            break
                    for i in torch.where(active)[0].tolist():
                        results.append({'success':False, 'failures':['step_limit'], 'characters':int(command.character_index[i])})
                    print(f'EVALUATED {len(results)}/{episodes}', flush=True)
            report = {'target':args.target if args.suite == 'phrase' else 'random13', 'suite':args.suite,
                      'episodes':episodes, 'seed':args.seed, 'domain_strength':strength, 'num_envs':env.num_envs,
                      'success_rate':sum(r['success'] for r in results) / episodes,
                      'max_steps':args.max_steps, 'elapsed_s':time.time() - started, 'results':results}
            write(args.output / 'evaluation.json', report)
            print(f"SUCCESS: {report['success_rate']:.2%}", flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
