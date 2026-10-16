# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a vision or state checkpoint across seeds, or view its exact USD rollouts in Kit RTX."""
from __future__ import annotations

from so101_typing.checkpoints import load_runner_checkpoint
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def restore_values(value):
    """Clear generated spawn paths and decode slices in archived configs."""
    if isinstance(value, dict):
        return {k: restore_values(v) for k, v in value.items() if k not in ('spawn_path', 'spawn_paths')}
    if isinstance(value, list):
        return [restore_values(v) for v in value]
    if isinstance(value, str) and re.fullmatch(r'slice\((None|-?\d+), (None|-?\d+), (None|-?\d+)\)', value):
        return slice(*(None if v == 'None' else int(v) for v in value[6:-1].split(', ')))
    return value


def merge_config(base, saved):
    # Runner construction removes class_name entries in-place before training
    # writes agent.json. Retain fresh defaults for these dispatch fields.
    for key, value in saved.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value
    return base


def evaluate(args):
    import torch
    from isaaclab.app import add_launcher_args, launch_simulation
    from rsl_rl.runners import OnPolicyRunner
    from agent import runner_config
    from runtime import create_env, make_config
    from record_replay_usd import ReplayUsdRecorder

    saved_env = read_json(args.checkpoint.parent / 'environment.json')
    saved_run = read_json(args.checkpoint.parent / 'run.json')
    args.target = args.target or saved_run.get('target', 'ISAAC LAB 3.0')
    args.domain_strength = (args.domain_strength if args.domain_strength is not None
                            else saved_run.get('domain_strength', 1.))
    args.num_envs = 1
    # Bootstrap the config's types with its saved nominal keyboard placement.
    # The full environment archive below restores dynamics, targets and rewards.
    pose = saved_env['scene']['keyboard']['init_state']
    args.task_config = args.output / 'task.json'
    args.task_config.write_text(json.dumps({'schema_version': 1, 'keyboard': {
        'position_m': pose['pos'], 'rotation_xyzw': pose['rot']}}, indent=2) + '\n')
    state_policy = saved_run.get('kind') == 'state'
    if state_policy:
        from isaaclab_tasks.core.dexsuite.config.so101.phrase_typing_env_cfg import SO101KeyboardMXGTCBerlinIsaacLabPhraseEnvCfg
        from isaaclab_tasks.utils.hydra import resolve_presets
        cfg = resolve_presets(SO101KeyboardMXGTCBerlinIsaacLabPhraseEnvCfg())
        cfg.seed = args.seed
        saved_env.pop('log_dir', None)  # Training output directory is not replay state.
        from public_task import apply_config, load_config
        apply_config(cfg, load_config(args.task_config))
        if args.target != 'ISAAC LAB 3.0':
            raise ValueError('This state checkpoint requires ISAAC LAB 3.0')
    else:
        cfg = make_config(args)
    cfg.from_dict(restore_values(saved_env))
    cfg.scene.num_envs = 1
    cfg.sim.device = 'cuda:0'
    cfg.seed = args.seed
    camera = read_json(args.viewer_camera)
    cfg.viewer.eye, cfg.viewer.lookat = tuple(camera['eye_m']), tuple(camera['lookat_m'])
    if len(cfg.commands.typing.letter_slots) != (38 if args.target == 'ISAAC LAB 3.0' else 26):
        raise ValueError('Target vocabulary differs from checkpoint training environment')
    from provenance import verify_asset_identity
    verify_asset_identity(args.checkpoint, cfg)
    app_parser = argparse.ArgumentParser(add_help=False)
    add_launcher_args(app_parser)
    app_args = app_parser.parse_args(['--headless', '--device', 'cuda:0'])
    torch.set_num_threads(8)
    saved_agent = read_json(args.checkpoint.parent / 'agent.json')
    config = saved_agent if state_policy else merge_config(runner_config(), saved_agent)
    with launch_simulation(cfg, app_args):
        env = create_env(cfg)
        raw = env.unwrapped
        recorder = None
        try:
            raw._vision_eval_word = args.target
            raw._vision_curriculum = args.domain_strength
            raw._vision_height_curriculum = args.domain_strength
            raw.reset(seed=args.seed)
            runner = OnPolicyRunner(env, config, log_dir=None, device='cuda:0')
            load_runner_checkpoint(runner, str(args.checkpoint), load_cfg={'actor': True, 'critic': True, 'optimizer': False})
            policy = runner.get_inference_policy()
            command = raw.command_manager.get_term('typing')
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + ('0123456789 .' if len(command.cfg.letter_slots) == 38 else '')
            episodes = []
            compute = raw.termination_manager.compute
            budget = args.max_steps or int(env.max_episode_length) + 1
            for index in range(args.episodes):
                seed = args.seed + index
                raw.termination_manager.compute = compute
                with torch.inference_mode():
                    raw.reset(seed=seed)
                episode_output = args.output / f'episode_{index + 1:03d}'
                episode_output.mkdir()
                recorder = ReplayUsdRecorder(raw, episode_output / 'replay.usdc')
                frames = []
                result = {}

                def capture_before_reset():
                    terminated = compute()
                    recorder.capture()
                    typed = ''.join(alphabet[int(command._slot_to_letter[slot])]
                                    for slot in command.typed[0].tolist() if slot >= 0)
                    frames.append({'frame': len(frames) + 1, 'typed': typed, 'phase': int(command.phase[0])})
                    if bool(terminated[0]):
                        failures = [name for name in raw.termination_manager.active_terms
                                    if name != 'success' and bool(raw.termination_manager.get_term(name)[0])]
                        result.update(typed=typed, success=bool(command.completed[0]) and typed == args.target
                                      and not failures, failures=failures)
                    return terminated

                # Capture physics and typing state before the environment auto-resets.
                raw.termination_manager.compute = capture_before_reset
                with torch.inference_mode():
                    obs = env.get_observations()
                    for _ in range(budget):
                        obs, _, _, _ = env.step(policy(obs))
                        if result:
                            break
                if not result:
                    result = {'typed': frames[-1]['typed'], 'success': False, 'failures': ['step_limit']}
                export = recorder.save()
                episode = {'episode': index + 1, 'seed': seed, 'steps': len(frames),
                           'replay_path': str(Path(episode_output.name) / 'replay.usdc'),
                           'replay': export, **result}
                episodes.append(episode)
                (episode_output / 'evaluation.json').write_text(json.dumps(episode, indent=2) + '\n')
                (episode_output / 'timeline.json').write_text(json.dumps(frames, indent=2) + '\n')
                print('EPISODE:', json.dumps(episode), flush=True)
            raw.termination_manager.compute = compute
            successes = sum(episode['success'] for episode in episodes)
            report = {'checkpoint': str(args.checkpoint),
                      'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      'policy_kind': 'state' if state_policy else 'vision',
                      'target': args.target, 'seed': args.seed, 'domain_strength': args.domain_strength,
                      'environment': str(args.checkpoint.parent / 'environment.json'),
                      'episode_count': len(episodes), 'successes': successes,
                      'success_rate_percent': 100. * successes / len(episodes), 'episodes': episodes}
            (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'EVALUATION: {successes}/{len(episodes)} completed '
                  f"({report['success_rate_percent']:.1f}%)", flush=True)
        finally:
            env.close()


def view(args):
    # Kit and standalone OVRTX use different USD runtimes: launch this mode in
    # the base simulator image, after the evaluation process has exited.
    import torch  # noqa: F401 -- load the runtime's PyTorch before Kit extensions
    from isaacsim import SimulationApp

    app = SimulationApp({'headless': False, 'renderer': 'RayTracedLighting', 'width': 1600, 'height': 900, 'fast_shutdown': False})
    try:
        import omni.timeline
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        context = omni.usd.get_context()
        if args.replay.suffix == '.json':
            report = read_json(args.replay)
            replays = [args.replay.parent / episode['replay_path'] for episode in report['episodes']]
        else:
            replays = [args.replay]
        if not replays:
            raise ValueError('Replay playlist is empty')
        started = time.monotonic()
        index = 0
        while app.is_running():
            replay = replays[index % len(replays)]
            if not context.open_stage(str(replay.resolve())):
                raise RuntimeError(f'Cannot open {replay}')
            for _ in range(600):
                app.update()
                stage = context.get_stage()
                if stage and stage.GetPrimAtPath('/RenderCamera').IsValid():
                    break
            else:
                raise RuntimeError('RTX stage load timed out')
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError('No active RTX viewport')
            viewport.camera_path = '/RenderCamera'
            timeline = omni.timeline.get_timeline_interface()
            timeline.pause()
            fps = stage.GetTimeCodesPerSecond()
            start, end = stage.GetStartTimeCode() / fps, stage.GetEndTimeCode() / fps
            duration = max(end - start, 1 / fps)
            for _ in range(30):
                app.update()
            print(f'RTX REPLAY READY: episode {index % len(replays) + 1}/{len(replays)}. '
                  'Orbit Alt+left drag; pan middle drag; zoom wheel. Close to exit.', flush=True)
            episode_started = time.monotonic()
            while app.is_running():
                if args.viewer_seconds and time.monotonic() - started >= args.viewer_seconds:
                    return
                elapsed = time.monotonic() - episode_started
                if elapsed >= duration + 1.:
                    break
                # Scrub captured transforms with physics paused. Hold the final
                # outcome for one second before advancing to the next episode.
                timeline.set_current_time(start + min(elapsed, duration))
                app.update()
                time.sleep(.01)
            index += 1
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--viewer-camera', type=Path, default=ROOT / 'configs/vision/evaluation_replay_camera.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--target', choices=('NEWTON', 'ISAAC LAB 3.0'))
    parser.add_argument('--seed', type=int, default=14001, help='First seed; each episode increments it')
    parser.add_argument('--episodes', type=int, default=1, help='Number of distinct-seed episodes (default: 1)')
    parser.add_argument('--domain-strength', type=float)
    parser.add_argument('--max-steps', type=int, default=0, help='0 uses the native episode budget')
    parser.add_argument('--replay', type=Path, help='Viewer mode: open evaluation.json playlist or a replay.usdc')
    parser.add_argument('--viewer-seconds', type=float, default=0., help='0 keeps the RTX viewer open until closed')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    if args.max_steps < 0 or not 0 <= args.viewer_seconds < float('inf'):
        parser.error('step and viewer limits must be nonnegative and finite')
    if args.domain_strength is not None and not 0 <= args.domain_strength <= 1:
        parser.error('domain strength must be in [0,1]')
    if args.replay:
        if not args.replay.is_file():
            parser.error('replay USD is missing')
        view(args)
    else:
        if args.checkpoint is None or not args.checkpoint.is_file() or args.output is None:
            parser.error('provide --checkpoint FILE and --output DIRECTORY')
        for name in ('environment.json', 'agent.json', 'run.json'):
            if not (args.checkpoint.parent / name).is_file():
                parser.error(f'keep {name} beside the checkpoint')
        if args.output.exists() and any(args.output.iterdir()):
            parser.error('use an empty output directory')
        args.output.mkdir(parents=True, exist_ok=True)
        evaluate(args)


if __name__ == '__main__':
    main()
