# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Loop the wrist-vision student in the interactive Newton viewer."""
from runtime import parser, make_config, create_env
from agent import runner_config
from state import write_json
from isaaclab.app import launch_simulation
from rsl_rl.runners import OnPolicyRunner
import hashlib
import json
import time
import torch


def main():
    p = parser()
    p.set_defaults(num_envs=1, visualizer=['newton'])
    p.add_argument('--domain-strength', type=float, default=1.)
    p.add_argument('--record-episodes', type=int, default=0)
    p.add_argument('--episode-seeds', type=int, nargs='+')
    args = p.parse_args()
    if args.num_envs != 1 or args.checkpoint is None:
        p.error('Replay requires one environment and --checkpoint')
    if args.record_episodes < 0:
        p.error('--record-episodes must be nonnegative')
    seeds = args.episode_seeds or [args.seed + i for i in range(args.record_episodes)]
    if args.record_episodes and (len(seeds) != args.record_episodes or len(set(seeds)) != len(seeds)):
        p.error('Provide one distinct seed per recorded episode')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    cfg = make_config(args)
    with launch_simulation(cfg, args):
        env = create_env(cfg)
        raw = env.unwrapped
        raw._vision_eval_word = 'NEWTON'
        raw._vision_curriculum = args.domain_strength
        runner = OnPolicyRunner(env, runner_config(), log_dir=None, device='cuda:0')
        runner.load(str(args.checkpoint), load_cfg={'actor': True, 'critic': True})
        policy = runner.get_inference_policy()
        command = raw.command_manager.get_term('typing')
        original_reset = command.reset
        episodes = []
        armed = False
        active_seed = args.seed

        def typed_word():
            return ''.join(chr(65 + int(command._slot_to_letter[slot]))
                           for slot in command.typed[0].tolist() if slot >= 0)

        def capture(env_ids=None):
            if armed:
                reasons = [name for name in raw.termination_manager.active_terms
                           if name != 'success' and bool(raw.termination_manager.get_term(name)[0])]
                exact = bool((command.typed[0, :6] == command.target[0, :6]).all())
                success = bool(command.completed[0]) and exact and not reasons and not bool(
                    command.wrong_key[0] or command.overlap[0] or command.initially_held[0])
                episode = dict(episode=len(episodes) + 1, seed=active_seed, typed=typed_word(), success=success,
                               failures=reasons, time=time.time())
                episodes.append(episode)
                write_json(args.output / 'episodes.json', episodes)
                print('REPLAY EPISODE', episode, flush=True)
            return original_reset(env_ids)

        command.reset = capture
        windows = [v._viewer for v in raw.sim._visualizers if getattr(v, '_viewer', None)]

        def caption(typed):
            title = f'Vision PPO | best checkpoint | NEWTON | typed: {typed or "-"} | episode {len(episodes)+1}'
            for viewer in windows:
                window = getattr(getattr(viewer, 'renderer', None), 'window', None)
                if window is not None and getattr(window, '_window', None) and hasattr(window, 'set_caption'):
                    window.set_caption(title)

        def draw_status(ui):
            ui.text('Vision PPO: best checkpoint')
            ui.text('Target: NEWTON')
            ui.text('Seed: ' + str(active_seed))
            ui.text('Typed: ' + (typed_word() or '-'))
            if episodes:
                ui.text('Last attempt: ' + episodes[-1]['typed'] + (' SUCCESS' if episodes[-1]['success'] else ' FAILED'))

        for viewer in windows:
            viewer.register_ui_callback(draw_status, position='side')

        write_json(args.output / 'replay.json', dict(
            checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            target='NEWTON', domain_strength=args.domain_strength, seed=args.seed,
            renderer='online OVRTX wrist observations; Newton overview viewer', started_at=time.time()))
        camera_path = args.output / 'viewer-camera.json'
        camera_mtime = None

        def update_camera():
            nonlocal camera_mtime
            if camera_path.exists() and camera_path.stat().st_mtime_ns != camera_mtime:
                camera = json.loads(camera_path.read_text())
                raw.sim.set_camera_view(tuple(camera['eye_m']), tuple(camera['lookat_m']))
                camera_mtime = camera_path.stat().st_mtime_ns

        with torch.inference_mode():
            raw.reset()
            raw.sim.set_camera_view(cfg.viewer.eye, cfg.viewer.lookat)
            update_camera()
            obs = env.get_observations()
            if args.record_episodes:
                import imageio.v2 as imageio
                import numpy as np
                from PIL import Image, ImageDraw, ImageFont
                movie = args.output / f'evaluation_{args.record_episodes}episodes.mp4'
                wrist_movie = args.output / f'evaluation_{args.record_episodes}episodes_wrist.mp4'
                fps = round(1. / float(raw.step_dt))
                writer = imageio.get_writer(movie, fps=fps, codec='libx264', quality=8,
                                            macro_block_size=2, ffmpeg_params=['-threads', '4', '-movflags', '+faststart'])
                wrist_writer = imageio.get_writer(wrist_movie, fps=fps, codec='libx264', quality=9,
                                                  macro_block_size=2, ffmpeg_params=['-threads', '2', '-movflags', '+faststart'])
                try:
                    font = ImageFont.truetype('DejaVuSans.ttf', 28)
                except OSError:
                    font = ImageFont.load_default(size=28)
                frame_count = 0
                video_timeline = []

                def wrist_frame():
                    # Read the already-rendered sensor output; no extra simulation step or augmentation draw.
                    return raw.scene['camera_wrist'].data.output['rgb'].torch[0, ..., :3].cpu().numpy().copy()

                def append_frame(pixels, wrist_pixels, text):
                    nonlocal frame_count
                    finished = text.startswith('Result:')
                    typed = text.split(': ', 1)[1].split(' | ', 1)[0]
                    typed = '' if typed == '-' else typed
                    event = dict(episode=len(episodes) + (0 if finished else 1), seed=active_seed, typed=typed, finished=finished)
                    if not video_timeline or any(video_timeline[-1][key] != value for key,value in event.items()):
                        video_timeline.append(dict(frame=frame_count, **event))
                    frame = Image.fromarray(pixels).resize((1920, 1080))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((18, 16, 900, 100), fill=(22, 26, 34))
                    draw.text((34, 25), f'NEWTON | Episode {len(episodes) + (0 if text.startswith("Result:") else 1)}/{args.record_episodes} | Seed {active_seed}', font=font, fill='white')
                    draw.text((34, 63), text, font=font, fill='white')
                    writer.append_data(np.asarray(frame))
                    wrist_writer.append_data(wrist_pixels)
                    frame_count += 1

                try:
                    for active_seed in seeds:
                        armed = False
                        raw.reset(seed=active_seed)
                        obs = env.get_observations()
                        raw.sim.render()
                        initial = windows[0].get_frame(render_ui=False).numpy()
                        initial_wrist = wrist_frame()
                        start_frame = frame_count
                        before = len(episodes)
                        keyboard = raw.scene['keyboard']
                        sampled_pose = dict(position_m=keyboard.data.root_pos_w.torch[0].tolist(),
                                            quaternion_wxyz=keyboard.data.root_quat_w.torch[0].tolist())
                        armed = True
                        caption('')
                        append_frame(initial, initial_wrist, 'Typed: -')
                        last_frame, last_wrist = initial, initial_wrist
                        for step in range(int(env.max_episode_length) + 2):
                            if not raw.sim.is_headless_or_exist_active_visualizer():
                                raise RuntimeError('Viewer closed before all requested episodes were recorded')
                            started = time.monotonic()
                            obs, _, _, _ = env.step(policy(obs))
                            if not raw.sim.is_headless_or_exist_active_visualizer():
                                raise RuntimeError('Viewer closed before all requested episodes were recorded')
                            if len(episodes) > before:
                                armed = False
                                result = episodes[-1]
                                outcome = 'SUCCESS' if result['success'] else 'FAILED'
                                for _ in range(fps):
                                    append_frame(last_frame, last_wrist, f'Result: {result["typed"] or "-"} | {outcome}')
                                result.update(start_frame=start_frame, end_frame_exclusive=frame_count,
                                              simulation_steps=step+1, keyboard_pose=sampled_pose)
                                write_json(args.output / 'episodes.json', episodes)
                                break
                            last_frame = windows[0].get_frame(render_ui=False).numpy()
                            last_wrist = wrist_frame()
                            append_frame(last_frame, last_wrist, 'Typed: ' + (typed_word() or '-'))
                            caption(typed_word())
                            time.sleep(max(0., float(raw.step_dt) - (time.monotonic() - started)))
                        else:
                            raise RuntimeError('Recorded episode exceeded the native task time budget')
                finally:
                    try:
                        writer.close()
                    finally:
                        wrist_writer.close()
                write_json(args.output / 'video_timeline.json', video_timeline)
                write_json(args.output / 'evaluation.json', dict(
                    target='NEWTON', episodes=len(episodes), seeds=seeds,
                    success_rate=sum(e['success'] for e in episodes)/len(episodes),
                    domain_strength=args.domain_strength, checkpoint=str(args.checkpoint),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    video=movie.name, wrist_video=wrist_movie.name,
                    wrist_resolution=[160, 120], wrist_source='Raw online OVRTX wrist sensor before policy image augmentation',
                    paired_frame_count=frame_count, fps=fps, frames=frame_count, duration_s=frame_count/fps,
                    camera=json.loads(camera_path.read_text()), results=episodes,
                    note='One second of result hold follows each episode; this five-episode recording does not replace the full convergence evaluation.'))
                print('RECORDING COMPLETE:', movie, flush=True)
            else:
                armed = True
                caption('')
                last_word = ''
                print('REPLAY READY: NEWTON; close the Newton window to finish.', flush=True)
                while raw.sim.is_headless_or_exist_active_visualizer():
                    started = time.monotonic()
                    update_camera()
                    obs, _, _, _ = env.step(policy(obs))
                    if not raw.sim.is_headless_or_exist_active_visualizer():
                        break
                    word = typed_word()
                    if word != last_word:
                        print('REPLAY TYPED:', word, flush=True)
                        last_word = word
                    caption(word)
                    time.sleep(max(0., float(raw.step_dt) - (time.monotonic() - started)))
        armed = False
        env.close()


if __name__ == '__main__':
    main()
