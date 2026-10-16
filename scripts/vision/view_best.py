#!/usr/bin/env python3
"""Host-only Docker launcher for the best evaluated wrist-vision checkpoint."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--record-episodes', type=int, default=0)
    p.add_argument('--episode-seeds', type=int, nargs='+')
    args = p.parse_args()
    out = args.run.resolve()
    state = json.loads((out / 'supervisor.json').read_text())
    best = max(state['evaluations'], key=lambda r: r['success_rate'])
    generation = (ROOT / Path(best['checkpoint']).relative_to('/repo')).parent
    runtime = json.loads((generation / 'launch.json').read_text())
    config = json.loads((generation / 'run.json').read_text())
    name = 'so101-vision-' + out.name.replace('_', '-')

    def alive(container):
        r = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', container],
                           capture_output=True, text=True)
        return r.returncode == 0 and r.stdout.strip() == 'true'

    if alive(name + '-viewer'):
        print('Viewer already running:', (out / 'viewer.json').read_text())
        return
    if alive(name + '-gpu1'):
        p.error('GPU 1 evaluation is still running; retry after it finishes.')
    authority = Path(os.environ.get('XAUTHORITY', f'/run/user/{os.getuid()}/gdm/Xauthority'))
    if not os.environ.get('DISPLAY') or not authority.is_file():
        p.error('An X11 DISPLAY and readable Xauthority are required.')
    directory = out / 'replays' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory.mkdir(parents=True)
    inside = lambda path: '/repo/' + str(path.relative_to(ROOT))
    camera = json.loads((ROOT / 'configs/vision/replay_camera.json').read_text())
    (directory / 'viewer-camera.json').write_text(json.dumps(camera, indent=2) + '\n')
    cmd = ['docker', 'run', '-d', '--name', name + '-viewer', '--gpus', 'device=1',
           '--ipc=host', '--shm-size=16g', '--no-healthcheck',
           '--log-opt', 'max-size=20m', '--log-opt', 'max-file=2',
           '-e', 'NVIDIA_DRIVER_CAPABILITIES=all', '-e', 'OMP_NUM_THREADS=8',
           '-e', 'DISPLAY=' + os.environ['DISPLAY'], '-e', 'XAUTHORITY=/tmp/typing.xauthority',
           '-v', f'{authority}:/tmp/typing.xauthority:ro', '-v', '/tmp/.X11-unix:/tmp/.X11-unix:ro',
           '-v', f'{ROOT}:/repo:ro', '-v', f'{out}:{inside(out)}',
           '-v', f'{out}/cache_gpu1:/root/.cache',
           '-v', f'{ROOT}/scripts/vision/replay.py:/workspace/isaaclab/scripts/vision/replay.py:ro']
    for folder in ('nv_shadercache', 'texturecache', 'DerivedDataCache'):
        cache = out / 'renderer_gpu1' / folder
        cache.mkdir(parents=True, exist_ok=True)
        cmd += ['-v', f'{cache}:/opt/vision/lib/python3.12/site-packages/ovrtx/bin/cache/{folder}']
    cmd += [runtime['image'], '/workspace/isaaclab/scripts/vision/replay.py',
            '--visualizer', 'newton', '--num-envs', '1', '--seed', str(best['seed']),
            '--checkpoint', best['checkpoint'], '--task-config', config['task_config'],
            '--output', inside(directory), '--domain-strength', '1']
    if args.record_episodes:
        cmd += ['--record-episodes', str(args.record_episodes)]
        if args.episode_seeds:
            cmd += ['--episode-seeds', *map(str, args.episode_seeds)]
    (directory / 'launch.json').write_text(json.dumps({'argv': cmd, 'best_evaluation': best}, indent=2) + '\n')
    subprocess.run(['docker', 'rm', name + '-viewer'], capture_output=True)
    subprocess.run(cmd, check=True)
    (out / 'viewer.json').write_text(json.dumps({'container': name + '-viewer', 'output': str(directory)}, indent=2) + '\n')
    print(directory)


if __name__ == '__main__':
    main()
