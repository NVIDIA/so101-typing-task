# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Host launcher: evaluate the state policy, then display its motion in an RTX grid."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from .cli import ROOT, IMAGE, BASE, docker, execute, inside, output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, default=2)
    p.add_argument('--cols', type=int, default=2)
    p.add_argument('--gpu', type=int, default=int(os.environ.get('SO101_VIEW_GPU', '0')))
    p.add_argument('--output', type=Path)
    p.add_argument('--no-viewer', action='store_true')
    p.add_argument('--viewer-seconds', type=float, default=0)
    p.add_argument('--plan', action='store_true')
    args = p.parse_args()
    if min(args.rows, args.cols) < 1 or args.rows * args.cols > 64 or args.gpu < 0 or args.viewer_seconds < 0:
        p.error('Choose 1–64 cells, a nonnegative GPU index and viewer duration')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = output_path(args.output or ROOT / 'output/rtx_examples' / stamp)
    if output.exists():
        p.error('Use a new output directory')
    authority = Path(os.environ.get('XAUTHORITY', f'/run/user/{os.getuid()}/gdm/Xauthority'))
    if not args.no_viewer and not args.plan and (not os.environ.get('DISPLAY') or not authority.is_file()):
        p.error('An X11 display is required; use --no-viewer to export only')
    command = [ROOT / 'so101', 'replay', '--model', 'models/isaac_lab_state/isaac_lab_ckpt.pt',
               '--no-viewer', '--output', output, '--gpu', str(args.gpu)]
    execute(command, args.plan)
    scene = output / 'grid.usdc'
    execute(docker(args.gpu, IMAGE, 'scripts/rendering/build_typing_grid.py',
                   [inside(output / 'episode_001/replay.usdc'), inside(scene),
                    '--rows', args.rows, '--cols', args.cols, '--resolution', 1920, 1080]), args.plan)
    print(f'Grid replay: {scene}. Cells repeat the same evaluated trajectory.', flush=True)
    if args.no_viewer:
        return
    view_grid(scene, args.gpu, args.viewer_seconds, args.plan)


def view_grid(scene, gpu, seconds=0, plan=False):
    authority = Path(os.environ.get('XAUTHORITY', f'/run/user/{os.getuid()}/gdm/Xauthority'))
    viewer_home = Path(scene).parent / '.kit-grid-viewer'
    mounts = []
    for name, destination in (('home','/tmp/so101-home'), ('cache','/isaac-sim/kit/cache'),
                              ('data','/isaac-sim/kit/data'), ('logs','/isaac-sim/kit/logs')):
        directory = viewer_home / name
        if not plan:
            directory.mkdir(parents=True, exist_ok=True)
        mounts += ['-v', f'{directory}:{destination}']
    execute(['docker', 'run', '--rm', '--init', '--gpus', f'device={gpu}',
             '--ipc=private', '--shm-size=16g', '--user', f'{os.getuid()}:{os.getgid()}',
             '-e', 'DISPLAY', '-e', 'NVIDIA_DRIVER_CAPABILITIES=all', '-e', 'OMNICLIENT_HUB_MODE=disabled',
             '-e', 'HOME=/tmp/so101-home', '-e', 'USER=so101', '-e', 'LOGNAME=so101', '-e', 'XAUTHORITY=/tmp/typing.xauthority',
             '-v', '/tmp/.X11-unix:/tmp/.X11-unix:ro', '-v', f'{authority}:/tmp/typing.xauthority:ro',
             '-v', f'{ROOT}:/repo:ro', *mounts, BASE, '-p', '/repo/scripts/vision/evaluate_policy.py',
             '--replay', inside(scene), '--viewer-seconds', str(seconds)], plan)



if __name__ == '__main__':
    main()
