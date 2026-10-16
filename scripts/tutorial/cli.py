"""Host orchestration for the tutorial. ML runs in the repository's Docker runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from . import artifacts

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'so101-tutorial:dev'
BASE = 'so101-typing-public:dev'
DEFAULT_CONFIG = ROOT / 'configs/tutorial/default.json'
RELEASES = ROOT / 'configs/tutorial/releases.json'
VOCABULARY = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .'


def handles(argv):
    return bool(argv) and (argv[0] in ('setup', 'debug', 'replay', 'hardware', 'bundle')
        or (argv[0] in ('train', 'evaluate') and len(argv) > 1 and argv[1] in ('teacher', 'vision'))
        or (argv[0] in ('download', 'deploy') and any(a == '--model' or a.startswith('--model=') for a in argv)))


def configuration(path):
    data = artifacts.read(path)
    train, evaluation = data['training'], data['evaluation']
    if data.get('schema_version') != 1 or train['sequence_length'] != 13 or train['vocabulary'] != VOCABULARY or train['random_sequences'] is not True:
        raise ValueError('This tutorial supports random 13-character training over the 38 documented keys')
    if evaluation['target'] != 'ISAAC LAB 3.0':
        raise ValueError('This tutorial evaluation preset targets ISAAC LAB 3.0')
    for key in ('num_envs', 'teacher_iterations', 'vision_iterations', 'save_interval'):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(train['warmup']) is not int or train['warmup'] < 0 or not 0 <= train['domain_strength'] <= 1 or not 0 < train['learning_rate'] < 1:
        raise ValueError('Invalid warmup, randomization strength or learning rate')
    if type(evaluation['episodes']) is not int or evaluation['episodes'] < 1:
        raise ValueError('evaluation episodes must be positive')
    return data


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Keep tutorial inputs inside this repository: {path}')
    return '/repo/' + str(path.relative_to(ROOT))


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'output'):
        raise ValueError('Generated output must be under output/')
    return path


def execute(command, plan=False, **kwargs):
    print(shlex.join(map(str, command)), flush=True)
    if not plan:
        subprocess.run(list(map(str, command)), cwd=ROOT, check=True, **kwargs)


def ensure_image(image, plan=False, force=False):
    if image != IMAGE:
        execute(['docker', 'image', 'inspect', image], plan, stdout=subprocess.DEVNULL)
        return
    if not force and not plan and subprocess.run(['docker','image','inspect',IMAGE], capture_output=True).returncode == 0:
        return
    if not plan and subprocess.run(['docker','image','inspect',BASE], capture_output=True).returncode:
        execute([ROOT / 'so101', 'build'])
    execute(['docker','build','-f','docker/vision-ppo.Dockerfile','-t',IMAGE,'.'], plan)


def docker(gpu, image, script, options):
    return ['docker','run','--rm','--init','--gpus',f'device={gpu}','--ipc=private','--shm-size=16g',
            '--user',f'{os.getuid()}:{os.getgid()}',
            '-e','HOME=/tmp','-e','USER=so101','-e','LOGNAME=so101',
            '-e','OMNICLIENT_HUB_MODE=disabled',
            '--tmpfs','/opt/vision/lib/python3.12/site-packages/ovrtx/bin/cache/nv_shadercache:rw,mode=1777',
            '--tmpfs','/opt/vision/lib/python3.12/site-packages/ovrtx/bin/cache/texturecache:rw,mode=1777',
            '--tmpfs','/opt/vision/lib/python3.12/site-packages/ovrtx/bin/cache/DerivedDataCache:rw,mode=1777',
            '--tmpfs','/opt/vision/lib/python3.12/site-packages/ovrtx/bin/mdl/omniverse_exts:rw,mode=1777',
            '-e','NVIDIA_DRIVER_CAPABILITIES=all','-e','OMP_NUM_THREADS=8',
            '-v',f'{ROOT}:/repo:ro','-v',f'{ROOT / "output"}:/repo/output',
            image,f'/repo/{script}',*map(str,options)]


def selected(args):
    if getattr(args, 'model', None):
        path = Path(args.model)
        if not path.exists():
            path = ROOT / '.artifacts/models' / args.model
        if path.is_dir() and (path / 'bundle.json').exists():
            manifest = artifacts.read(path / 'bundle.json')
            for name, checksum in manifest['files'].items():
                member = (path / name).resolve()
                if not member.is_relative_to(path.resolve()) or artifacts.digest(member) != checksum:
                    raise ValueError('Model bundle verification failed')
            return artifacts.checkpoint(path / manifest['checkpoint'])
        return artifacts.checkpoint(path, getattr(args, 'checkpoint', 'best'), getattr(args, 'suite', None) or 'phrase')
    return artifacts.checkpoint(args.run, args.checkpoint, args.suite or 'phrase')


def debug(args):
    checks = {}
    for name, command in (
        ('docker', ['docker','info','--format','{{.ServerVersion}}']),
        ('gpu', ['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']),
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            checks[name] = {'ok':result.returncode == 0, 'detail':(result.stdout + result.stderr).strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks[name] = {'ok':False, 'detail':str(exc)}
    checks['disk'] = {'ok':os.access(ROOT, os.W_OK), 'free_gib':round(shutil.disk_usage(ROOT).free / 1024**3, 1)}
    if args.viewer:
        authority = Path(os.environ.get('XAUTHORITY', f'/run/user/{os.getuid()}/gdm/Xauthority'))
        checks['display'] = {'ok':bool(os.environ.get('DISPLAY')) and authority.is_file(), 'detail':str(authority)}
    if checks['docker']['ok']:
        command = ['docker','run','--rm','--gpus',f'device={args.gpu}',args.image,'-c',
                   'import torch,ovrtx,ovstage; from pathlib import Path; '
                   'assert torch.cuda.is_available(); '
                   'assert Path("/opt/so101/assets/workshop_so101.usdc").is_file(); '
                   'print("CUDA, OVRTX, USD imports and Workshop asset available")']
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            checks['runtime'] = {'ok':result.returncode == 0, 'detail':(result.stdout + result.stderr)[-2000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks['runtime'] = {'ok':False, 'detail':str(exc)}
    print(json.dumps(checks, indent=2))
    return 0 if all(c['ok'] for c in checks.values()) else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog='./so101', description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('setup', help='Build or pull runtime; optionally install hardware dependencies')
    setup.add_argument('--hardware', action='store_true')
    setup.add_argument('--build', action='store_true', help='Rebuild the tutorial image after source changes')
    setup.add_argument('--release', type=Path, default=RELEASES)
    setup.add_argument('--plan', action='store_true')
    doc = sub.add_parser('debug', help='Check Docker, GPU, assets, imports and optionally X11')
    doc.add_argument('--viewer', action='store_true')
    doc.add_argument('--gpu', type=int, default=0)
    doc.add_argument('--image', default=IMAGE)
    for operation in ('train', 'evaluate', 'replay'):
        q = sub.add_parser(operation)
        if operation != 'replay':
            q.add_argument('kind', choices=('teacher','vision'))
        q.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        q.add_argument('--gpu', type=int, default=0)
        q.add_argument('--image', default=IMAGE)
        q.add_argument('--plan', action='store_true')
        if operation == 'train':
            q.add_argument('--teacher', type=Path)
            q.add_argument('--checkpoint', default='best', help='Teacher selection: best, latest, or filename')
            q.add_argument('--output', type=Path)
            q.add_argument('--suite', choices=('phrase','random'), default='random')
        else:
            source = q.add_mutually_exclusive_group(required=True)
            source.add_argument('--run', type=Path)
            source.add_argument('--model')
            q.add_argument('--checkpoint', default='best' if operation == 'replay' else 'latest')
            q.add_argument('--suite', choices=('phrase','random'))
            q.add_argument('--output', type=Path)
            q.add_argument('--episodes', type=int)
            q.add_argument('--max-steps', type=int, default=0)
            if operation == 'replay':
                q.add_argument('--no-viewer', action='store_true')
                q.add_argument('--video', action='store_true', help='Also render an overview MP4 headlessly')
                q.add_argument('--viewer-seconds', type=float, default=0)
    download = sub.add_parser('download')
    download.add_argument('--model', required=True)
    download.add_argument('--manifest', type=Path, default=RELEASES)
    bundle = sub.add_parser('bundle')
    bundle.add_argument('operation', choices=('pack','import'))
    bundle.add_argument('--checkpoint', type=Path)
    bundle.add_argument('--archive', type=Path, required=True)
    bundle.add_argument('--sha256')
    bundle.add_argument('--name')
    hardware = sub.add_parser('hardware')
    hardware.add_argument('operation', choices=('configure','calibrate','save-rest','check'))
    hardware.add_argument('--profile', type=Path, default=ROOT / 'local/robot.json')
    hardware.add_argument('--model')
    hardware.add_argument('--plan', action='store_true')
    deploy = sub.add_parser('deploy')
    deploy.add_argument('--model', required=True)
    deploy.add_argument('--profile', type=Path, required=True)
    deploy.add_argument('--target', default='ISAAC LAB 3.0')
    deploy.add_argument('--tolerate-expected-repeat', action='store_true')
    deploy.add_argument('--plan', action='store_true')
    args = p.parse_args(argv)
    try:
        if args.command == 'debug':
            return debug(args)
        if args.command == 'setup':
            release = artifacts.read(args.release)
            if release['images'] and not args.build:
                for tag, role in ((BASE,'base'),(IMAGE,'vision')):
                    ref = release['images'][role]
                    if '@sha256:' not in ref:
                        raise ValueError('Release images must be pinned by digest')
                    execute(['docker','pull',ref], args.plan)
                    execute(['docker','tag',ref,tag], args.plan)
            else:
                ensure_image(IMAGE, args.plan, force=args.build)
            if args.hardware:
                execute([ROOT / 'scripts/setup_hardware.sh'], args.plan)
            return 0
        if args.command in ('bundle','download'):
            if args.command == 'bundle' and args.operation == 'pack':
                if not args.checkpoint:
                    raise ValueError('bundle pack requires --checkpoint')
                output_path(args.archive)
                artifacts.pack(args.checkpoint, args.archive, ROOT / 'configs/vision/evaluation_replay_camera.json')
                print(f'Local bundle: {args.archive}; no upload performed')
                return 0
            name = args.model if args.command == 'download' else args.name
            if not name or Path(name).name != name or name in ('.','..'):
                raise ValueError('Provide a simple model name')
            destination = ROOT / '.artifacts/models' / name
            if args.command == 'download':
                item = artifacts.read(args.manifest)['models'].get(name)
                if not item:
                    raise ValueError(f'No published URL configured for {name}. Train locally or import a bundle; this checkout does not upload models.')
                result = artifacts.download(item['url'], item['sha256'], destination)
            else:
                if not args.sha256:
                    raise ValueError('bundle import requires --sha256')
                result = artifacts.unpack(args.archive, destination, args.sha256)
            print(result)
            return 0
        if args.command in ('hardware','deploy'):
            from .hardware import dispatch
            return dispatch(args)
        settings = configuration(args.config)
        if args.gpu < 0 or getattr(args, 'max_steps', 0) < 0 or (getattr(args, 'episodes', None) is not None and args.episodes < 1):
            raise ValueError('GPU/step indices must be nonnegative; episodes must be positive')
        if args.command == 'train':
            output = output_path(args.output or ROOT / 'output' / args.kind)
            if output.exists() and any(output.iterdir()):
                raise ValueError(f'Use an empty output directory: {output}')
            if args.kind == 'vision' and args.teacher is None:
                raise ValueError('vision training requires --teacher RUN_OR_CHECKPOINT')
            task = ROOT / settings['task_config']
            options = ['--headless','--output',inside(output),'--task-config',inside(task)]
            if args.kind == 'teacher':
                script = 'scripts/tutorial/runtime.py'
                options += ['--mode','train-teacher','--config',inside(args.config)]
            else:
                chosen = artifacts.checkpoint(args.teacher, args.checkpoint, args.suite)
                if (chosen.parent / 'run.json').is_file() and artifacts.read(chosen.parent / 'run.json').get('kind') != 'teacher':
                    raise ValueError('Use a state teacher checkpoint from ./so101 train teacher')
                saved = artifacts.read(chosen.parent / 'environment.json')
                expected = artifacts.read(task)['keyboard']
                pose = saved['scene']['keyboard']['init_state']
                if not all(math.isclose(a,b,rel_tol=0,abs_tol=1e-10) for a,b in zip(pose['pos']+pose['rot'], expected['position_m']+expected['rotation_xyzw'], strict=True)):
                    raise ValueError('Teacher and requested task poses differ; use its matching task_config')
                script = 'scripts/vision/train_simple.py'
                options += ['--teacher',inside(chosen),'--target',settings['evaluation']['target']]
                for field, flag in (('vision_iterations','iterations'),('warmup','warmup'),('num_envs','num-envs'),
                                    ('learning_rate','learning-rate'),('save_interval','save-interval'),
                                    ('domain_strength','domain-strength'),('seed','seed')):
                    options += ['--'+flag, str(settings['training'][field])]
            ensure_image(args.image, args.plan)
            image_id = args.image
            if not args.plan:
                output.mkdir(parents=True, exist_ok=True)
                image_id = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',args.image], text=True).strip()
            execute(docker(args.gpu,image_id,script,options), args.plan)
            if not args.plan:
                artifacts.write(output / 'runtime.json', {'image':args.image, 'image_id':image_id})
                artifacts.write(output / 'tutorial.json', settings)
                artifacts.write(output / 'task.json', artifacts.read(task))
            return 0
        chosen = selected(args)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        output = output_path(args.output or ROOT / 'output/tutorial' / (stamp + '-' + args.command))
        if not args.plan:
            output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == 'replay':
            if artifacts.read(chosen.parent / 'run.json').get('kind') == 'teacher':
                raise ValueError('Replay requires a vision checkpoint; use evaluate teacher for state policies')
            command = [ROOT / 'scripts/vision/evaluate_policy.sh', chosen, '--output', output,
                       '--episodes',str(args.episodes or 1),'--max-steps',str(args.max_steps),
                       '--viewer-seconds',str(args.viewer_seconds)]
            if args.no_viewer or args.video:
                command += ['--no-viewer']
            camera = chosen.parent / 'evaluation_replay_camera.json'
            if camera.exists():
                command += ['--viewer-camera',inside(camera)]
            ensure_image(args.image,args.plan)
            execute(command, args.plan, env=os.environ | {'SO101_VIEW_GPU':str(args.gpu), 'SO101_EVAL_IMAGE':args.image})
            if args.video:
                ensure_image(args.image,args.plan)
                execute(docker(args.gpu,args.image,'scripts/tutorial/render_video.py',
                               ['--evaluation',inside(output / 'evaluation.json')]), args.plan)
            return 0
        suite = args.suite or ('random' if args.kind == 'teacher' else 'phrase')
        # Reconstruct nominal geometry from the checkpoint, never from an unrelated preset.
        options = ['--mode','evaluate','--kind',args.kind,'--headless','--config',inside(args.config),
                   '--checkpoint',inside(chosen),'--task-config',inside(ROOT / settings['task_config']),
                   '--output',inside(output),'--suite',suite,'--max-steps',str(args.max_steps)]
        if args.episodes:
            options += ['--episodes',str(args.episodes)]
        ensure_image(args.image, args.plan)
        execute(docker(args.gpu,args.image,'scripts/tutorial/runtime.py',options), args.plan)
        if not args.plan and not args.max_steps:
            report = artifacts.read(output / 'evaluation.json')
            report.pop('results', None)
            artifacts.record_evaluation(chosen.parent, chosen, report, suite)
        return 0
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
