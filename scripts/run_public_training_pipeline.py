#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run staged public training, held-out evaluation and optional software-only export.

This host-side supervisor uses the public ./so101 CLI. It never enables hardware.
Each invocation owns a new output directory and can run in a detached tmux session.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/reinforcement_learning/rsl_rl'))
from public_physics import load_config as load_physics_config
from public_task import load_config as load_task_config


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def checkpoint_artifacts(run):
    configs = list(run.glob('rsl_rl/*/*/params/env.yaml'))
    if len(configs) != 1:
        raise RuntimeError(f'Expected one saved training environment in {run}, found {len(configs)}')
    checkpoints = list(configs[0].parent.parent.glob('model_*.pt'))
    if not checkpoints:
        raise RuntimeError(f'No checkpoint was saved in {run}')
    # Select the completed run's final checkpoint, then assess its quality by evaluation.
    checkpoint = max(checkpoints, key=lambda p: int(p.stem.removeprefix('model_')))
    return checkpoint, configs[0]


def passes_gate(report, minimum):
    # Exact typing alone does not establish release/clearance or valid contacts.
    score = float(report['overall_success_rate'])
    return 0 <= score <= 1 and score >= minimum


def validated_inputs(args):
    """Read validated inputs before creating a run or spending a training budget."""
    inputs = {}
    for kind, loader in (('task', load_task_config), ('physics', load_physics_config)):
        source = getattr(args, kind + '_config', None)
        if source is None:
            continue
        source = source.expanduser().resolve()
        payload = source.read_bytes()
        loader(source)
        if source.read_bytes() != payload:
            raise ValueError(f'{kind} configuration changed during validation: {source}')
        inputs[kind] = (source, payload)
    if args.joint_calibration is not None:
        # This package path also works in an uninstalled source archive; the
        # calibration loader and its constants use only the standard library.
        sys.path.insert(0, str(ROOT / 'source'))
        from so101_typing.hardware.joint_calibration import load
        source = args.joint_calibration.expanduser().resolve()
        payload = source.read_bytes()
        try:
            _, validated_sha256 = load(source, args.robot_id)
        except (KeyError, TypeError) as error:
            raise ValueError(f'invalid joint calibration: {error}') from error
        if (validated_sha256 != hashlib.sha256(payload).hexdigest()
                or source.read_bytes() != payload):
            raise ValueError(f'joint calibration changed during validation: {source}')
        inputs['joint_calibration'] = (source, payload)
    return inputs


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.root = args.out_dir.expanduser().resolve()
        inputs = validated_inputs(args)
        revision = None
        if (ROOT / '.git').exists():
            revision = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()
        self.root.mkdir(parents=True, exist_ok=False)
        self.status = {
            'schema_version': 1, 'started_at': now(), 'status': 'starting',
            'source_commit': revision,
            'source_distribution': 'git' if revision else 'archive',
            'supervisor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'hardware_motion_enabled': False, 'stages': [],
            'inputs': {},
        }
        if inputs:
            (self.root / 'inputs').mkdir()
        for kind, (source, payload) in inputs.items():
            key = kind if kind == 'joint_calibration' else kind + '_config'
            filename = 'joint_calibration.json' if kind == 'joint_calibration' else f'public_{kind}.json'
            snapshot = self.root / 'inputs' / filename
            snapshot.write_bytes(payload)
            self.status['inputs'][key] = {
                'source_path': str(source), 'snapshot_path': str(snapshot),
                'sha256': hashlib.sha256(payload).hexdigest(),
            }
        self.save()
        self.stop = threading.Event()
        self.monitor = threading.Thread(target=self.monitor_gpu, daemon=True)

    def save(self):
        self.status['updated_at'] = now()
        write_json(self.root / 'status.json', self.status)

    def monitor_gpu(self):
        with (self.root / 'gpu.csv').open('w') as log:
            log.write('timestamp, index, memory_used_mib, memory_total_mib, utilization_percent, power_watts\n')
            while not self.stop.is_set():
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits',
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    log.write(result.stdout)
                    log.flush()
                self.stop.wait(10)

    def run(self, name, arguments, simulation=True):
        run = self.root / name
        command = [str(ROOT / 'so101'), *map(str, arguments)]
        if simulation:
            command += ['--output-dir', str(run)]
            if getattr(self.args, 'image', None):
                command += ['--image', self.args.image]
        entry = {'name': name, 'command': command, 'started_at': now(), 'status': 'running'}
        self.status['status'] = 'running'
        self.status['active_stage'] = name
        self.status['stages'].append(entry)
        self.save()
        print(f'[{now()}] Starting {name}', flush=True)
        start = time.monotonic()
        with (self.root / f'{name}.launcher.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry.update(exit_code=result.returncode, elapsed_seconds=round(time.monotonic()-start, 2),
                     finished_at=now(), status='passed' if result.returncode == 0 else 'failed')
        self.save()
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; see {name}.launcher.log')
        return run

    def train(self, name, stage, iterations, checkpoint=None):
        command = ['train', '--actuator', 'anchorbench', '--stage', stage,
                   '--num-envs', self.args.num_envs, '--iterations', iterations, '--seed', self.args.seed]
        for kind in ('task', 'physics'):
            config = self.status['inputs'].get(kind + '_config')
            if config:
                command += ['--' + kind + '-config', config['snapshot_path']]
        if checkpoint:
            command += ['--checkpoint', checkpoint]
        return checkpoint_artifacts(self.run(name, command))

    def evaluate(self, name, stage, artifacts, seed):
        checkpoint, config = artifacts
        run = self.run(name, ['evaluate', '--actuator', 'anchorbench', '--stage', stage,
            '--checkpoint', checkpoint, '--env-config', config, '--num-envs', self.args.eval_envs,
            '--seed', seed])
        report = json.loads((run / 'evaluation.json').read_text())
        entry = self.status['stages'][-1]
        entry['overall_success_rate'] = report['overall_success_rate']
        entry['typed_exact_rate'] = report['typed_exact_rate']
        entry['quality_gate_passed'] = passes_gate(report, self.args.minimum_success)
        self.save()
        return entry['quality_gate_passed']

    def video(self, name, stage, artifacts, target):
        checkpoint, config = artifacts
        self.run(name, ['video', '--actuator', 'anchorbench', '--stage', stage,
            '--checkpoint', checkpoint, '--env-config', config, '--target', target])

    def execute(self):
        self.monitor.start()
        try:
            p1a = self.train('01-p1a-train', 'p1a', self.args.p1a_iterations)
            qualified = self.evaluate('02-p1a-evaluate', 'p1a', p1a, self.args.seed + 1000)
            self.video('03-p1a-video', 'p1a', p1a, 'HE')
            if not qualified:
                self.status['status'] = 'p1a_quality_gate_failed'
                self.status['next_action'] = 'Inspect evaluation and video before extending training or advancing curriculum.'
                return 2
            transit = self.train('04-transit15-train', 'transit15', self.args.transit_iterations, p1a[0])
            gates = [self.evaluate(f'05-transit15-evaluate-seed-{seed}', 'transit15', transit, seed)
                     for seed in (self.args.seed + 1000, self.args.seed + 2000)]
            self.video('06-transit15-video', 'transit15', transit, 'NEWTON')
            if self.args.prepare_deployment and all(gates):
                bundle = self.root / 'deployment'
                calibration = self.status['inputs'].get('joint_calibration')
                mapping = (['--joint-calibration', calibration['snapshot_path']] if calibration
                           else ['--encoder-convention', self.args.encoder_convention])
                self.run('07-prepare-deployment', ['prepare-deployment', '--checkpoint', transit[0],
                    '--env-config', transit[1], '--robot-id', self.args.robot_id,
                    '--out-dir', bundle, *mapping], simulation=False)
                self.run('08-deployment-dry-run', ['deploy', bundle, 'NEWTON'], simulation=False)
            self.status['status'] = 'software_quality_gates_passed' if all(gates) else 'transit15_quality_gate_failed'
            self.status['final_checkpoint'] = str(transit[0])
            self.status['next_action'] = 'Inspect saved rollout videos and traces; physical reproduction requires operator validation.'
            return 0 if all(gates) else 2
        except KeyboardInterrupt:
            self.status['status'] = 'interrupted'
            raise
        except Exception as error:
            self.status['status'] = 'execution_failed'
            self.status['error'] = str(error)
            raise
        finally:
            self.stop.set()
            self.monitor.join(timeout=15)
            self.status['finished_at'] = now()
            self.status.pop('active_stage', None)
            self.save()
            print(f'[{now()}] {self.status["status"]}: {self.root / "status.json"}', flush=True)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--task-config', type=Path, help='Keyboard pose/DR JSON; copied into this run before training')
    parser.add_argument('--physics-config', type=Path, help='Newton MJWarp JSON; copied into this run before training')
    parser.add_argument('--image', help='Docker image for all simulation stages; defaults to the public CLI image')
    parser.add_argument('--num-envs', type=positive, default=4096)
    parser.add_argument('--eval-envs', type=positive, default=1024)
    parser.add_argument('--p1a-iterations', type=positive, default=500)
    parser.add_argument('--transit-iterations', type=positive, default=2000)
    parser.add_argument('--seed', type=int, default=1307)
    parser.add_argument('--minimum-success', type=float, default=0.90)
    parser.add_argument('--prepare-deployment', action='store_true', help='Requires ./so101 setup-hardware; performs dry run only')
    parser.add_argument('--robot-id', help='Your calibrated robot ID; required with --prepare-deployment')
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument('--encoder-convention', choices=('lerobot', 'benchmark'), help='Verified mapping for your robot/scene')
    mapping.add_argument('--joint-calibration', type=Path, help='Measured encoder-to-model fit for your robot')
    args = parser.parse_args()
    if args.prepare_deployment and not args.robot_id:
        parser.error('--prepare-deployment requires --robot-id for your calibrated robot')
    if args.prepare_deployment and not (args.encoder_convention or args.joint_calibration):
        parser.error('--prepare-deployment requires --encoder-convention or --joint-calibration for your verified alignment')
    if not 0 < args.minimum_success <= 1:
        parser.error('--minimum-success must be in (0, 1]')
    try:
        return Pipeline(args).execute()
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
