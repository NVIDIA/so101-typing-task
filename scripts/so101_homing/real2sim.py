"""Fit and check a typing scene from hand-guided recordings; never moves hardware."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from scripts.fixture_layout import load_geometry
from .audit_scene_mapping import validate_identities
from .calibrate_scene import calibrate, key_down_samples
from .fk_probe_trajectory import ROOT, digest
from .scene_check import load_recording


def inputs(directory=None, paths=()):
    paths = sorted(directory.glob('*/recording.json')) if directory else list(paths)
    paths = [p.resolve() for p in paths]
    if len(paths) < 2:
        raise ValueError('collect at least two recordings; repeat real2sim.sh record with the same output root')
    if len({digest(p) for p in paths}) != len(paths):
        raise ValueError('duplicate recording contents; use distinct captures')
    records = [load_recording(p) for p in paths]
    validate_identities(records, load_geometry())
    for r in records:
        if r.get('status') != 'recorded':
            raise ValueError('a recording is incomplete; select completed captures with --recording')
        if r['task_config'] != records[0]['task_config']:
            raise ValueError('recordings have different reference layouts')
        if any(not 0 <= s['lerobot_position'][5] <= 2. for s in r['samples']):
            raise ValueError('keep the moving jaw closed throughout recording')
        key_down_samples(r)
    if {r['key'] for r in key_down_samples(records[0])} != set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        raise ValueError('the first recording must cover A-Z; collect all letters before fitting')
    return paths, records


def contact_pass(result):
    expected = [r['key'] for r in result.get('contacts', [])]
    best = result.get('best', {})
    return bool(expected) and best.get('total') == len(expected) and best.get('correct_first_contacts') == len(expected) and best.get('first_keys') == expected


def run(paths, records, out, rounds):
    if out.exists():
        raise ValueError(f'refusing to overwrite {out}; choose a new output directory')
    out.mkdir(parents=True)
    reference = out/'measured-reference.json'
    reference.write_text(json.dumps(records[0]['task_config'], indent=2)+'\n')
    summary = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'fitting', 'criterion': 'reset_between_keys_then_correct_first_contact',
               'all_recordings_used_for_fitting': True, 'physical_deployment_validated': False,
               'inputs': [{'sha256': digest(p), 'kind': r['kind']} for p, r in zip(paths, records)]}
    status = out/'scene-calibration.json'
    def save():
        status.write_text(json.dumps(summary, indent=2)+'\n')
    save()
    try:
        print('Fitting starting scenes from recorded contacts...', flush=True)
        # One conservative start and one all-joint/full-pose start. The refinement
        # below uses every recording: none are advertised as held-out evidence.
        calibrate(reference, paths[0], paths[1:], out/'initial-fit')
        calibrate(reference, paths[0], [], out/'all-joint-fit', additional_fit_paths=paths[1:],
                  fit_pan=True, fit_roll=True, free_yaw=True, tilt=True)
        print('Refining all five offsets and keyboard pose with Newton per-key resets; log: '+str(out/'refinement.log'), flush=True)
        command = [sys.executable, '-m', 'scripts.so101_homing.refine_scene_contacts',
                   '--seed-report', str(out/'initial-fit/mapping-report.json'),
                   '--seed-report', str(out/'all-joint-fit/mapping-report.json'),
                   '--out-dir', str(out/'refined'), '--rounds', str(rounds)]
        for path in paths:
            command += ['--recording', str(path)]
        with (out/'refinement.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result_path = out/'refined/contact-result.json'
        result = json.loads(result_path.read_text())
        for name in ('task.json', 'mapping-report.json'):
            shutil.copyfile(out/'refined'/name, out/name)
        summary.update(status='passed_per_key_replay' if contact_pass(result) else 'failed_per_key_replay',
                       correct_first_contacts=result['best']['correct_first_contacts'], total=result['best']['total'],
                       task_sha256=digest(out/'task.json'), mapping_sha256=digest(out/'mapping-report.json'),
                       contact_report='refined/contact-result.json', contact_report_sha256=digest(result_path))
        save()
        if not contact_pass(result):
            print(f"Per-key replay FAILED: {summary['correct_first_contacts']}/{summary['total']}. Inspect {result_path}; do not treat this scene as calibrated.")
            return 1
        print(f"Per-key replay PASSED: {summary['correct_first_contacts']}/{summary['total']}.")
        print('Scene:', out/'task.json')
        print('Keep these robot-specific offsets for deployment:', out/'mapping-report.json')
        print('Train in the model coordinates with this scene (choose a budget and evaluate early):')
        print(shlex.join(['./so101', 'train', '--actuator', 'anchorbench', '--stage', 'p1a',
                          '--task-config', str(out/'task.json'), '--num-envs', '4096', '--iterations', '500']))
        return 0
    except Exception as error:
        summary.update(status='failed', error=str(error)); save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--recordings', type=Path, help='root used by the record command; includes its timestamped captures')
    source.add_argument('--recording', type=Path, action='append', help='explicit capture path; repeat for each recording')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    try:
        if not 1 <= args.rounds <= 10:
            raise ValueError('rounds must be 1..10')
        paths, records = inputs(args.recordings, args.recording or ())
        return run(paths, records, args.out_dir.resolve(), args.rounds)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.exit(main())
