"""Refine freshly fitted scene candidates using actual Newton first-key contacts.

Every supplied recording becomes fitting data. Continuous replay is a separate
optional diagnostic; these outputs are never installed on hardware.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from .calibrate_scene import key_down_samples, target_points
from .audit_scene_mapping import batch_fk, metrics, validate_identities
from .fk_probe_trajectory import digest
from .scene_check import load_recording
from .run_fk_probe import simulate
from scripts.fixture_layout import load_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-report', type=Path, action='append', required=True)
    parser.add_argument('--recording', type=Path, action='append', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--continuous', action='store_true', help='optimize complete down/up sequences instead of reset first-contact probes')
    parser.add_argument('--resume-export', action='store_true', help='export an existing completed search after verifying all input hashes')
    args = parser.parse_args()
    try:
        if args.out_dir.exists() and not args.resume_export:
            raise ValueError('refusing to overwrite contact-refinement directory')
        if not 1 <= args.rounds <= 10:
            raise ValueError('rounds must be 1..10')
        recordings = [load_recording(p) for p in args.recording]
        if any(r.get('status') != 'recorded' for r in recordings):
            raise ValueError('use completed recordings only')
        if any(not 0 <= s['lerobot_position'][5] <= 2. for r in recordings for s in r['samples']):
            raise ValueError('recordings must keep the moving jaw closed (0-2 percent)')
        for recording in recordings:
            key_down_samples(recording)
        geometry = load_geometry()
        validate_identities(recordings, geometry)
        seeds = [json.loads(p.read_text()) for p in args.seed_report]
        for seed in seeds:
            if seed['status'] != 'diagnostic_only_not_approved_for_deployment':
                raise ValueError('expected diagnostic seed report')
            if any(seed['identities'][k] != recordings[0][k] for k in seed['identities']):
                raise ValueError('seed and recording identities differ')
            if any(seed['reference_task'] != r['task_config'] for r in recordings):
                raise ValueError('seed and recording reference scenes differ')
        hashes = [digest(p) for p in args.recording]
        if len(set(hashes)) != len(hashes):
            raise ValueError('duplicate recordings')
        specification = args.out_dir/'search.json'
        if args.resume_export:
            if (args.out_dir/'mapping-report.json').exists():
                raise ValueError('refusing to overwrite exported mapping')
            spec = json.loads(specification.read_text())
            for paths, names in [(args.recording, spec['recordings']), (args.seed_report, spec['starting_reports'])]:
                if len(paths) != len(names) or any(digest(p) != digest(args.out_dir/n) for p, n in zip(paths, names)):
                    raise ValueError('search inputs changed')
        else:
            args.out_dir.mkdir(parents=True)
            spec = {'recordings': [], 'starting_reports': []}
            for i, p in enumerate(args.recording):
                name = f'recording-{i+1}/recording.json'
                (args.out_dir/name).parent.mkdir()
                shutil.copyfile(p, args.out_dir/name)
                spec['recordings'].append(name)
            for i, p in enumerate(args.seed_report):
                name = f'seed-{i+1}.json'; shutil.copyfile(p, args.out_dir/name)
                spec['starting_reports'].append(name)
            specification.write_text(json.dumps(spec, indent=2)+'\n')
            script = 'refine_so101_continuous_contacts.py' if args.continuous else 'refine_so101_scene_contacts.py'
            code = simulate(specification.resolve(), headless=True,
                            script='scripts/reinforcement_learning/rsl_rl/'+script,
                            report_name='contact-result.json', extra_args=['--rounds', str(args.rounds)])
            if code:
                raise ValueError('Newton refinement failed; inspect sim-console.log')
        result = json.loads((args.out_dir/'contact-result.json').read_text())
        if result['specification_sha256'] != digest(specification):
            raise ValueError('result does not match search specification')
        if args.continuous != ('edit_distances' in result['best']):
            raise ValueError('select the matching search mode with --continuous')
        x = np.array(result['best']['parameters'])
        report = copy.deepcopy(seeds[0])
        report.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                      method='contact_refined_joint_and_rigid_scene_fit', candidate_name='joint_scene',
                      historical_parameters_used=any(s.get('historical_parameters_used', True) for s in seeds),
                      fixed_offsets_deg={}, fit_shoulder_pan=True, fit_wrist_roll=True, fit_tilt=True,
                      keyboard_yaw_correction_fixed=False,
                      pan_yaw_gauge='contact-equivalent rigid rotations are not unique; no mechanical zero claim',
                      replay={'status': 'not_run'},
                      contact_refinement={'report_sha256': digest(args.out_dir/'contact-result.json'),
                                          'metric': 'complete_sequence' if args.continuous else 'reset_first_contact',
                                          'result': {k: v for k, v in result['best'].items() if k not in ('parameters', 'simulated_sequences', 'first_keys')},
                                          'all_recordings_used_for_fitting': True,
                                          'seed_sha256': [digest(p) for p in args.seed_report]},
                      script_sha256=digest(Path(__file__)))
        report.pop('optimizer', None)
        task = copy.deepcopy(report['reference_task'])
        task['keyboard'] = {'position_m': (x[5:8]/1000).tolist(),
                            'rotation_xyzw': Rotation.from_euler('xyz', x[8:11], degrees=True).as_quat().tolist()}
        report['task_config'] = task
        (args.out_dir/'task.json').write_text(json.dumps(task, indent=2)+'\n')
        report['task_config_sha256'] = digest(args.out_dir/'task.json')
        from scripts.fixture_layout import transform
        floor = min(transform(p, geometry, task)[2] for p in geometry['case_hull_m'])
        report['lowest_case_relative_to_robot_underside_mm'] = (floor-geometry['datum_m'][2])*1000
        scores = []
        for recording in recordings:
            rows = key_down_samples(recording)
            q = np.array([r['q_rad'] for r in rows])+np.r_[np.deg2rad(x[:5]), 0.]
            errors = (batch_fk(q)-target_points(rows, geometry, task, report['key_down_depression_assumption_m']))*1000
            scores.append({**metrics(errors), 'contacts': [{**r, 'residual_xyz_mm': e.tolist()} for r, e in zip(rows, errors)]})
        report['candidates'] = {'joint_scene': {'offset_deg': x[:5].tolist(), 'fit': scores[0],
                                              'additional_fit': scores[1:], 'validation': []}}
        report['recordings'] = [{'role': 'fit', 'name': name, 'sha256': h, 'kind': r['kind'],
                                 'contacts': scores[i]['count']} for i, (name, h, r) in
                                enumerate(zip(spec['recordings'], hashes, recordings))]
        (args.out_dir/'mapping-report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Refined offsets:', x[:5].tolist())
        print('Refined keyboard:', task['keyboard'])
        print('Contact result:', report['contact_refinement']['result'])
        print('Per-key fitting does not qualify physical deployment; see docs/release/ALIGNMENT.md.')
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
