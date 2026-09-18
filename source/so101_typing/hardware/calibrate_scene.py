"""Fit a new effective scene from measured key-downs; offline, no hardware I/O.

Fit lift/elbow/wrist flex and keyboard XYZ/yaw by default. Optional pan fitting
replaces free keyboard yaw; wrist roll can also be fitted.
No historical offset or pose estimates are used as optimization inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scripts.fixture_layout import load_geometry, transform
from scripts.reinforcement_learning.rsl_rl.public_task import load_config, nominal_geometry
from .audit_scene_mapping import batch_fk, metrics, validate_identities, IDENTITY_FIELDS
from .fk_probe_trajectory import ROOT, digest
from .scene_check import load_recording
from .so101_kinematics import ARM_JOINT_NAMES, FIXED_JAW_TYPING_TIP_SITE_POS


def key_down_samples(recording, max_gap_s=.05):
    """Keep every repeated press; nearest encoder sample to recorded down time.

    Existing September recordings store USB receipt times, not kernel edge times.
    Preserve that limitation rather than synthesizing edge precision.
    """
    times = np.array([s['time_s'] for s in recording['samples']])
    rows, active = [], {}
    for event in recording['events']:
        key = (event.get('char') or '').upper()
        if len(key) != 1 or key not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            raise ValueError('fit recordings must contain only A-Z down/up events')
        if event['edge'] == 'down':
            if active:
                raise ValueError('overlapping key-downs in physical recording')
            index = int(np.argmin(np.abs(times-event['time_s'])))
            delta = float(times[index]-event['time_s'])
            if abs(delta) > max_gap_s:
                raise ValueError(f'{key}: nearest encoder read exceeds {max_gap_s:g} s')
            row = {'key': key, 'down_s': event['time_s'], 'sample_index': index,
                   'sample_dt_ms': delta*1000,
                   'q_rad': recording['samples'][index]['q_model_rad']}
            active[event['code']] = row
        elif event['edge'] == 'up':
            if event['code'] not in active:
                raise ValueError('key-up has no matching down')
            row = active.pop(event['code'])
            if row['key'] != key or event['time_s'] <= row['down_s']:
                raise ValueError('invalid key release')
            rows.append({**row, 'up_s': event['time_s']})
        else:
            raise ValueError('expected down/up edges')
    if active or not rows:
        raise ValueError('recording lacks complete presses')
    return rows


def target_points(rows, geometry, task, depression_m):
    keys = {k['label'].upper(): k for k in geometry['keys']
            if k['slot'].startswith('main_') and len(k['label']) == 1}
    _, rotation, _, _ = nominal_geometry(geometry['reference_position_m'],
                                         geometry['reference_rotation_xyzw'], [], task.get('keyboard', {}))
    asset = ROOT/'source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/keyboards/data/mx_keys/mx_keys_asset.npz'
    with np.load(asset) as data:
        normal = Rotation.from_quat(rotation).apply(
            Rotation.from_quat(data['deck_quat_xyzw']).apply([0, 0, 1]))
    return np.array([transform(keys[r['key']]['rim_center_m'], geometry, task)
                     for r in rows])-depression_m*normal


def unpack(values, tilt=False, fit_pan=False, fit_roll=False, free_yaw=False):
    offsets = np.array([values[6] if fit_pan else 0., *values[:3], values[7] if fit_roll else 0.])
    yaw_index = 7+int(fit_roll)
    yaw = values[yaw_index] if fit_pan and free_yaw else (0. if fit_pan else values[6])
    tilt_index = yaw_index+int(fit_pan and free_yaw)
    angles = [values[tilt_index], values[tilt_index+1], yaw] if tilt else [0., 0., yaw]
    return offsets, Rotation.from_euler('xyz', angles, degrees=True), values[3:6]/1000


def residual(values, q, targets, pivot, tilt=False, fit_pan=False, fit_roll=False, free_yaw=False):
    offsets, rotation, translation = unpack(values, tilt, fit_pan, fit_roll, free_yaw)
    tips = batch_fk(q+np.r_[np.deg2rad(offsets), 0.])
    posed_targets = rotation.apply(targets-pivot)+pivot+translation
    return (tips-posed_targets)*1000


def fit(q, targets, keys, pivot, *, tilt=False, fit_pan=False, fit_roll=False, free_yaw=False):
    # Equal weight per key so repeated letters do not dominate the scene.
    counts = {k: keys.count(k) for k in set(keys)}
    weights = np.array([1/np.sqrt(counts[k]) for k in keys])
    size = 7+int(fit_roll)+int(fit_pan and free_yaw)+2*int(tilt)
    lower = np.array([-15.]*3+[-40.]*3+[-15.]+([-15.] if fit_roll else [])
                     +([-15.] if fit_pan and free_yaw else [])+([-8., -8.] if tilt else []))
    upper = -lower
    def objective(x):
        errors = residual(x, q, targets, pivot, tilt, fit_pan, fit_roll, free_yaw)*weights[:, None]
        # Same weak zero-centred regularization concept as the original fitter.
        return np.r_[errors.ravel(), .025*unpack(x, tilt, fit_pan, fit_roll, free_yaw)[0]]
    rng = np.random.default_rng(20260916)
    starts = [np.zeros(size)]+[rng.uniform(-1., 1., size) for _ in range(5)]
    fits = [least_squares(objective, start, bounds=(lower, upper), loss='soft_l1',
                          f_scale=1.5, max_nfev=2000, x_scale='jac',
                          ftol=1e-10, xtol=1e-10, gtol=1e-10) for start in starts]
    best = min(fits, key=lambda f: f.cost)
    jac = np.column_stack([(residual(best.x+a*1e-4, q, targets, pivot, tilt, fit_pan, fit_roll, free_yaw).ravel()
                            -residual(best.x-a*1e-4, q, targets, pivot, tilt, fit_pan, fit_roll, free_yaw).ravel())/2e-4
                           for a in np.eye(size)])
    singular = np.linalg.svd(jac, compute_uv=False)
    return best.x, {'cost': float(best.cost), 'converged': bool(best.success),
                    'parameter_order': ['shoulder_lift_deg', 'elbow_flex_deg', 'wrist_flex_deg',
                                        'keyboard_dx_mm', 'keyboard_dy_mm', 'keyboard_dz_mm',
                                        'shoulder_pan_deg' if fit_pan else 'keyboard_dyaw_deg']
                                       +(['wrist_roll_deg'] if fit_roll else [])
                                       +(['keyboard_dyaw_deg'] if fit_pan and free_yaw else [])
                                       +(['keyboard_droll_deg', 'keyboard_dpitch_deg'] if tilt else []),
                    'boundary_indices': np.flatnonzero(np.abs(best.x) >= upper-1e-4).tolist(),
                    'jacobian_singular_values': singular.tolist(),
                    'starts': [{'cost': float(f.cost), 'values': f.x.tolist(),
                                'converged': bool(f.success)} for f in fits]}


def calibrate(task_path, fit_path, validation_paths, out_dir, *, depression_m=.0009, tilt=False,
              fit_pan=False, fit_roll=False, free_yaw=False, additional_fit_paths=()):
    if out_dir.exists():
        raise ValueError(f'refusing to overwrite {out_dir}')
    if not np.isfinite(depression_m) or not 0 <= depression_m <= .0018:
        raise ValueError('key-down depression must be finite, between 0 and 1.8 mm')
    fit_count = 1+len(additional_fit_paths)
    paths = [fit_path, *additional_fit_paths, *validation_paths]
    hashes = [digest(p) for p in paths]
    if len(paths) < 2 or len(set(hashes)) != len(hashes):
        raise ValueError('require at least two distinct recording contents')
    recordings = [load_recording(p) for p in paths]
    geometry, task = load_geometry(), load_config(task_path)
    validate_identities(recordings, geometry)
    if any(r.get('status') != 'recorded' for r in recordings):
        raise ValueError('use completed recordings only')
    if task != recordings[0]['task_config']:
        raise ValueError('starting task must match the measured reference in the recordings')
    if any(not 0 <= s['lerobot_position'][5] <= 2. for r in recordings for s in r['samples']):
        raise ValueError('recordings must keep the moving jaw closed (0-2 percent)')
    if any(r['task_config'] != recordings[0]['task_config'] for r in recordings):
        raise ValueError('recordings describe different physical-reference tasks')
    rows = [key_down_samples(r) for r in recordings]
    if set(r['key'] for r in rows[0]) != set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        raise ValueError('fitting recording must cover all A-Z letters')
    targets = [target_points(r, geometry, task, depression_m) for r in rows]
    qs = [np.array([r['q_rad'] for r in rr]) for rr in rows]
    position, rotation, _, _ = nominal_geometry(geometry['reference_position_m'],
        geometry['reference_rotation_xyzw'], [], task.get('keyboard', {}))
    pivot = np.array(position)
    values, optimizer = fit(np.concatenate(qs[:fit_count]), np.concatenate(targets[:fit_count]),
                            [r['key'] for rr in rows[:fit_count] for r in rr], pivot,
                            tilt=tilt, fit_pan=fit_pan, fit_roll=fit_roll, free_yaw=free_yaw)
    offsets, correction, translation = unpack(values, tilt, fit_pan, fit_roll, free_yaw)
    candidate = {**task, 'keyboard': {'position_m': (pivot+translation).tolist(),
        'rotation_xyzw': (correction*Rotation.from_quat(rotation)).as_quat().tolist()}}
    scores = []
    for r, q, target in zip(rows, qs, targets):
        errors = residual(values, q, target, pivot, tilt, fit_pan, fit_roll, free_yaw)
        scores.append({**metrics(errors), 'contacts': [
            {**row, 'residual_xyz_mm': error.tolist()} for row, error in zip(r, errors)]})
    floor = min(transform(p, geometry, candidate)[2] for p in geometry['case_hull_m'])
    report = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'diagnostic_only_not_approved_for_deployment', 'method': 'event_associated_joint_and_rigid_scene_fit',
        'identities': {k: recordings[0][k] for k in IDENTITY_FIELDS},
        'joint_order': list(ARM_JOINT_NAMES), 'sampling': 'nearest_measured_sample_to_recorded_down_time',
        'event_clock_note': 'September recordings use USB receipt time; not kernel-synchronised physical contact time.',
        'fixed_jaw_site_m': list(FIXED_JAW_TYPING_TIP_SITE_POS),
        'fixed_offsets_deg': {**({} if fit_pan else {'shoulder_pan': 0.}),
                              **({} if fit_roll else {'wrist_roll': 0.})},
        'fit_shoulder_pan': fit_pan, 'fit_wrist_roll': fit_roll,
        'keyboard_yaw_correction_fixed': fit_pan and not free_yaw,
        'pan_yaw_gauge': 'weak zero-offset prior chooses among equivalent poses; not uniquely identifiable' if fit_pan and free_yaw else 'one rotation reference fixed',
        'historical_parameters_used': False, 'key_down_depression_assumption_m': depression_m,
        'fit_tilt': tilt, 'reference_task': task, 'reference_task_sha256': digest(task_path),
        'task_config': candidate, 'effective_scene_not_literal_measurement': True,
        'lowest_case_relative_to_robot_underside_mm': (floor-geometry['datum_m'][2])*1000,
        'optimizer': optimizer, 'script_sha256': digest(Path(__file__)),
        'recordings': [{'role': 'fit' if i < fit_count else 'validation', 'name': p.name,
                       'folder': ('fit' if i == 0 else f'fit-{i+1}') if i < fit_count else f'validation-{i-fit_count+1}',
                       'sha256': hashes[i], 'kind': r['kind'], 'contacts': len(rows[i])}
                      for i, (p, r) in enumerate(zip(paths, recordings))],
        'candidate_name': 'joint_scene' if fit_pan or fit_roll else 'fit3_pose',
        'candidates': {('joint_scene' if fit_pan or fit_roll else 'fit3_pose'):
                       {'offset_deg': offsets.tolist(), 'fit': scores[0],
                        'additional_fit': scores[1:fit_count], 'validation': scores[fit_count:]}},
        'replay': {'status': 'not_run'},
        'limitations': 'Contact centres and depression are assumptions. Fit residuals are not contact replay or hardware success. Validation sessions previously inspected are regression evidence.'}
    out_dir.mkdir(parents=True)
    (out_dir/'task.json').write_text(json.dumps(candidate, indent=2)+'\n')
    report['task_config_sha256'] = digest(out_dir/'task.json')
    (out_dir/'mapping-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for i, p in enumerate(paths):
        folder = out_dir/report['recordings'][i]['folder']
        folder.mkdir()
        shutil.copyfile(p, folder/'recording.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-config', type=Path, required=True)
    parser.add_argument('--fit-recording', type=Path, required=True)
    parser.add_argument('--validation-recording', type=Path, action='append', default=[])
    parser.add_argument('--additional-fit-recording', type=Path, action='append', default=[])
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--key-down-depression-mm', type=float, default=.9)
    parser.add_argument('--fit-tilt', action='store_true', help='also fit effective roll/pitch; diagnostic')
    parser.add_argument('--fit-shoulder-pan', action='store_true', help='fit pan instead of keyboard yaw correction')
    parser.add_argument('--fit-wrist-roll', action='store_true', help='also fit the wrist-roll offset')
    parser.add_argument('--free-keyboard-yaw', action='store_true', help='fit yaw together with pan; weak prior resolves the coordinate gauge, not physical identification')
    parser.add_argument('--replay', action='store_true', help='run headless Newton on unchanged copies of all recordings')
    args = parser.parse_args()
    try:
        report = calibrate(args.task_config, args.fit_recording, args.validation_recording,
                           args.out_dir, depression_m=args.key_down_depression_mm/1000, tilt=args.fit_tilt,
                           fit_pan=args.fit_shoulder_pan, fit_roll=args.fit_wrist_roll,
                           free_yaw=args.free_keyboard_yaw, additional_fit_paths=args.additional_fit_recording)
        c = report['candidates'][report['candidate_name']]
        print('New offsets (deg):', c['offset_deg'], flush=True)
        print('Effective keyboard:', report['task_config']['keyboard'], flush=True)
        print('Mean residual mm, fit / validation:', c['fit']['mean_mm'],
              [s['mean_mm'] for s in c['validation']], flush=True)
        print('Parameter bounds reached:', report['optimizer']['boundary_indices'], flush=True)
        if args.replay:
            replays = []
            for row in report['recordings']:
                folder = row['folder']
                command = [sys.executable, '-m', 'so101_typing.hardware.scene_check', 'replay', '--headless',
                    '--recording', str((args.out_dir/folder/'recording.json').resolve()),
                    '--replay-task-config', str((args.out_dir/'task.json').resolve()),
                    '--mapping-report', str((args.out_dir/'mapping-report.json').resolve()),
                    '--mapping-candidate', report['candidate_name']]
                with (args.out_dir/f'replay-{folder}.log').open('x') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'replay failed; inspect {args.out_dir}/replay-{folder}.log')
                paths = list((args.out_dir/folder).glob('placement-*/scene-report.json'))
                if len(paths) != 1:
                    raise ValueError('expected exactly one contact replay report')
                data = json.loads(paths[0].read_text())
                item = {'role': folder, 'report': str(paths[0].relative_to(args.out_dir)),
                    'key_sequence_matches': data['key_sequence_matches'],
                    'physical_edges': len(data['physical_events']), 'simulated_edges': len(data['simulation_events']),
                    'max_tip_drift_within_physics_step_m': data['max_tip_drift_within_physics_step_m']}
                replays.append(item)
                print('Replay:', item, flush=True)
            # Keep mapping-report immutable: replay artifacts pin its content hash.
            summary = {'status': 'passed' if all(r['key_sequence_matches'] for r in replays) else 'failed',
                       'replays': replays, 'physical_deployment_validated': False}
            (args.out_dir/'replay-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            if summary['status'] != 'passed':
                print('Contact replay FAILED; this candidate is not ready for training or deployment.', flush=True)
                sys.exit(1)
        print('Saved diagnostic calibration:', args.out_dir.resolve())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
