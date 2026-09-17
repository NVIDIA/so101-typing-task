"""Prepare a scene mapping and qualify it from recorded evidence; no hardware I/O.

Qualification checks representative contacts, not policy performance. Evidence
hashes detect drift; they are not signatures or independent physical metrology.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.fixture_layout import load_geometry
from scripts.reinforcement_learning.rsl_rl.public_task import load_config, nominal_geometry
from . import joint_calibration
from .fixed_cartesian_policy import PHYSICAL_CALIBRATED_20260718_PROFILE as REFERENCE
from .fk_probe_trajectory import ROOT, RESET, digest, load as load_trajectory, tip
from .robot import require_calibration
from .scene_check import load_recording, validate_mapping_offsets
from .so101_kinematics import ARM_JOINT_NAMES, FIXED_JAW_TYPING_TIP_SITE_POS

PROBE_KEYS = frozenset('QGP')
CLOSED_MAX_PERCENT = 2.0
CLEARANCE_M = .015
CLEARANCE_TICKS = 2


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f'{Path(path).name} must contain a JSON object')
    return data


def model_identity():
    geometry = load_geometry()
    sources = {**geometry['source_sha256'],
               'scripts/so101_homing/so101_kinematics.py': digest(ROOT/'scripts/so101_homing/so101_kinematics.py')}
    return {'geometry_sha256': digest(ROOT/'docs/fixtures/model_geometry.json'),
            'robot_usd_sha256': geometry['robot_usd_sha256'], 'source_sha256': sources,
            'fixed_tip_site_m': list(FIXED_JAW_TYPING_TIP_SITE_POS),
            'moving_jaw_goal_percent': 0., 'moving_jaw_closed_max_percent': CLOSED_MAX_PERCENT}


def prepare(task_path, robot_id, out_dir, *, convention=None, mapping_report=None,
            mapping_candidate=None, joint_fit=None, validation_recordings=(), experimental=False):
    if sum((convention is not None, mapping_report is not None, joint_fit is not None)) != 1:
        raise ValueError('choose exactly one encoder convention, mapping report or joint calibration')
    if bool(mapping_report) != bool(mapping_candidate):
        raise ValueError('--mapping-report and --mapping-candidate are required together')
    if validation_recordings and not mapping_report:
        raise ValueError('--validation-recording requires a mapping report')
    if experimental and not mapping_report:
        raise ValueError('--experimental requires a mapping report')
    task = load_config(task_path)
    calibration = require_calibration(robot_id)
    contract = {'robot_id': robot_id, 'calibration_sha256': digest(calibration),
                'task_config': task, 'task_config_sha256': digest(task_path),
                'joint_order': list(ARM_JOINT_NAMES), **model_identity()}
    if mapping_report:
        report = read_json(mapping_report)
        if mapping_candidate not in ('fit4', 'fit3_pose', 'joint_scene') or report.get('status') != 'diagnostic_only_not_approved_for_deployment':
            raise ValueError('recording-based alignment requires diagnostic fit4, fit3_pose or joint_scene')
        for field in ('robot_id', 'calibration_sha256', 'robot_usd_sha256', 'source_sha256'):
            if report.get('identities', {}).get(field) != contract[field]:
                raise ValueError(f'mapping report identity differs: {field}')
        if report.get('task_config') != task or report.get('joint_order') != list(ARM_JOINT_NAMES):
            raise ValueError('mapping report differs from frozen task or joint order')
        rows = report.get('recordings', [])
        fitting = [r['sha256'] for r in rows if r.get('role') == 'fit']
        reserved = [r['sha256'] for r in rows if r.get('role') == 'validation']
        for path in validation_recordings:
            recording = load_recording(path)
            check_reserved_recording(contract, recording, report.get('reference_task', task))
            reserved.append(digest(path))
        if not fitting or (not reserved and not experimental) or len(set(fitting+reserved)) != len(fitting+reserved):
            raise ValueError('mapping needs distinct fit and reserved validation recordings')
        offsets = validate_mapping_offsets(report['candidates'][mapping_candidate]['offset_deg']).tolist()
        if mapping_candidate != 'joint_scene' and offsets[4] != 0:
            raise ValueError('recording fit must leave wrist roll at zero')
        if mapping_candidate == 'fit3_pose':
            if (offsets[0] != 0 or report.get('method') != 'event_associated_joint_and_rigid_scene_fit'
                    or not isinstance(report.get('reference_task'), dict)):
                raise ValueError('fit3_pose requires fixed pan and measured reference task provenance')
        if mapping_candidate == 'joint_scene' and (report.get('method') not in
                ('event_associated_joint_and_rigid_scene_fit', 'contact_refined_joint_and_rigid_scene_fit')
                or not isinstance(report.get('reference_task'), dict)):
            raise ValueError('joint_scene requires joint/scene method and reference task provenance')
        mapping = {'kind': 'recording_'+mapping_candidate, 'source_sha256': digest(mapping_report),
                   'fit_recording_sha256': fitting[0], 'fit_recordings_sha256': fitting,
                   'reserved_recording_sha256': reserved}
        if experimental:
            mapping['experimental'] = True
        if mapping_candidate in ('fit3_pose', 'joint_scene'):
            mapping['reference_task'] = report['reference_task']
    elif joint_fit:
        offsets, source_hash = joint_calibration.load(joint_fit, robot_id)
        mapping = {'kind': 'measured_joint_fit', 'source_sha256': source_hash}
    else:
        if convention not in ('lerobot', 'benchmark'):
            raise ValueError('select lerobot or benchmark explicitly')
        offsets = [0.]*5 if convention == 'lerobot' else list(REFERENCE.joint_zero_offset_deg)
        mapping = {'kind': convention}
    contract['joint_zero_offset_deg'] = validate_mapping_offsets(list(offsets)).tolist()
    contract['mapping'] = mapping
    result = {'schema_version': 1, 'kind': 'so101_alignment_candidate',
              'status': 'unqualified', 'contract': contract,
              'contract_sha256': fingerprint(contract), 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    validate_candidate(result, robot_id)
    out_dir.mkdir(parents=True, exist_ok=False)
    if mapping_report:
        shutil.copy2(mapping_report, out_dir/'mapping-report.json')
    for index, path in enumerate(validation_recordings, 1):
        shutil.copy2(path, out_dir/f'validation-{index}.json')
    if joint_fit:
        shutil.copy2(joint_fit, out_dir/'joint-calibration.json')
    (out_dir/'alignment.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def validate_candidate(data, robot_id):
    if data.get('schema_version') != 1 or data.get('kind') != 'so101_alignment_candidate' or data.get('status') != 'unqualified':
        raise ValueError('expected an unqualified scene alignment candidate')
    c = data['contract']
    if fingerprint(c) != data.get('contract_sha256'):
        raise ValueError('alignment contract hash changed')
    if c.get('robot_id') != robot_id or c.get('calibration_sha256') != digest(require_calibration(robot_id)):
        raise ValueError('alignment robot or LeRobot calibration changed')
    for field, value in model_identity().items():
        if c.get(field) != value:
            raise ValueError(f'alignment model/jaw identity changed: {field}')
    if c.get('joint_order') != list(ARM_JOINT_NAMES):
        raise ValueError('alignment joint order changed')
    validate_mapping_offsets(c['joint_zero_offset_deg'])
    return c


def candidate(path, robot_id):
    data = read_json(path)
    validate_candidate(data, robot_id)
    return data


def check_plan(contract, plan):
    for field in ('task_config', 'source_sha256', 'robot_usd_sha256'):
        if plan.get(field) != contract[field]:
            raise ValueError(f'probe differs from alignment: {field}')


def key_sequence(events, *, physical):
    sequence = []
    for event in events:
        key = event.get('char' if physical else 'key', '')
        if not isinstance(key, str) or len(key) != 1 or key.upper() not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            raise ValueError('alignment evidence contains a non-letter/unknown key')
        if event.get('edge') not in ('down', 'up'):
            raise ValueError('alignment evidence contains an invalid key edge')
        sequence.append([key.upper(), event['edge']])
    if len(sequence) % 2 or any(a[1] != 'down' or b != [a[0], 'up'] for a, b in zip(sequence[::2], sequence[1::2])):
        raise ValueError('alignment requires complete, non-overlapping press/release pairs')
    if not PROBE_KEYS.issubset({key for key, _ in sequence}):
        raise ValueError('reserved replay must cover Q, G and P')
    return sequence


def check_reserved_recording(contract, recording, reference_task):
    if recording.get('status') != 'recorded':
        raise ValueError('reserved recording did not complete')
    for field in ('robot_id', 'calibration_sha256', 'source_sha256', 'robot_usd_sha256'):
        if recording.get(field) != contract[field]:
            raise ValueError(f'reserved recording differs from alignment: {field}')
    if recording.get('task_config') != reference_task:
        raise ValueError('reserved recording differs from alignment: task_config')
    if any(not 0 <= s['lerobot_position'][5] <= CLOSED_MAX_PERCENT for s in recording['samples']):
        raise ValueError('reserved recording does not verify the closed moving jaw')
    return key_sequence(recording['events'], physical=True)


def check_replay(contract, recording_path, report_path):
    recording, report = load_recording(recording_path), read_json(report_path)
    mapping = contract['mapping']
    expected = check_reserved_recording(contract, recording, mapping.get('reference_task', contract['task_config']))
    record_hash = digest(recording_path)
    if mapping['kind'] in ('recording_fit4', 'recording_fit3_pose', 'recording_joint_scene') and record_hash not in mapping['reserved_recording_sha256']:
        raise ValueError('replay is not from the reserved validation recordings')
    if (report.get('status') != 'replayed' or report.get('recording_sha256') != record_hash
            or report.get('task_config') != contract['task_config']
            or report.get('joint_zero_offset_deg') != contract['joint_zero_offset_deg']):
        raise ValueError('replay status, recording, scene or mapping differs from alignment')
    actual = key_sequence(report['simulation_events'], physical=False)
    if expected != actual or report.get('key_sequence_matches') is not True:
        raise ValueError('reserved replay has missed, extra or wrong contacts')
    return {'recording_sha256': record_hash, 'presses': len(expected)//2}


def check_probe(contract, contract_hash, trajectory_path, simulation_path, log_path):
    plan, sim = load_trajectory(trajectory_path), read_json(simulation_path)
    check_plan(contract, plan)
    trajectory_hash = digest(trajectory_path)
    if (sim.get('status') != 'passed' or sim.get('trajectory_sha256') != trajectory_hash
            or [e['edge'] for e in sim.get('events', [])] != ['down', 'up']
            or sim.get('steps') != len(plan['rows'])):
        raise ValueError('probe needs a complete simulation pass for its exact trajectory')
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not rows or rows[0].get('kind') != 'header' or rows[-1].get('kind') != 'result':
        raise ValueError('incomplete physical probe log')
    header = rows[0]
    expected = {'trajectory_sha256': trajectory_hash, 'alignment_contract_sha256': contract_hash,
                'calibration_sha256': contract['calibration_sha256'], 'robot_id': contract['robot_id'],
                'joint_zero_offset_deg': contract['joint_zero_offset_deg'], 'alignment_probe_schema': 1}
    if any(header.get(k) != v for k, v in expected.items()):
        raise ValueError('physical probe identity differs from alignment/trajectory')
    if rows[-1].get('status') != 'passed' or rows[-1].get('edges') != ['down', 'up']:
        raise ValueError('physical probe failed or did not press and release')
    commands, edges, cleared, streak, previous_time = [], [], False, 0, None
    offsets = np.array(contract['joint_zero_offset_deg'])
    for row in rows[1:-1]:
        if row.get('kind') == 'key':
            edge = row.get('edge')
            if str(row.get('char', '')).upper() != plan['letter'] or edge not in ('down', 'up'):
                raise ValueError('physical probe contains a wrong key')
            edges.append(edge)
            if edges != ['down', 'up'][:len(edges)]:
                raise ValueError('physical probe contains repeated/unordered key edges')
            if edge == 'down' and row.get('phase') not in ('descend', 'press', 'lift'):
                raise ValueError('physical key-down occurred outside the probe press')
        elif row.get('kind') == 'command':
            q = np.asarray(row['q_command_rad'], dtype=float)
            measured = np.asarray(row['q_measured_encoder_deg'], dtype=float)+offsets
            timestamp = float(row['monotonic_s'])
            grip = row.get('gripper_percent')
            if (q.shape != (5,) or measured.shape != (5,) or not np.isfinite(q).all()
                    or not np.isfinite(measured).all() or not np.isfinite(timestamp)
                    or previous_time is not None and not 0 < timestamp-previous_time <= .12
                    or type(grip) not in (int, float) or not 0 <= grip <= CLOSED_MAX_PERCENT):
                raise ValueError('physical probe has invalid timing, measurements or jaw closure')
            previous_time = timestamp
            if (np.max(np.abs(np.rad2deg(q)-measured)) > 12
                    or not np.allclose(row['q_command_encoder_deg'], np.rad2deg(q)-offsets, atol=1e-8, rtol=0)
                    or not np.allclose(row['q_measured_model_deg'], measured, atol=1e-8, rtol=0)):
                raise ValueError('physical probe tracking or inverse conversion differs')
            if row['step'] >= 0:
                commands.append(row)
            if edges == ['down', 'up'] and tip(np.deg2rad(measured))[2] >= plan['rim_xyz_b_m'][2]+CLEARANCE_M:
                streak += 1
                cleared |= streak >= CLEARANCE_TICKS
            else:
                streak = 0
        else:
            raise ValueError('unknown physical probe log event')
    if edges != ['down', 'up'] or not cleared or len(commands) != len(plan['rows']):
        raise ValueError('physical probe lacks complete trajectory, key release or sustained clearance')
    for i, (row, planned) in enumerate(zip(commands, plan['rows'])):
        if row['step'] != i or row['phase'] != planned['phase'] or not np.allclose(row['q_command_rad'], planned['q_rad'], atol=1e-9, rtol=0):
            raise ValueError('physical commands differ from the simulated trajectory')
    return plan['letter']


def validate_evidence(data, paths, robot_id):
    required = {'recording', 'replay'} | {f'{key}_{part}' for key in PROBE_KEYS for part in ('trajectory', 'simulation', 'hardware')}
    if set(paths) != required:
        raise ValueError('qualification requires one reserved replay and complete Q/G/P probe evidence')
    c = validate_candidate(data['candidate'], robot_id)
    replay = check_replay(c, paths['recording'], paths['replay'])
    keys = [check_probe(c, data['candidate']['contract_sha256'],
                        paths[f'{key}_trajectory'], paths[f'{key}_simulation'], paths[f'{key}_hardware'])
            for key in sorted(PROBE_KEYS)]
    if set(keys) != PROBE_KEYS or len(set(keys)) != len(keys):
        raise ValueError('qualification requires distinct physical Q/G/P probes')
    return c, replay


def qualify(alignment_path, recording_path, replay_path, probe_paths, out_dir):
    data = read_json(alignment_path)
    robot_id = data['contract']['robot_id']
    validate_candidate(data, robot_id)
    paths = {'recording': recording_path, 'replay': replay_path}
    for trajectory, simulation, hardware in probe_paths:
        letter = load_trajectory(trajectory)['letter']
        if letter not in PROBE_KEYS or f'{letter}_trajectory' in paths:
            raise ValueError('provide each of Q, G and P exactly once')
        paths.update({f'{letter}_trajectory': trajectory, f'{letter}_simulation': simulation, f'{letter}_hardware': hardware})
    result = {'schema_version': 1, 'kind': 'so101_alignment_qualification', 'status': 'qualified',
              'candidate': data, 'scope': 'Q/G/P alignment probes and reserved measured-motion replay; not policy qualification',
              'created_at_utc': datetime.now(timezone.utc).isoformat()}
    _, result['reserved_replay'] = validate_evidence(result, paths, robot_id)
    result['evidence'] = {name: {'file': name+('.jsonl' if name.endswith('hardware') else '.json'),
                               'sha256': digest(path)} for name, path in paths.items()}
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, path in paths.items():
        shutil.copy2(path, out_dir/result['evidence'][name]['file'])
    target = out_dir/'qualification.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    load_qualified(target, robot_id)
    return result


def load_qualified(path, robot_id):
    data = read_json(path)
    if data.get('schema_version') != 1 or data.get('kind') != 'so101_alignment_qualification' or data.get('status') != 'qualified':
        raise ValueError('physical deployment needs a qualified alignment, not a diagnostic candidate')
    paths = {}
    for name, item in data['evidence'].items():
        relative = Path(item['file'])
        source = (path.parent/relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(path.parent.resolve()) or digest(source) != item['sha256']:
            raise ValueError('alignment evidence path/hash changed')
        paths[name] = source
    contract, _ = validate_evidence(data, paths, robot_id)
    return contract, data, paths


def validate_environment(contract, profile, env):
    geometry = load_geometry()
    position, rotation, points, _ = nominal_geometry(geometry['reference_position_m'], geometry['reference_rotation_xyzw'],
                                                    REFERENCE.az_xyz_b_m, contract['task_config'].get('keyboard', {}))
    state = env['scene']['keyboard']['init_state']
    if (not np.allclose(np.asarray(state['pos'], dtype=float), position, atol=1e-8, rtol=0)
            or not (np.allclose(np.asarray(state['rot'], dtype=float), rotation, atol=1e-8, rtol=0)
                    or np.allclose(np.asarray(state['rot'], dtype=float), -np.array(rotation), atol=1e-8, rtol=0))
            or not np.allclose(profile.az_xyz_b_m, points, atol=1e-8, rtol=0)
            or not np.allclose(profile.q_reset_rad, RESET, atol=1e-8, rtol=0)
            or not np.allclose(profile.joint_zero_offset_deg, contract['joint_zero_offset_deg'], atol=1e-9, rtol=0)):
        raise ValueError('qualified scene, reset or mapping differs from trained environment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare', help='freeze an unqualified mapping; no hardware access')
    p.add_argument('--robot-id', required=True)
    p.add_argument('--task-config', type=Path, required=True)
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--encoder-convention', choices=('lerobot', 'benchmark'))
    m.add_argument('--joint-calibration', type=Path)
    m.add_argument('--mapping-report', type=Path)
    p.add_argument('--mapping-candidate', choices=('fit4', 'fit3_pose', 'joint_scene'))
    p.add_argument('--validation-recording', type=Path, action='append', default=[],
                   help='reserve a new completed capture excluded from fitting; repeat for more captures')
    p.add_argument('--experimental', action='store_true',
                   help='prepare a recording-fit candidate before reserving validation; cannot qualify without reserved replay')
    p.add_argument('--out-dir', type=Path, required=True)
    p = sub.add_parser('qualify', help='verify and copy reserved replay and physical Q/G/P evidence')
    p.add_argument('--alignment', type=Path, required=True)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--replay-report', type=Path, required=True)
    p.add_argument('--probe', nargs=3, type=Path, action='append', required=True, metavar=('TRAJECTORY', 'SIM_REPORT', 'HARDWARE_LOG'))
    p.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            prepare(args.task_config, args.robot_id, args.out_dir, convention=args.encoder_convention,
                    mapping_report=args.mapping_report, mapping_candidate=args.mapping_candidate,
                    joint_fit=args.joint_calibration, validation_recordings=args.validation_recording,
                    experimental=args.experimental)
            print(f'UNQUALIFIED candidate: {args.out_dir/"alignment.json"}')
        else:
            qualify(args.alignment, args.recording, args.replay_report, args.probe, args.out_dir)
            print(f'Qualified representative alignment: {args.out_dir/"qualification.json"}')
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
