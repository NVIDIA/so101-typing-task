"""Synthetic evidence only: exercise qualification/export without serial access."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.so101_homing import scene_alignment as sa, deployment_bundle as bundle
from scripts.so101_homing import fk_probe_trajectory as trajectory
from scripts.so101_homing.so101_kinematics import JOINT_NAMES, GRIPPER_MODEL_MIN_RAD


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')
    return path


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    task = tmp_path/'task.json'
    task.write_bytes((sa.ROOT/'configs/tasks/physical-reference-20260915.json').read_bytes())
    calibration = dump(tmp_path/'robot.json', {name: {'id': i+1, 'drive_mode': 0, 'homing_offset': 0,
                              'range_min': 0, 'range_max': 4095} for i, name in enumerate(JOINT_NAMES)})
    def require(robot_id):
        if robot_id != 'test_robot':
            raise ValueError('unknown test robot')
        return calibration
    monkeypatch.setattr(sa, 'require_calibration', require)
    monkeypatch.setattr(bundle, 'require_calibration', require)
    candidate = sa.prepare(task, 'test_robot', tmp_path/'candidate', convention='lerobot')
    c = candidate['contract']
    events = [{'char': key.lower(), 'key_name': 'KEY_'+key, 'edge': edge, 'time_s': .1+i*.4}
              for i, (key, edge) in enumerate((k, e) for k in 'QGP' for e in ('down', 'up'))]
    recording = {'schema_version': 1, 'kind': 'so101_torque_off_recording', 'status': 'recorded',
                 **{k: c[k] for k in ('robot_id', 'calibration_sha256', 'task_config', 'task_config_sha256', 'source_sha256', 'robot_usd_sha256')},
                 'joint_names': list(JOINT_NAMES), 'joint_zero_offset_deg': [0]*5, 'events': events,
                 'samples': [{'time_s': i*.2, 'q_model_rad': [*trajectory.RESET, GRIPPER_MODEL_MIN_RAD],
                              'lerobot_position': [*np.rad2deg(trajectory.RESET), 0.]} for i in range(16)]}
    record_path = dump(tmp_path/'recording.json', recording)
    replay = {'status': 'replayed', 'recording_sha256': sa.digest(record_path), 'task_config': c['task_config'],
              'joint_zero_offset_deg': c['joint_zero_offset_deg'], 'key_sequence_matches': True,
              'simulation_events': [{'key': e['char'].upper(), 'edge': e['edge']} for e in events]}
    replay_path = dump(tmp_path/'scene-report.json', replay)
    probes = []
    for key in 'QGP':
        folder = tmp_path/key
        path = folder/'trajectory.json'
        plan = trajectory.generate(task, path, letter=key)
        sim_path = dump(folder/'sim-report.json', {'status': 'passed', 'trajectory_sha256': sa.digest(path),
                      'steps': len(plan['rows']), 'events': [{'edge': 'down'}, {'edge': 'up'}]})
        rows = [{'kind': 'header', 'alignment_probe_schema': 1,
                 'trajectory_sha256': sa.digest(path), 'alignment_contract_sha256': candidate['contract_sha256'],
                 'calibration_sha256': c['calibration_sha256'], 'robot_id': c['robot_id'],
                 'joint_zero_offset_deg': c['joint_zero_offset_deg']}]
        down = up = False
        for i, row in enumerate(plan['rows']):
            if row['phase'] == 'press' and not down:
                rows.append({'kind': 'key', 'char': key.lower(), 'edge': 'down', 'phase': 'press'})
                down = True
            if row['phase'] == 'released' and not up:
                rows.append({'kind': 'key', 'char': key.lower(), 'edge': 'up', 'phase': 'released'})
                up = True
            deg = np.rad2deg(row['q_rad']).tolist()
            rows.append({'kind': 'command', 'step': i, 'phase': row['phase'], 'monotonic_s': 1+i*.04,
                         'q_command_rad': row['q_rad'], 'q_command_encoder_deg': deg,
                         'q_measured_encoder_deg': deg, 'q_measured_model_deg': deg, 'gripper_percent': 0.})
        rows.append({'kind': 'result', 'status': 'passed', 'edges': ['down', 'up']})
        hardware = folder/'hardware.jsonl'
        hardware.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        probes.append((path, sim_path, hardware))
    return SimpleNamespace(root=tmp_path, calibration=calibration, task=task, candidate=candidate,
                           alignment=tmp_path/'candidate/alignment.json', recording=record_path,
                           replay=replay_path, probes=probes)


def qualify(e):
    out = e.root/'qualified'
    sa.qualify(e.alignment, e.recording, e.replay, e.probes, out)
    return out/'qualification.json'


def environment(c):
    g = sa.load_geometry()
    pos, rot, points, _ = sa.nominal_geometry(g['reference_position_m'], g['reference_rotation_xyzw'],
                                              sa.REFERENCE.az_xyz_b_m, c['task_config']['keyboard'])
    return {'sim': {'dt': .01}, 'decimation': 4, 'keyboard_profile': 'test', 'actuator_profile': 'anchorbench',
            'task_contract': 'test_scene', 'target_reference_sha256': 'test', 'target_manifest_sha256': 'test',
            'scene': {'keyboard': {'init_state': {'pos': pos, 'rot': rot}}},
            'commands': {'typing': {'q_reset_ref': trajectory.RESET.tolist(), 'letter_xyz_b_m': points,
                         'target_xyz_mean_b_m': [.2, .1, .01], 'target_xyz_std_b_m': [.01, .02, .001], 'letter_length': [6, 6]}},
            'actions': {'action': {'joint_names': list(sa.ARM_JOINT_NAMES),
                                  'velocity_limits_rad_s': list(bundle.JOINT_RATE_LIMITS_RAD_S), 'q_reset_ref': trajectory.RESET.tolist()}}}


def export(e, qualification, monkeypatch, env=None, candidate=False):
    checkpoint = e.root/'model.pt'; checkpoint.write_bytes(b'synthetic-policy-test-only')
    env_path = e.root/'env.yaml'; env_path.write_text(yaml.safe_dump(env or environment(e.candidate['contract'])))
    monkeypatch.setattr(bundle, 'FixedCartesianPolicy', lambda p: SimpleNamespace(sha256=sa.digest(p)))
    kwargs = {'alignment_candidate_path' if candidate else 'alignment_path': qualification}
    return bundle.prepare(checkpoint, env_path, 'test_robot', 'auto', e.root/'bundle', **kwargs)


def test_candidate_bundle_preserves_offsets_and_requires_explicit_experimental_mode(evidence, monkeypatch):
    e = evidence
    offsets = [3.12651, .74869, -9.61276, .54019, 0.]
    e.candidate['contract']['joint_zero_offset_deg'] = offsets
    e.candidate['contract']['mapping'] = {'kind': 'recording_fit4', 'fit_recording_sha256': 'fit',
                                         'reserved_recording_sha256': [sa.digest(e.recording)]}
    e.candidate['contract_sha256'] = sa.fingerprint(e.candidate['contract'])
    dump(e.alignment, e.candidate)
    manifest = export(e, e.alignment, monkeypatch, candidate=True)
    root = e.root/'bundle'
    profile, data = bundle.validate_manifest(root/'deployment.json', root/'checkpoint.pt', root/'env.yaml', 'test_robot')
    assert list(profile.joint_zero_offset_deg) == offsets
    assert manifest['alignment_qualified'] is False
    assert data['mapping_selection'] == 'unqualified_alignment'
    with pytest.raises(ValueError, match='requires a qualified'):
        bundle.require_qualified_alignment(data)
    assert bundle.validate_execution_mode(data, experimental=True) == 'experimental_unqualified_mapping'
    assert data['alignment_qualified'] is False
    e.calibration.write_text(e.calibration.read_text()+'\n')
    with pytest.raises(ValueError, match='calibration changed'):
        bundle.validate_manifest(root/'deployment.json', root/'checkpoint.pt', root/'env.yaml', 'test_robot')


def test_qualified_evidence_is_portable_and_revalidated_on_export(evidence, monkeypatch):
    e = evidence
    q = qualify(e)
    manifest = export(e, q, monkeypatch)
    assert manifest['alignment_qualified'] is True
    assert manifest['hardware_validated'] is False  # No policy typing claim.
    shutil.rmtree(q.parent)
    for path in (e.recording, e.replay, *(p for group in e.probes for p in group)):
        path.unlink()
    root = e.root/'bundle'
    _, data = bundle.validate_manifest(root/'deployment.json', root/'checkpoint.pt', root/'env.yaml', 'test_robot')
    bundle.require_qualified_alignment(data)
    (root/'alignment/Q_hardware.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='evidence path/hash'):
        bundle.validate_manifest(root/'deployment.json', root/'checkpoint.pt', root/'env.yaml', 'test_robot')


@pytest.mark.parametrize('mutation', ['wrong_key', 'replay_failed', 'jaw_open', 'offset_changed', 'missing_probe', 'wrong_trajectory', 'tracking', 'gap', 'missing_commands', 'false_pass_flag', 'no_clearance'])
def test_bad_evidence_cannot_create_qualification(evidence, mutation):
    e = evidence
    if mutation in ('wrong_key', 'replay_failed'):
        d = sa.read_json(e.replay)
        if mutation == 'wrong_key': d['simulation_events'][0]['key'] = 'N'
        else: d['status'] = 'failed'
        dump(e.replay, d)
    elif mutation == 'missing_probe':
        e.probes.pop()
    else:
        path = e.probes[0][2]
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        commands = [r for r in rows if r.get('kind') == 'command']
        if mutation == 'jaw_open': commands[0]['gripper_percent'] = 15
        if mutation == 'offset_changed': rows[0]['joint_zero_offset_deg'][2] = -10
        if mutation == 'wrong_trajectory': rows[0]['trajectory_sha256'] = 'wrong'
        if mutation == 'tracking': commands[30]['q_measured_encoder_deg'][2] += 14
        if mutation == 'gap': commands[30]['monotonic_s'] += .2
        if mutation == 'missing_commands': rows.remove(commands[30])
        if mutation == 'false_pass_flag': rows = [rows[0], rows[-1]]
        if mutation == 'no_clearance':
            # Release only at the end leaves no two subsequent clearance ticks.
            up = next(r for r in rows if r.get('kind') == 'key' and r['edge'] == 'up')
            rows.remove(up); rows.insert(-1, up)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises((ValueError, KeyError)):
        qualify(e)
    assert not (e.root/'qualified').exists()


def test_nonzero_fit_mapping_remains_consistent_through_probe_and_export(evidence, monkeypatch):
    e = evidence
    offsets = [3., .7, -9.6, .5, 0.]
    c = e.candidate['contract']
    c['joint_zero_offset_deg'] = offsets
    c['mapping'] = {'kind': 'recording_fit4', 'source_sha256': 'synthetic-test-only',
                    'fit_recording_sha256': 'different-fit', 'reserved_recording_sha256': [sa.digest(e.recording)]}
    e.candidate['contract_sha256'] = sa.fingerprint(c)
    dump(e.alignment, e.candidate)
    replay = sa.read_json(e.replay); replay['joint_zero_offset_deg'] = offsets; dump(e.replay, replay)
    for _, _, hardware in e.probes:
        rows = [json.loads(line) for line in hardware.read_text().splitlines()]
        rows[0]['alignment_contract_sha256'] = e.candidate['contract_sha256']
        rows[0]['joint_zero_offset_deg'] = offsets
        for row in rows:
            if row.get('kind') == 'command':
                row['q_command_encoder_deg'] = (np.array(row['q_command_encoder_deg'])-offsets).tolist()
                row['q_measured_encoder_deg'] = (np.array(row['q_measured_encoder_deg'])-offsets).tolist()
        hardware.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    q = qualify(e)
    manifest = export(e, q, monkeypatch)
    assert manifest['profile']['joint_zero_offset_deg'] == tuple(offsets)
    rest = sa.read_json(e.root/'bundle/rest_pose.json')
    np.testing.assert_allclose([rest['pose_deg'][j] for j in sa.ARM_JOINT_NAMES], np.rad2deg(trajectory.RESET)-offsets)


@pytest.mark.parametrize('changed', ['calibration', 'layout', 'reset', 'draft'])
def test_export_rejects_changed_identity_before_creating_bundle(evidence, monkeypatch, changed):
    e = evidence
    q = qualify(e)
    env = environment(e.candidate['contract'])
    if changed == 'calibration': e.calibration.write_text(e.calibration.read_text()+'\n')
    elif changed == 'layout': env['scene']['keyboard']['init_state']['pos'][0] += .001
    elif changed == 'reset': env['commands']['typing']['q_reset_ref'][0] += .01; env['actions']['action']['q_reset_ref'][0] += .01
    elif changed == 'draft': q = e.alignment
    with pytest.raises(ValueError):
        export(e, q, monkeypatch, env)
    assert not (e.root/'bundle').exists()


def test_fit4_requires_separate_reserved_replay(evidence):
    e = evidence; c = e.candidate['contract']
    report = {'status': 'diagnostic_only_not_approved_for_deployment', 'task_config': c['task_config'],
              'joint_order': c['joint_order'], 'identities': {k: c[k] for k in ('robot_id', 'calibration_sha256', 'robot_usd_sha256', 'source_sha256')},
              'recordings': [{'role': 'fit', 'sha256': sa.digest(e.recording)}, {'role': 'validation', 'sha256': 'reserved-other'}],
              'candidates': {'fit4': {'offset_deg': [3., .7, -9.6, .5, 0.]}}}
    source = dump(e.root/'fit4.json', report)
    data = sa.prepare(e.task, 'test_robot', e.root/'fit-candidate', mapping_report=source, mapping_candidate='fit4')
    assert data['status'] == 'unqualified'
    with pytest.raises(ValueError, match='reserved validation'):
        sa.check_replay(data['contract'], e.recording, e.replay)
    report['recordings'][1]['sha256'] = report['recordings'][0]['sha256']
    dump(source, report)
    with pytest.raises(ValueError, match='distinct fit'):
        sa.prepare(e.task, 'test_robot', e.root/'bad-candidate', mapping_report=source, mapping_candidate='fit4')


def test_one_candidate_routes_identically_to_measured_pose_replay(evidence, monkeypatch):
    from scripts.so101_homing import scene_check, run_fk_probe
    e = evidence
    captured = {}
    def simulate(path, **kwargs):
        captured.update(path=path, **kwargs)
        return 0
    monkeypatch.setattr(run_fk_probe, 'simulate', simulate)
    monkeypatch.setattr('sys.argv', ['scene_check', 'replay', '--recording', str(e.recording), '--alignment', str(e.alignment), '--headless'])
    assert scene_check.main() == 0
    assert captured['headless'] is True
    assert captured['read_only_mounts'] == [(e.alignment.resolve(), '/alignment.json')]
    record = sa.read_json(e.recording)
    offsets, identity = scene_check.load_alignment_candidate(record, e.alignment, record['task_config'])
    assert offsets == e.candidate['contract']['joint_zero_offset_deg']
    assert identity['alignment_contract_sha256'] == e.candidate['contract_sha256']
    record['calibration_sha256'] = 'different'
    with pytest.raises(ValueError, match='calibration_sha256'):
        scene_check.load_alignment_candidate(record, e.alignment, record['task_config'])


def test_unqualified_execute_does_not_launch_runner(monkeypatch, tmp_path):
    from scripts.so101_homing import deploy_bundle
    dump(tmp_path/'deployment.json', {'robot_id': 'test_robot'})
    monkeypatch.setattr(deploy_bundle, 'validate_manifest', lambda *a: (None, {'robot_id': 'test_robot', 'encoder_convention': 'lerobot', 'actuator_profile': 'anchorbench', 'alignment_qualified': True}))
    monkeypatch.setattr(deploy_bundle, 'require_calibration', lambda *_: None)
    monkeypatch.setattr(deploy_bundle.subprocess, 'call', lambda *_: pytest.fail('must not launch hardware runner'))
    monkeypatch.setattr('sys.argv', ['deploy_bundle', str(tmp_path), 'NVIDIA', '--execute', '--port', '/dev/test', '--keyboard-device', '/dev/test-keys'])
    with pytest.raises(SystemExit) as error:
        deploy_bundle.main()
    assert error.value.code == 2


@pytest.mark.parametrize('execute', [False, True])
def test_experimental_candidate_mode_forwards_exact_target_and_motion_opt_in(evidence, monkeypatch, execute):
    from scripts.so101_homing import deploy_bundle
    e = evidence
    export(e, e.alignment, monkeypatch, candidate=True)
    monkeypatch.setattr(deploy_bundle, 'require_calibration', lambda *_: e.calibration)
    commands = []
    monkeypatch.setattr(deploy_bundle.subprocess, 'call', lambda command: commands.append(command) or 0)
    argv = ['deploy_bundle', str(e.root/'bundle'), 'NEWTON', '--experimental']
    if execute:
        argv += ['--execute', '--port', '/dev/test', '--keyboard-device', '/dev/test-keys']
    monkeypatch.setattr('sys.argv', argv)
    assert deploy_bundle.main() == 0
    command = commands[0]
    assert command[command.index('--target')+1] == 'NEWTON'
    assert '--experimental' in command
    assert ('--enable-robot' in command) is execute
    assert ('--dry-run' in command) is not execute
    # Experimental mode still rejects changed calibration before launching a runner.
    e.calibration.write_text(e.calibration.read_text()+'\n')
    with pytest.raises(SystemExit):
        deploy_bundle.main()
    assert len(commands) == 1


@pytest.mark.parametrize('manifest', [
    {'encoder_convention': 'benchmark'},
    {'encoder_convention': 'candidate', 'alignment_candidate_sha256': 'hash'},
])
def test_experimental_execution_requires_a_calibration_pinned_candidate(manifest):
    with pytest.raises(ValueError, match='pinned calibration'):
        bundle.validate_execution_mode(manifest, experimental=True)
