"""A fitted effective scene preserves and checks the original recording task."""
import copy
import json

import pytest

from test_scene_alignment import evidence, dump
from scripts.so101_homing import scene_alignment as sa


@pytest.mark.parametrize('candidate_name,offsets', [('fit3_pose', [0., .8, -10., 1., 0.]),
                                                   ('joint_scene', [2., .8, -10., 1., -3.])])
def test_fitted_scene_qualifies_only_its_reserved_source_scene(evidence, candidate_name, offsets):
    e = evidence
    c = e.candidate['contract']
    fitted_task = copy.deepcopy(c['task_config'])
    fitted_task['keyboard']['position_m'][0] += .002
    task_path = dump(e.root/'fitted-task.json', fitted_task)
    report = {'status': 'diagnostic_only_not_approved_for_deployment',
              'method': 'event_associated_joint_and_rigid_scene_fit',
              'task_config': fitted_task, 'reference_task': c['task_config'],
              'joint_order': c['joint_order'],
              'identities': {k: c[k] for k in ('robot_id', 'calibration_sha256', 'robot_usd_sha256', 'source_sha256')},
              'recordings': [{'role': 'fit', 'sha256': 'different-fit'},
                             {'role': 'validation', 'sha256': sa.digest(e.recording)}],
              'candidates': {candidate_name: {'offset_deg': offsets}}}
    source = dump(e.root/'fit3.json', report)
    data = sa.prepare(task_path, 'test_robot', e.root/'scene-candidate',
                      mapping_report=source, mapping_candidate=candidate_name)
    replay = json.loads(e.replay.read_text())
    replay.update(task_config=fitted_task, joint_zero_offset_deg=offsets)
    dump(e.replay, replay)
    assert sa.check_replay(data['contract'], e.recording, e.replay)['presses'] == 3
    changed = copy.deepcopy(data['contract'])
    changed['mapping']['reference_task'] = fitted_task
    with pytest.raises(ValueError, match='task_config'):
        sa.check_replay(changed, e.recording, e.replay)
    data['contract']['mapping']['reserved_recording_sha256'] = ['different']
    with pytest.raises(ValueError, match='reserved validation'):
        sa.check_replay(data['contract'], e.recording, e.replay)


@pytest.mark.parametrize('invalid', [None, 'fit_overlap', 'robot', 'scene', 'incomplete', 'keys', 'jaw'])
def test_reserve_new_capture_after_contact_fitting(evidence, invalid):
    e = evidence
    c = e.candidate['contract']
    report = {'status': 'diagnostic_only_not_approved_for_deployment',
              'method': 'contact_refined_joint_and_rigid_scene_fit',
              'task_config': c['task_config'], 'reference_task': c['task_config'],
              'joint_order': c['joint_order'],
              'identities': {k: c[k] for k in ('robot_id', 'calibration_sha256', 'robot_usd_sha256', 'source_sha256')},
              'recordings': [{'role': 'fit', 'sha256': 'fitting-capture'}],
              'candidates': {'joint_scene': {'offset_deg': [0.] * 5}}}
    recording = json.loads(e.recording.read_text())
    if invalid == 'robot':
        recording['robot_id'] = 'different_robot'
    elif invalid == 'scene':
        recording['task_config']['keyboard']['position_m'][0] += .01
    elif invalid == 'incomplete':
        recording['status'] = 'interrupted'
    elif invalid == 'keys':
        recording['events'] = recording['events'][:2]
    elif invalid == 'jaw':
        recording['samples'][0]['lerobot_position'][5] = 10.
    dump(e.recording, recording)
    if invalid == 'fit_overlap':
        report['recordings'][0]['sha256'] = sa.digest(e.recording)
    source = dump(e.root/'all-fit.json', report)
    original = source.read_bytes()
    out = e.root/'new-candidate'
    kwargs = dict(mapping_report=source, mapping_candidate='joint_scene', validation_recordings=[e.recording])
    if invalid:
        with pytest.raises(ValueError):
            sa.prepare(e.task, 'test_robot', out, **kwargs)
        assert not out.exists()
    else:
        data = sa.prepare(e.task, 'test_robot', out, **kwargs)
        assert data['status'] == 'unqualified'
        assert data['contract']['mapping']['reserved_recording_sha256'] == [sa.digest(e.recording)]
        assert (out/'validation-1.json').read_bytes() == e.recording.read_bytes()
        assert sa.check_replay(data['contract'], e.recording, e.replay)['presses'] == 3
        assert source.read_bytes() == original
