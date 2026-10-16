# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Recorded-motion scene fitting, joint mapping and replay validation."""

from copy import deepcopy
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytest

from so101_typing.hardware import (
    audit_scene_mapping as audit,
    calibrate_scene as cs,
    fit_scene_recording as fit,
    real2sim as workflow,
)
from so101_typing.hardware.constants import ARM_JOINT_NAMES
from so101_typing.hardware.joint_calibration import (
    fit as fit_joint_mapping,
    load as load_joint_mapping,
)
from so101_typing.hardware.so101_kinematics import FIXED_JAW_TYPING_TIP_SITE_POS, site_position


# Mapping audit and held-out evidence

def synthetic_presses(offsets, seed=4, count=12):
    rng = np.random.default_rng(seed)
    samples = [rng.uniform([-.7, -.6, -.4, .2, -.8, 0], [.7, .5, .8, 1.3, .8, 1.], (5, 6))
               for _ in range(count)]
    delta = np.r_[np.deg2rad(offsets), 0.]
    rows = [{"key": chr(65+i), "pressed_key_xyz_m": audit.batch_fk(q+delta).mean(0).tolist()}
            for i, q in enumerate(samples)]
    return audit.Presses(rows, samples)


def test_batched_fk_matches_canonical_scalar_for_fixed_jaw():
    q = np.random.default_rng(33).uniform(-1.5, 1.5, (40, 6))
    expected = [site_position(row, site_pos=FIXED_JAW_TYPING_TIP_SITE_POS) for row in q]
    np.testing.assert_allclose(audit.batch_fk(q), expected, atol=2e-15)
    q[:, 5] += .5
    np.testing.assert_allclose(audit.batch_fk(q), expected, atol=2e-15)


def test_average_of_fk_samples_is_used_instead_of_fk_at_mean_angle():
    presses = synthetic_presses([0]*5, count=1)
    np.testing.assert_allclose(presses.predict([0]*5), presses.targets, atol=1e-15)
    assert np.linalg.norm(audit.batch_fk(presses.q.mean(0)[None])-presses.targets) > .001


@pytest.mark.parametrize("family,offsets", [
    ("fit3", [0., 1.9, -10.7, 1.4, 0.]),
    ("fit4", [2.1, 1.9, -10.7, 1.4, 0.]),
    ("fit5_diagnostic", [2.1, 1.9, -10.7, 1.4, 3.2]),
])
def test_recovers_known_offsets_and_predicts_held_out_poses(family, offsets):
    fit = synthetic_presses(offsets)
    held_out = synthetic_presses(offsets, seed=44, count=8)
    result = audit.fit_candidates(fit)[family]
    np.testing.assert_allclose(result["offset_deg"], offsets, atol=1e-6)
    assert audit.score(held_out, result["offset_deg"])["max_mm"] < 1e-6
    assert not result["boundary_hits"]
    assert all(start["converged"] for start in result["multistarts"])
    assert result["jacobian_condition_number"] is not None


@pytest.mark.parametrize("field", audit.IDENTITY_FIELDS)
def test_rejects_mismatched_identity(field):
    first = {"robot_id": "unit", "calibration_sha256": "calibration", "robot_usd_sha256": "asset",
             "source_sha256": {"a.py": "hash"}}
    second = copy.deepcopy(first)
    second[field] = "different"
    with pytest.raises(ValueError, match=field):
        audit.validate_identities([first, second], {"robot_usd_sha256": "asset"})


def test_validation_cannot_change_fit_and_frozen_task_is_preserved(tmp_path, monkeypatch):
    task = tmp_path/"frozen.json"
    task.write_text(json.dumps({"schema_version": 1, "keyboard": {"position_m": [.2, .01, .007]}}))
    original_bytes = task.read_bytes()
    fit_path, validation_path = tmp_path/"fit.json", tmp_path/"validation.json"
    fit_path.write_text("fit")
    validation_path.write_text("validation")
    fit = synthetic_presses([0., 1.9, -10.7, 1.4, 0.])
    held_out = synthetic_presses([0., 1.9, -10.7, 1.4, 0.], seed=100)
    metadata = {"robot_id": "unit", "calibration_sha256": "cal", "robot_usd_sha256": "asset",
                "source_sha256": {"a.py": "hash"}, "task_config": {}, "task_config_sha256": "old-task"}
    monkeypatch.setattr(audit, "load_geometry", lambda: {"robot_usd_sha256": "asset"})
    monkeypatch.setattr(audit, "load_recording", lambda p: {**metadata, "label": p.stem})
    monkeypatch.setattr(audit, "recording_presses", lambda r, *_: fit if r["label"] == "fit" else held_out)
    before = audit.audit(task, fit_path, [validation_path], tmp_path/"before")
    held_out.targets += [.10, -.04, .03]
    after = audit.audit(task, fit_path, [validation_path], tmp_path/"after")
    for name in audit.FAMILIES:
        assert before["candidates"][name]["offset_deg"] == after["candidates"][name]["offset_deg"]
        assert after["candidates"][name]["validation"][0]["mean_mm"] > 100
    assert task.read_bytes() == original_bytes
    assert set(p.name for p in (tmp_path/"after").iterdir()) == {"mapping-report.json"}
    assert after["selected_candidate"] is None
    with pytest.raises(ValueError, match="overwrite"):
        audit.audit(task, fit_path, [validation_path], tmp_path/"after")
    with pytest.raises(ValueError, match="duplicate content"):
        audit.audit(task, fit_path, [fit_path], tmp_path/"duplicate")


def test_sampling_selection_and_frozen_scene_override_do_not_mutate_input(monkeypatch):
    q = [[.1+i*.01, -.2, .3, 1., .05, 0.] for i in range(5)]
    recording = {"task_config": {"keyboard": {"position_m": [99., 0, 0]}},
                 "samples": [{"time_s": t, "q_model_rad": row} for t, row in zip([0., .02, .04, .06, .08], q)]}
    original = copy.deepcopy(recording)
    frozen = {"keyboard": {"position_m": [.2, .01, .007]}}
    def contact_rows(r, geometry):
        assert r["task_config"] == frozen
        return [{"key": "F", "down_s": .025, "up_s": .08, "samples": 3,
                 "tip_xyz_m": [0.]*3, "tip_span_mm": [0.]*3, "pressed_key_xyz_m": [.2, .1, .01]}]
    monkeypatch.setattr(audit, "contacts", contact_rows)
    interval = audit.recording_presses(recording, {}, frozen)
    prior = audit.recording_presses(recording, {}, frozen, "pre_down")
    np.testing.assert_array_equal(interval.q, q[2:])
    np.testing.assert_array_equal(prior.q, q[1:2])
    assert recording == original


# Joint and scene optimization

@pytest.mark.parametrize('pan,roll', [(False, False), (True, False), (False, True), (True, True)])
def test_recovers_new_offsets_and_pose_on_independent_samples(pan, roll):
    rng = np.random.default_rng(52)
    q = rng.uniform([-.7, -.6, -.4, .2, -.8, 0], [.7, .5, .8, 1.3, .8, 1.], (52, 6))
    expected = np.array([2., -9., 1., 3., -7., 2., 4.]+([3.] if roll else []))
    offsets, rotation, translation = cs.unpack(expected, fit_pan=pan, fit_roll=roll)
    pivot = np.array([.26, .01, .007])
    tips = cs.batch_fk(q+np.r_[np.deg2rad(offsets), 0.])
    targets = rotation.inv().apply(tips-pivot-translation)+pivot
    fitted, info = cs.fit(q[:26], targets[:26], list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), pivot, fit_pan=pan, fit_roll=roll)
    np.testing.assert_allclose(fitted, expected, atol=.03)
    assert np.max(np.linalg.norm(cs.residual(fitted, q[26:], targets[26:], pivot, fit_pan=pan, fit_roll=roll), axis=1)) < .02
    assert not info['boundary_indices']


def test_keeps_repeated_short_presses_and_nearest_encoder_sample():
    recording = {'samples': [{'time_s': t, 'q_model_rad': [t]*6} for t in [0., .04, .08, .12]],
                 'events': [{'char': 'q', 'code': 16, 'edge': edge, 'time_s': t}
                            for edge, t in [('down', .03), ('up', .045), ('down', .075), ('up', .10)]]}
    original = copy.deepcopy(recording)
    rows = cs.key_down_samples(recording)
    assert [r['sample_index'] for r in rows] == [1, 2]
    assert [r['key'] for r in rows] == ['Q', 'Q']
    assert recording == original
    recording['events'][2]['time_s'] = .5
    with pytest.raises(ValueError, match='nearest encoder'):
        cs.key_down_samples(recording)


def test_overlapping_contact_is_not_silently_assigned():
    recording = {'samples': [{'time_s': .01, 'q_model_rad': [0]*6}],
                 'events': [{'char': key, 'code': code, 'edge': 'down', 'time_s': .01}
                            for key, code in [('q', 16), ('w', 17)]]}
    with pytest.raises(ValueError, match='overlapping'):
        cs.key_down_samples(recording)


def test_all_five_offsets_and_full_pose_predict_new_samples_despite_gauge():
    rng = np.random.default_rng(52)
    q = rng.uniform(-.7, .7, (60, 6))
    expected = np.array([2., -9., 1., 3., -7., 2., 4., 3., 5., .2, -.3])
    offsets, rotation, translation = cs.unpack(expected, True, True, True, True)
    pivot = np.array([.26, .01, .007])
    tips = cs.batch_fk(q+np.r_[np.deg2rad(offsets), 0.])
    targets = rotation.inv().apply(tips-pivot-translation)+pivot
    fitted, info = cs.fit(q[:30], targets[:30], [str(i) for i in range(30)], pivot,
                          tilt=True, fit_pan=True, fit_roll=True, free_yaw=True)
    errors = cs.residual(fitted, q[30:], targets[30:], pivot, True, True, True, True)
    assert np.max(np.linalg.norm(errors, axis=1)) < .1
    singular = info['jacobian_singular_values']
    assert singular[-1]/singular[0] < 1e-5


# Planar scene fitting

def test_recovers_xy_yaw_without_using_target_height():
    source = np.array([[0.,0,0],[.1,0,0],[0,.2,0],[.1,.2,0]])
    expected = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    shift = np.array([.2, -.1, .03])
    rotation, translation = fit.planar_fit(source, source@expected.T+shift)
    np.testing.assert_allclose(rotation, expected, atol=1e-12)
    np.testing.assert_allclose(translation, [.2, -.1, 0], atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.)


def test_rejects_collinear_contact_points():
    points = np.array([[0.,0,0],[1,0,0],[2,0,0]])
    with pytest.raises(ValueError, match="collinear"):
        fit.planar_fit(points, points)


def test_arbitrary_target_z_cannot_change_fit_or_source_tilt():
    source = np.array([[0.,0,.01],[.1,0,.02],[0,.2,.03],[.1,.2,.04]])
    target = source + [.03, -.02, 0]
    before = fit.planar_fit(source, target)
    target[:, 2] = [10., -50., 20., 80.]
    after = fit.planar_fit(source, target)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose((source@after[0].T+after[1])[:, 2], source[:, 2], atol=1e-15)


def test_held_out_observations_cannot_change_training_fit(tmp_path, monkeypatch):
    recording = {"task_config": {"schema_version":1,"keyboard":{}}, "robot_id":"test",
                 "calibration_sha256":"test", "samples":[]}
    geometry = {"reference_position_m":[0,0,0], "reference_rotation_xyzw":[0,0,0,1],
                "case_hull_m":[[0,0,0]], "datum_m":[0,0,0]}
    rows = []
    for i, key in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        point = np.array([i%9*.02, i//9*.02, 0.])
        rows.append({"key":key, "pressed_key_xyz_m":point.tolist(),
                     "tip_xyz_m":(point+[.1,.2,.03]).tolist()})
    path = tmp_path/"recording.json"
    path.write_text("{}")
    monkeypatch.setattr(fit, "load_recording", lambda p: recording)
    monkeypatch.setattr(fit, "load_geometry", lambda: geometry)
    monkeypatch.setattr(fit, "contacts", lambda r,g,**kw: rows)
    before = fit.fit_recording(path, tmp_path/"before")
    for row in rows:
        if row["key"] in fit.HELD_OUT:
            row["tip_xyz_m"][0] += .2
    after = fit.fit_recording(path, tmp_path/"after")
    assert before["task_fit18"] == after["task_fit18"]
    assert before["held_out"]["mean_xy_mm"] < 1e-9
    assert before["held_out"]["mean_signed_z_mm"] == pytest.approx(-30.)
    assert after["held_out"]["mean_xy_mm"] == pytest.approx(200.)
    assert set(after["fit_keys"]).isdisjoint(after["held_out_keys"])


def test_offset_contact_fk_matches_replay_without_mutating_recording():
    import copy
    from so101_typing.hardware.scene_check import replay_joint_positions
    from so101_typing.hardware.so101_kinematics import site_position, FIXED_JAW_TYPING_TIP_SITE_POS
    recording = {"task_config": {},
                 "samples": [{"time_s": t, "q_model_rad": [.1,-.2,.3,1.,.05,0.]}
                             for t in (0., .02, .04, .06)],
                 "events": [{"char":"f", "code":33, "edge":"down", "time_s":0.},
                            {"char":"f", "code":33, "edge":"up", "time_s":.06}]}
    before = copy.deepcopy(recording)
    geometry = fit.load_geometry()
    direct = fit.contacts(recording, geometry)
    offset = fit.contacts(recording, geometry, benchmark_offsets=True)
    q, _ = replay_joint_positions(recording, True)
    expected = site_position(q[0], site_pos=FIXED_JAW_TYPING_TIP_SITE_POS)
    np.testing.assert_allclose(offset[0]['tip_xyz_m'], expected, atol=1e-12)
    assert not np.allclose(direct[0]['tip_xyz_m'], offset[0]['tip_xyz_m'])
    assert direct[0]['pressed_key_xyz_m'] == offset[0]['pressed_key_xyz_m']
    assert recording == before


# Joint-zero calibration

class JointCalibrationTests(unittest.TestCase):
    def measurements(self):
        return {'robot_id': 'test_robot', 'arm_joint_names': list(ARM_JOINT_NAMES),
                'samples': [{'model_deg': [angle+offset for offset in [1,2,3,4,5]],
                             'encoder_deg': [angle]*5, 'measurement_reference': f'test-jig-{angle}'}
                            for angle in [-10, 0, 10]]}

    def test_offset_direction_and_identity(self):
        result = fit_joint_mapping(self.measurements())
        self.assertEqual(result['joint_zero_offset_deg'], [1,2,3,4,5])
        self.assertFalse(result['hardware_geometry_validated'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calibration.json'
            path.write_text(json.dumps(result))
            self.assertEqual(load_joint_mapping(path, 'test_robot')[0], (1,2,3,4,5))
            with self.assertRaises(ValueError):
                load_joint_mapping(path, 'different_robot')
            result['joint_zero_offset_deg'][0] += 1
            path.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                load_joint_mapping(path, 'test_robot')

    def test_nonconstant_mapping_and_missing_evidence_rejected(self):
        for bad in ['angle', 'evidence']:
            data = deepcopy(self.measurements())
            if bad == 'angle':
                data['samples'][0]['model_deg'][0] += 5
            else:
                del data['samples'][0]['measurement_reference']
            with self.assertRaises(ValueError):
                fit_joint_mapping(data)

    def test_malformed_measurements_and_artifacts_raise_readable_errors(self):
        for data in (None, [], {**self.measurements(), 'samples': None},
                     {**self.measurements(), 'samples': [None]*3}):
            with self.assertRaises(ValueError):
                fit_joint_mapping(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calibration.json'
            for data in (None, []):
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'JSON object'):
                    load_joint_mapping(path, 'test_robot')


# Record, fit and replay workflow

@pytest.mark.parametrize('change', ['wrong_key', 'multiple', 'missing', 'empty', 'wrong_count'])
def test_per_key_check_requires_actual_matching_contacts(change):
    result = {'contacts': [{'key': 'Q'}, {'key': 'G'}],
              'best': {'total': 2, 'correct_first_contacts': 2, 'first_keys': ['Q', 'G']}}
    assert workflow.contact_pass(result)
    if change == 'wrong_key': result['best']['first_keys'][1] = 'H'
    if change == 'multiple': result['best']['first_keys'][1] = 'MULTIPLE'
    if change == 'missing': result['best']['first_keys'].pop()
    if change == 'empty': result = {'contacts': [], 'best': {'total': 0, 'correct_first_contacts': 0, 'first_keys': []}}
    if change == 'wrong_count': result['best']['correct_first_contacts'] = 1
    assert not workflow.contact_pass(result)


@pytest.mark.parametrize('passed', [False, True])
def test_preserves_reference_and_only_prints_training_on_pass(tmp_path, monkeypatch, capsys, passed):
    paths = [tmp_path/'capture-a.json', tmp_path/'capture-b.json']
    for i, path in enumerate(paths): path.write_text(str(i))
    records = [{'kind': 'so101_torque_off_recording', 'task_config': {'schema_version': 1}}]*2
    original = copy.deepcopy(records)
    monkeypatch.setattr(workflow, 'calibrate', lambda *a, **k: None)
    def refinement(command, **kwargs):
        assert command[1:3] == ['-m', 'so101_typing.hardware.refine_scene_contacts']
        folder = Path(command[command.index('--out-dir')+1]); folder.mkdir()
        result = {'contacts': [{'key': 'Q'}], 'best': {'total': 1,
                  'correct_first_contacts': int(passed), 'first_keys': ['Q' if passed else 'P']}}
        for name, content in [('contact-result.json', result), ('task.json', records[0]['task_config']), ('mapping-report.json', {'offsets': 'synthetic'})]:
            (folder/name).write_text(json.dumps(content))
    monkeypatch.setattr(workflow.subprocess, 'run', refinement)
    out = tmp_path/'scene'
    assert workflow.run(paths, records, out, 2) == (0 if passed else 1)
    text = capsys.readouterr().out
    assert ('./so101 train' in text) == passed
    status = json.loads((out/'scene-calibration.json').read_text())
    assert status['all_recordings_used_for_fitting']
    assert not status['physical_deployment_validated']
    assert status['task_sha256'] == workflow.digest(out/'task.json')
    assert records == original
    with pytest.raises(ValueError, match='overwrite'):
        workflow.run(paths, records, out, 2)
