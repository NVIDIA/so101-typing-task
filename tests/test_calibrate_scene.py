"""Scene fitting recovers geometry independently of validation and keeps repeats."""
import copy

import numpy as np
import pytest

from scripts.so101_homing import calibrate_scene as cs


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
