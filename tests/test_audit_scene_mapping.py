"""Diagnostic recovery, averaging, identity checks and held-out independence."""
import copy
import json

import numpy as np
import pytest

from so101_typing.hardware import audit_scene_mapping as audit
from so101_typing.hardware.so101_kinematics import site_position, FIXED_JAW_TYPING_TIP_SITE_POS


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
