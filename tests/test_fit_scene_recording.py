"""Planar fitting, fixed height/tilt and held-out independence."""
import numpy as np
import pytest

from so101_typing.hardware import fit_scene_recording as fit


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
