"""Manual capture must never send joint goals or enable torque."""
import io
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from so101_typing.hardware import scene_check
from so101_typing.hardware.so101_kinematics import JOINT_NAMES


class ReadBus:
    # Intentionally no write-goal or enable-torque methods.
    disabled = False
    reads = 0

    def disable_torque(self, motors, num_retry):
        assert motors == list(JOINT_NAMES)
        self.disabled = True

    def sync_read(self, register, motors, normalize, num_retry):
        assert self.disabled
        if register == "Torque_Enable":
            assert normalize is False
            return dict.fromkeys(JOINT_NAMES, 0)
        assert register == "Present_Position" and normalize is True
        self.reads += 1
        return dict(zip(JOINT_NAMES, [10, 20, 30, 40, 50, 0]))


def test_capture_only_disables_and_reads(monkeypatch):
    bus = ReadBus()
    monkeypatch.setattr(scene_check.time, "sleep", lambda _: None)
    keyboard = SimpleNamespace(poll=lambda: ([{"char": "f", "key_name": "KEY_F", "edge": "down",
        "received_monotonic_s": time.monotonic()}] if bus.reads == 2 else [], bus.reads == 3))
    stream = io.StringIO()
    rows, events, reason = scene_check.capture(SimpleNamespace(bus=bus), keyboard, seconds=2, hz=50, stream=stream)
    assert len(rows) == 3 and reason == "esc"
    assert len(events) == 1 and events[0]["char"] == "f"
    np.testing.assert_allclose(rows[0]["q_model_rad"], np.deg2rad([10,20,30,40,50,-10]))
    assert len(stream.getvalue().splitlines()) == 4


def test_refuses_capture_if_any_motor_has_torque(monkeypatch):
    bus = ReadBus()
    monkeypatch.setattr(bus, "sync_read", lambda *a, **kw: dict.fromkeys(JOINT_NAMES, 1))
    with pytest.raises(RuntimeError, match="torque off"):
        scene_check.capture(SimpleNamespace(bus=bus), None, seconds=2, hz=50, stream=io.StringIO())
    assert bus.reads == 0


@pytest.mark.parametrize("times,q,match", [
    ([0, 0], [[0]*6]*2, "increasing"),
    ([0, .5], [[0]*6]*2, "sampling gap"),
    ([0, .02], [[0]*6, [float('nan')]*6], "finite"),
])
def test_replay_rejects_invalid_samples(tmp_path, times, q, match):
    data = {"kind": "so101_torque_off_recording", "schema_version": 1, "joint_names": list(JOINT_NAMES),
            "joint_zero_offset_deg": [0]*5, "source_sha256": {},
            "samples": [{"time_s": t, "q_model_rad": p} for t,p in zip(times,q)]}
    path = tmp_path/"recording.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        scene_check.load_recording(path)


def test_benchmark_offsets_apply_on_read_once_without_changing_recording():
    q = np.array([[.1, -.2, .3, -.4, .5, -.17], [.2, -.3, .4, -.5, .6, -.16]])
    data = {"samples": [{"q_model_rad": row.tolist()} for row in q]}
    before = json.dumps(data)
    replay, offsets = scene_check.replay_joint_positions(data, True)
    np.testing.assert_allclose(np.rad2deg(replay[:, :5]-q[:, :5]),
                               [[0, 1.9336272481, -10.7735791533, 1.3647648917, 0]]*2)
    np.testing.assert_array_equal(replay[:, 5], q[:, 5])
    assert json.dumps(data) == before
    direct, zero = scene_check.replay_joint_positions(data)
    np.testing.assert_array_equal(direct, q)
    assert zero == [0]*5


def test_powered_replay_uses_follower_measurements_not_leader_or_goals(tmp_path):
    measured = [[10, 20, 30, 40, 50, 0], [11, 21, 31, 41, 51, 0]]
    data = {"kind": "so101_powered_teleop_recording", "schema_version": 1,
            "joint_names": list(JOINT_NAMES), "joint_zero_offset_deg": [0]*5,
            "measurement_source": "follower_present_position", "source_sha256": {},
            "samples": [{"time_s": i*.02, "lerobot_position": p,
                         "q_model_rad": scene_check.lerobot_position_to_model_rad(p).tolist(),
                         "leader_position": [80]*6, "goal_arm_deg": [70]*5}
                        for i, p in enumerate(measured)]}
    path = tmp_path/"recording.json"
    path.write_text(json.dumps(data))
    loaded = scene_check.load_recording(path)
    replay, offsets = scene_check.replay_joint_positions(loaded)
    np.testing.assert_allclose(replay[:, :5], np.deg2rad(np.array(measured)[:, :5]))
    assert offsets == [0]*5
    data["samples"][0]["q_model_rad"][0] = np.deg2rad(70)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differ from recorded follower"):
        scene_check.load_recording(path)
    data["measurement_source"] = "leader_targets"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="measured follower"):
        scene_check.load_recording(path)


@pytest.mark.parametrize("flags", [["--benchmark-offsets"], ["--replay-task-config", "different-task.json"]])
def test_replay_overrides_cannot_be_used_during_recording(monkeypatch, flags):
    monkeypatch.setattr("sys.argv", ["scene_check", "record", '--robot-id', 'test',
                                   '--port', '/dev/test', '--keyboard-device', '/dev/keys',
                                   '--task-config', 'scene.json', *flags])
    monkeypatch.setattr(scene_check, "record", lambda args: pytest.fail("recorder should not be entered"))
    with pytest.raises(SystemExit) as error:
        scene_check.main()
    assert error.value.code == 2


def test_recording_requires_explicit_devices_before_hardware_access(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['scene_check', 'record'])
    monkeypatch.setattr(scene_check, 'record', lambda _: pytest.fail('must not enter recorder'))
    with pytest.raises(SystemExit) as error:
        scene_check.main()
    assert error.value.code == 2
    assert 'record requires --robot-id, --port, --keyboard-device, --task-config' in capsys.readouterr().err


@pytest.fixture
def diagnostic_mapping(tmp_path):
    task = {"schema_version": 1, "keyboard": {"position_m": [.26, .014, .0076]}}
    recording = {"robot_id": "robot", "calibration_sha256": "calibration",
                 "robot_usd_sha256": "usd", "source_sha256": {"kinematics.py": "source"},
                 "samples": [{"q_model_rad": [0.1]*6}, {"q_model_rad": [0.2]*6}]}
    report = {"schema_version": 1, "status": "diagnostic_only_not_approved_for_deployment",
              "joint_order": list(JOINT_NAMES[:5]), "task_config": task,
              "task_config_sha256": "frozen", "sampling": "interval_mean",
              "identities": {k: recording[k] for k in ("robot_id", "calibration_sha256", "robot_usd_sha256", "source_sha256")},
              "candidates": {"fit3": {"offset_deg": [0., 2., -10., 1., 0.]},
                             "fit4": {"offset_deg": [2.5, 1.5, -9., 1., 0.]}}}
    path = tmp_path/"mapping-report.json"
    path.write_text(json.dumps(report))
    return recording, task, report, path


def test_mapping_candidate_selection_and_once_only_copy(diagnostic_mapping):
    recording, task, report, path = diagnostic_mapping
    before = json.dumps(recording)
    offsets, provenance = scene_check.load_mapping_candidate(recording, path, "fit4", task)
    assert offsets == report["candidates"]["fit4"]["offset_deg"]
    assert provenance["candidate"] == "fit4" and provenance["report_sha256"] == scene_check.digest(path)
    assert provenance["diagnostic_only"]
    original = np.array([row["q_model_rad"] for row in recording["samples"]])
    for _ in range(2):
        replay, applied = scene_check.replay_joint_positions(recording, mapping_offsets_deg=offsets)
        np.testing.assert_allclose(replay[:, :5], original[:, :5]+np.deg2rad(offsets))
        np.testing.assert_array_equal(replay[:, 5], original[:, 5])
        assert applied == offsets
    assert json.dumps(recording) == before
    with pytest.raises(ValueError, match="mutually exclusive"):
        scene_check.replay_joint_positions(recording, True, mapping_offsets_deg=offsets)


@pytest.mark.parametrize("field", ["robot_id", "calibration_sha256", "robot_usd_sha256", "source_sha256"])
def test_mapping_rejects_identity_mismatch(diagnostic_mapping, field):
    recording, task, report, path = diagnostic_mapping
    report["identities"][field] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=field):
        scene_check.load_mapping_candidate(recording, path, "fit4", task)


@pytest.mark.parametrize("offsets", [[0.]*4, [0.]*6, [16., 0., 0., 0., 0.],
                                    [float("nan"), 0., 0., 0., 0.], [True, 0., 0., 0., 0.]])
def test_mapping_rejects_invalid_offsets(diagnostic_mapping, offsets):
    recording, task, report, path = diagnostic_mapping
    report["candidates"]["fit4"]["offset_deg"] = offsets
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="five finite offsets"):
        scene_check.load_mapping_candidate(recording, path, "fit4", task)


def test_mapping_rejects_wrong_task_status_joint_order_and_candidate(diagnostic_mapping):
    recording, task, report, path = diagnostic_mapping
    with pytest.raises(ValueError, match="frozen task"):
        scene_check.load_mapping_candidate(recording, path, "fit4", {})
    with pytest.raises(ValueError, match="explicit replay task"):
        scene_check.load_mapping_candidate(recording, path, "fit4", None)
    with pytest.raises(ValueError, match="not found"):
        scene_check.load_mapping_candidate(recording, path, "unlisted", task)
    with pytest.raises(ValueError, match="invalid mapping candidate"):
        scene_check.load_mapping_candidate(recording, path, "../fit4", task)
    report["joint_order"] = list(reversed(JOINT_NAMES[:5]))
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="joint order"):
        scene_check.load_mapping_candidate(recording, path, "fit4", task)
    report["status"] = "passed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="diagnostic-only"):
        scene_check.load_mapping_candidate(recording, path, "fit4", task)


@pytest.mark.parametrize("flags", [
    ["--mapping-report", "report.json"], ["--mapping-candidate", "fit4"],
    ["--mapping-report", "report.json", "--mapping-candidate", "fit4"],
    ["--mapping-report", "report.json", "--mapping-candidate", "fit4", "--benchmark-offsets"],
    ["--headless", "--video"],
])
def test_mapping_cli_rejects_invalid_flag_combinations(monkeypatch, flags):
    monkeypatch.setattr("sys.argv", ["scene_check", "replay", *flags])
    monkeypatch.setattr(scene_check, "load_recording", lambda _: pytest.fail("must reject before loading input"))
    with pytest.raises(SystemExit) as error:
        scene_check.main()
    assert error.value.code == 2


def test_mapping_launcher_mounts_readonly_preserves_task_and_refuses_overwrite(diagnostic_mapping, monkeypatch):
    from so101_typing.hardware import run_fk_probe
    recording, task, _, mapping_path = diagnostic_mapping
    task_path = mapping_path.parent/"task-input.json"
    task_path.write_text(json.dumps(task, separators=(",", ":")))
    trajectory_path = mapping_path.parent/"recording.json"
    trajectory_path.write_text(json.dumps(recording))
    monkeypatch.setattr(scene_check, "load_recording", lambda _: recording)
    launches = []
    monkeypatch.setattr(run_fk_probe, "simulate", lambda path, **kwargs: launches.append((path, kwargs)) or 0)
    monkeypatch.setattr("sys.argv", ["scene_check", "replay", "--recording", str(trajectory_path),
        "--mapping-report", str(mapping_path), "--mapping-candidate", "fit4",
        "--replay-task-config", str(task_path), "--headless"])
    assert scene_check.main() == 0
    _, kwargs = launches[0]
    assert kwargs["read_only_mounts"] == [(mapping_path.resolve(), "/mapping-report.json")]
    assert kwargs["headless"] is True
    destination = trajectory_path.parent/kwargs["report_name"]
    assert scene_check.digest(mapping_path)[:12] in destination.parent.name
    assert scene_check.digest(task_path)[:12] in destination.parent.name
    assert destination.parent.name.endswith("-fit4")
    assert (destination.parent/"task.json").read_bytes() == task_path.read_bytes()
    assert kwargs["extra_args"][:4] == ["--mapping-report", "/mapping-report.json", "--mapping-candidate", "fit4"]
    with pytest.raises(SystemExit) as error:
        scene_check.main()
    assert error.value.code == 2 and len(launches) == 1
