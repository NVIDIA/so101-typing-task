"""Record a torque-off robot or replay measured follower poses in Newton."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np

from .fk_probe_trajectory import ROOT, TASK, digest
from .source_identity import identity_matches, source_hash_matches
from .so101_kinematics import JOINT_NAMES, lerobot_position_to_model_rad


def replay_joint_positions(recording, benchmark_offsets=False, *, mapping_offsets_deg=None):
    """Apply one selected encoder-to-model convention only to a replay copy."""
    if benchmark_offsets and mapping_offsets_deg is not None:
        raise ValueError("benchmark offsets and diagnostic mapping are mutually exclusive")
    offsets_deg = np.zeros(6)
    if benchmark_offsets:
        from .fixed_cartesian_policy import PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG
        offsets_deg[:5] = PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG
    if mapping_offsets_deg is not None:
        offsets_deg[:5] = validate_mapping_offsets(mapping_offsets_deg)
    positions = np.array([r["q_model_rad"] for r in recording["samples"]])
    return positions + np.deg2rad(offsets_deg), offsets_deg[:5].tolist()


def validate_mapping_offsets(values):
    if (not isinstance(values, list) or len(values) != 5
            or any(type(value) not in (int, float) for value in values)):
        raise ValueError("diagnostic mapping needs five finite offsets within +/-15 degrees")
    offsets = np.asarray(values, dtype=float)
    if not np.isfinite(offsets).all() or (np.abs(offsets) > 15).any():
        raise ValueError("diagnostic mapping needs five finite offsets within +/-15 degrees")
    return offsets


def load_mapping_candidate(recording, report_path, candidate, task):
    """Validate diagnostic provenance for simulation only; never install calibration."""
    if report_path is None or not candidate:
        raise ValueError("--mapping-report and --mapping-candidate are required together")
    if task is None:
        raise ValueError("diagnostic mapping requires an explicit replay task configuration")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        raise ValueError("invalid mapping candidate name")
    report = json.loads(report_path.read_text())
    if (report.get("schema_version") != 1
            or report.get("status") != "diagnostic_only_not_approved_for_deployment"):
        raise ValueError("expected a diagnostic-only scene mapping report")
    if report.get("joint_order") != list(JOINT_NAMES[:5]):
        raise ValueError("mapping report joint order differs from recording")
    for field in ("robot_id", "calibration_sha256", "robot_usd_sha256", "source_sha256"):
        if not recording.get(field) or not identity_matches(field, report.get("identities", {}).get(field), recording[field]):
            raise ValueError(f"mapping report differs from recording: {field}")
    if task != report.get("task_config"):
        raise ValueError("explicit replay task differs from mapping report frozen task")
    candidates = report.get("candidates", {})
    if candidate not in candidates:
        raise ValueError(f"mapping candidate {candidate!r} not found; available: {', '.join(candidates)}")
    offsets = validate_mapping_offsets(candidates[candidate].get("offset_deg"))
    return offsets.tolist(), {"report_sha256": digest(report_path), "candidate": candidate,
                              "report_status": report["status"],
                              "report_task_config_sha256": report.get("task_config_sha256"),
                              "report_sampling": report.get("sampling"),
                              "diagnostic_only": True}


def load_recording(path):
    data = json.loads(path.read_text())
    kinds = {"so101_torque_off_recording", "so101_powered_teleop_recording"}
    if data.get("kind") not in kinds or data.get("schema_version") != 1:
        raise ValueError("expected a measured torque-off or powered-teleoperation scene recording")
    if data.get("joint_names") != list(JOINT_NAMES) or data.get("joint_zero_offset_deg") != [0]*5:
        raise ValueError("recording joint convention mismatch")
    samples = data["samples"]
    times = np.array([r["time_s"] for r in samples])
    q = np.array([r["q_model_rad"] for r in samples])
    if len(times) < 2 or not np.isfinite(times).all() or times[0] < 0 or (np.diff(times) <= 0).any():
        raise ValueError("recording needs at least two samples with increasing timestamps")
    if q.shape != (len(samples), 6) or not np.isfinite(q).all():
        raise ValueError("recording needs six finite measured joint positions per sample")
    if data["kind"] == "so101_powered_teleop_recording":
        if data.get("measurement_source") != "follower_present_position":
            raise ValueError("powered replay requires measured follower Present_Position, not leader targets")
    try:
        measured = np.asarray([r["lerobot_position"] for r in samples], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("recording needs measured follower positions") from error
    if measured.shape != q.shape or not np.isfinite(measured).all():
        raise ValueError("recording needs six finite measured follower positions per sample")
    expected = np.array([lerobot_position_to_model_rad(row) for row in measured])
    if not np.allclose(q, expected, rtol=0, atol=1e-10):
        raise ValueError("replay poses differ from recorded follower measurements")
    if times[-1] > 600 or np.max(np.diff(times)) > .25:
        raise ValueError("recording exceeds 10 minutes or has a sampling gap above 250 ms")
    for name, expected in data["source_sha256"].items():
        source = (ROOT/name).resolve()
        if not source.is_relative_to(ROOT) or not source_hash_matches(name, digest(source), expected):
            raise ValueError(f"recorded model/scene source changed: {name}")
    return data


def load_alignment_candidate(recording, path, task):
    """Simulation-side identity check; live calibration is checked on the host."""
    from .scene_alignment import fingerprint
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or data.get('kind') != 'so101_alignment_candidate' or data.get('status') != 'unqualified':
        raise ValueError('expected an unqualified alignment candidate')
    c = data['contract']
    if fingerprint(c) != data.get('contract_sha256') or c.get('task_config') != task:
        raise ValueError('alignment contract or replay scene changed')
    for field in ('robot_id', 'calibration_sha256', 'robot_usd_sha256', 'source_sha256'):
        if not identity_matches(field, c.get(field), recording.get(field)):
            raise ValueError(f'alignment differs from recording: {field}')
    if c.get('joint_order') != list(JOINT_NAMES[:5]):
        raise ValueError('alignment joint order changed')
    return validate_mapping_offsets(c['joint_zero_offset_deg']).tolist(), {
        'alignment_contract_sha256': data['contract_sha256'], 'diagnostic_only': True}


def capture(robot, keyboard, *, seconds, hz, stream):
    """The only motor writes are disabling torque. No goal or enable-torque writes."""
    robot.bus.disable_torque(list(JOINT_NAMES), num_retry=3)
    torque = robot.bus.sync_read("Torque_Enable", list(JOINT_NAMES), normalize=False, num_retry=3)
    if any(int(v) != 0 for v in torque.values()) or set(torque) != set(JOINT_NAMES):
        raise RuntimeError("could not verify torque off on all six motors")
    print("TORQUE OFF. Guide the fixed tip onto keys and release each key. Esc or Ctrl-C saves and stops.", flush=True)
    samples, events = [], []
    start = time.monotonic()
    reason = "duration"
    try:
        while time.monotonic()-start < seconds:
            tick = time.monotonic()
            values = robot.bus.sync_read("Present_Position", list(JOINT_NAMES), normalize=True, num_retry=3)
            position = [float(values[name]) for name in JOINT_NAMES]
            q = lerobot_position_to_model_rad(position)
            row = {"time_s": time.monotonic()-start, "q_model_rad": q.tolist(),
                   "lerobot_position": position}
            samples.append(row)
            stream.write(json.dumps({"kind": "sample", **row})+"\n")
            edges, stop = keyboard.poll()
            for event in edges:
                event = {**event, "time_s": event["received_monotonic_s"]-start}
                events.append(event)
                stream.write(json.dumps({"kind": "key", **event})+"\n")
                print(f"REAL {event['key_name']} {event['edge'].upper()} at {event['time_s']:.2f}s", flush=True)
            stream.flush()
            if stop:
                reason = "esc"
                break
            time.sleep(max(0, 1/hz-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        reason = "ctrl_c"
    return samples, events, reason


def record(args):
    from scripts.fixture_layout import load_geometry
    from scripts.reinforcement_learning.rsl_rl.public_task import load_config
    from .robot import require_calibration, resolve_robot_port, connect_robot
    from .policy_runtime import KeyboardMonitor
    calibration = require_calibration(args.robot_id)
    port = resolve_robot_port(args.port)
    task = load_config(args.task_config)
    geometry = load_geometry()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = args.output_root.resolve()/stamp
    directory.mkdir(parents=True, exist_ok=False)
    source_files = ["scripts/so101_homing/so101_kinematics.py", *geometry["source_sha256"]]
    data = {"schema_version": 1, "kind": "so101_torque_off_recording", "task": TASK,
            "task_config": task, "task_config_sha256": digest(args.task_config),
            "source_sha256": {name: digest(ROOT/name) for name in source_files},
            "robot_usd_sha256": "c6c82840925ace388b0ff0acb7d8538c2b419d92fbe01dc70fe833b974d6d462",
            "robot_id": args.robot_id, "port": port, "calibration": json.loads(calibration.read_text()),
            "calibration_sha256": digest(calibration), "keyboard_device": args.keyboard_device,
            "joint_names": list(JOINT_NAMES), "joint_zero_offset_deg": [0]*5,
            "requested_hz": args.hz, "samples": [], "events": [], "status": "failed"}
    robot = None
    try:
        with KeyboardMonitor(args.keyboard_device) as keyboard:
            print("Support the arm: torque will turn OFF in 3 seconds. Keep using the fixed typing tip.", flush=True)
            time.sleep(3)
            keyboard.poll()  # Discard the terminal command's Enter release.
            if keyboard.device.active_keys():
                raise ValueError("release keyboard keys before recording")
            robot = connect_robot(port=port, robot_id=args.robot_id, enable_torque=False)
            with (directory/"capture.jsonl").open("x") as stream:
                samples, events, reason = capture(robot, keyboard, seconds=args.seconds, hz=args.hz, stream=stream)
            data.update(samples=samples, events=events, stop_reason=reason, status="recorded")
    except BaseException as error:
        data["error"] = str(error)
        raise
    finally:
        if robot is not None:
            robot.disconnect()  # Does not re-enable torque or command a return pose.
        path = directory/"recording.json"
        path.write_text(json.dumps(data, indent=2)+"\n")
        print(f"Recording: {path}\nTorque remains off; support the arm.", flush=True)
    load_recording(path)
    print("Replay with: ./scripts/scene_check.sh replay --recording " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "replay"))
    parser.add_argument("--robot-id", help="your follower's LeRobot calibration ID (record mode)")
    parser.add_argument("--port", help="explicit follower serial device (record mode)")
    parser.add_argument("--keyboard-device", help="explicit keyboard evdev device (record mode)")
    parser.add_argument("--task-config", type=Path, help="your measured scene (record mode)")
    parser.add_argument("--output-root", type=Path, default=ROOT/"output/scene-recordings")
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--hz", type=float, default=50)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--headless", action="store_true", help="replay without the viewer or video rendering")
    parser.add_argument("--benchmark-offsets", action="store_true",
                        help="replay only: add the historical joint offsets; save separate comparison outputs")
    parser.add_argument("--mapping-report", type=Path,
                        help="replay only: use an unapproved diagnostic mapping report in simulation")
    parser.add_argument("--mapping-candidate", help="candidate name from --mapping-report")
    parser.add_argument('--alignment', type=Path, help='replay only: frozen alignment candidate from ./so101 alignment prepare')
    parser.add_argument("--replay-task-config", type=Path,
                        help="replay only: test a different keyboard pose and save separate placement outputs")
    args = parser.parse_args()
    try:
        if args.alignment and (args.mapping_report or args.mapping_candidate or args.benchmark_offsets):
            parser.error('--alignment cannot be combined with other mapping selections')
        if bool(args.mapping_report) != bool(args.mapping_candidate):
            parser.error("--mapping-report and --mapping-candidate are required together")
        if args.mapping_report and args.benchmark_offsets:
            parser.error("--mapping-report and --benchmark-offsets are mutually exclusive")
        if args.mapping_report and not args.replay_task_config:
            parser.error("--mapping-report requires explicit --replay-task-config")
        if args.headless and args.video:
            parser.error("choose --headless or --video")
        if not 1 <= args.seconds <= 600 or not 10 <= args.hz <= 100:
            parser.error("--seconds must be 1..600 and --hz must be 10..100")
        if args.mode == "record":
            if args.alignment:
                parser.error('--alignment is only for replay; recordings preserve measured angles')
            missing = [name for name in ('robot_id', 'port', 'keyboard_device', 'task_config') if not getattr(args, name)]
            if missing:
                parser.error('record requires ' + ', '.join('--'+name.replace('_', '-') for name in missing))
            if args.headless:
                parser.error("--headless is only for replay")
            if args.mapping_report:
                parser.error("diagnostic mappings are only for simulation replay")
            if args.replay_task_config:
                parser.error("--replay-task-config is only for replay")
            if args.benchmark_offsets:
                parser.error("--benchmark-offsets is only for replay; recordings keep the original encoder readings")
            record(args)
            return 0
        path = args.recording
        if path is None:
            paths = sorted(args.output_root.glob("*/recording.json"), reverse=True)
            path = next((p for p in paths if json.loads(p.read_text()).get("status") == "recorded"), None)
        if path is None:
            parser.error("no recording found; run record first")
        path = path.resolve()
        recording = load_recording(path)
        print(f"Replaying measured poses from: {path}", flush=True)
        report_name = "benchmark-offsets/scene-report.json" if args.benchmark_offsets else "scene-report.json"
        extra_args = ["--benchmark-offsets"] if args.benchmark_offsets else []
        simulation_options = {}
        if args.headless:
            simulation_options["headless"] = True
        if args.alignment:
            from .scene_alignment import candidate
            from scripts.reinforcement_learning.rsl_rl.public_task import load_config
            candidate(args.alignment, recording['robot_id'])
            task = load_config(args.replay_task_config) if args.replay_task_config else recording['task_config']
            _, mapping = load_alignment_candidate(recording, args.alignment, task)
            folder = f"alignment-{mapping['alignment_contract_sha256'][:12]}"
            destination = path.parent/folder
            destination.mkdir(parents=True, exist_ok=False)
            (destination/'task.json').write_text(json.dumps(task, indent=2)+'\n')
            extra_args += ['--alignment', '/alignment.json', '--task-config', f'/results/{folder}/task.json']
            simulation_options['read_only_mounts'] = [(args.alignment.resolve(), '/alignment.json')]
            report_name = f'{folder}/scene-report.json'
        elif args.replay_task_config:
            from scripts.reinforcement_learning.rsl_rl.public_task import load_config
            task = load_config(args.replay_task_config)
            if args.mapping_report:
                _, mapping = load_mapping_candidate(recording, args.mapping_report, args.mapping_candidate, task)
                folder = (f"placement-{digest(args.replay_task_config)[:12]}-mapping-"
                          f"{mapping['report_sha256'][:12]}-{args.mapping_candidate}")
                extra_args += ["--mapping-report", "/mapping-report.json", "--mapping-candidate", args.mapping_candidate]
                simulation_options["read_only_mounts"] = [(args.mapping_report.resolve(), "/mapping-report.json")]
            else:
                condition = "benchmark" if args.benchmark_offsets else "direct"
                folder = f"placement-{digest(args.replay_task_config)[:12]}-{condition}"
            destination = path.parent/folder
            destination.mkdir(parents=True, exist_ok=not bool(args.mapping_report))
            if args.mapping_report:
                (destination/"task.json").write_bytes(args.replay_task_config.read_bytes())
            else:
                (destination/"task.json").write_text(json.dumps(task, indent=2)+"\n")
            extra_args += ["--task-config", f"/results/{folder}/task.json"]
            report_name = f"{folder}/scene-report.json"
        from .run_fk_probe import simulate
        return simulate(path, video=args.video,
                        script="scripts/reinforcement_learning/rsl_rl/replay_so101_scene_recording.py",
                        report_name=report_name, extra_args=extra_args, **simulation_options)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
