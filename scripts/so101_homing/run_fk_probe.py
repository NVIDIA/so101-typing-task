"""Generate, simulate, or physically replay ONE saved joint-command trajectory.

Default: offline hardware preflight. Only --execute can connect to the robot.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import time

import numpy as np

from .fk_probe_trajectory import DT, RATES, ROOT, digest, generate, load


def verified_plan(path):
    plan = load(path)
    report = json.loads(path.with_name("sim-report.json").read_text())
    if report.get("status") != "passed" or report.get("trajectory_sha256") != digest(path):
        raise ValueError("this exact trajectory has not passed simulation; run --sim first")
    return plan


def simulate(path, *, video=False,
             script="scripts/reinforcement_learning/rsl_rl/replay_so101_fk_probe.py",
             report_name="sim-report.json", extra_args=(), read_only_mounts=(), headless=False):
    launcher = runpy.run_path(str(ROOT/"so101"))
    image = launcher["IMAGE"]
    try:
        source = subprocess.check_output(["docker", "image", "inspect", image, "--format",
            '{{index .Config.Labels "org.so101.source-sha256"}}'], text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot inspect runtime image; run ./so101 build") from error
    if source != launcher["source_fingerprint"]():
        raise ValueError("runtime image does not match this checkout; run ./so101 build")
    if not (video or headless) and not os.environ.get("DISPLAY"):
        raise ValueError("Newton viewer needs DISPLAY; run from your desktop terminal or use --video")
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home()/".cache")))/"so101-typing"
    cache.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", "HOME=/tmp/so101-home", "-e", "USER=so101", "-e", "LOGNAME=so101",
           "-e", "OMNICLIENT_HUB_MODE=disabled", "-v", f"{cache}:/tmp/so101-home",
           "-v", f"{path.parent}:/results", "-v", f"{path.parent}:/workspace/isaaclab/logs"]
    if not (video or headless):
        cmd += ["-e", f"DISPLAY={os.environ['DISPLAY']}", "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
                "-v", "/tmp/.X11-unix:/tmp/.X11-unix:ro"]
        authority = os.environ.get("XAUTHORITY")
        if authority and Path(authority).is_file():
            cmd += ["-v", f"{authority}:/tmp/typing.xauthority:ro", "-e", "XAUTHORITY=/tmp/typing.xauthority"]
    for host_path, container_path in read_only_mounts:
        cmd += ["-v", f"{Path(host_path).resolve()}:{container_path}:ro"]
    cmd += [image, "-p", script,
            "--trajectory", f"/results/{path.name}", "--out", f"/results/{report_name}"]
    if video:
        cmd += ["--headless", "--video"]
    elif headless:
        cmd += ["--headless"]
    else:
        cmd += ["--visualizer", "newton", "--step-sleep", str(DT)]
    cmd += list(extra_args)
    cmd += ["physics=newton_mjwarp"]
    report_path = path.parent/report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.unlink(missing_ok=True)
    with (report_path.parent/"sim-console.log").open("w") as log:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def execute(plan, path, args, calibration):
    from .policy_runtime import KeyboardMonitor, append_jsonl
    from .robot import connect_robot, prepare_closed_typing_jaw, read_arm_q_deg, send_typing_q_deg
    benchmark_offsets = bool(getattr(args, "benchmark_offsets", False))
    alignment = getattr(args, 'alignment_data', None)
    if alignment:
        offsets_deg = np.asarray(alignment['contract']['joint_zero_offset_deg'], dtype=float)
    elif benchmark_offsets:
        from .fixed_cartesian_policy import PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG
        offsets_deg = np.asarray(PHYSICAL_CALIBRATED_20260718_JOINT_ZERO_OFFSET_DEG, dtype=float)
    else:
        offsets_deg = np.zeros(5)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log = path.parent/"hardware"/(stamp+".jsonl")
    print(f"Robot: {args.robot_id}; {args.port}\nReplay: {path}\n"
          f"Joint offsets (model = encoder + offset): {offsets_deg.tolist()} degrees. "
          f"{len(plan['rows'])*DT:.2f}s of saved commands after startup.\n"
          "Clear the sweep and keep motor power ready to stop. Esc or Ctrl-C stops commands; torque stays on.")
    print("Starting the supervised replay because --execute was provided.", flush=True)
    append_jsonl(log, {"kind": "header", "trajectory_sha256": digest(path),
                      "calibration_sha256": digest(calibration), "robot_id": args.robot_id,
                      "port": args.port, "keyboard_device": args.keyboard_device,
                      "alignment_probe_schema": 1 if alignment else None,
                      "alignment_contract_sha256": alignment['contract_sha256'] if alignment else None,
                      "joint_zero_offset_deg": offsets_deg.tolist(), "control_dt_s": DT})
    edges = []
    clearance_ticks = 0
    clearance_verified = False
    robot = None
    status = "aborted"
    try:
        with KeyboardMonitor(args.keyboard_device) as keyboard:
            # Allow the operator to release Enter after the terminal confirmation.
            deadline = time.monotonic()+2
            while keyboard.device.active_keys() and time.monotonic() < deadline:
                _, stop = keyboard.poll()
                if stop:
                    raise KeyboardInterrupt("Esc")
                time.sleep(.02)
            if keyboard.device.active_keys():
                raise ValueError("release all keys before starting the probe")
            _, stop = keyboard.poll()
            if stop:
                raise KeyboardInterrupt("Esc")
            keyboard.pressed_codes.clear()
            robot = connect_robot(port=args.port, robot_id=args.robot_id, enable_torque=True)
            prepare_closed_typing_jaw(robot)

            def step(q, phase, index):
                nonlocal clearance_ticks, clearance_verified
                start = time.monotonic()
                events, stop = keyboard.poll()
                for event in events:
                    append_jsonl(log, {"kind": "key", "phase": phase, "step": index, **event})
                    char = (event["char"] or "").upper()
                    if char != plan["letter"]:
                        raise RuntimeError(f"unexpected key: {event['key_name']}")
                    edges.append(event["edge"])
                    print(f"PHYSICAL {char} {event['edge'].upper()}", flush=True)
                    if edges != ["down", "up"][:len(edges)] or (event["edge"] == "down" and phase not in ("descend", "press", "lift")):
                        raise RuntimeError(f"unexpected physical key sequence for {plan['letter']}: {edges} "
                                           f"during {phase} at step {index}; expected one down then one up")
                if stop:
                    raise KeyboardInterrupt("Esc")
                measured_encoder = read_arm_q_deg(robot)
                measured_model = measured_encoder + offsets_deg
                gripper_percent = None
                if alignment:
                    from .scene_alignment import CLOSED_MAX_PERCENT, CLEARANCE_M, CLEARANCE_TICKS
                    from .fk_probe_trajectory import tip
                    gripper_percent = float(robot.bus.sync_read('Present_Position', ['gripper'], normalize=True,
                                                               num_retry=0)['gripper'])
                    if not 0 <= gripper_percent <= CLOSED_MAX_PERCENT:
                        raise RuntimeError('moving jaw is not measured closed; stopping alignment probe')
                    if edges == ['down', 'up'] and tip(np.deg2rad(measured_model))[2] >= plan['rim_xyz_b_m'][2]+CLEARANCE_M:
                        clearance_ticks += 1
                        clearance_verified |= clearance_ticks >= CLEARANCE_TICKS
                    else:
                        clearance_ticks = 0
                if np.max(np.abs(np.rad2deg(q)-measured_model)) > 12:
                    raise RuntimeError("joint tracking error exceeds 12 degrees; stopping commands")
                command_encoder = np.rad2deg(q) - offsets_deg
                send_typing_q_deg(robot, command_encoder)
                append_jsonl(log, {"kind": "command", "step": index, "phase": phase,
                                  "monotonic_s": start, "q_command_rad": np.asarray(q).tolist(),
                                  "q_command_encoder_deg": command_encoder.tolist(),
                                  "q_measured_encoder_deg": measured_encoder.tolist(),
                                  "q_measured_model_deg": measured_model.tolist(),
                                  "gripper_percent": gripper_percent,
                                  "joint_zero_offset_deg": offsets_deg.tolist()})
                elapsed = time.monotonic()-start
                if elapsed > .12:
                    raise RuntimeError(f"control tick took {elapsed:.3f}s; stopping commands")
                time.sleep(max(0, DT-elapsed))

            current_encoder = read_arm_q_deg(robot)
            current = np.deg2rad(current_encoder + offsets_deg)
            initial = np.array(plan["initial_q_rad"])
            count = max(100, int(np.ceil(np.max(np.abs(initial-current)/(RATES*.45*DT))*1.5)))
            print("Moving to shared start pose.", flush=True)
            for i, a in enumerate(np.linspace(0, 1, count+1)[1:]):
                step(current+(initial-current)*(a*a*(3-2*a)), "startup", -count+i)
            for _ in range(25):
                step(initial, "startup_settle", -1)
            startup_error_deg = float(np.max(np.abs(np.rad2deg(initial)-(read_arm_q_deg(robot)+offsets_deg))))
            if startup_error_deg > 4:
                raise RuntimeError(f"start pose is {startup_error_deg:.2f} degrees from the shared initial pose")
            if startup_error_deg > 1:
                print(f"Startup servo tracking error: {startup_error_deg:.2f} degrees; continuing within 4 degree limit.", flush=True)
            print("Replaying the exact simulation trajectory.", flush=True)
            previous_phase = None
            for i, row in enumerate(plan["rows"]):
                if row["phase"] != previous_phase:
                    print(row["phase"].upper(), flush=True)
                    previous_phase = row["phase"]
                step(np.array(row["q_rad"]), row["phase"], i)
            if edges != ["down", "up"] or keyboard.pressed_codes:
                raise RuntimeError(
                    f"motion finished, but {plan['letter']} down/up was not verified: {edges}"
                )
            if alignment and not clearance_verified:
                raise RuntimeError('release lacks two measured 15 mm clearance ticks')
            status = "passed"
            print(f"Physical {plan['letter']} press and release verified; returned to shared start pose.")
    finally:
        if robot is not None:
            robot.disconnect()
        append_jsonl(log, {"kind": "result", "status": status, "edges": edges,
                          "clearance_verified": clearance_verified})
        print(f"Log: {log}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--letter", choices=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), default="F",
                        help="letter to generate when using --generate")
    parser.add_argument("--robot-id")
    parser.add_argument("--port")
    parser.add_argument("--keyboard-device")
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument("--benchmark-offsets", action="store_true",
                        help="apply the historical model-to-LeRobot inverse conversion")
    mapping.add_argument('--alignment', type=Path, help='unqualified alignment.json for supervised mapping probes')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--sim", action="store_true", help="replay in Newton viewer")
    mode.add_argument("--video", action="store_true", help="headless simulation with video")
    mode.add_argument("--execute", action="store_true", help="supervised physical replay")
    args = parser.parse_args()
    path = args.trajectory.expanduser().resolve()
    try:
        if args.generate:
            if args.task_config is None:
                parser.error("--generate needs --task-config")
            plan = generate(args.task_config, path, letter=args.letter)
            print(f"Saved {len(plan['rows'])} joint commands ({len(plan['rows'])*DT:.2f}s): {path}")
            return 0
        plan = load(path)
        if args.task_config and digest(args.task_config) != plan["task_config_sha256"]:
            raise ValueError("task config changed; generate a new trajectory and verify it in sim")
        if args.sim or args.video:
            return simulate(path, video=args.video)
        verified_plan(path)
        if not args.robot_id or not args.port or not args.keyboard_device:
            parser.error("hardware preflight needs --robot-id, --port and --keyboard-device")
        from .robot import require_calibration, resolve_robot_port
        from .keyboard_events import _open_device
        calibration = require_calibration(args.robot_id)
        if args.alignment:
            from . import scene_alignment
            args.alignment_data = scene_alignment.candidate(args.alignment, args.robot_id)
            scene_alignment.check_plan(args.alignment_data['contract'], plan)
            offsets = np.asarray(args.alignment_data['contract']['joint_zero_offset_deg'])
            goals = np.rad2deg(np.asarray([plan['initial_q_rad']]+[r['q_rad'] for r in plan['rows']]))-offsets
            limits = json.loads(calibration.read_text())
            for i, name in enumerate(args.alignment_data['contract']['joint_order']):
                half_span = (limits[name]['range_max']-limits[name]['range_min'])*180/4095
                if np.max(np.abs(goals[:, i])) > half_span:
                    raise ValueError(f'aligned trajectory exceeds calibrated {name} range')
        args.port = resolve_robot_port(args.port)
        device = _open_device(args.keyboard_device)
        device.close()
        if not args.execute:
            mapping = f"unqualified alignment {args.alignment_data['contract_sha256']}" if args.alignment else \
                "historical benchmark offsets" if args.benchmark_offsets else "direct LeRobot/model mapping"
            print(f"PREFLIGHT PASSED; no robot connection or motion.\nRobot: {args.robot_id}\n"
                  f"Calibration: {calibration}\nVerified trajectory: {path}\n"
                  f"SHA256: {digest(path)}\nMapping: {mapping}. Add --execute for physical replay.")
            return 0
        return execute(plan, path, args, calibration)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("Stopped. Commands ended; support the arm before switching off motor power.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
