"""Prescribed measured-pose replay for scene inspection, not an actuator/policy test.

Robot joint states are set from the recording at every physics step. Keyboard
joints remain simulated. Contact results are diagnostic; compare with USB events.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.so101_homing.scene_check import load_alignment_candidate, load_mapping_candidate, load_recording, replay_joint_positions
from scripts.so101_homing.fk_probe_trajectory import TASK, digest
from scripts.so101_homing.so101_kinematics import JOINT_NAMES, FIXED_JAW_TYPING_TIP_SITE_POS, site_position
from public_task import apply_config, load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trajectory", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--task", default=TASK)
parser.add_argument("--video", action="store_true")
parser.add_argument("--step-sleep", type=float, default=0)
parser.add_argument("--benchmark-offsets", action="store_true")
parser.add_argument("--mapping-report", type=Path)
parser.add_argument("--mapping-candidate")
parser.add_argument('--alignment', type=Path)
parser.add_argument("--task-config", type=Path)
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
if args.alignment and (args.mapping_report or args.mapping_candidate or args.benchmark_offsets):
    parser.error('--alignment cannot be combined with other mappings')
if bool(args.mapping_report) != bool(args.mapping_candidate):
    parser.error("--mapping-report and --mapping-candidate are required together")
if args.mapping_report and args.benchmark_offsets:
    parser.error("--mapping-report and --benchmark-offsets are mutually exclusive")
if args.mapping_report and not args.task_config:
    parser.error("diagnostic mapping requires explicit --task-config")
sys.argv = [sys.argv[0]]+remaining
if args.video:
    args.enable_cameras = True


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    recording = load_recording(args.trajectory)
    if digest(Path(os.environ["SO101_ROBOT_USD"])) != recording["robot_usd_sha256"]:
        raise ValueError("robot USD differs from recorded model")
    task = load_config(args.task_config) if args.task_config else recording["task_config"]
    mapping_offsets, diagnostic_mapping = None, None
    if args.alignment:
        mapping_offsets, diagnostic_mapping = load_alignment_candidate(recording, args.alignment, task)
    if args.mapping_report:
        mapping_offsets, diagnostic_mapping = load_mapping_candidate(
            recording, args.mapping_report, args.mapping_candidate, task)
    if args.mapping_report or args.alignment:
        if any(p.exists() for p in (args.out, args.out.parent/"scene-trace.jsonl", args.out.parent/"scene-replay.mp4")):
            raise ValueError("refusing to overwrite diagnostic mapping replay artifacts")
    apply_config(env_cfg, task)
    samples = recording["samples"]
    times = np.array([s["time_s"] for s in samples])
    recorded_positions = np.array([s["q_model_rad"] for s in samples])
    positions, offsets_deg = replay_joint_positions(recording, args.benchmark_offsets,
                                                    mapping_offsets_deg=mapping_offsets)
    times -= times[0]
    velocity = np.gradient(positions, times, axis=0)
    events_real = [{**e, "time_s": e["time_s"]-samples[0]["time_s"]} for e in recording["events"]]
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 1307
    env_cfg.decimation = 1
    # The policy action manager is unused; align its constructor's timestep check
    # with this diagnostic's one-physics-step replay cadence.
    env_cfg.actions.action.control_dt_s = float(env_cfg.sim.dt)
    # Construct the articulation at its valid model default. A hand-guided
    # recording can extend beyond the model limits and must not become its
    # validated default pose. The loop below prescribes every measured sample,
    # including the first, without clipping or changing the recording.
    if args.device:
        env_cfg.sim.device = args.device
    if args.video:
        env_cfg.video_recorder.eye = (.62, .48, .42)
        env_cfg.video_recorder.lookat = (.21, .08, .06)
    report = {"status": "failed", "mode": "prescribed_measured_pose_geometry_replay",
              "recording_kind": recording["kind"],
              "measurement_source": recording.get("measurement_source", "follower_present_position"),
              "recording_sha256": digest(args.trajectory), "robot_id": recording["robot_id"],
              "task_config_sha256": digest(args.task_config) if args.task_config else recording["task_config_sha256"],
              "recorded_task_config_sha256": recording["task_config_sha256"],
              "task_config": task, "keyboard_pose_overridden": args.task_config is not None,
              "joint_zero_offset_deg": offsets_deg,
              "encoder_convention": "diagnostic_mapping" if diagnostic_mapping else "benchmark" if args.benchmark_offsets else "direct",
              "mapping": "q_replay_model_rad = q_recorded_rad + radians(joint_zero_offset_deg)",
              "robot_joint_states_prescribed": True,
              "keyboard_joint_states_prescribed": False, "physical_events": events_real,
              "simulation_events": [], "max_pose_error_rad": 0., "max_fk_error_m": 0.,
              "max_robot_drift_within_physics_step_rad": 0.,
              "max_tip_drift_within_physics_step_m": 0.,
              "max_key_fraction_change_during_display_repose": 0.,
              "joint_order": list(JOINT_NAMES),
              "measurement_boundaries": {
                  "after_physics_before_repose": "Copied after scene.update, before reapplying the displayed robot pose.",
                  "existing_tip_and_key_fraction_fields": "Measured after reapplying the displayed robot pose and sim.forward; keyboard joints are never prescribed.",
                  "drift_reference": "Interpolated recorded pose and its FK tip at the end of the physics step; not the displayed pose error."},
              "note": "Visual geometry/contact comparison; not actuator tracking or policy qualification."}
    if diagnostic_mapping:
        report["diagnostic_mapping"] = diagnostic_mapping
    writer = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with launch_simulation(env_cfg, args):
            base = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
            try:
                base.reset()
                env = base.unwrapped
                robot = env.scene["robot"]
                command = env.command_manager.get_term("typing")
                joint_ids = [robot.joint_names.index(name) for name in JOINT_NAMES]
                ids = torch.tensor([0], device=env.device)
                dt = float(env.physics_dt)
                report["physics_dt_s"] = dt
                report["duration_s"] = float(times[-1])
                report["key_event_metric"] = {"sample_rate_hz": 1/dt, "consecutive_samples": 2,
                                              "down_depression_fraction": .5, "up_depression_fraction": .25,
                                              "sample_boundary": "after_display_robot_repose"}
                report["slot_labels"] = list(command.cfg.slot_labels)
                held = np.zeros(len(command.cfg.slot_labels), dtype=bool)
                downs = np.zeros(len(held), dtype=int)
                ups = downs.copy()
                lower = command.keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 0]
                upper = command.keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 1]
                frame_every = max(1, round(.04/dt))
                if args.video:
                    import imageio.v2 as imageio
                    writer = imageio.get_writer(str(args.out.parent/"scene-replay.mp4"), fps=1/(frame_every*dt))

                def at(t, array):
                    return np.array([np.interp(t, times, array[:, j]) for j in range(6)])

                def pose(q, dq):
                    qt = torch.tensor(q[None], device=env.device, dtype=torch.float32)
                    dqt = torch.tensor(dq[None], device=env.device, dtype=torch.float32)
                    robot.write_joint_position_to_sim_index(position=qt, joint_ids=joint_ids, env_ids=ids)
                    robot.write_joint_velocity_to_sim_index(velocity=dqt, joint_ids=joint_ids, env_ids=ids)
                    robot.set_joint_position_target_index(target=qt, joint_ids=joint_ids, env_ids=ids)
                    env.sim.forward()

                real_index = 0
                with (args.out.parent/"scene-trace.jsonl").open("w") as stream, torch.inference_mode():
                    for i, t in enumerate(np.arange(0, times[-1], dt)):
                        tick = time.monotonic()
                        pose(at(t, positions), at(t, velocity))
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(dt=dt)
                        next_t = min(t+dt, times[-1])
                        expected = at(next_t, positions)
                        # Save the physics result before the display pose overwrites robot state.
                        actual_physics = robot.data.joint_pos.torch[0, joint_ids].cpu().numpy().copy()
                        tip_physics = command.fixed_tip_pos_w()[0].cpu().numpy().copy()
                        fractions_physics = ((upper-command.keyboard.data.joint_pos.torch[command._inst_idx, command._col_idx])/(upper-lower))[0].cpu().numpy().copy()
                        expected_tip = site_position(expected, site_pos=FIXED_JAW_TYPING_TIP_SITE_POS)
                        joint_drift = actual_physics-expected
                        drift = float(np.max(np.abs(joint_drift)))
                        tip_drift = float(np.linalg.norm(tip_physics-expected_tip))
                        report["max_robot_drift_within_physics_step_rad"] = max(report["max_robot_drift_within_physics_step_rad"], drift)
                        report["max_tip_drift_within_physics_step_m"] = max(report["max_tip_drift_within_physics_step_m"], tip_drift)
                        # Reapply recorded robot pose for the displayed geometry. Never pose keyboard joints.
                        pose(expected, at(next_t, velocity))
                        actual = robot.data.joint_pos.torch[0, joint_ids].cpu().numpy()
                        tip = command.fixed_tip_pos_w()[0].cpu().numpy()
                        pose_error = float(np.max(np.abs(actual-expected)))
                        fk_error = float(np.linalg.norm(tip-expected_tip))
                        report["max_pose_error_rad"] = max(report["max_pose_error_rad"], pose_error)
                        report["max_fk_error_m"] = max(report["max_fk_error_m"], fk_error)
                        if pose_error > 1e-5 or fk_error > .001:
                            raise ValueError("measured pose replay does not match robot model")
                        fractions = ((upper-command.keyboard.data.joint_pos.torch[command._inst_idx, command._col_idx])/(upper-lower))[0].cpu().numpy()
                        fraction_change = float(np.max(np.abs(fractions-fractions_physics)))
                        report["max_key_fraction_change_during_display_repose"] = max(report["max_key_fraction_change_during_display_repose"], fraction_change)
                        downs = np.where(fractions >= .5, downs+1, 0)
                        ups = np.where(fractions <= .25, ups+1, 0)
                        for slot in range(len(held)):
                            edge = "down" if not held[slot] and downs[slot] >= 2 else "up" if held[slot] and ups[slot] >= 2 else None
                            if edge:
                                held[slot] = edge == "down"
                                event = {"time_s": float(next_t), "key": str(command.cfg.slot_labels[slot]).upper(), "edge": edge}
                                report["simulation_events"].append(event)
                                print(f"SIM  {event['key']} {edge.upper()} at {next_t:.2f}s", flush=True)
                        while real_index < len(events_real) and events_real[real_index]["time_s"] <= next_t:
                            e = events_real[real_index]
                            print(f"REAL {e['key_name']} {e['edge'].upper()} at {e['time_s']:.2f}s", flush=True)
                            real_index += 1
                        stream.write(json.dumps({"time_s": float(next_t),
                                     "q_recorded_rad": at(next_t, recorded_positions).tolist(),
                                     "q_replay_model_rad": expected.tolist(),
                                     "nominal_tip_xyz_b_m": expected_tip.tolist(),
                                     "after_physics_before_repose": {
                                         "q_rad": actual_physics.tolist(), "tip_xyz_w_m": tip_physics.tolist(),
                                         "joint_drift_rad": joint_drift.tolist(), "tip_drift_m": tip_drift,
                                         "key_depression_fraction": fractions_physics.tolist()},
                                     "tip_xyz_b_m": tip.tolist(), "key_depression_fraction": fractions.tolist()})+"\n")
                        if i % frame_every == 0 and (not args.headless or args.video):
                            frame = base.render()
                            if writer is not None:
                                writer.append_data(frame)
                        if args.step_sleep:
                            time.sleep(max(0, dt-(time.monotonic()-tick)))
                real_sequence = [(e["char"].upper(), e["edge"]) for e in events_real if e.get("char") and e["char"].isalpha()]
                sim_sequence = [(e["key"], e["edge"]) for e in report["simulation_events"]]
                report["physical_letter_sequence"] = real_sequence
                report["simulated_key_sequence"] = sim_sequence
                report["key_sequence_matches"] = bool(real_sequence) and real_sequence == sim_sequence
                report["status"] = "replayed"
                print(f"Replay complete. Physical/simulated key sequence matches: {report['key_sequence_matches']}", flush=True)
            finally:
                if writer is not None:
                    writer.close()
                base.close()
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        args.out.write_text(json.dumps(report, indent=2)+"\n")
        print(f"Report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
