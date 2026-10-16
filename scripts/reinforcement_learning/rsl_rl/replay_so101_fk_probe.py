# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay saved position commands through Newton actuators; require real F contact."""
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks.core.dexsuite.config.so101  # noqa: F401 -- register the typing tasks
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from so101_typing.hardware.fk_probe_trajectory import TASK, digest, load, tip
from public_task import apply_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trajectory", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--task", default=TASK)
parser.add_argument("--video", action="store_true")
parser.add_argument("--step-sleep", type=float, default=0)
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining
if args.video:
    args.enable_cameras = True


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    plan = load(args.trajectory)
    letter = plan["letter"]
    if args.task != plan["task"]:
        raise ValueError("trajectory task mismatch")
    if digest(Path(os.environ["SO101_TYPING_USD"])) != plan["robot_usd_sha256"]:
        raise ValueError("robot USD differs from the planned model")
    apply_config(env_cfg, plan["task_config"])
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 1307
    # This is a slow geometry probe. Retain contact/scrape/joint safety checks,
    # but allow its approach and dwell to exceed the task-policy phase deadlines.
    env_cfg.commands.typing.phase_timeouts_s = (120., 120., 120.)
    env_cfg.episode_length_s = len(plan["rows"])*plan["control_dt_s"] + 5
    if args.device is not None:
        env_cfg.sim.device = args.device
    if args.video:
        env_cfg.video_recorder.eye = (.62, .48, .42)
        env_cfg.video_recorder.lookat = (.21, .08, .06)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "trajectory_sha256": digest(args.trajectory),
              "physics": "newton_mjwarp", "actuator": "gtc_berlin", "policy_used": False,
              "keyboard_state_written": False, "joint_zero_offset_deg": [0]*5,
              "phase_timeouts_s": [120]*3, "events": [], "max_command_error_rad": 0.,
              "max_fk_model_error_m": 0., "max_contact_force_n": 0.,
              f"max_{letter}_depression_fraction": 0., "max_other_depression_fraction": 0.}
    try:
        with launch_simulation(env_cfg, args):
            base = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
            if args.video:
                base = gym.wrappers.RecordVideo(base, video_folder=str(args.out.parent/"videos"),
                    step_trigger=lambda n: n == 0, video_length=len(plan["rows"]),
                    disable_logger=True, name_prefix=f"fk-{letter}")
            try:
                env = RslRlVecEnvWrapper(base, clip_actions=None)
                env.reset()
                command = env.unwrapped.command_manager.get_term("typing")
                action = env.unwrapped.action_manager.get_term("action")
                labels = [str(s).upper() for s in command.cfg.slot_labels]
                slot = labels.index(plan["letter"])
                command.target[0].fill_(slot)
                command.target_len[0] = command.max_len
                command.distance[0] = float(command.max_len)
                dt = float(env.unwrapped.step_dt)
                if abs(dt-plan["control_dt_s"]) > 1e-8:
                    raise ValueError("sim control timestep differs from trajectory")
                if not np.allclose(action.joint_command[0].cpu().numpy(), plan["initial_q_rad"], atol=1e-6):
                    raise ValueError("sim initial command differs from trajectory")
                lower = command.keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 0]
                upper = command.keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx, 1]
                held = False
                down_count = up_count = 0
                with (args.out.parent/"sim-trace.jsonl").open("w") as stream, torch.inference_mode():
                    for i, row in enumerate(plan["rows"]):
                        goal = torch.tensor([row["q_rad"]], device=env.unwrapped.device, dtype=torch.float32)
                        native = (goal-action.joint_command)/(action.velocity_limits_rad_s*dt)
                        if float(native.abs().max()) > 1.00001:
                            raise ValueError("saved command exceeds sim action rate limit")
                        _, _, done, _ = env.step(native)
                        error = float((goal-action.joint_command).abs().max())
                        q = command.robot.data.joint_pos.torch[0, command._arm_joint_ids].cpu().numpy()
                        sim_tip = command.fixed_tip_pos_w()[0].cpu().numpy()
                        fk_error = float(np.linalg.norm(tip(q)-sim_tip))
                        depression = ((upper-command.keyboard.data.joint_pos.torch[command._inst_idx, command._col_idx])/(upper-lower))[0]
                        f = float(depression[slot])
                        other = float(torch.cat((depression[:slot], depression[slot+1:])).max())
                        force = float(env.unwrapped.scene["robot_contact"].data.net_forces_w_history.torch.norm(dim=-1).max())
                        report["max_command_error_rad"] = max(report["max_command_error_rad"], error)
                        report["max_fk_model_error_m"] = max(report["max_fk_model_error_m"], fk_error)
                        report["max_contact_force_n"] = max(report["max_contact_force_n"], force)
                        report[f"max_{letter}_depression_fraction"] = max(
                            report[f"max_{letter}_depression_fraction"], f
                        )
                        report["max_other_depression_fraction"] = max(report["max_other_depression_fraction"], other)
                        edge = None
                        down_count = down_count+1 if f >= command.cfg.key_down_fraction else 0
                        up_count = up_count+1 if f <= command.cfg.key_up_fraction else 0
                        if not held and down_count >= command.cfg.stable_samples:
                            held, edge = True, "down"
                        elif held and up_count >= command.cfg.stable_samples:
                            held, edge = False, "up"
                        if edge:
                            event = {"step": i, "time_s": (i+1)*dt, "edge": edge, "phase": row["phase"]}
                            report["events"].append(event)
                            print(f"SIM {plan['letter']} {edge.upper()} at {(i+1)*dt:.2f}s ({row['phase']})", flush=True)
                        record = {"step": i, "phase": row["phase"], "q_command_rad": row["q_rad"],
                                  "q_measured_rad": q.tolist(), "tip_xyz_b_m": sim_tip.tolist(),
                                  f"{letter}_depression_fraction": f, "other_depression_fraction": other,
                                  "contact_force_n": force}
                        stream.write(json.dumps(record)+"\n")
                        if error > 1e-6 or fk_error > .001:
                            raise ValueError(f"command/FK parity failed at step {i}: {error=}, {fk_error=}")
                        if other >= command.cfg.key_down_fraction or force > 10 or f > 1.3:
                            raise ValueError(f"unexpected contact at step {i}: {other=}, {force=}, {f=}")
                        if edge == "down" and row["phase"] not in ("descend", "press"):
                            raise ValueError(f"{letter} pressed outside planned descent/press")
                        if bool(done.any()):
                            raise ValueError(f"sim terminated at step {i}: {env.unwrapped.termination_manager}")
                        if args.step_sleep:
                            time.sleep(args.step_sleep)
                if [e["edge"] for e in report["events"]] != ["down", "up"] or held:
                    raise ValueError(f"trajectory did not produce exactly one {letter} down followed by release")
                report["steps"] = len(plan["rows"])
                report["duration_s"] = len(plan["rows"])*dt
                report["status"] = "passed"
            finally:
                base.close()
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        args.out.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
