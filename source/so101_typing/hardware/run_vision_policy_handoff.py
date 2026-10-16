# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the GTC Berlin wrist-vision typing actor on the physical SO-101.

All camera and keyboard readers start before the five-second motion countdown.
The same wrist frames are logged and fed to the actor. Esc stops the rollout.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .constants import ARM_JOINT_NAMES
from .fixed_cartesian_policy import (CONTROL_HZ,
                                     FixedCartesianTypingState, integrate_command)
from .deployment_bundle import validate_manifest, validate_execution_mode
from .keyboard_debounce import DEFAULT_KEYBOARD_DEBOUNCE_MS, KeyboardDebouncer
from .keyboard_events import _suppress_terminal_echo
from .policy_runtime import KeyboardMonitor, PolicyContractError, PressGate, append_jsonl, load_rest_pose
from .robot import connect_robot, prepare_closed_typing_jaw, resolve_robot_port, send_typing_q_deg
from .run_fixed_cartesian_policy_handoff import (
    ALL_MOTOR_NAMES, DEFAULT_POST_EVAL_HOME, _append_telemetry_register,
    _load_post_eval_home, _model_goal_to_arm_q_deg,
    _move_to_fixed_reset_with_retries, _read_closed_typing_q_rad,
    _return_home_and_disable, _start_error_deg, _validate_policy_keyboard_device,
    _validate_rest_pose,
)
from .so101_kinematics import FIXED_JAW_TYPING_TIP_SITE_POS, GRIPPER_MODEL_MIN_RAD, site_position
from .typing_calibration_record import TELEMETRY_REGISTERS
from .vision_policy import CameraRecorder, VisionActor, VisionTypingState, retime_cameras, vision_observation

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ACTOR_DIR = ROOT / 'output/vision_ppo/real_preflight'
DEFAULT_ENV = ROOT / 'output/vision_ppo/20261013_newton_ovrtx/generation_001/environment.json'
DEFAULT_MAPPING_DIR = ROOT / 'output/gtc_camera_top_experimental_bundle_20261013'
DEFAULT_OUTPUT = ROOT / 'output/vision_ppo/real_evals'


def load_mapping(mapping_dir: Path, environment: dict, robot_id: str, *, experimental=False):
    # A state-policy bundle supplies the measured joint/scene mapping, not the
    # vision actor. Validate its own checkpoint, environment and alignment
    # evidence before comparing the physical contract with the vision task.
    profile, bundle = validate_manifest(mapping_dir/'deployment.json',
                                        mapping_dir/'checkpoint.pt', mapping_dir/'env.yaml', robot_id)
    if not bundle.get('lerobot_calibration_sha256'):
        raise PolicyContractError('mapping must pin the follower calibration')
    validate_execution_mode(bundle, experimental=experimental)
    if bundle.get('robot_id') != robot_id or bundle.get('actuator_profile') != 'gtc_berlin':
        raise PolicyContractError('camera-top mapping robot or actuator profile mismatch')
    if bundle.get('task_contract', '').split('_task_')[-1] != environment['task_contract'].split('_task_')[-1]:
        raise PolicyContractError('camera-top mapping task geometry differs from vision training')
    action_cfg = environment['actions']['action']
    if not np.allclose(profile.q_reset_rad, action_cfg['q_reset_ref'], atol=1e-12, rtol=0):
        raise PolicyContractError('camera-top mapping reset differs from vision checkpoint')
    if not np.allclose(action_cfg['velocity_limits_rad_s'], [.30, 1.10, .75, .20, .10], atol=1e-12, rtol=0):
        raise PolicyContractError('trained joint velocity limits differ from hardware integrator')
    if abs(float(action_cfg['control_dt_s']) - 1/CONTROL_HZ) > 1e-12:
        raise PolicyContractError('trained policy timestep differs from hardware loop')
    rest_path = mapping_dir/'rest_pose.json'
    rest = load_rest_pose(rest_path)
    _validate_rest_pose(rest, profile)
    return profile, rest, bundle


def _raw_keyboard_event(out: Path, step: int, event: dict, gate: PressGate):
    overlap = gate.process(event)
    append_jsonl(out, {'kind':'keyboard_edge_raw', 'step':step, 'received_unix_ns':time.time_ns(),
                       **event, 'multiple_key_press':overlap})
    return overlap


def _is_expected_repeat(state: FixedCartesianTypingState, char: object, transition: object) -> bool:
    return (transition in ('press_before_release', 'press_before_clearance') and
            isinstance(char, str) and len(char) == 1 and char.upper() == state.active_letter())


def _prepare(args, out: Path):
    actor = VisionActor(args.actor, args.actor_metadata, args.environment)
    experimental = getattr(args, 'experimental', False)
    profile, rest, bundle = load_mapping(args.mapping_dir, actor.environment, args.id,
                                         experimental=experimental)
    state = VisionTypingState.from_environment(args.target, profile, actor.environment, actor.goal_characters)
    if args.max_duration > float(actor.environment['episode_length_s']):
        raise PolicyContractError('Requested duration exceeds trained episode limit')
    post_home = _load_post_eval_home(args.post_eval_home, args.id)
    node = _validate_policy_keyboard_device(args.device, args.post_eval_home)
    required_node = getattr(args, 'required_keyboard_node', '/dev/input/event9')
    if required_node and node != required_node:
        raise PolicyContractError(f'typing keyboard moved from required {required_node} to {node}')
    manifest = {
        'kind':'vision_physical_eval', 'target':state.target, 'robot_id':args.id,
        'actor_sha256':actor.metadata['actor_sha256'],
        'checkpoint_sha256':actor.metadata['source_checkpoint_sha256'],
        'task_contract':actor.environment['task_contract'],
        'environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest(),
        'mapping_sha256':hashlib.sha256((args.mapping_dir/'deployment.json').read_bytes()).hexdigest(),
        'mapping_status':bundle.get('mapping_selection'),
        'execution_mode':validate_execution_mode(bundle, experimental=experimental),
        'keyboard_event_node':node, 'wrist_camera':args.wrist_device,
        'external_camera':args.external_url, 'control_hz':CONTROL_HZ,
        'return_home':str(args.post_eval_home),
        'goal_characters':actor.goal_characters,'clearance_m':state.clearance_m,
        'phase_timeouts_s':state.timeouts_s,'max_duration_s':args.max_duration,
        'tolerate_expected_repeat':args.tolerate_expected_repeat, 'started_unix_ns':time.time_ns(),
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return actor, profile, rest, state, post_home, manifest


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware-profile', type=Path, help='Machine-local devices, calibration mapping and rest pose')
    p.add_argument('--interactive-reset', action='store_true', help='Prompt for reset-view confirmation in the terminal')
    p.add_argument('--actor',type=Path,default=DEFAULT_ACTOR_DIR/'vision_actor.pt')
    p.add_argument('--actor-metadata',type=Path,default=DEFAULT_ACTOR_DIR/'vision_actor.json')
    p.add_argument('--environment',type=Path,default=DEFAULT_ENV)
    p.add_argument('--mapping-dir',type=Path,default=DEFAULT_MAPPING_DIR)
    p.add_argument('--post-eval-home',type=Path,default=DEFAULT_POST_EVAL_HOME)
    p.add_argument('--target',default='NEWTON')
    p.add_argument('--id',default='my_typing_robot')
    p.add_argument('--port',default=None)
    p.add_argument('--device',default='/dev/input/by-path/pci-0000:0c:00.0-usb-0:10:1.0-event-kbd')
    p.add_argument('--wrist-device',default='/dev/video0')
    p.add_argument('--external-url',default='http://127.0.0.1:8765/stream.mjpg')
    p.add_argument('--no-external-camera',action='store_true',help='Explicitly record wrist only')
    p.add_argument('--max-duration',type=float,default=61.0)
    p.add_argument('--keyboard-debounce-ms',type=float,default=DEFAULT_KEYBOARD_DEBOUNCE_MS)
    p.add_argument('--return-to-rest-s',type=float,default=6.0)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--preflight-only',action='store_true',help='Check actor, keyboard and both cameras; no robot connection')
    p.add_argument('--experimental',action='store_true',help='Explicitly use a validated unqualified mapping for a supervised experiment')
    p.add_argument('--review-reset',action='store_true',help='Hold at trained reset for wrist-frame inspection before policy')
    p.add_argument('--tolerate-expected-repeat',action='store_true',help='Allow repeated press of current target key during release/clearance')
    args=p.parse_args()
    if args.hardware_profile:
        from .tutorial_profile import apply_profile
        apply_profile(args)
    if args.interactive_reset and not args.review_reset:
        p.error('--interactive-reset requires --review-reset')
    if args.no_external_camera:
        args.external_url = ""
    if not 0 < args.max_duration <= 150:
        p.error('--max-duration must be in (0, 150] seconds')
    if args.keyboard_debounce_ms < 0 or args.keyboard_debounce_ms > 100:
        p.error('invalid keyboard debounce')
    out_dir=args.out or DEFAULT_OUTPUT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    out_dir.mkdir(parents=True,exist_ok=False)
    log=out_dir/'events.jsonl'
    robot=None
    torque_enabled=False
    summary={'kind':'summary','status':'preflight','target':args.target.upper(),'out':str(out_dir)}
    try:
        actor,profile,rest,state,post_home,manifest=_prepare(args,out_dir)
        with ExitStack() as stack:
            cameras=stack.enter_context(CameraRecorder(out_dir/'cameras',args.wrist_device,args.external_url))
            keyboard=stack.enter_context(KeyboardMonitor(args.device))
            history,previous_rgb,frame_info=cameras.wrist_rgb_history(None)
            observation=vision_observation(np.asarray(profile.q_reset_rad),np.zeros(5),
                                           np.asarray(profile.q_reset_rad),state,0.0)
            action=actor.action(observation,history)
            append_jsonl(log,{'kind':'preflight','actor_action':action.tolist(),'wrist_frame':frame_info,
                              'cameras':cameras.sources,'keyboard_event_node':manifest['keyboard_event_node']})
            print(f'Preflight passed: actor, selected keyboard, configured camera frames; output={out_dir}',flush=True)
            if args.preflight_only:
                summary['status']='preflight_passed'
                return 0
            robot=connect_robot(port=resolve_robot_port(args.port),robot_id=args.id,enable_torque=False)
            for register in TELEMETRY_REGISTERS:
                _append_telemetry_register(robot,log,register,step=-1)
            cameras.check()
            print('Recording is active. Reset motion begins in 5 seconds; press Ctrl-C to abort.',flush=True)
            for remaining in range(5,0,-1):
                print(f'{remaining}...',flush=True)
                time.sleep(1)
                cameras.check()
            # The keyboard monitor already holds event9; no arm write precedes this point.
            robot.bus.enable_torque(list(ARM_JOINT_NAMES),num_retry=5)
            torque_enabled=True
            prepare_closed_typing_jaw(robot,gripper_deg=0.0)
            reset,previous_q=_move_to_fixed_reset_with_retries(
                robot,target_q_deg=rest,profile=profile,transition_s=3.0,epsilon_deg=.5,
                settle_timeout_s=3.0,max_attempts=3)
            append_jsonl(log,{'kind':'startup','reset':reset})
            if not reset['converged'] or previous_q is None or _start_error_deg(previous_q,profile)>.5:
                raise PolicyContractError('physical reset did not converge to trained vision start')
            if args.review_reset:
                import cv2
                _, _, reset_frame=cameras.wrist_rgb_history(None)
                with cameras.lock:
                    current_rgb=cameras.latest_wrist[2].copy()
                preview=out_dir/'reset_wrist.jpg'
                if not cv2.imwrite(str(preview),cv2.cvtColor(current_rgb,cv2.COLOR_RGB2BGR)):
                    raise RuntimeError('cannot save reset wrist preview')
                append_jsonl(log,{'kind':'reset_view_ready','preview':str(preview),'wrist_frame':reset_frame})
                flag=out_dir/'continue_after_reset.flag'
                print(f'RESET_VIEW_READY {preview}; waiting up to 90 seconds for {flag}',flush=True)
                if args.interactive_reset:
                    import os, subprocess
                    if os.environ.get('DISPLAY'):
                        subprocess.Popen(['xdg-open',str(preview.resolve())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print('Inspect reset_wrist.jpg, then type start and Enter within 90 seconds to run the policy.', flush=True)
                deadline=time.monotonic()+90
                while not flag.exists() and time.monotonic()<deadline:
                    cameras.check()
                    if args.interactive_reset:
                        import select
                        if select.select([sys.stdin],[],[],0)[0] and sys.stdin.readline().strip() == 'start':
                            flag.touch()
                    _,stop=keyboard.poll()
                    if stop:
                        raise RuntimeError('Esc pressed during reset view review')
                    time.sleep(.05)
                if not flag.exists():
                    raise RuntimeError('reset view review timed out; policy not started')
            print('Policy inference and motion begin in 5 seconds.',flush=True)
            for remaining in range(5,0,-1):
                print(f'policy {remaining}...',flush=True)
                time.sleep(1)
                cameras.check()
            command=np.asarray(profile.q_reset_rad,dtype=np.float64).copy()
            gate=PressGate()
            debouncer=KeyboardDebouncer(args.keyboard_debounce_ms)
            raw_down=[]
            processed_down=[]
            previous_sample=time.perf_counter()
            started=time.monotonic()
            state.phase_started_s=started
            previous_rgb=None
            stop_reason='max_duration'
            step=0
            with _suppress_terminal_echo(enabled=True):
                while time.monotonic()-started<args.max_duration:
                    tick=time.perf_counter()
                    cameras.check()
                    raw,stop=keyboard.poll()
                    if stop:
                        stop_reason='escape';break
                    for event in raw:
                        if _raw_keyboard_event(log,step,event,gate):stop_reason='multiple_key_press'
                        if event['value']==1 and isinstance(event.get('char'),str):raw_down.append(event['char'].upper())
                    if stop_reason!='max_duration':break
                    for event in debouncer.process(raw,now_s=time.time()):
                        char=event.get('char')
                        if not isinstance(char,str) or len(char)!=1 or char.upper() not in actor.goal_characters:
                            transition='unmapped_key'
                        elif event['value']==1:
                            processed_down.append(char.upper())
                            transition=state.key_down(char,now_s=time.monotonic())
                        else:
                            transition=state.key_up(char,now_s=time.monotonic())
                        append_jsonl(log,{'kind':'keyboard_edge','step':step,'transition':transition,
                                          'typed':state.typed,**event})
                        expected_repeat=(args.tolerate_expected_repeat and
                                         _is_expected_repeat(state,char,transition))
                        if expected_repeat:
                            state.failure = None
                            append_jsonl(log,{'kind':'expected_key_repeat','step':step,'char':char.upper(),
                                              'phase':state.phase_name,'transition':transition})
                        elif transition in ('wrong_key','press_before_release','press_before_clearance','unmapped_key'):
                            stop_reason=transition;break
                    if stop_reason!='max_duration':break
                    sample=time.perf_counter()
                    q_full=_read_closed_typing_q_rad(robot,profile)
                    q=q_full[:5]
                    dt=sample-previous_sample
                    velocity=np.zeros(5) if step==0 else (q-previous_q[:5])/dt
                    now=time.monotonic()
                    transition=state.clearance_tick(tip_z_b_m=float(site_position(q_full,site_pos=FIXED_JAW_TYPING_TIP_SITE_POS)[2]),
                                                     any_key_held=bool(keyboard.pressed_codes),now_s=now)
                    if transition:
                        append_jsonl(log,{'kind':'phase_transition','step':step,'transition':transition,'typed':state.typed})
                    if transition=='target_complete':stop_reason='target_complete';break
                    if state.phase_timed_out(now):stop_reason=f'phase_timeout:{state.phase_name}';break
                    observation=vision_observation(q,velocity,command,state,now)
                    history,previous_rgb,frame_info=cameras.wrist_rgb_history(previous_rgb)
                    action=actor.action(observation,history)
                    before=command.copy()
                    command=integrate_command(command,action)
                    if np.degrees(np.max(np.abs(command-q)))>15:
                        raise RuntimeError('joint command tracking error exceeds 15 degrees')
                    goal=np.r_[command,GRIPPER_MODEL_MIN_RAD]
                    send_typing_q_deg(robot,_model_goal_to_arm_q_deg(goal,profile),gripper_deg=0.0)
                    append_jsonl(log,{'kind':'policy_tick','step':step,'sample_monotonic_s':sample,
                                      'received_unix_ns':time.time_ns(),'active_letter':state.active_letter(),
                                      'phase':state.phase_name,'typed':state.typed,'observation':observation.tolist(),
                                      'wrist_frame':frame_info,'action':action.tolist(),'q_rad':q.tolist(),
                                      'q_command_before_rad':before.tolist(),'q_command_after_rad':command.tolist(),
                                      'joint_sample_dt_s':dt,'tick_pre_log_s':time.perf_counter()-tick})
                    _append_telemetry_register(robot,log,TELEMETRY_REGISTERS[step%len(TELEMETRY_REGISTERS)],step=step)
                    previous_q=q_full
                    previous_sample=sample
                    step+=1
                    remaining=1/CONTROL_HZ-(time.perf_counter()-tick)
                    if remaining>0:time.sleep(remaining)
            summary.update(status='completed' if state.complete else 'stopped',stop_reason=stop_reason,
                           typed=state.typed,raw_key_downs=''.join(raw_down),
                           debounced_key_downs=''.join(processed_down),steps=step,
                           exact_match=state.complete and state.typed==state.target,
                           policy_duration_s=time.monotonic()-started)
            return 0 if summary['exact_match'] else 2
    except BaseException as exc:
        summary.update(status='error',error=f'{type(exc).__name__}: {exc}')
        print(f'ERROR: {summary["error"]}',file=sys.stderr,flush=True)
        return 1
    finally:
        if robot is not None:
            try:
                if torque_enabled:
                    result=_return_home_and_disable(robot,post_home,args.return_to_rest_s)
                else:
                    robot.bus.sync_write('Torque_Enable',dict.fromkeys(ALL_MOTOR_NAMES,0),normalize=False,num_retry=3)
                    observed=robot.bus.sync_read('Torque_Enable',list(ALL_MOTOR_NAMES),normalize=False,num_retry=3)
                    result={'torque_enable_after':observed,'torque_off_verified':observed==dict.fromkeys(ALL_MOTOR_NAMES,0)}
                summary['post_eval_cleanup']=result
                if not result.get('torque_off_verified'):
                    summary['status']='cleanup_failed'
            except BaseException as exc:
                summary['post_eval_cleanup_error']=f'{type(exc).__name__}: {exc}'
                summary['status']='cleanup_failed'
            finally:
                robot.disconnect()
        if (out_dir/'cameras'/'wrist.mp4').is_file():
            try:
                summary['realtime_videos']=retime_cameras(out_dir/'cameras', names=('wrist',) if args.no_external_camera else ('wrist','external'))
            except BaseException as exc:
                summary['video_retime_error']=f'{type(exc).__name__}: {exc}'
        summary['ended_unix_ns']=time.time_ns()
        append_jsonl(log,summary)
        (out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
