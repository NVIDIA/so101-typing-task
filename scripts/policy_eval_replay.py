"""Build the familiar two-camera browser replay from a physical policy trial."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from scripts.typing_calibration_replay import serve
from so101_typing.hardware.so101_kinematics import GRIPPER_MODEL_MIN_RAD
from so101_typing.hardware.fixed_cartesian_policy import (
    PHYSICAL_CALIBRATED_20260718_PROFILE,
    profile_model_rad_to_lerobot_position,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'scripts/templates/typing_calibration_playback.html'
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
REGISTERS = ['Present_Current', 'Present_Voltage', 'Present_Load', 'Present_Temperature']


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def clip_camera(camera_dir: Path, out: Path, stem: str, start_ns: int, end_ns: int) -> tuple[float, float]:
    stamps = rows(camera_dir / f'{stem}_timestamps.jsonl')
    selected = [s for s in stamps if start_ns <= s['received_unix_ns'] <= end_ns]
    if not selected:
        raise ValueError(f'{stem} has no frames in the policy interval')
    source = cv2.VideoCapture(str(camera_dir / f'{stem}.mp4'))
    if not source.isOpened():
        raise RuntimeError(f'cannot read {stem}.mp4')
    source.set(cv2.CAP_PROP_POS_FRAMES, selected[0]['frame'])
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = source.get(cv2.CAP_PROP_FPS)
    target = out / f'{stem}.webm'
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'VP80'), fps, (width, height))
    if not writer.isOpened():
        source.release()
        raise RuntimeError(f'cannot write {target}')
    count = 0
    try:
        for stamp in selected:
            ok, frame = source.read()
            if not ok:
                raise RuntimeError(f'{stem}: video ended before timestamped frame {stamp["frame"]}')
            writer.write(frame)
            count += 1
    finally:
        source.release()
        writer.release()
    (out / f'{stem}_timestamps.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in selected))
    print(f'{stem}: {count} frames -> {target}', flush=True)
    return ((selected[0]['received_unix_ns'] - start_ns) / 1e9,
            (selected[-1]['received_unix_ns'] - start_ns) / 1e9)


def build(camera_dir: Path, policy_log: Path, *, keyboard_source_invalid: bool = False) -> Path:
    records = rows(policy_log)
    terminal = next(r for r in records if r.get('kind') == 'policy_terminal')
    summary = next(r for r in records if r.get('kind') == 'so101_fixed_cartesian_handoff')
    start_ns = round(terminal['policy_started_at_unix_s'] * 1e9) - 2_000_000_000
    end_ns = round(terminal['policy_terminal_at_unix_s'] * 1e9) + 2_000_000_000
    out = camera_dir.parent / 'policy_replay'
    out.mkdir(parents=True, exist_ok=True)
    arm_start, arm_end = clip_camera(camera_dir, out, 'typing_calibration', start_ns, end_ns)
    external_start, _ = clip_camera(camera_dir, out, 'external_camera', start_ns, end_ns)
    duration = (end_ns - start_ns) / 1e9
    samples = []
    for r in records:
        if r.get('kind') != 'policy_tick':
            continue
        q = np.asarray([*r['q_rad'], GRIPPER_MODEL_MIN_RAD], dtype=float)
        joint = profile_model_rad_to_lerobot_position(q, PHYSICAL_CALIBRATED_20260718_PROFILE)
        samples.append([round((r['received_unix_ns'] - start_ns) / 1e9, 4),
                        [round(float(x), 3) for x in joint]])
    if not samples:
        raise ValueError('policy log has no joint samples')
    raw_edges = [r for r in records if r.get('kind') == 'keyboard_edge_raw']
    key_edges = [r for r in records if r.get('kind') == 'keyboard_edge']
    presses = []
    held = {}
    for edge in key_edges:
        label = (edge.get('char') or edge.get('key_name', '?').removeprefix('KEY_')).upper()
        at = round((edge['received_unix_ns'] - start_ns) / 1e9, 4)
        if edge['value'] == 1:
            held[edge['code']] = len(presses)
            presses.append([at, label, at])
        elif edge['value'] == 0 and edge['code'] in held:
            presses[held.pop(edge['code'])][2] = at
    captured = {r['register'] for r in records if r.get('kind') == 'telemetry_register'}
    available = [name for name in REGISTERS if name in captured]
    if not available:
        raise ValueError('policy log has no electrical telemetry')
    latest = {}
    telemetry = []
    for r in records:
        if r.get('kind') != 'telemetry_register' or r['register'] not in available:
            continue
        latest[r['register']] = r['values_raw']
        if len(latest) != len(available):
            continue
        values = []
        for name in available:
            scale = 6.5 if name == 'Present_Current' else 0.1 if name == 'Present_Voltage' else 1
            values.append([round(latest[name][motor] * scale, 1) for motor in NAMES])
        telemetry.append([round((r['received_unix_ns'] - start_ns) / 1e9, 4), values])
    display = {'Present_Current': ('Current', 'mA'), 'Present_Voltage': ('Voltage', 'V'),
               'Present_Load': ('Load', 'raw'), 'Present_Temperature': ('Temperature', '°C')}
    data = {'source_started_utc': camera_dir.name, 'duration_s': duration,
            'camera_start_s': arm_start, 'camera_end_s': arm_end,
            'video_start_s': {'arm': arm_start, 'external': external_start},
            'joint_names': NAMES, 'units': ['deg'] * 5 + ['percent'],
            'samples': samples, 'leader_available': False, 'key_presses': presses,
            'keyboard_debounce_ms': summary.get('keyboard_debounce_ms', 50),
            'raw_key_down_count': sum(r['value'] == 1 for r in raw_edges),
            'debounced_key_down_count': len(presses),
            'target': summary.get('target'), 'stop_reason': terminal['stop_reason'],
            'keyboard_source_invalid': keyboard_source_invalid,
            'policy_start_s': round((round(terminal['policy_started_at_unix_s'] * 1e9) - start_ns) / 1e9, 3),
            'policy_end_s': round((round(terminal['policy_terminal_at_unix_s'] * 1e9) - start_ns) / 1e9, 3)}
    (out / 'playback_data.json').write_text(json.dumps(data, separators=(',', ':')) + '\n')
    (out / 'telemetry_data.json').write_text(json.dumps({
        'role': 'follower', 'joint_names': NAMES,
        'metrics': [display[name][0] for name in available],
        'units': [display[name][1] for name in available], 'samples': telemetry,
        'missing_metrics': [display[name][0] for name in REGISTERS if name not in available],
    }, separators=(',', ':')) + '\n')
    html = TEMPLATE.read_text()
    html = html.replace('SO-101 typing calibration playback', 'SO-101 physical policy evaluation replay')
    html = html.replace('Camera views, debounced keyboard events, joint angles, and follower telemetry from the same take.',
                        f'Physical policy target {summary.get("target")}; stopped: {terminal["stop_reason"]}. Cameras show two seconds before and after policy control.')
    html = html.replace('<span><i class="line" style="background:#f8a34f"></i>Leader</span>', '')
    html = html.replace('const all = data.samples.flatMap(s=>[s[1][j],s[2][j]]);',
                        'const all = data.samples.flatMap(s=>data.leader_available === false ? [s[1][j]] : [s[1][j],s[2][j]]);')
    html = html.replace('for(const [idx,color] of [[1,colors.follower],[2,colors.leader]]){',
                        'for(const [idx,color] of [[1,colors.follower],[2,colors.leader]].filter(([idx])=>idx===1 || data.leader_available !== false)){')
    html = html.replace('values[j].textContent=`Follower ${near[1][j].toFixed(1)}${unit}  ·  Leader ${near[2][j].toFixed(1)}${unit}  ·  Δ ${(near[2][j]-near[1][j]).toFixed(1)}${unit}`;',
                        'values[j].textContent=data.leader_available === false ? `Follower ${near[1][j].toFixed(1)}${unit}` : `Follower ${near[1][j].toFixed(1)}${unit}  ·  Leader ${near[2][j].toFixed(1)}${unit}  ·  Δ ${(near[2][j]-near[1][j]).toFixed(1)}${unit}`;')
    keyboard_note = ('Wrong keyboard was read; physical key presses in this trial are unknown.'
                     if keyboard_source_invalid else
                     ('Typing keyboard recorded '+''.join(p[1] for p in presses)+'.' if presses
                      else 'No key-down appears in the selected keyboard log.'))
    html = html.replace('Arm video, keyboard presses, follower motor telemetry, and joint angles are from the same recorded take.',
                        'Policy ran from '+str(data['policy_start_s'])+' to '+str(data['policy_end_s'])+' s. '+keyboard_note+' Joint angles and servo telemetry cover policy control only.')
    html = html.replace('Arm video, keyboard presses, follower motor telemetry, and joint angles cover the take.',
                        'Two cameras cover this clip; joints and servo telemetry cover policy control only. '+keyboard_note)
    if keyboard_source_invalid:
        html = html.replace('Physical keyboard input', 'Keyboard capture unavailable')
        html = html.replace('No key pressed at this time', 'Typing keyboard was not recorded')
        html = html.replace('Debounced key-down events', 'Wrong input device selected for this trial')
        html = html.replace("document.getElementById('debounce-note').textContent=`${data.keyboard_debounce_ms || 0} ms debounce · ${data.debounced_key_down_count ?? data.key_presses.length} presses (${data.raw_key_down_count ?? data.key_presses.length} raw)`;", '')
    html = html.replace('for(let m=0;m<4;m++)', 'for(let m=0;m<telemetry.metrics.length;m++)')
    html = html.replace('m===1?0.2:1', "telemetry.metrics[m]==='Voltage'?0.2:1")
    html = html.replace('m===1?1:0', "telemetry.metrics[m]==='Voltage'?1:0")
    html = html.replace('m<2?1:0', "['Current','Voltage'].includes(telemetry.metrics[m])?1:0")
    html = html.replace('<a href="recording.json">Full joint and electrical telemetry</a> · <a href="keyboard_events.jsonl">Raw keyboard events</a> · <a href="playback_keyboard_events_debounced.jsonl">Debounced events</a> · ',
                        '<a href="policy_log.jsonl">Policy log and electrical telemetry</a> · ')
    (out / 'playback.html').write_text(html)
    shutil.copy2(policy_log, out / 'policy_log.jsonl')
    print(f'replay={out / "playback.html"} policy_samples={len(samples)} telemetry_samples={len(telemetry)}', flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('camera_dir', type=Path)
    parser.add_argument('policy_log', type=Path)
    parser.add_argument('--port', type=int, default=8783)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--keyboard-source-invalid', action='store_true',
                        help='mark this trial as having read the wrong physical keyboard')
    args = parser.parse_args()
    out = build(args.camera_dir, args.policy_log, keyboard_source_invalid=args.keyboard_source_invalid)
    if args.serve:
        serve(out, args.port)


if __name__ == '__main__':
    main()
