"""Build and serve the SO-101 typing calibration browser replay."""
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re

from so101_typing.hardware.keyboard_debounce import DEFAULT_KEYBOARD_DEBOUNCE_MS, debounce_events

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'scripts/templates/typing_calibration_playback.html'


def build_assets(recording_path: Path, *, keyboard_debounce_ms: float | None = None) -> Path:
    run = recording_path.resolve().parent
    recording = json.loads(recording_path.read_text())
    names = recording['joint_names']
    samples = recording['samples'][::5]
    if not samples:
        raise ValueError('recording has no joint samples')
    if samples[-1] is not recording['samples'][-1]:
        samples.append(recording['samples'][-1])
    with (run / 'typing_calibration_timestamps.jsonl').open() as stream:
        first = json.loads(next(stream))
        last = first
        for line in stream:
            last = json.loads(line)
    start = recording['recording_start_monotonic_s']
    video_start = first['received_monotonic_ns'] / 1e9 - start
    video_end = last['received_monotonic_ns'] / 1e9 - start
    key_events = [json.loads(line) for line in (run / 'keyboard_events.jsonl').open()]
    if keyboard_debounce_ms is None:
        keyboard_debounce_ms = recording.get('keyboard_debounce_ms', DEFAULT_KEYBOARD_DEBOUNCE_MS)
    raw_events = key_events
    key_events = debounce_events(raw_events, keyboard_debounce_ms)
    (run / 'playback_keyboard_events_debounced.jsonl').write_text(
        ''.join(json.dumps(event) + '\n' for event in key_events))
    presses = []
    open_presses = {}
    for event in key_events:
        label = (event.get('char') or event['key_name'].removeprefix('KEY_')).upper()
        at = round(event['time_s'], 4)
        code = event['code']
        if event['edge'] == 'down':
            open_presses[code] = len(presses)
            presses.append([at, label, at])
        elif event['edge'] == 'up' and code in open_presses:
            presses[open_presses.pop(code)][2] = at
    joint_data = {
        'source_started_utc': recording['started_utc'],
        'duration_s': round(recording.get('trim', {}).get('duration_s', max(samples[-1]['time_s'], video_end)), 4),
        'camera_start_s': round(video_start, 4),
        'camera_end_s': round(video_end, 4),
        'video_start_s': {'arm': round(video_start, 4)},
        'joint_names': names,
        'units': ['deg'] * 5 + ['percent'],
        'samples': [[round(s['time_s'], 4),
                     [round(x, 3) for x in s['lerobot_position']],
                     [round(x, 3) for x in s['leader']['lerobot_position']]] for s in samples],
        'key_presses': presses,
        'keyboard_debounce_ms': keyboard_debounce_ms,
        'raw_key_down_count': sum(e['edge'] == 'down' for e in raw_events),
        'debounced_key_down_count': len(presses),
    }
    external = run / 'external_camera.mp4'
    if external.exists():
        stamps = [json.loads(line) for line in (run/'external_camera_timestamps.jsonl').open()]
        joint_data['video_start_s']['external'] = stamps[0]['received_monotonic_ns']/1e9-start
        joint_data['external_video_end_s'] = stamps[-1]['received_monotonic_ns']/1e9-start
    telemetry = []
    for sample in recording['telemetry']:
        raw = sample['follower']['registers_raw']
        telemetry.append([round(sample['time_s'], 4), [
            [round(raw['Present_Current'][name] * 6.5, 1) for name in names],
            [round(raw['Present_Voltage'][name] * 0.1, 1) for name in names],
            [raw['Present_Load'][name] for name in names],
            [raw['Present_Temperature'][name] for name in names],
        ]])
    telemetry_data = {
        'role': 'follower', 'joint_names': names,
        'metrics': ['Current', 'Voltage', 'Load', 'Temperature'],
        'units': ['mA', 'V', 'raw', '°C'],
        'semantics': ['Servo-reported current', 'Servo-reported voltage',
                      'Signed raw servo load, not calibrated force', 'Servo temperature'],
        'samples': telemetry,
    }
    (run / 'playback_data.json').write_text(json.dumps(joint_data, separators=(',', ':')) + '\n')
    (run / 'telemetry_data.json').write_text(json.dumps(telemetry_data, separators=(',', ':')) + '\n')
    (run / 'playback.html').write_text(TEMPLATE.read_text())
    for stem in ('typing_calibration', 'external_camera'):
        if not (run/f'{stem}.mp4').exists():
            continue
        video = run / f'{stem}.webm'
        mp4 = run / f'{stem}.mp4'
        if not video.exists() or video.stat().st_mtime < mp4.stat().st_mtime:
            import cv2
            src = cv2.VideoCapture(str(mp4))
            if not src.isOpened():
                raise RuntimeError(f'cannot open {mp4}')
            width = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = src.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'VP80'), fps, (width, height))
            if not writer.isOpened():
                src.release()
                raise RuntimeError(f'cannot write {video}')
            count = 0
            try:
                while True:
                    ok, frame = src.read()
                    if not ok:
                        break
                    writer.write(frame)
                    count += 1
            finally:
                writer.release()
                src.release()
            if not count:
                raise RuntimeError('converted WebM has no frames')
    print(f'replay={run / "playback.html"} key_presses={len(presses)}', flush=True)
    return run


def serve(run: Path, port: int) -> None:
    videos = {'/typing_calibration.webm': run/'typing_calibration.webm',
              '/external_camera.webm': run/'external_camera.webm'}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(run), **kwargs)

        def do_GET(self):
            if self.path.split('?', 1)[0] in videos:
                self.serve_video(head_only=False)
            else:
                super().do_GET()

        def do_HEAD(self):
            if self.path.split('?', 1)[0] in videos:
                self.serve_video(head_only=True)
            else:
                super().do_HEAD()

        def serve_video(self, *, head_only: bool):
            video = videos[self.path.split('?', 1)[0]]
            if not video.exists():
                self.send_error(404)
                return
            size = video.stat().st_size
            first, last = 0, size - 1
            header = self.headers.get('Range')
            if header:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
                if not match or not any(match.groups()):
                    self.send_error(416, 'Invalid byte range')
                    return
                start, end = match.groups()
                if start:
                    first = int(start)
                    last = min(int(end), size - 1) if end else size - 1
                else:
                    first = max(0, size - int(end))
                if first >= size or last < first:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.end_headers()
                    return
            remaining = last - first + 1
            self.send_response(206 if header else 200)
            self.send_header('Content-Type', 'video/webm')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(remaining))
            if header:
                self.send_header('Content-Range', f'bytes {first}-{last}/{size}')
            self.end_headers()
            if head_only:
                return
            try:
                with video.open('rb') as stream:
                    stream.seek(first)
                    while remaining:
                        block = stream.read(min(remaining, 1 << 16))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

    print(f'viewer=http://127.0.0.1:{port}/playback.html', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8771)
    parser.add_argument('--keyboard-debounce-ms', type=float, default=None,
                        help='override recorded debounce interval (default 50 ms for older takes); 0 shows raw events')
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    run = build_assets(args.recording, keyboard_debounce_ms=args.keyboard_debounce_ms)
    if not args.build_only:
        serve(run, args.port)


if __name__ == '__main__':
    main()
