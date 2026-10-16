# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Combine two successful Newton replays into plain and annotated side-by-side movies."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from compose_replay_video import extract_presses, keyboard_panel


def _reader(path, fps):
    import imageio_ffmpeg
    stream = imageio_ffmpeg.read_frames(str(path))
    metadata = next(stream)
    if abs(metadata['fps'] - fps) > 1e-6:
        raise ValueError(f'Video fps mismatch: {path}')
    return stream, tuple(metadata['size'])


def _encoder(path, size, fps):
    import imageio_ffmpeg
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f'Output already exists: {path}')
    process = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    return process


def main():
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sides = [root / 'left_isaac', root / 'right_nvidia']
    reports = [json.loads((side / 'evaluation.json').read_text()) for side in sides]
    targets, presses = zip(*(extract_presses(report) for report in reports))
    fps = reports[0]['replay_export']['fps']
    if any(report['replay_export']['fps'] != fps for report in reports):
        raise ValueError('Replay fps mismatch')
    counts = [report['steps_executed'] for report in reports]
    readers, sizes = [], []
    for side, count in zip(sides, counts):
        for name in ('overview-1080p.mp4', 'wrist-camera-ovrtx.mp4'):
            path = side / 'videos' / name
            video_report = json.loads(path.with_suffix('.json').read_text())
            if video_report['frames'] != count or video_report['fps'] != fps or video_report['start_state_index'] != 1:
                raise ValueError(f'Render does not match rollout: {path}')
            reader, size = _reader(path, fps)
            readers.append(reader)
            sizes.append(size)
    if sizes != [(1920, 1080), (640, 480)] * 2:
        raise ValueError(f'Unexpected camera resolutions: {sizes}')
    width, height = 1280, 720
    out_size = (2 * width, height)
    videos = root / 'videos'
    plain = videos / 'two-envs-plain.mp4'
    annotated = videos / 'two-envs-with-overlays.mp4'
    if plain.exists() or annotated.exists():
        parser.error('Output videos already exist')
    encoders = [_encoder(path, out_size, fps) for path in (plain, annotated)]
    regular = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    fonts = {name: ImageFont.truetype(path, size) for name, path, size in (
        ('header', bold, 13), ('key', bold, 17), ('small', regular, 11), ('word', bold, 27))}
    title_font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf', 34)
    title = Image.new('RGBA', (224, 58), (0, 0, 0, 0))
    pen = ImageDraw.Draw(title)
    pen.rounded_rectangle((0, 0, 223, 57), radius=11, fill=(19, 25, 35, 238), outline=(159, 172, 188, 150))
    pen.rounded_rectangle((13, 13, 17, 44), radius=2, fill=(146, 202, 88))
    pen.text((28, 28), 'Two Envs', anchor='lm', font=title_font, fill=(245, 248, 252))
    cache = {}
    previous = [None] * 4
    try:
        for frame in range(max(counts)):
            current = []
            for i, reader in enumerate(readers):
                side = i // 2
                if frame < counts[side]:
                    pixels = next(reader, None)
                    if pixels is None:
                        raise ValueError(f'Camera {i} ended before its rollout')
                    previous[i] = pixels
                current.append(previous[i])
            overview = [Image.frombytes('RGB', sizes[2 * i], current[2 * i]).resize((width, height), Image.Resampling.LANCZOS) for i in range(2)]
            base = Image.new('RGB', out_size)
            for side, picture in enumerate(overview):
                base.paste(picture, (side * width, 0))
            base.paste(title, (14, height - title.height - 14), title)
            encoders[0].stdin.write(base.tobytes())
            display = base.copy()
            for side in range(2):
                x = side * width
                wrist = Image.frombytes('RGB', sizes[2 * side + 1], current[2 * side + 1])
                wrist = ImageOps.expand(wrist.resize((280, 210), Image.Resampling.LANCZOS), border=3, fill='white')
                display.paste(wrist, (x + width - wrist.width - 16, 16))
                active = next((p['key'] for p in presses[side] if p['down_frame'] <= frame < p['up_frame']), None)
                typed = ''.join(p['key'] for p in presses[side] if p['down_frame'] <= frame)
                state = (side, active, typed)
                if state not in cache:
                    cache[state] = keyboard_panel(active, typed, targets[side], fonts).resize((300, 182), Image.Resampling.LANCZOS)
                panel = cache[state]
                display.paste(panel, (x + 16, 16), panel)
            encoders[1].stdin.write(display.tobytes())
            if frame == 0 or (frame + 1) % 25 == 0 or frame + 1 == max(counts):
                print(f'Combined frame {frame + 1}/{max(counts)}', flush=True)
        for encoder in encoders:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoder failed')
    finally:
        for reader in readers:
            reader.close()
        for encoder in encoders:
            if encoder.poll() is None:
                encoder.stdin.close()
                encoder.wait()
    metadata = {'title': 'Two Envs', 'fps': fps, 'frames': max(counts),
                'resolution': list(out_size), 'layout': 'ISAAC LAB 3.0 left, NVIDIA right',
                'finished_side_behavior': 'hold final captured frame',
                'robots': [{'target': target, 'successful': True, 'frames': count,
                            'evaluation_sha256': hashlib.sha256((side / 'evaluation.json').read_bytes()).hexdigest(),
                            'key_events': events} for side, target, count, events in zip(sides, targets, counts, presses)]}
    for path, overlays in ((plain, False), (annotated, True)):
        path.with_suffix('.json').write_text(json.dumps(dict(metadata, overlays=overlays), indent=2) + '\n')
        print(f'TWO ENVS VIDEO: {path}', flush=True)


if __name__ == '__main__':
    main()
