# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Add optional event-driven wrist and keyboard overlays to one two-bench movie."""

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
        raise ValueError(f'Frame rate mismatch: {path}')
    return stream, tuple(metadata['size'])


def _encoder(path, size, fps):
    import imageio_ffmpeg
    if path.exists():
        raise ValueError(f'Video already exists: {path}')
    return subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def main():
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    scene = json.loads((root / 'two-envs.json').read_text())
    sides = [root / 'left_isaac', root / 'right_nvidia']
    reports = [json.loads((side / 'evaluation.json').read_text()) for side in sides]
    targets, presses = zip(*(extract_presses(report) for report in reports))
    fps = scene['fps']
    counts = [report['steps_executed'] for report in reports]
    frames = max(counts)
    if scene['frames'] != frames + 1 or any(report['replay_export']['fps'] != fps for report in reports):
        raise ValueError('Shared scene and evaluation timing differ')
    videos = root / 'videos'
    overview = videos / 'shared-overview.mp4'
    wrist_paths = [side / 'videos/wrist-camera-ovrtx.mp4' for side in sides]
    paths = [overview] + wrist_paths
    render_reports = [json.loads(path.with_suffix('.json').read_text()) for path in paths]
    if (render_reports[0]['frames'] != frames or render_reports[0]['start_state_index'] != 1
            or any(report['frames'] != count or report['start_state_index'] != 1
                   for report, count in zip(render_reports[1:], counts))):
        raise ValueError('Camera videos do not match the saved rollouts')
    readers, sizes = zip(*(_reader(path, fps) for path in paths))
    size = sizes[0]
    if list(size) != scene['resolution'] or sizes[1:] != ((640, 480), (640, 480)):
        raise ValueError(f'Unexpected camera resolutions: {sizes}')
    width, height = size
    half = width // 2
    plain = videos / 'two-envs-plain.mp4'
    annotated = videos / 'two-envs-with-overlays.mp4'
    encoders = [_encoder(path, size, fps) for path in (plain, annotated)]
    regular = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    fonts = {name: ImageFont.truetype(path, points) for name, path, points in (
        ('header', bold, 13), ('key', bold, 17), ('small', regular, 11), ('word', bold, 27))}
    title = Image.new('RGBA', (315, 78), (0, 0, 0, 0))
    pen = ImageDraw.Draw(title)
    pen.rounded_rectangle((0, 0, 314, 77), radius=13, fill=(19, 25, 35, 238), outline=(159, 172, 188, 150))
    pen.rounded_rectangle((19, 17, 24, 60), radius=3, fill=(118, 185, 0))
    pen.text((42, 38), 'Two Envs', anchor='lm',
             font=ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf', 47),
             fill=(245, 248, 252))
    last_wrist = [None, None]
    cache = {}
    try:
        for frame in range(frames):
            pixels = next(readers[0], None)
            if pixels is None:
                raise ValueError(f'Shared overview ended at frame {frame}')
            wrist_pixels = []
            for side in range(2):
                if frame < counts[side]:
                    current = next(readers[side + 1], None)
                    if current is None:
                        raise ValueError(f'Wrist view {side} ended at frame {frame}')
                    last_wrist[side] = current
                wrist_pixels.append(last_wrist[side])
            base = Image.frombytes('RGB', size, pixels)
            encoders[0].stdin.write(base.tobytes())
            display = base.copy()
            display.paste(title, (width - title.width - 27, height - title.height - 27), title)
            for side in range(2):
                offset = side * half
                wrist = Image.frombytes('RGB', (640, 480), wrist_pixels[side])
                wrist = ImageOps.expand(wrist.resize((420, 315), Image.Resampling.LANCZOS), border=4, fill='white')
                display.paste(wrist, (offset + half - wrist.width - 25, 25))
                active = next((p['key'] for p in presses[side] if p['down_frame'] <= frame < p['up_frame']), None)
                typed = ''.join(p['key'] for p in presses[side] if p['down_frame'] <= frame)
                state = (side, active, typed)
                if state not in cache:
                    cache[state] = keyboard_panel(active, typed, targets[side], fonts).resize((440, 266), Image.Resampling.LANCZOS)
                panel = cache[state]
                # The lime arm rises into the top-left of its half of this
                # shared camera view, so its key graphic sits to its right.
                display.paste(panel, (offset + (25 if side == 0 else 280), 25), panel)
            encoders[1].stdin.write(display.tobytes())
            if frame == 0 or (frame + 1) % 25 == 0 or frame + 1 == frames:
                print(f'Composed shared scene frame {frame + 1}/{frames}', flush=True)
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
    metadata = {'title': 'Two Envs', 'fps': fps, 'frames': frames, 'resolution': list(size),
                'scene': str(root / 'two-envs.usdc'), 'camera': '/RenderCamera',
                'layout': 'two physically adjacent tabletops in one USD scene and one camera',
                'finished_side_behavior': 'last recorded wrist frame and robot pose held',
                'targets': list(targets), 'events': presses,
                'scene_sha256': hashlib.sha256((root / 'two-envs.usdc').read_bytes()).hexdigest()}
    for path, overlays in ((plain, False), (annotated, True)):
        path.with_suffix('.json').write_text(json.dumps(dict(metadata, overlays=overlays), indent=2) + '\n')
        print(f'TWO ENVS SHARED VIDEO: {path}', flush=True)


if __name__ == '__main__':
    main()
