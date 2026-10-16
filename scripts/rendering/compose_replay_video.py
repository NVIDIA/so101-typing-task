# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Combine matching replay cameras with event-driven key highlights and a title."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile


def extract_presses(report):
    """Require unambiguous successful single-env events; never infer from video."""
    if report['num_envs'] != 1 or report['successes'] != 1:
        raise ValueError('This presentation requires a successful single-environment replay')
    target = report['target_descriptor']['target']
    if not target or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.' for c in target):
        raise ValueError('Keyboard graphic supports uppercase A-Z, digits, space, and period')
    presses, active, previous = [], None, -1
    for event in report['event_trace_env0']:
        step = event['step']
        if not isinstance(step, int) or not previous < step < report['steps_executed']:
            raise ValueError('Event frames must be ordered within the recorded rollout')
        previous = step
        if event['wrong_key'] or event['overlap']:
            raise ValueError('Cannot label wrong-key or overlapping contacts from target-only events')
        if event['target_down'] or event['target_up']:
            index = event['character_index']
            if index != len(presses) or index >= len(target):
                raise ValueError('Key events do not follow the reported target sequence')
            key = target[index]
            if event['target_down']:
                if active is not None:
                    raise ValueError('Duplicate key-down without release')
                active = {'key': key, 'down_frame': step}
            if event['target_up']:
                if active is None or active['key'] != key or step <= active['down_frame']:
                    raise ValueError('Key-up must follow the matching key-down')
                presses.append(dict(active, up_frame=step))
                active = None
    if active is not None or ''.join(p['key'] for p in presses) != target:
        raise ValueError('Incomplete key press/release evidence')
    return target, presses


def keyboard_panel(active_key, typed, target, fonts):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGBA', (400, 242), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, 399, 241), radius=14, fill=(19, 25, 35, 239),
                           outline=(159, 172, 188, 185), width=1)
    if active_key:
        draw.text((386, 15), f'{active_key.strip() or "Space"} pressed', anchor='ra',
                  font=fonts['header'], fill=(52, 231, 202))
    extended = any(char in target for char in '0123456789.')
    if extended:
        rows = (('1234567890', 32), ('QWERTYUIOP', 64),
                ('ASDFGHJKL', 96), ('ZXCVBNM.', 128))
        key_width, key_height, stride = 31, 27, 35
        space_top, space_bottom = 164, 183
    else:
        rows = (('QWERTYUIOP', 43), ('ASDFGHJKL', 80), ('ZXCVBNM', 117))
        key_width, key_height, stride = 33, 30, 38
        space_top, space_bottom = 158, 178
    for letters, y in rows:
        start = (400 - (len(letters)*key_width + (len(letters)-1)*(stride-key_width))) // 2
        for col, letter in enumerate(letters):
            x, down = start + col*stride, letter == active_key
            draw.rounded_rectangle((x, y, x+key_width, y+key_height), radius=5,
                                   fill=(52, 231, 202) if down else (46, 55, 70),
                                   outline=(166, 255, 239) if down else (93, 106, 125), width=2 if down else 1)
            draw.text((x+key_width//2, y+key_height//2), letter, anchor='mm', font=fonts['key'],
                      fill=(9, 44, 41) if down else (238, 243, 250))
    draw.rounded_rectangle((105, space_top, 294, space_bottom), radius=4,
                           fill=(52, 231, 202) if active_key == ' ' else (46, 55, 70),
                           outline=(166, 255, 239) if active_key == ' ' else (93, 106, 125))
    draw.text((200, (space_top+space_bottom)//2), 'SPACE', anchor='mm', font=fonts['small'],
              fill=(9, 44, 41) if active_key == ' ' else (159, 172, 190))
    draw.line((14, 189, 386, 189), fill=(83, 96, 115, 200))
    draw.text((200, 214), typed or '—', anchor='mm', font=fonts['word'],
              fill=(52, 231, 202) if typed == target else (245, 248, 252))
    return canvas


def main():
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import imageio_ffmpeg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay_dir', type=Path)
    parser.add_argument('--overview', type=Path, default=Path('videos/evaluation-step-0.mp4'))
    parser.add_argument('--wrist', type=Path, default=Path('videos/wrist-camera-ovrtx.mp4'))
    parser.add_argument('--output', type=Path, default=Path('videos/single-env.mp4'))
    args = parser.parse_args()
    root = args.replay_dir.resolve()
    overview, wrist, output = [root / path for path in (args.overview, args.wrist, args.output)]
    report_path = root / 'evaluation.json'
    if output.exists() or output.with_suffix('.json').exists():
        parser.error(f'Output already exists: {output}')
    report = json.loads(report_path.read_text())
    target, presses = extract_presses(report)
    fps, frames = report['replay_export']['fps'], report['steps_executed']
    wrist_report = json.loads(wrist.with_suffix('.json').read_text())
    if (wrist_report['start_state_index'] != 1 or wrist_report['frames'] != frames
            or wrist_report['fps'] != fps):
        parser.error('Wrist render must start at state 1 and match the full evaluation frame count and fps')
    regular = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    fonts = {name: ImageFont.truetype(path, size) for name, path, size in (
        ('header', bold, 13), ('key', bold, 17), ('small', regular, 11), ('word', bold, 27))}
    title = Image.new('RGBA', (278, 66), (0, 0, 0, 0))
    draw = ImageDraw.Draw(title)
    draw.rounded_rectangle((0, 0, 277, 65), radius=12, fill=(19, 25, 35, 238), outline=(159, 172, 188, 150))
    draw.rounded_rectangle((15, 16, 18, 49), radius=2, fill=(52, 231, 202, 255))
    draw.text((34, 32), 'Single Env', anchor='lm', fill=(245, 248, 252),
              font=ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf', 36))
    readers = [imageio_ffmpeg.read_frames(str(path)) for path in (overview, wrist)]
    temporary, encoder = None, None
    try:
        metas = [next(reader) for reader in readers]
        if any(abs(meta['fps'] - fps) > 1e-6 for meta in metas):
            raise ValueError('Both camera videos must match the evaluation fps')
        width, height = metas[0]['size']
        wrist_width, wrist_height = metas[1]['size']
        if width < 960 or height < 540 or width % 2 or height % 2:
            raise ValueError('Overview must have even dimensions and be at least 960 x 540')
        scale = width / 1280
        scaled = lambda pair: tuple(round(v * scale) for v in pair)
        if wrist_width * 3 != wrist_height * 4:
            raise ValueError('Expected the Workshop wrist camera aspect ratio of 4:3')
        margin, border = round(16*scale), max(1, round(4*scale))
        title = title.resize(scaled(title.size), Image.Resampling.LANCZOS)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=output.stem+'-', suffix='.mp4', dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
        encoder = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-threads', '4',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
        cache, count = {}, 0
        for frame, pair in enumerate(itertools.zip_longest(*readers)):
            if any(pixels is None for pixels in pair) or frame >= frames:
                raise ValueError('Camera frame counts differ from each other or the evaluation')
            picture = Image.frombytes('RGB', (width, height), pair[0])
            inset = Image.frombytes('RGB', (wrist_width, wrist_height), pair[1]).resize(scaled((352, 264)), Image.Resampling.LANCZOS)
            inset = ImageOps.expand(inset, border=border, fill='white')
            picture.paste(inset, (width-inset.width-margin, margin))
            down = next((p['key'] for p in presses if p['down_frame'] <= frame < p['up_frame']), None)
            typed = ''.join(p['key'] for p in presses if p['down_frame'] <= frame)
            state = (down, typed)
            if state not in cache:
                cache[state] = keyboard_panel(down, typed, target, fonts).resize(scaled((400, 242)), Image.Resampling.LANCZOS)
            panel = cache[state]
            picture.paste(panel, (margin, margin), panel)
            picture.paste(title, (margin, height-title.height-margin), title)
            encoder.stdin.write(picture.tobytes())
            count += 1
        encoder.stdin.close()
        if encoder.wait() != 0 or count != frames:
            raise ValueError(f'Incomplete video: expected {frames} frames, received {count}')
        temporary.chmod(0o644)
        temporary.replace(output)
    finally:
        for reader in readers:
            reader.close()
        if encoder is not None and encoder.poll() is None:
            encoder.stdin.close()
            encoder.wait()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    for press in presses:
        press.update(down_video_s=press['down_frame']/fps, up_video_s=press['up_frame']/fps)
    metadata = {'overview': str(overview), 'wrist': str(wrist), 'output': output.name,
                'evaluation_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                'fps': fps, 'frames': count, 'resolution': [width, height], 'duration_seconds': count/fps,
                'final_word': target, 'events': presses, 'title': 'Single Env', 'title_font': 'Noto Sans Bold',
                'layout': 'keyboard top left, typed word beneath, wrist top right, title bottom left',
                'highlight_rule': 'target_down inclusive through target_up exclusive; no artificial hold',
                'timing': 'video frame zero is first control step; event step equals video frame index'}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'COMBINED REPLAY: {output} ({count} frames at {fps} fps)', flush=True)


if __name__ == '__main__':
    main()
