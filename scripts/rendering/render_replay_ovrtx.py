"""Render a wrist-camera movie from an exported Newton rollout in isolated OVRTX."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import ovrtx
import ovstage
from PIL import Image

from render_tabletop_ovrtx import copy_color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--start-frame', type=int, default=1, help='State index; step-triggered overview movies start after the first step')
    parser.add_argument('--max-frames', type=int, default=0, help='Optional short preview')
    args = parser.parse_args()
    if args.samples < 1 or args.max_frames < 0 or args.start_frame < 0:
        parser.error('samples must be positive and max-frames nonnegative')
    metadata = json.loads(args.scene.with_suffix('.replay.json').read_text())
    available = metadata['frames'] - args.start_frame
    if available < 1:
        parser.error('start-frame must be within the exported rollout')
    count = min(available, args.max_frames or available)
    fps = metadata['fps']
    width, height = metadata['wrist_resolution']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    encoders = list(Path('/workspace/isaaclab/_isaac_sim/kit/python/lib').glob(
        'python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-*'))
    if len(encoders) != 1:
        raise RuntimeError('Expected the project runtime imageio ffmpeg encoder')
    encoder = subprocess.Popen([
        str(encoders[0]), '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)], stdin=subprocess.PIPE)
    start = time.monotonic()
    renderer = ovrtx.Renderer(ovrtx.RendererConfig(log_file_path=str(args.output.with_suffix('.log'))))
    stage = ovstage.Stage('so101.recorded-wrist-replay')
    renderer.attach_ovstage(stage)
    try:
        ovstage.population.open_usd(stage, str(args.scene.resolve()), ordinal=1, time_code=args.start_frame / fps)
        stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        for frame in range(count):
            ordinal = frame + 1
            if frame:
                ovstage.population.update_from_usd_time(stage, ordinal=ordinal, time_code=(frame + args.start_frame) / fps)
                stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            samples = max(64, args.samples) if frame == 0 else args.samples
            for sample in range(samples):
                products = renderer.step(render_products={'/Render/Wrist'}, delta_time=1 / fps, ordinal=ordinal)
                if sample == samples - 1:
                    pixels = copy_color(products)
                del products
            if pixels is None or pixels.shape[:2] != (height, width) or not pixels.any():
                raise RuntimeError(f'No valid wrist-camera frame at index {frame}')
            rgb = np.ascontiguousarray(pixels[:, :, :3], dtype=np.uint8)
            encoder.stdin.write(rgb.tobytes())
            if frame in (0, count // 2, count - 1):
                Image.fromarray(rgb).save(args.output.with_name(f'{args.output.stem}-frame-{frame:04d}.jpg'))
            if frame == 0 or (frame + 1) % 25 == 0 or frame == count - 1:
                print(f'OVRTX wrist frame {frame + 1}/{count}', flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Video encoder failed')
        report = {'renderer': 'ovrtx', 'mode': 'PathTracing',
                  'versions': {name: importlib.metadata.version(name) for name in ('ovrtx', 'ovstage')},
                  'scene_sha256': hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                  'camera': metadata['wrist_camera'], 'frames': count, 'fps': fps,
                  'duration_seconds': count / fps, 'start_state_index': args.start_frame, 'resolution': [width, height],
                  'samples_per_frame': args.samples, 'first_frame_samples': max(64, args.samples),
                  'elapsed_seconds': time.monotonic() - start,
                  'source': 'recorded Newton world body transforms; no physics rerun'}
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'OVRTX WRIST VIDEO: {args.output}', flush=True)
    finally:
        if encoder.poll() is None:
            encoder.stdin.close()
            encoder.wait()
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


if __name__ == '__main__':
    main()
