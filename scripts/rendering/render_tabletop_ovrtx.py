# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Render a posed USD snapshot with OVRTX in its isolated Docker Python."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import ovrtx
import ovstage
from PIL import Image


def copy_color(products):
    """Copy to owned CPU memory and release mappings before renderer shutdown."""
    for product in products.values():
        for frame in product.frames:
            for name, buffer in frame.render_vars.items():
                if str(name).endswith('LdrColor'):
                    mapping = buffer.map(device=ovrtx.Device.CPU)
                    try:
                        view = np.from_dlpack(mapping)
                        pixels = view.copy()
                        del view
                        return pixels
                    finally:
                        mapping.unmap()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=64)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    renderer = ovrtx.Renderer(ovrtx.RendererConfig(log_file_path=str(args.output.with_suffix('.log'))))
    stage = ovstage.Stage('so101.recorded-tabletop')
    renderer.attach_ovstage(stage)
    try:
        ovstage.population.open_usd(stage, str(args.scene.resolve()), ordinal=1)
        stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        print(f'Rendering {args.frames} frames for convergence...', flush=True)
        pixels = None
        for i in range(args.frames):
            products = renderer.step(render_products={'/Render/Tabletop'}, delta_time=1 / 60, ordinal=1)
            if i == args.frames - 1:
                pixels = copy_color(products)
            del products
            if (i + 1) % 16 == 0:
                print(f'Rendered {i + 1}/{args.frames}', flush=True)
        if pixels is None or pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
            raise RuntimeError(f'No RGB image returned: {None if pixels is None else pixels.shape}')
        Image.fromarray(pixels).save(args.output)
        report = {'renderer': 'ovrtx', 'mode': 'PathTracing',
                  'versions': {name: importlib.metadata.version(name) for name in ('ovrtx', 'ovstage')},
                  'scene_sha256': hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                  'output': args.output.name, 'shape': list(pixels.shape), 'frames': args.frames,
                  'elapsed_seconds': time.monotonic() - start}
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'OVRTX IMAGE: {args.output}', flush=True)
    finally:
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


if __name__ == '__main__':
    main()
