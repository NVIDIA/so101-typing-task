# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Render captured evaluation trajectories as headless overview MP4s."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rendering'))
from render_tabletop_ovrtx import copy_color


def main():
    import imageio_ffmpeg
    import numpy as np
    import ovrtx
    import ovstage
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    args=p.parse_args()
    report=json.loads(args.evaluation.read_text())
    for episode in report['episodes']:
        scene=args.evaluation.parent/episode['replay_path']
        metadata=json.loads(scene.with_suffix('.replay.json').read_text())
        output=scene.with_name('overview.mp4')
        if output.exists():
            raise ValueError(f'Refusing to overwrite {output}')
        fps=metadata['fps']
        renderer=ovrtx.Renderer(ovrtx.RendererConfig(log_file_path=str(output.with_suffix('.log'))))
        stage=ovstage.Stage('tutorial.replay')
        renderer.attach_ovstage(stage)
        encoder=None
        try:
            ovstage.population.open_usd(stage,str(scene.resolve()),ordinal=1,time_code=0.)
            for frame in range(metadata['frames']):
                ordinal=frame+1
                if frame:
                    ovstage.population.update_from_usd_time(stage,ordinal=ordinal,time_code=frame/fps)
                stage.advance_write_floor(ordinal,ovstage.Scope.ALL).wait()
                for _ in range(32 if frame==0 else 4):
                    products=renderer.step(render_products={'/Render/Tabletop'},delta_time=1/fps,ordinal=ordinal)
                pixels=copy_color(products)
                del products
                if pixels is None:
                    raise RuntimeError(f'Missing frame {frame}')
                rgb=np.ascontiguousarray(pixels[:,:,:3],dtype=np.uint8)
                if encoder is None:
                    height,width=rgb.shape[:2]
                    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24',
                        '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
                encoder.stdin.write(rgb.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('Video encoder failed')
            print(f'VIDEO: {output}',flush=True)
        finally:
            if encoder and encoder.poll() is None:
                encoder.stdin.close()
                encoder.wait(timeout=30)
            renderer.detach_ovstage()
            stage.destroy()
            renderer.destroy()


if __name__=='__main__':
    main()
