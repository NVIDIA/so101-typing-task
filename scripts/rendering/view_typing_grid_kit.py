# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Open an exported typing grid in Isaac Sim's interactive RTX viewport."""

import argparse
from pathlib import Path
import time

# Load the runtime's PyTorch before Kit's deprecated ML extension can select
# its older bundled copy with an incompatible NCCL dependency.
import torch  # noqa: F401
from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('scene', type=Path)
parser.add_argument('--frame', type=int, default=1)
args = parser.parse_args()
if not args.scene.is_file() or args.frame < 0:
    parser.error('scene must exist and frame must be nonnegative')

app = SimulationApp({'headless': False, 'renderer': 'RayTracedLighting',
                     'width': 1600, 'height': 900})
try:
    import omni.timeline
    import omni.usd
    from omni.kit.viewport.utility import get_active_viewport

    context = omni.usd.get_context()
    if not context.open_stage(str(args.scene.resolve())):
        raise RuntimeError(f'Could not open scene: {args.scene}')
    for _ in range(600):
        app.update()
        stage = context.get_stage()
        if stage and stage.GetPrimAtPath('/RenderCamera').IsValid():
            break
    else:
        raise RuntimeError('Timed out loading grid stage')

    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    timeline.set_current_time(args.frame / 25.0)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError('No active Kit viewport')
    viewport.camera_path = '/RenderCamera'
    for _ in range(60):
        app.update()
    print(f'INTERACTIVE GRID READY: {args.scene} at frame {args.frame}; '
          'orbit with Alt+left drag, pan with middle drag, zoom with wheel.', flush=True)
    while app.is_running():
        app.update()
        time.sleep(0.01)
finally:
    app.close()
