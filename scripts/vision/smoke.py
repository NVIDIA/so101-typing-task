# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from runtime import parser, make_config, create_env
from isaaclab.app import launch_simulation
import torch
from PIL import Image
import json

torch.set_num_threads(8)
import faulthandler
faulthandler.dump_traceback_later(90, repeat=True)
args = parser().parse_args()
args.output.mkdir(parents=True, exist_ok=True)
cfg = make_config(args)
with launch_simulation(cfg, args):
    env = create_env(cfg)
    obs = env.get_observations()
    print('VISION_OBS', {k: list(v.shape) for k, v in obs.items()}, flush=True)
    frames=[]
    for step in range(args.steps):
        obs, reward, done, info = env.step(torch.tensor([.2,0,0,0,0],device='cuda:0').expand(args.num_envs,-1))
        if step in (0,args.steps-1):
            raw=env.unwrapped.scene['camera_wrist'].data.output['rgb']
            if hasattr(raw,'torch'): raw=raw.torch
            image=raw[0,:,:,:3].cpu().numpy()
            Image.fromarray(image).save(args.output/f'wrist_{step:03d}.png')
            frames.append(float(image.std()))
    assert min(frames)>2, frames
    (args.output/'smoke.json').write_text(json.dumps({'shapes':{k:list(v.shape) for k,v in obs.items()},'pixel_std':frames}))
    env.unwrapped._vision_curriculum=1.
    env.unwrapped.reset()
    for _ in range(5): env.step(torch.zeros((args.num_envs,5),device='cuda:0'))
    print('FULL_DOMAIN_RESET_OK', flush=True)
    env.close()
