# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge portable bundles to the documented LeRobot hardware runtime."""
import os
from pathlib import Path


def dispatch(args):
    from .cli import ROOT, IMAGE, selected, execute, docker, ensure_image
    from .artifacts import read
    python = ROOT / '.venv-hardware/bin/python'
    if not python.is_file() and not args.plan:
        raise ValueError('Run ./so101 setup --hardware first')
    env = os.environ | {'PYTHONPATH':str(ROOT / 'source')}
    if args.command == 'hardware' and args.operation != 'check':
        execute([python,'-m','so101_typing.hardware.tutorial_profile',args.operation,
                 '--profile',args.profile],args.plan,env=env)
        return 0
    if not args.model:
        raise ValueError('hardware check requires --model BUNDLE_OR_CHECKPOINT')
    chosen = selected(args)
    if read(chosen.parent / 'run.json').get('kind') in ('teacher', 'state'):
        raise ValueError('Hardware deployment requires a vision checkpoint')
    profile = read(args.profile)
    for key in ('mapping_dir','post_eval_home'):
        if not profile.get(key):
            raise ValueError(f'Configure {key} in {args.profile} before deployment')
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = ROOT / 'output/hardware' / stamp
    actor_dir = output / 'actor'
    ensure_image(IMAGE,args.plan)
    if not args.plan:
        output.mkdir(parents=True)
    from .cli import inside
    execute(docker(0,IMAGE,'scripts/vision/export_actor.py',[
        '--checkpoint',inside(chosen),'--output',inside(actor_dir)]),args.plan)
    target = getattr(args,'target',None) or read(chosen.parent / 'run.json').get('target','ISAAC LAB 3.0')
    command = [python,'-m','so101_typing.hardware.run_vision_policy_handoff',
               '--actor',actor_dir/'vision_actor.pt','--actor-metadata',actor_dir/'vision_actor.json',
               '--environment',chosen.parent/'environment.json','--hardware-profile',args.profile,
               '--target',target,'--max-duration',str(read(chosen.parent/'environment.json')['episode_length_s'])]
    if getattr(args, 'experimental', False):
        command += ['--experimental']
    if args.command == 'hardware':
        execute(command+['--preflight-only','--out',output/'preflight'],args.plan,env=env)
    else:
        execute(command+['--preflight-only','--out',output/'preflight'],args.plan,env=env)
        if not args.plan and input('Preflight passed. Type start to enable the reset countdown: ').strip() != 'start':
            print('No motion requested. Preflight recording retained.')
            return 0
        if args.tolerate_expected_repeat:
            command+=['--tolerate-expected-repeat']
        execute(command+['--review-reset','--interactive-reset','--out',output/'evaluation'],args.plan,env=env)
    return 0
