# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Strict target-sequence evaluation on held-out full-domain camera episodes."""
from so101_typing.checkpoints import load_runner_checkpoint
from runtime import parser, make_config, create_env
from agent import runner_config, load_teacher
from state import write_json
from isaaclab.app import launch_simulation
from rsl_rl.runners import OnPolicyRunner
import torch
import hashlib
import time
from collections import Counter

p=parser()
p.add_argument('--episodes',type=int,default=1024)
p.add_argument('--domain-strength',type=float,default=1.)
p.add_argument('--teacher-eval',action='store_true')
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8)
cfg=make_config(args)
results=[]
failures=Counter()
started=time.time()
with launch_simulation(cfg,args):
    env=create_env(cfg)
    raw=env.unwrapped
    raw._vision_eval_word=args.target
    raw._vision_curriculum=args.domain_strength
    command=raw.command_manager.get_term('typing')
    from PIL import Image
    Image.fromarray(raw.scene['camera_wrist'].data.output['rgb'].torch[0].cpu().numpy()).save(args.output/'wrist_initial.png')
    if args.teacher_eval:
        policy,_=load_teacher(env.get_observations(),args.teacher,'cuda:0')
        path=args.teacher
    else:
        runner=OnPolicyRunner(env,runner_config(),log_dir=None,device='cuda:0')
        load_runner_checkpoint(runner, str(args.checkpoint),load_cfg={'actor':True,'critic':True})
        policy=runner.get_inference_policy()
        path=args.checkpoint
    original_reset=command.reset
    pending=torch.zeros(env.num_envs,dtype=torch.bool,device='cuda:0')
    def capture(env_ids=None):
        ids=torch.arange(env.num_envs,device='cuda:0') if env_ids is None else torch.as_tensor(env_ids,device='cuda:0')
        ids=ids[pending[ids]]
        for i in ids.tolist():
            length=int(command.target_len[i])
            exact=bool((command.typed[i,:length]==command.target[i,:length]).all())
            reasons=[name for name in raw.termination_manager.active_terms
                     if name!='success' and bool(raw.termination_manager.get_term(name)[i])]
            success=bool(command.completed[i]) and exact and not reasons and not bool(command.wrong_key[i] or command.overlap[i] or command.initially_held[i])
            results.append({'success':success,'exact':exact,'characters':int(command.character_index[i]),'failures':reasons})
            failures.update(reasons)
        pending[ids]=False
        return original_reset(env_ids)
    command.reset=capture
    with torch.inference_mode():
        while len(results)<args.episodes:
            pending[:]=False
            raw.reset()
            pending[:min(env.num_envs,args.episodes-len(results))]=True
            obs=env.get_observations()
            for step in range(int(env.max_episode_length)+2):
                obs,_,_,_=env.step(policy(obs))
                if not bool(pending.any()): break
            if bool(pending.any()):
                raise RuntimeError('Evaluation episode did not terminate within its native time budget')
            write_json(args.output/'progress.json',{'episodes':len(results),'successes':sum(r['success'] for r in results),'time':time.time()})
            print(f'EVAL {len(results)}/{args.episodes}: {sum(r["success"] for r in results)/len(results):.3%}',flush=True)
    report={'target':args.target,'episodes':len(results),'success_rate':sum(r['success'] for r in results)/len(results),
            'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'domain_strength':args.domain_strength,'seed':args.seed,'started_at':started,'finished_at':time.time(),
            'failure_counts':dict(failures),'teacher':args.teacher_eval,'results':results}
    write_json(args.output/'evaluation.json',report)
    env.close()
