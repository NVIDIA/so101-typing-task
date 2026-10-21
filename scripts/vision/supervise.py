#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Host Docker orchestration only. All simulation and ML execute in containers."""
import argparse
import fcntl
import json
import subprocess
import time
import os
from pathlib import Path
from state import write_json, converged, stalled, adjustment

ROOT=Path(__file__).resolve().parents[2]

def docker(*args,check=True):
    return subprocess.run(['docker',*map(str,args)],check=check,text=True,capture_output=True)

def inside(path):
    return '/repo/'+str(Path(path).resolve().relative_to(ROOT))

def alive(name):
    result=docker('inspect','-f','{{.State.Running}}',name,check=False)
    return result.returncode==0 and result.stdout.strip()=='true'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--task-config',type=Path,required=True)
    p.add_argument('--image',default='so101-vision-ppo:dev')
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--eval-envs',type=int,default=64)
    p.add_argument('--eval-episodes',type=int,default=1024)
    p.add_argument('--eval-interval',type=int,default=600)
    p.add_argument('--warmup',type=int,default=300)
    p.add_argument('--warmup-learning-rate',type=float,default=3e-4)
    p.add_argument('--target',choices=('NEWTON','ISAAC LAB 3.0'),default='NEWTON')
    p.add_argument('--initial-checkpoint',type=Path)
    p.add_argument('--initial-learning-rate',type=float,default=1e-4)
    p.add_argument('--resume-full-domain',action='store_true')
    p.add_argument('--initial-height-strength',type=float,default=0.,choices=(0.,1.))
    p.add_argument('--height-curriculum-iterations',type=int,default=1000)
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    image=docker('image','inspect','--format','{{.Id}}',args.image).stdout.strip()
    name='so101-vision-'+out.name.replace('_','-')
    train_name=name+'-gpu0';eval_name=name+'-gpu1'
    state_path=out/'supervisor.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {
        'generation':0,'started_at':time.time(),'generation_started_at':time.time(),
        'evaluations':[],'adjustments':[],'image':image,'status':'starting','failures':0}
    control=out/'control.json'
    def launch(script,gpu,directory,extra):
        cache=out/f'cache_gpu{gpu}';cache.mkdir(exist_ok=True)
        renderer=out/f'renderer_gpu{gpu}';renderer.mkdir(exist_ok=True)
        for cache_name in ('nv_shadercache','texturecache','DerivedDataCache'):
            (renderer/cache_name).mkdir(exist_ok=True)
        container=train_name if gpu==0 else eval_name
        docker('rm',container,check=False)
        command=['run','--cap-drop=ALL','--security-opt=no-new-privileges','-d','--name',container,'--gpus',f'device={gpu}', '--ipc=private',
            '--shm-size=16g','--user',f'{os.getuid()}:{os.getgid()}',
            '-e','HOME=/tmp','-e','USER=so101','-e','LOGNAME=so101','--no-healthcheck','--log-opt','max-size=50m','--log-opt','max-file=4',
            '-e','NVIDIA_DRIVER_CAPABILITIES=all','-e','OMP_NUM_THREADS=8',
            '-v',f'{ROOT}:/repo:ro','-v',f'{out}:{inside(out)}',
            '-v',f'{cache}:/tmp/.cache',
            *[item for cache_name in ('nv_shadercache','texturecache','DerivedDataCache')
              for item in ('-v',f'{renderer}/{cache_name}:/opt/vision/lib/python3.12/site-packages/ovrtx/bin/cache/{cache_name}')],
            image,f'/workspace/isaaclab/scripts/vision/{script}.py','--headless',
            '--task-config',inside(args.task_config),'--teacher',inside(args.teacher),
            '--output',inside(directory),*extra]
        result=docker(*command)
        write_json(directory/'launch.json',{'argv':command,'container':container,'image':image,'time':time.time()})
        return result.stdout.strip()
    def recovery_change():
        # Read the previous trainer before start_train changes train_dir.
        directory=Path(state.get('train_dir',out))
        rate=args.initial_learning_rate
        for filename in ('progress.json','run.json'):
            path=directory/filename
            if path.exists():
                record=json.loads(path.read_text())
                if 'learning_rate' in record:
                    rate=record['learning_rate']
                    break
        return adjustment(state['evaluations'],state['generation'],current_learning_rate=rate)

    def start_train(resume=None,change=None):
        directory=out/f'generation_{state["generation"]:03d}'
        directory.mkdir(exist_ok=True)
        if control.exists(): control.unlink()
        change=change or {'learning_rate':args.initial_learning_rate,'curriculum_iterations':1000,'recovery_warmup':args.warmup}
        extra=['--num-envs',str(args.num_envs),'--seed',str(4811+state['generation']),
               '--target',args.target,'--warmup-learning-rate',str(args.warmup_learning_rate),
               '--control',inside(control),'--warmup',str(change['recovery_warmup']),
               '--learning-rate',str(change['learning_rate']),
               '--curriculum-iterations',str(change['curriculum_iterations'])]
        if resume: extra+=['--checkpoint',str(resume)]
        if args.resume_full_domain:
            extra+=['--resume-full-domain','--height-curriculum-iterations',str(args.height_curriculum_iterations),
                    '--initial-height-strength',str(args.initial_height_strength if state['generation']==0 else 1.)]
        state['train_dir']=str(directory)
        state['train_container']=train_name
        state['train_id']=launch('train',0,directory,extra)
        state['status']='training';state['generation_started_at']=time.time()
        state['image']=image
        write_json(state_path,state)
    if state['status']=='converged': return
    if not alive(train_name):
        resume=None
        previous=Path(state.get('train_dir',out))/'latest.json'
        if previous.exists():
            resume=json.loads(previous.read_text())['checkpoint']
            state['generation']+=1
        elif args.initial_checkpoint and state['generation']==0:
            resume=inside(args.initial_checkpoint)
        start_train(resume,recovery_change() if previous.exists() else None)
    pending=state.get('pending_eval');last_eval=state.get('last_eval_started',0.)
    while True:
        now=time.time()
        current=Path(state['train_dir'])
        if pending and not alive(eval_name):
            evaldir=Path(pending['output'])
            report=evaldir/'evaluation.json'
            logs=docker('logs',eval_name,check=False)
            (evaldir/'container.log').write_text(logs.stdout+logs.stderr)
            if report.exists():
                result=json.loads(report.read_text())
                episodes=result.pop('results',[])
                result['mean_completed_letters']=sum(e['characters'] for e in episodes)/max(len(episodes),1)
                state['evaluations'].append(result)
                write_json(out/'evaluation_history.json',state['evaluations'])
                if converged(state['evaluations'],episodes=args.eval_episodes,target=args.target):
                    write_json(control,{'action':'stop','reason':f'Three distinct checkpoints >98% on held-out {args.target}'})
                    state['status']='converged';state['best_checkpoint']=result['checkpoint']
                    # Wait for the trainer to finish its current update and save.
                    while alive(train_name):
                        write_json(state_path,state|{'heartbeat':time.time()});time.sleep(10)
                    write_json(state_path,state);return
            else:
                state.setdefault('evaluation_errors',[]).append({'output':str(evaldir),'time':now})
            pending=None;state['pending_eval']=None
        if not alive(train_name):
            logs=docker('logs',train_name,check=False)
            (current/'container.log').write_text(logs.stdout+logs.stderr)
            state['failures']+=1
            latest=current/'latest.json'
            # Back off, retain all evidence, and recover from the last atomic save.
            state['status']='recovering';write_json(state_path,state)
            if state['failures']>=3:
                # Repeated startup failures need code repair, not silent training claims.
                state['status']='runtime_error';write_json(state_path,state)
                raise RuntimeError('Three training crashes; see container.log')
            time.sleep(30)
            checkpoint=json.loads(latest.read_text())['checkpoint'] if latest.exists() else None
            state['generation']+=1
            start_train(checkpoint,recovery_change())
            continue
        progress_path=current/'progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        if now-progress.get('time',state['generation_started_at'])>1200:
            state.setdefault('watchdog_events',[]).append({'time':now,'reason':'No training progress for 20 minutes'})
            docker('stop','-t','30',train_name,check=False)
            write_json(state_path,state)
            continue
        latest=current/'latest.json'
        # A visible replay temporarily owns GPU 1; resume evaluations when it closes.
        state['evaluation_paused_for_viewer']=(alive(name+'-viewer') or (out/'pause_gpu1_evaluations.flag').exists())
        if latest.exists() and pending is None and not state['evaluation_paused_for_viewer'] and now-last_eval>=args.eval_interval:
            metadata=json.loads(latest.read_text())
            known={r['checkpoint_sha256'] for r in state['evaluations']}
            ready = metadata['phase']=='ppo' or metadata['iteration']>=50
            if ready and metadata['checkpoint_sha256'] not in known:
                evaldir=out/'evaluations'/f'{len(state["evaluations"]):04d}_g{state["generation"]}_{metadata["iteration"]}'
                evaldir.mkdir(parents=True,exist_ok=True)
                launch('evaluate',1,evaldir,['--num-envs',str(args.eval_envs),'--episodes',str(args.eval_episodes),
                      '--target',args.target,
                      '--seed',str(900001+len(state['evaluations'])*1031),'--checkpoint',metadata['checkpoint']])
                pending={'output':str(evaldir),'checkpoint':metadata['checkpoint']}
                state['pending_eval']=pending;last_eval=now;state['last_eval_started']=now
        # Plateau clock restarts at each generation; evaluation conditions stay fixed.
        if pending is None and stalled(state['evaluations'],state['generation_started_at']):
            latest=json.loads((current/'latest.json').read_text()) if (current/'latest.json').exists() else None
            if latest:
                write_json(control,{'action':'restart','reason':'four_hour_plateau'})
                while alive(train_name):time.sleep(10)
                state['generation']+=1
                change=recovery_change()
                change['time']=time.time();state['adjustments'].append(change)
                best=max(state['evaluations'],key=lambda r:(r['success_rate'],r.get('mean_completed_letters',0),r.get('finished_at',0)),default=latest)
                start_train(best['checkpoint'],change)
        state['heartbeat']=now
        write_json(state_path,state)
        time.sleep(10)

if __name__=='__main__':main()
