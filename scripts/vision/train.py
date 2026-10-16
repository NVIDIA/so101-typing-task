"""Online OVRTX teacher warmup followed by asymmetric PPO; externally supervised."""
from runtime import parser, make_config, create_env
from agent import runner_config, load_teacher
from state import write_json
from checkpoint import expand_goal_vocabulary
from isaaclab.app import launch_simulation
from rsl_rl.runners import OnPolicyRunner
import torch
import time
import json
import hashlib
from pathlib import Path

p=parser()
p.add_argument('--warmup',type=int,default=300)
p.add_argument('--warmup-learning-rate',type=float,default=3e-4)
p.add_argument('--initial-domain-strength',type=float,default=0.)
p.add_argument('--iterations',type=int,default=1000000000)
p.add_argument('--learning-rate',type=float,default=1e-4)
p.add_argument('--curriculum-iterations',type=int,default=1000)
p.add_argument('--control',type=Path,required=True)
p.add_argument('--resume-full-domain',action='store_true')
p.add_argument('--height-curriculum-iterations',type=int,default=1000)
p.add_argument('--initial-height-strength',type=float,default=-1.)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=True
cfg=make_config(args)

class RequestedStop(Exception):
    pass

with launch_simulation(cfg,args):
    env=create_env(cfg)
    env.unwrapped._vision_curriculum=1. if args.resume_full_domain else args.initial_domain_strength
    if args.initial_height_strength >= 0:
        env.unwrapped._vision_height_curriculum=args.initial_height_strength
    env.unwrapped.reset()
    train_cfg=runner_config(args.seed,args.learning_rate)
    runner=OnPolicyRunner(env,train_cfg,log_dir=str(args.output),device='cuda:0')
    raw=env.unwrapped
    teacher, teacher_ckpt=load_teacher(env.get_observations(),args.teacher,'cuda:0')
    if args.checkpoint:
        source=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        source_actor=source['actor_state_dict']
        target_actor=runner.alg._raw_actor.state_dict()
        if source_actor['obs_normalizer._mean'].shape != target_actor['obs_normalizer._mean'].shape:
            if args.target != 'ISAAC LAB 3.0':
                raise ValueError('checkpoint and vision observation dimensions differ')
            adapted=expand_goal_vocabulary(source_actor,target_actor)
            runner.alg._raw_actor.load_state_dict(adapted)
            runner.alg._raw_critic.load_state_dict(teacher_ckpt['critic_state_dict'])
            runner.current_learning_iteration=int(source['iter'])+1
            write_json(args.output/'transfer.json',{'source_checkpoint':str(args.checkpoint),
                'source_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'teacher_checkpoint':str(args.teacher),'old_goal_channels':26,'new_goal_channels':38,
                'critic_initialized_from_phrase_teacher':True,
                'wrist_cnn_weights_preserved':True})
        else:
            runner.load(str(args.checkpoint),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':True})
            runner.current_learning_iteration+=1
    else:
        runner.alg._raw_critic.load_state_dict(teacher_ckpt['critic_state_dict'])
    write_json(args.output/'agent.json',train_cfg)
    write_json(args.output/'environment.json',cfg.to_dict())
    write_json(args.output/'run.json',vars(args)|{'output':str(args.output),'teacher':str(args.teacher),
        'task_config':str(args.task_config),'checkpoint':str(args.checkpoint),'control':str(args.control)})
    start_it=runner.current_learning_iteration
    original_save=runner.save
    def save(path,infos=None):
        path=Path(path)
        checkpoint=runner.alg.save()|{'iter':runner.current_learning_iteration,'infos':{'phase':phase,'agent':train_cfg}}
        tmp=path.with_suffix('.pt.tmp')
        torch.save(checkpoint,tmp); tmp.replace(path)
        write_json(args.output/'latest.json',{'checkpoint':str(path),'iteration':runner.current_learning_iteration,
             'phase':phase,'created_at':time.time(),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    runner.save=save
    def control():
        if args.control.is_file():
            request=json.loads(args.control.read_text())
            if request.get('action') in ('stop','restart'):
                save(args.output/f'model_{runner.current_learning_iteration}_final.pt')
                raise RequestedStop(request['action'])
    phase='warmup'
    try:
        # Supervised on-policy dataset aggregation, initially using teacher actions.
        # RGB and proprioception alone feed the student, including during imitation.
        optimizer=torch.optim.Adam(runner.alg._raw_actor.parameters(),lr=args.warmup_learning_rate)
        obs=env.get_observations()
        for iteration in range(args.warmup):
            loss_sum=0.
            for step in range(32):
                actor=runner.alg._raw_actor
                actor.update_normalization(obs)
                with torch.no_grad():
                    targets=teacher(obs)
                prediction=actor(obs)
                loss=torch.nn.functional.mse_loss(prediction,targets)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
                image_gradient=sum(float(p.grad.detach().norm()) for p in actor.cnns.parameters() if p.grad is not None)
                if not image_gradient > 0:
                    raise RuntimeError('No learning signal reaches the wrist image encoder')
                optimizer.step()
                with torch.no_grad():
                    # Retain teacher coverage while exposing the student to its errors.
                    probability=1.-.5*iteration/max(args.warmup-1,1)
                    action=torch.where(torch.rand((env.num_envs,1),device='cuda:0')<probability,
                                       targets,prediction.detach())
                    obs,_,_,_=env.step(action)
                loss_sum+=float(loss.detach())
            runner.current_learning_iteration=start_it+iteration
            write_json(args.output/'progress.json',{'phase':phase,'iteration':iteration,
                'mse':loss_sum/32,'image_gradient_norm':image_gradient,'time':time.time(),'domain_strength':0.})
            print(f'WARMUP {iteration} mse={loss_sum/32:.6f}',flush=True)
            if iteration%50==0: save(args.output/f'warmup_{iteration}.pt')
            control()
        save(args.output/'warmup_final.pt')
        phase='ppo'
        runner.current_learning_iteration=start_it+args.warmup
        ppo_start=runner.current_learning_iteration
        raw._vision_curriculum=1. if args.resume_full_domain else .05
        if args.initial_height_strength >= 0:
            raw._vision_height_curriculum=args.initial_height_strength
        original_log=runner.logger.log
        def log(**kw):
            original_log(**kw)
            raw._vision_curriculum=(1. if args.resume_full_domain else
                                    min(1.,.05+(kw['it']-ppo_start)/args.curriculum_iterations))
            if args.initial_height_strength >= 0:
                raw._vision_height_curriculum=min(1.,args.initial_height_strength+
                    (kw['it']-ppo_start)/args.height_curriculum_iterations)
            write_json(args.output/'progress.json',{'phase':phase,'iteration':kw['it'],
                 'time':time.time(),'domain_strength':raw._vision_curriculum,
                 'height_strength':getattr(raw,'_vision_height_curriculum',raw._vision_curriculum),
                 'loss':kw['loss_dict'],'learning_rate':kw['learning_rate']})
            control()
        runner.logger.log=log
        runner.learn(args.iterations)
    except RequestedStop as exc:
        write_json(args.output/'stopped.json',{'reason':str(exc),'time':time.time()})
    finally:
        env.close()
