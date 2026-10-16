"""Export and validate the vision CNN for CPU inference; never opens hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from tensordict import TensorDict
from rsl_rl.models import CNNModel
from agent import runner_config


class VisionActor(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor.as_jit()

    def forward(self, proprio: torch.Tensor, rgb_history: torch.Tensor) -> torch.Tensor:
        return self.actor(proprio, [rgb_history])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(124)
    cfg = runner_config()['actor'].copy()
    cfg.pop('class_name')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dim = int(checkpoint['actor_state_dict']['obs_normalizer._mean'].shape[-1])
    if dim not in (45, 57):
        raise ValueError(f'Unsupported proprio dimension: {dim}')
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + ('0123456789 .' if dim == 57 else '')
    n = len(characters)
    obs = TensorDict({'proprio': torch.zeros(1,dim), 'rgb': torch.zeros(1,6,120,160)}, batch_size=[1])
    model = CNNModel(obs, {'actor':['proprio','rgb']}, 'actor', 5, **cfg)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    exported = torch.jit.script(VisionActor(model).eval())
    path = args.output/'vision_actor.pt'
    exported.save(str(path))
    loaded = torch.jit.load(str(path), map_location='cpu').eval()
    errors, timings = [], []
    with torch.inference_mode():
        for _ in range(64):
            x = torch.randn(1,dim)*.1
            x[:,15:15+n]=0; x[:,15+int(torch.randint(n,(1,)))]=1
            x[:,15+n:18+n]=0; x[:,15+n+int(torch.randint(3,(1,)))]=1
            rgb = torch.rand(1,6,120,160)
            reference=model(TensorDict({'proprio':x,'rgb':rgb}, batch_size=[1]))
            started=time.perf_counter();actual=loaded(x,rgb);timings.append(time.perf_counter()-started)
            errors.append(float((reference-actual).abs().max()))
            assert torch.isfinite(actual).all() and actual.abs().max()<=1
    assert max(errors)<1e-6, errors
    report={'schema_version':1, 'kind':'vision_cpu_export', 'hardware_motion':False,
        'source_checkpoint':str(args.checkpoint),
        'source_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'iteration':int(checkpoint['iter']), 'actor_file':path.name,
        'actor_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'proprio_dim':dim,'goal_characters':characters,'rgb_shape':[6,120,160], 'action_dim':5,'control_hz':25,
        'rgb_order':'RGB float32 [0,1]; previous frame followed by current frame',
        'proprio_order':['q minus reset (5)','joint velocity (5)','issued command minus reset (5)',
                        f'active goal onehot ({n})','phase onehot (3)','phase elapsed seconds (1)'],
        'validation_cases':64,'max_action_error':max(errors),
        'cpu_inference_ms_mean':sum(timings[4:])/len(timings[4:])*1000,
        'cpu_inference_ms_max':max(timings[4:])*1000,
        'limitations':'Numerical export parity only; hardware camera/observation/control integration still requires preflight.'}
    (args.output/'vision_actor.json').write_text(json.dumps(report,indent=2)+'\n')
    path.chmod(0o644)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
