# Included policy checkpoints

| Directory | Policy input | Target | Recorded simulation result |
| --- | --- | --- | --- |
| [newton_vision](newton_vision/) | Wrist RGB and proprioception | NEWTON | 1,016/1,024 |
| [isaac_lab_state](isaac_lab_state/) | Simulator state | ISAAC LAB 3.0 | 256/256 |

Each directory keeps its weights, agent/environment/run settings, evaluation
evidence and replay camera together. The evaluation protocols differ.

## NEWTON vision-based PPO

`newton_vision/newton_ckpt.pt` is a byte-identical copy of iteration 38,000 from
`output/vision_ppo/20261014_newton_ovrtx_height2mm_from24950/generation_000/`.
It uses the GTC Berlin actuator model, online OVRTX wrist images, full domain
randomization and shared key-height variation of up to ±2 mm.

Recorded evaluation: **1,016/1,024 completed (99.21875%)**, seed 926807.
This is the highest recorded NEWTON student success rate in the available
supervisor histories; checkpoints were evaluated with different seed batches.
The evaluation does not establish the same success rate on physical hardware.

```bash
./so101 setup
./so101 replay --model models/newton_vision/newton_ckpt.pt
```

Keep `environment.json`, `agent.json`, `run.json`, and the camera preset beside
the weights. `evaluation.json` records the original evaluation and SHA256.
`run.json` provides portable replay settings and source provenance; original
machine-specific teacher/output paths are not runtime dependencies.
The checkpoint retains actor, critic and optimizer state. Replay loads the
actor/critic without restoring the optimizer and does not require the teacher.

SHA256: `d6d78ca17dcb943d37bdc0dfdb19f4abfdd751e612831abde54c44f39d725d21`

This checkpoint is stored directly in Git, not through Git LFS. Generated models
remain ignored; only the two selected checkpoints are weight-file exceptions.

## ISAAC LAB 3.0 state-based PPO

`isaac_lab_state/isaac_lab_ckpt.pt` is the unchanged `model_5999.pt` from the
state-based phrase run. Its 256-environment strict evaluation completed
**256/256** episodes (seed 1307, no keyboard-pose randomization).
This is a different evaluation protocol from the NEWTON vision result.
The actor consumes 22 simulator-state values and produces five joint actions;
it does not consume wrist images. The RTX viewer renders the recorded motion.

```bash
./so101 replay --model models/isaac_lab_state/isaac_lab_ckpt.pt
```

Each model subfolder contains the checkpoint, JSON versions of its original YAML
agent/environment settings, replay target, camera preset, and evaluation report.
The evaluator restores these state-policy settings without adding vision-domain
randomization. The vision hardware deployment command rejects state checkpoints.

SHA256: `788c3ede8ad62ce683e9dcaeb838778213a88c3e591b3261efe1ed81c47c552a`
