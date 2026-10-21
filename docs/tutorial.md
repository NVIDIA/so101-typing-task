# SO-101 tutorial: from simulation to a recorded hardware trial

Use the `gtc_berlin` branch. Commands execute simulation in Docker and hardware
access in the pinned `.venv-hardware` environment. Legacy `./so101 train --stage`
and deployment-bundle commands remain available; the tutorial routes use
`train teacher`, `train vision`, `evaluate teacher`, and `evaluate vision`.

## 1. Get a working runtime

Run `./so101 setup`, then `./so101 debug --viewer`. The debug command checks Docker,
GPU discovery, container CUDA access, OVRTX/USD imports, the Workshop asset,
free disk space, and X11 credentials when requested. Its JSON output identifies
which check failed. Import checks do not prove rendering or policy success; use
the smoke workflow below to exercise simulation and replay.

`setup --hardware` additionally invokes the existing uv installer using
`runtime/hardware.lock`. It does not connect to or move a robot. Avoid creating
another host simulation environment: the container pins the incompatible USD
runtimes and the replay launcher starts Kit in a separate process.

## 2. Run a short end-to-end installation check

```bash
./so101 train teacher --config configs/tutorial/smoke.json \
  --output output/smoke-teacher
./so101 evaluate teacher --config configs/tutorial/smoke.json \
  --run output/smoke-teacher --suite random --max-steps 2
./so101 train vision --config configs/tutorial/smoke.json \
  --teacher output/smoke-teacher --checkpoint latest --output output/smoke-vision
./so101 evaluate vision --config configs/tutorial/smoke.json \
  --run output/smoke-vision --max-steps 2
./so101 replay --run output/smoke-vision --checkpoint latest --max-steps 2
```

Expected milestones: PPO logs report two updates; model files and JSON sidecars
appear in each output directory; evaluation creates `evaluation.json`; replay
opens a movable RTX viewport after exporting the episode. Truncated smoke
rollouts count as failures and do not update `best_*` checkpoint selections.
The initial simulator/renderer startup can take minutes; the two-update budget
is intentionally too short to learn the task.

For a desktop-free artifact, add `--video` to replay. The command exports exact
Newton motion and renders `episode_001/overview.mp4`. This is the same captured
episode, not another physics rollout. `--no-viewer` exports the USD without
rendering video. Full replay defaults to one episode and its native time budget.

## 3. Train a useful teacher and vision policy

Copy `configs/tutorial/default.json` to `output/my-config.json`. It is JSON so
host orchestration needs no Python packages. Set `task_config` to a measured
scene JSON, then use the same config for both training stages:

```bash
./so101 train teacher --config output/my-config.json --output output/my-teacher
./so101 evaluate teacher --config output/my-config.json --run output/my-teacher \
  --checkpoint model_4999.pt --suite random
./so101 train vision --config output/my-config.json --teacher output/my-teacher \
  --checkpoint best --output output/my-vision
./so101 evaluate vision --config output/my-config.json --run output/my-vision \
  --checkpoint latest --suite phrase
./so101 evaluate vision --config output/my-config.json --run output/my-vision \
  --checkpoint latest --suite random
./so101 replay --run output/my-vision --checkpoint best
```

Replace checkpoint names with actual saved files. Teacher and vision actors
share five actions, but their inputs differ: the teacher receives privileged
state; the vision actor receives the live wrist image history and proprioception.
The teacher initialization remains frozen during imitation. The separate critic
can access privileged state during PPO. Changing target length/vocabulary beyond
the supported 13/38 preset requires an environment/model change; the CLI rejects
unsupported combinations rather than silently ignoring them.

Evaluation restores the selected checkpoint's saved environment and architecture.
The tutorial config sets evaluation batch size, seed and randomization strength.
`--suite random` samples held-out random strings; `--suite phrase` repeats the
fixed demonstration phrase. `best_random.json` and `best_phrase.json` record
checkpoint hashes and evaluation protocols. They are independent rankings.
A different protocol cannot silently overwrite an existing selection. A run
with a broader training objective may initially lose fixed-phrase performance;
evaluate both suites before selecting a deployment model.

A complete vision checkpoint directory includes `model_N.pt`, `environment.json`,
`agent.json`, and `run.json`. Keep these together. Teacher runs also provide
`params/env.yaml` and `params/public_task.json` for the existing calibration and
mapping tools. No fixed training budget guarantees convergence.

## 4. Prepare your hardware

Follow [motor/scene preparation](release/REAL2SIM.md),
[manual keyboard measurements](release/TASK_CONFIGURATION.md#measure-your-fixture),
or the [agent-assisted keyboard fitting workflow](keyboard_calibration_astra.md).
Motor calibration, keyboard placement and encoder-to-simulator mapping are
separate steps. The agent-assisted guide is a workflow, not a one-command
multimodal optimizer. Preserve the measured actuator model and calibrated wrist
orientation when fitting keyboard placement.

Run `./so101 hardware configure --profile local/robot.json`. Select a stable
`/dev/serial/by-id` port, wrist/external camera devices, and the keyboard actually
pressed by the robot. Confirm a key event and inspect the camera previews.
Nothing moves during this wizard. Camera frame reads require working USB access.
Prefer `/dev/v4l/by-id` and `/dev/input/by-id` entries; by-path aliases also work
when devices have no unique serial number.

Run `hardware calibrate` to invoke LeRobot's interactive follower motor
calibration. Prepare the matching deployment mapping with the existing
`./so101 alignment` and `./so101 prepare-deployment` tools described in the
[alignment guide](release/REAL2SIM.md), then set the resulting directory as
`mapping_dir` in the profile. It must contain `deployment.json` and `rest_pose.json`
whose robot ID, GTC Berlin actuator profile, keyboard geometry and reset agree
with the checkpoint. An arbitrary simulation checkpoint cannot infer this
physical mapping. Existing qualified mappings are accepted; experimental
unqualified mappings retain the current runner's explicit reset-review workflow.

Place the follower in a suitable return pose with torque off, then run
`./so101 hardware save-rest --profile local/robot.json`. This only reads position
and torque registers, verifies all six motors are already off, and saves a
calibration-bound rest pose. It does not move the arm or disable an existing
controller. The optional `required_keyboard_node` field can pin a local event
node (for example event9); by default stable identity is used instead.

## 5. Check and deploy

```bash
./so101 hardware check --profile local/robot.json --model output/my-vision
./so101 deploy --profile local/robot.json --model output/my-vision
```

The check exports the actor with numerical parity validation, checks checkpoint
compatibility, validates the keyboard identity and rest calibration, and records
fresh camera frames without opening the robot connection. Deployment repeats
preflight before offering the start prompt. Inspect the reset wrist image and
confirm in the terminal when ready. Each motion stage has a five-second countdown.
The mapping directory must contain its complete prepared deployment bundle,
including `checkpoint.pt`, `env.yaml`, and alignment evidence. Preflight rechecks
the follower calibration and evidence hashes. Qualified alignment is required
by default. For a supervised candidate-mapping experiment, explicitly pass
`--experimental` to both `hardware check` and `deploy`; calibration and integrity
checks still apply.

Esc on the selected keyboard or Ctrl-C stops execution. The reset review expires
after 90 seconds, returning home and disabling torque.

Recordings go to `output/hardware/<timestamp>/evaluation/`. They include raw and
retimed videos, timestamps, keyboard edges, policy inputs/actions, and available
servo telemetry. `summary.json` reports the typed sequence, stop reason, rest-pose
convergence and all-six-motor torque-off verification. An external camera is
optional; the wrist camera is required. Duplicate handling stays strict unless
`--tolerate-expected-repeat` is supplied explicitly for a diagnostic trial.

## Troubleshooting

| Symptom | Next action |
| --- | --- |
| Docker cannot access GPU | Check host driver and NVIDIA Container Toolkit; rerun debug |
| Missing runtime image | Run setup; release manifests must contain valid image digests |
| OVRTX/USD import error | Use the provided images; keep Kit replay separate from inference |
| Long initial pause | Watch simulator logs; first startup compiles kernels and initializes RTX |
| CUDA out of memory | Reduce training.num_envs; run the smoke preset first |
| No X11 display | Use replay --video or --no-viewer on a server |
| Missing best selection | Evaluate a checkpoint first, or explicitly select latest/model_N.pt |
| Missing checkpoint JSON files | Restore the complete bundle/run directory |
| Camera missing after reconnect | Reconfigure stable device identity and inspect previews |
| Wrong keyboard selected | Rerun hardware configure and press a key on the intended keyboard |
| Mapping/reset mismatch | Revisit scene/joint calibration; do not substitute another robot's mapping |
| Simulation success but hardware failure | Inspect synchronized wrist/external video, key events and telemetry |
| Return pose or torque check failed | Read summary.json before another trial; the runner reports cleanup failure |

The current simulation results and hardware trials are separate evidence. Recent
hardware tests can fail after a short prefix despite high fixed-phrase simulation
success. This tutorial demonstrates a reproducible workflow, not a claim of
reliable deployment across uncalibrated benches.


## Outputs

Each run saves:

- `model_initial.pt`: random vision actor and teacher-initialized critic.
- `warmup_final.pt`: student actor after imitation; PPO starts from these weights.
- `warmup.json`: per-update imitation loss and wrist-CNN gradient measurements.
- `model_*.pt`: periodic and final runner checkpoints with optimizer state.
- `agent.json`, `environment.json`, `run.json`: architecture, task and run settings.
- TensorBoard event files: reward, loss and episode metrics.

The implementation is [train_simple.py](../scripts/vision/train_simple.py);
[train_simple.sh](../scripts/vision/train_simple.sh) only builds and runs Docker.
Generated checkpoints, recordings, telemetry logs and reports remain under
ignored `output/`. This revision does not track model weights. Provenance for
the withheld NEWTON and ISAAC LAB 3.0 checkpoints is in
[models/README.md](../models/README.md).

The original state-policy setup is covered in [training documentation](release/TRAINING.md) and
[hardware preparation](release/HARDWARE_PREPARATION.md).
