# New-user workflow

Use Linux x86-64, an NVIDIA GPU, Docker with NVIDIA Container Toolkit and host
Python 3. Building uses the pinned NVIDIA Isaac Lab image and its applicable
license terms. Runtime dependencies are public, and no additional source checkout
or cluster account is required. Simulation does not receive robot devices.
See [STATUS.md](STATUS.md) for measured validation.

## Build

```bash
./so101 build
```

Choose workflow 1 below to train for your own layout. No artifact download is
required. Workflow 2 is an optional existing policy for the reference fixture.

First startup compiles Warp kernels; the launcher retains a user-owned runtime cache.
Runtime source changes require `./so101 build` again. See the README budget table
for disk space, installation estimates and the roughly 20.4-hour full training
example. Install `uv` before using `./so101 setup-hardware`.

## Workflow 1: train with your chosen configuration

Copy `configs/tasks/keyboard-pose.json` to `configs/tasks/my-keyboard.json` and
edit it for your intended physical layout. The first command below uses that
nominal pose; the other commands illustrate alternative actuator/DR choices.

```bash
./so101 train --actuator anchorbench --stage p1a \
  --task-config configs/tasks/my-keyboard.json --num-envs 64 --iterations 10
./so101 train --actuator usd --stage p1a --num-envs 64 --iterations 10

./so101 train --actuator usd --stage p1a \
  --physics-config configs/physics/mjwarp-soft-drives.json \
  --task-config configs/tasks/keyboard-domain-randomization.json --iterations 10
```

These are short setup checks. Use [TRAINING.md](TRAINING.md) for the staged
P1A-to-Transit15 recipe and training-budget limitations. Add `--plan` to inspect
the command without starting simulation or writing outputs.

Newton MJWarp supports solver substeps, iterations, line-search iterations and
tolerance. Actuator JSON supports stiffness, damping, armature, friction,
dynamic/viscous friction and effort/velocity limits. Values may be scalars or
complete six-joint maps. Unknown fields and invalid values are rejected.
Sparse VBD is not exposed in this qualified interface.

[TASK_CONFIGURATION.md](TASK_CONFIGURATION.md) covers nominal keyboard XYZ/yaw,
XY/yaw randomization and robot reset-joint noise. The benchmark physical fixture
is a reference setup. Sampled pose perturbations are not supplied to the actor.

For physical placement, use the [bird’s-eye measurement guide](../../FIXTURE_SETUP.md)
and `./so101 fixture` to create the task JSON from Q/P/Z measurements. The helper
needs only host Python 3 and works before building Docker.

## Evaluate your training

```bash
./so101 evaluate --actuator usd --stage p1a \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --num-envs 64
```

Select the same actuator and stage used for that checkpoint. Keep the complete
`params/` directory: evaluation automatically restores `public_physics.json` and
`public_task.json` beside `env.yaml`. Their hashes and nominal geometry are checked
against the saved environment. Arbitrary edits to YAML outside the supported JSON
interfaces are not a configuration-restoration mechanism.

Runs live under `output/public/`. Requests record image/source identity, command,
seed and input hashes; training saves the resolved environment and agent YAML.
Each run also saves its console log and exit code. Compare policies with matched
sample budgets, seeds, configurations and sequence banks.

## Workflow 2: optional ~19k reference policy

Read [the fixture requirements](../../FIXTURE_SETUP.md) before physical use.
The pretrained policy assumes its trained keyboard transform and fixed jaw.
Calibration does not compensate for a different keyboard placement.

```bash
./so101 download
./so101 video --actuator anchorbench --stage transit15 \
  --checkpoint .artifacts/so101-keyboard-typing-benchmark/checkpoints/mjwarp-anchorbench-19k/model_19000.pt \
  --env-config .artifacts/so101-keyboard-typing-benchmark/configs/mjwarp-anchorbench-19k.env.yaml
```

Use `evaluate` for headless episodes or `play` for an interactive Newton viewer
(X11 `DISPLAY`/`XAUTHORITY` required). The optional `--all` download includes the
supported-model benchmark and development evidence. Verify downloaded files from the
artifact directory with `sha256sum -c DOWNLOAD_SHA256SUMS`.

## Both workflows: user calibration and deployment

Follow [HARDWARE_PREPARATION.md](HARDWARE_PREPARATION.md): install the isolated CPU
LeRobot environment, run the standard SO-101 follower calibration, export a
checkpoint/geometry bundle, and run its software dry run before operator-enabled
hardware execution. The bundle uses the trained nominal map, including changed
keyboard geometry and custom physics contracts. Standard LeRobot coordinates and
the benchmark robot's extra offsets are explicit, separate choices.

This software path has been tested locally. A new-user physical reproduction on
a second SO-101 remains pending.
