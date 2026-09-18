# New-user workflow

Start with the [README installation commands](../../README.md#install) to clone
the repository. Run all commands from its root. Replace device/robot placeholders
with your own values; use the host artifact paths printed by completed commands.

Use Linux x86-64, an NVIDIA GPU, Docker with NVIDIA Container Toolkit and host
Python 3. Building uses the pinned NVIDIA Isaac Lab image and its applicable
license terms. Runtime dependencies are public, and no additional source checkout
or cluster account is required. Simulation does not receive robot devices.
See [RESULTS.md](RESULTS.md) for the tested configuration and measured results.

## Build

```bash
./so101 build
```

Choose workflow 1 below to train for your own layout. No artifact download is
required. Workflow 2 plays either released policy in its matching reference scene.

First startup compiles Warp kernels; the launcher retains a user-owned runtime cache.
Runtime source changes require `./so101 build` again. See the README budget table
for disk space and installation estimates. The September 500 + 2,000-update
pipeline took 2 h 38 min 21 s on an RTX 5090, including evaluations/videos;
the larger 20,000-update budget is optional. Install `uv` before
using `./so101 setup-hardware`.

## Workflow 1: train with your chosen configuration

For physical typing, follow the [real-to-sim calibration guide](REAL2SIM.md):
measure the starting placement, record torque-off hand-guided key presses, fit
the keyboard pose and joint offsets, and check first contacts with resets between
keys. Train using its saved `output/my-scene/task.json`. The recordings calibrate
the scene; they are not policy training demonstrations. Separate reserved replay
and physical contact checks qualify the mapping for deployment afterward.
Complete your robot's LeRobot calibration before recording and keep it unchanged
through deployment. The short commands below are software checks that can run
before hardware is ready; they do not create a calibrated physical scene.

### Simulation-only startup check

Use a separate example task and output folder so this check cannot replace the
measured `my-fixture` or fitted `my-scene` used for physical typing:

```bash
mkdir -p output/sim-example
cp configs/tasks/keyboard-pose.json output/sim-example/task.json
# Optionally edit this simulated pose, then inspect it.
./so101 fixture --task-config output/sim-example/task.json --out-dir output/sim-example-view
./so101 probe --task-config output/sim-example/task.json
./so101 train --actuator anchorbench --stage p1a \
  --task-config output/sim-example/task.json --num-envs 64 --iterations 10
```

These are short setup checks. Use [TRAINING.md](TRAINING.md) for the staged
P1A-to-Transit15 recipe and training-budget limitations. Add `--plan` to simulation
commands to inspect the launch without starting simulation or writing outputs.

To inspect the scene in the Newton viewer, run
`./so101 scene --task-config output/sim-example/task.json` from an X11 desktop
terminal. It stays open until you close the window and does not run a policy.

Newton MJWarp supports solver substeps, iterations, line-search iterations and
tolerance. Actuator JSON supports stiffness, damping, armature, friction,
dynamic/viscous friction and effort/velocity limits. Values may be scalars or
complete six-joint maps. Unknown fields and invalid values are rejected.
Sparse VBD is not exposed in this qualified interface.

[TASK_CONFIGURATION.md](TASK_CONFIGURATION.md) covers nominal keyboard XYZ/yaw,
XY/yaw randomization and robot reset-joint noise. The benchmark physical fixture
is a reference setup. Sampled pose perturbations are not supplied to the actor.

Use the [model layout guide](../../FIXTURE_SETUP.md) and `./so101 fixture` to draw
the scene or fit a candidate task JSON from Q/P/Z measurements. It needs only host
Python 3. Its fitted reference geometry is not a verified physical setup template;
robot joint alignment and physical contacts need independent checks.

## Evaluate your training

```bash
./so101 evaluate --actuator anchorbench --stage p1a \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --num-envs 64
```

Select the same actuator and stage used for that checkpoint. Keep the complete
`params/` directory: evaluation automatically restores `public_physics.json` and
`public_task.json` beside `env.yaml`. Their hashes and nominal geometry are checked
against the saved environment. Arbitrary edits to YAML outside the supported JSON
interfaces are not a configuration-restoration mechanism.
For a Transit15 checkpoint, use `--stage transit15` and a six-letter video target;
P1A videos use two letters such as `HE`. A ten-update startup checkpoint is not
expected to type successfully.

Runs live under `output/public/`. Requests record image/source identity, command,
seed and input hashes; training saves the resolved environment and agent YAML.
Each run also saves its console log and exit code. Compare policies with matched
sample budgets, seeds, configurations and sequence banks.

## Workflow 2: optional reference policies

Read [the fixture requirements](../../FIXTURE_SETUP.md) before physical use.
The default ~19k policy assumes its trained keyboard transform and fixed jaw.
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

The additional [September calibrated-scene example](../../README.md#september-calibrated-scene-example)
is selected with `./so101 download --calibrated`. It includes `model_2498.pt`, its
complete `params/` folder and sanitized evidence for one physical NEWTON trial.
Its robot-specific mapping must not be substituted for your own scene calibration.

## Both workflows: user calibration and deployment

Workflow 1 already installed the hardware environment and calibrated the robot
before recording; reuse that calibration. For workflow 2 physical use, complete
[hardware preparation](HARDWARE_PREPARATION.md) now. Then follow the
[alignment qualification](ALIGNMENT.md) for the policy's frozen scene. Export a bundle
with `--alignment qualification.json` and run its software dry run before
operator-enabled hardware execution. The bundle checks the trained nominal map,
including changed keyboard geometry, against the qualified scene. Keep the same
robot calibration throughout validation and deployment.

For software-only exports without alignment evidence, only the exact historical
19k checkpoint/environment pair selects its mapping automatically. The September
checkpoint and new training require an explicit mapping. The downloaded reference
mapping is not a calibration for a different robot. If fitting your setup changes
the trained keyboard scene, return to workflow 1 and train for that scene.
