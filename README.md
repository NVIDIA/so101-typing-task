# SO-101 keyboard typing: setup, training and deployment

Train an SO-101 to type on a fixed keyboard using Newton, Isaac Lab and PPO,
then deploy the policy with LeRobot. The workflow covers physical setup,
recorded-motion scene calibration, training, evaluation and hardware deployment.
Start with workflow 1 for your own setup, or use workflow 2 to try a released policy
in simulation.

[Watch the typing timelapse](https://huggingface.co/datasets/nvidia/Anchor-Lab/blob/main/media/typing_abc_common7500_timelapse_hud_3x.mp4).

## Requirements

| Use | Requirements |
| --- | --- |
| Simulation and training | Linux x86-64, NVIDIA GPU, Docker with NVIDIA Container Toolkit, host Python 3, and `curl` or `wget` |
| Tested GPU | RTX 5090 with 32 GB VRAM |
| Physical typing | SO-101 follower with the supported fixed typing jaw, Logitech MX Keys keyboard, and [uv](https://docs.astral.sh/uv/getting-started/installation/) for the hardware environment |
| Disk | Built image is about 32.6 GB; budget at least 70 GB free for build layers, caches and runs |
| Example training budget | About 2 h 38 min for 500 two-letter + 2,000 six-letter updates, including evaluation/video, on RTX 5090 |

The container supplies pinned Isaac Lab, Newton and RSL-RL dependencies.
Hardware setup uses a separate LeRobot environment. See [training budgets](docs/release/TRAINING.md)
for runtime estimates and larger runs; the disk allowance is planning headroom.

## Install

This repository currently has Internal visibility in NVIDIA's GitHub enterprise.
The runtime dependencies and checkpoint downloads are public.

```bash
git clone https://github.com/NVIDIA-dev/so101-keyboard-typing-internal.git
cd so101-keyboard-typing-internal
./so101 build
```

Run all commands from the repository root. Source ZIP/tarball downloads also work.
The first build downloads dependencies, and the first simulation launch compiles
Warp kernels; later runs reuse the cache.

## Workflow 1: train for your setup

The policy uses a fixed keyboard map. Keep the real base, keyboard and typing jaw
fixed throughout calibration, training and deployment.

1. [Set up and calibrate your robot](docs/release/HARDWARE_PREPARATION.md#install-and-calibrate) with LeRobot.
2. [Measure the fixture](FIXTURE_SETUP.md) and save the starting layout as `output/my-fixture/task.json`.
3. [Record and fit the scene](docs/release/REAL2SIM.md): hand-guide key presses with torque off, fit joint offsets and the simulated keyboard pose, then check the recorded contacts. Keep `output/my-scene/task.json` and `mapping-report.json` together.
4. [Train and evaluate](docs/release/TRAINING.md) in the fitted scene: two-letter P1A, then six-letter Transit15.
5. [Qualify alignment](docs/release/ALIGNMENT.md) with reserved recordings and physical Q/G/P checks using the same mapping.
6. [Export and deploy](#both-workflows-qualify-alignment-and-deploy) the evaluated policy.

The recordings calibrate the scene; they are not PPO training demonstrations.
The fitted scene and joint mapping compensate for observed real-to-sim contact
differences. The [calibration guide](docs/release/REAL2SIM.md#why-recorded-motion-calibration-comes-before-training)
explains the encoder/model and contact-geometry limits.

After scene fitting, inspect the scene and start two-letter training:

```bash
./so101 fixture --task-config output/my-scene/task.json --out-dir output/my-scene-view
./so101 probe --task-config output/my-scene/task.json
./so101 train --actuator anchorbench --stage p1a \
  --task-config output/my-scene/task.json \
  --num-envs 4096 --iterations 500 --seed 1307
```

Follow the [training recipe](docs/release/TRAINING.md#train-for-your-configured-keyboard-pose)
to evaluate P1A and continue to six-letter typing. For an installation check
without hardware, use the [simulation-only quickstart](docs/release/QUICKSTART.md#simulation-only-startup-check).
Choose `anchorbench` or `usd` actuators; both use Newton MJWarp. See
[task and physics configuration](docs/release/TASK_CONFIGURATION.md) for custom layouts and randomization.

After the six-letter stage, evaluate the saved checkpoint and inspect a rollout:

```bash
./so101 evaluate --actuator anchorbench --stage transit15 \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --num-envs 1024
./so101 video --actuator anchorbench --stage transit15 \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --target NVIDIA
```

The launcher prints checkpoint and environment paths. Keep the full `params/`
folder so continuation, evaluation and export use the saved scene and physics.
Runs and videos are written under `output/`.

## Workflow 2: optionally try a released checkpoint

Each policy comes with its matching scene and environment. Simulation playback
needs no physical robot.

| Download | Policy and fixture | Physical evidence |
| --- | --- | --- |
| `./so101 download` | Historical `model_19000.pt`; original fitted scene and 4.44 mm keyboard mat | 89/100 six-letter trials on the original robot/fixture |
| `./so101 download --calibrated` | `model_2498.pt`; September fitted scene, no mat | One successful NEWTON trial |

The default historical checkpoint uses the [original fixture and joint mapping](docs/fixtures/reference_policy.md):

```bash
./so101 download
./so101 video --actuator anchorbench --stage transit15 \
  --checkpoint .artifacts/so101-keyboard-typing-benchmark/checkpoints/mjwarp-anchorbench-19k/model_19000.pt \
  --env-config .artifacts/so101-keyboard-typing-benchmark/configs/mjwarp-anchorbench-19k.env.yaml
```

### September calibrated-scene example

This checkpoint used the clamped robot base and keyboard directly on the same
flat tabletop, with a roughly 130 mm front gap and the right-side base edge
aligned to the right Ctrl key. See the [physical fixture and fitted simulation pose](docs/fixtures/physical_reference_20260915.md)
for measurements and bird's-eye views. The real fixture stayed fixed while the
simulation pose and joint mapping were fitted from recorded contacts.

```bash
./so101 download --calibrated
bundle=.artifacts/so101-keyboard-typing-benchmark/releases/calibrated-20260917
./so101 video --actuator anchorbench --stage transit15 \
  --checkpoint "$bundle/model_2498.pt" \
  --env-config "$bundle/params/env.yaml" --target NEWTON
```

Downloads verify pinned checksums. Keep the complete bundle, including `params/`.
`./so101 download --all` additionally fetches the supported development artifacts.
See [measured results](docs/release/RESULTS.md) for evaluation and physical evidence.

For physical use, calibrate your own robot and verify its alignment against the
chosen policy's scene. The released offsets are setup-specific. If your fitted
scene differs, use workflow 1 to train for it.

## Both workflows: qualify alignment and deploy

Use your robot's own LeRobot calibration and keep it unchanged after scene fitting.
Complete [alignment qualification](docs/release/ALIGNMENT.md) to produce
`output/my-fixture/qualified/qualification.json`. Then export your evaluated
checkpoint and run the deployment dry run:

```bash
./so101 prepare-deployment \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml \
  --robot-id my_typing_robot \
  --alignment output/my-fixture/qualified/qualification.json \
  --out-dir output/my_typing_robot
./so101 deploy output/my_typing_robot NVIDIA
```

For a released policy, substitute its checkpoint and matching environment paths.
The dry run checks the bundle and local calibration without connecting to the
robot. Follow [operator-run deployment](docs/release/HARDWARE_PREPARATION.md#operator-run-deployment)
for the physical `--execute` command, controller port and keyboard device.
The alignment guide also documents [supervised candidate tests](docs/release/ALIGNMENT.md#optional-supervised-experiments).

## Commands

| Entry point | Purpose |
| --- | --- |
| `./so101 build` | Build the simulator runtime |
| `./scripts/real2sim.sh record` / `fit` | Record hand-guided contacts and fit the scene/mapping |
| `./so101 fixture` / `scene` / `probe` | Draw, view and check a simulation scene |
| `./so101 train` / `evaluate` / `video` | Train a policy, measure performance and inspect rollouts |
| `./so101 download` | Download matched reference policies and evidence |
| `./so101 setup-hardware` / `alignment` | Prepare the hardware environment and alignment evidence |
| `./so101 prepare-deployment` / `deploy` | Export, dry-run and execute a policy |

Use `./so101 COMMAND --help` or `./scripts/real2sim.sh record --help` for options.

## Documentation

- [Quickstart](docs/release/QUICKSTART.md)
- [Fixture measurements and drawings](FIXTURE_SETUP.md) · [Scene calibration](docs/release/REAL2SIM.md)
- [Training](docs/release/TRAINING.md) · [Task and physics configuration](docs/release/TASK_CONFIGURATION.md)
- [Hardware preparation](docs/release/HARDWARE_PREPARATION.md) · [Alignment](docs/release/ALIGNMENT.md)
- [Results](docs/release/RESULTS.md) · [Artifacts](https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark)
- [Runtime provenance](runtime/README.md) · [Development checks](docs/DEVELOPMENT.md)
