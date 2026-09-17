# SO-101 keyboard typing: train and deploy

Train an SO-101 typing policy in Newton for your keyboard placement, evaluate it,
and export it for deployment with your robot's calibration and a verified joint
mapping. An established AnchorBench ~19k checkpoint is available for simulation
and **physical reference-fixture experiments**. Complete physical reproduction
on a new user's setup remains to be qualified.

[Watch the typing timelapse](https://huggingface.co/datasets/nvidia/Anchor-Lab/blob/main/media/typing_abc_common7500_timelapse_hud_3x.mp4)
in the [Anchor-Lab media collection](https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/main/media).

## Install

Requirements: Linux x86-64, an NVIDIA GPU, Docker with NVIDIA Container Toolkit,
host Python 3, and `curl` or `wget` for artifact downloads. Hardware setup
additionally requires `uv`; install it before
`./so101 setup-hardware` ([uv installation](https://docs.astral.sh/uv/getting-started/installation/)).
The pinned container supplies Isaac Lab, Newton and RSL-RL.
The tested GPU is an RTX 5090 with 32 GB; other GPUs and minimum requirements
remain to be qualified. Runtime dependencies and benchmark artifacts are publicly
available. No additional source checkout or cluster account is required.
The NVIDIA-dev repository is currently private, so cloning it requires repository
access. The runtime and artifact downloads themselves do not require that access.

| Budget | Planning guidance |
| --- | --- |
| Disk | Built image: about 32.6 GB (30.4 GiB). Budget at least 70 GB free for image/build layers, caches and runs; this is headroom guidance, not a measured peak. |
| First install | Download/build and first kernel compilation depend on bandwidth, CPU and cache state; allow tens of minutes to hours. This is an estimate, not a measured cold-install time. |
| Optional ~19k playback | No training required; evaluation follows installation and kernel startup. |
| Example 4,000 + 16,000 updates, 4096 environments | Roughly 20.4 hours of training-loop time on an RTX 5090, extrapolated from 36.72 s / 10 updates. Excludes startup and evaluation; not a measured full run. |

See [training budgets](docs/release/TRAINING.md) for early evaluation and timing logs.
ZIP/tarball source downloads are supported: run commands from the extracted folder;
provenance records content hashes with no Git revision when `.git` is absent.

```bash
git clone https://github.com/NVIDIA-dev/so101-keyboard-typing.git
cd so101-keyboard-typing
./so101 build
```

## Workflow 1: train for your setup

For physical typing, establish and validate the scene **before task training**:
measure keyboard placement from datums on the robot base, freeze that placement,
record torque-off hand-guided key presses, then replay the measured trajectories
in Newton. These recordings are **scene/mapping validation data, not policy
training demonstrations**. If needed, fit shoulder-pan/lift/elbow/wrist-flex
offsets with the keyboard fixed and validate on separate recordings before
adopting them. Follow the [pre-training plan](FIXTURE_SETUP.md#before-policy-training-verify-the-real-scene),
including policy-free physical contact checks.

The nominal keyboard position/orientation is relative to the simulated robot.
Your physical fixture must match that trained geometry at deployment.
The task JSON supports nominal XYZ/full orientation and optional XY/yaw placement randomization.
Use [task configuration](docs/release/TASK_CONFIGURATION.md) to choose a simulated
layout. The [fixture helper](FIXTURE_SETUP.md) draws that model and can fit a
candidate layout from measurements; it does not establish real-to-sim alignment.

```bash
cp configs/tasks/keyboard-pose.json configs/tasks/my-keyboard.json
# Edit the nominal simulated pose, then draw and inspect it.
./so101 fixture --task-config configs/tasks/my-keyboard.json --out-dir output/my-fixture
./so101 probe --task-config configs/tasks/my-keyboard.json

# Startup check; this budget is not intended to produce a deployable policy.
./so101 train --actuator anchorbench --stage p1a \
  --task-config configs/tasks/my-keyboard.json --num-envs 64 --iterations 10
```

The short startup check above can run before hardware is available; it does not
qualify the scene for physical typing. Before task training, check the robot's
joint mapping and keyboard contacts against this scene. Changing the task JSON
does not measure either on hardware.
Use the [shared F contact probe](FIXTURE_SETUP.md#shared-f-contact-probe) to generate
one hover–press–lift trajectory, verify its contact in Newton, and replay the same
joint commands on your calibrated robot. An FK calculation alone does not prove
physical alignment; the physical F press and release are the comparison.

Continue with the [P1A-to-six-letter training recipe](docs/release/TRAINING.md).
Choose `anchorbench` or `usd` actuators. Optional `--physics-config`
JSON configures Newton MJWarp and actuator parameters. Sparse VBD is not exposed
by this interface. No pretrained download is needed to train from scratch.

At the end of training, the launcher prints the host paths of the latest saved
checkpoint and its matching environment. Use those paths in the commands below,
with the stage used for that checkpoint:

```bash
./so101 evaluate --actuator anchorbench --stage transit15 \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --num-envs 1024
./so101 video --actuator anchorbench --stage transit15 \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --target NVIDIA
```

Keep the full `params/` folder. Resume and evaluation restore saved task/physics
JSONs; export uses the trained nominal key map and joint reference. Use `p1a` and
a two-letter target for P1A checkpoints. Runs and videos are saved under `output/`.

## Workflow 2: optionally try the established ~19k checkpoint

This AnchorBench checkpoint was trained for a specific keyboard pose and fixed
jaw. Simulation playback needs no physical robot. The historical setup used the
Logitech MX keyboard, closed moving jaw, fixed typing tip and a 4.44 mm keyboard
mat. Its 89/100 physical six-letter result was measured on the original benchmark
robot/fixture; it is not a new-robot success guarantee.

The scene's joint mapping and keyboard pose were calibrated from recorded IRL
contacts and then frozen in the matching environment YAML. The [historical
provenance record](docs/fixtures/historical_offset_provenance_20260718.md)
documents the measured contacts, the rejected diagnostic fits, the locked
contract, and its 78/78 direct replay gate. The [model drawing](FIXTURE_SETUP.md)
shows that frozen scene; its dimensions are not physical ruler instructions.
Physical use needs alignment checks, and moving the real keyboard does not update
the policy's frozen map.

```bash
./so101 download
./so101 video --actuator anchorbench --stage transit15 \
  --checkpoint .artifacts/so101-keyboard-typing-benchmark/checkpoints/mjwarp-anchorbench-19k/model_19000.pt \
  --env-config .artifacts/so101-keyboard-typing-benchmark/configs/mjwarp-anchorbench-19k.env.yaml
```

The default download contains only these weights, their environment and evaluation
report, verified against pinned hashes. `./so101 download --all` explicitly fetches
the supported-model benchmark/development artifacts. Existing downloaded files are retained.

## Both workflows: calibrate your robot and deploy

**You perform standard LeRobot calibration on your own robot.** Calibration
establishes encoder coordinates; it does not measure the keyboard pose. Do not
copy another robot's calibration or assume its additional fitted offsets apply.
For workflow 1, complete this setup before collecting the scene-validation
recordings; reuse that calibration for deployment rather than recalibrating
between validation and export.

```bash
./so101 setup-hardware
source .venv-hardware/bin/activate
lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/YOUR_CONTROLLER --robot.id=my_typing_robot

# Workflow 2: the exact released checkpoint/environment pair selects its mapping automatically.
./so101 prepare-deployment \
  --checkpoint .artifacts/so101-keyboard-typing-benchmark/checkpoints/mjwarp-anchorbench-19k/model_19000.pt \
  --env-config .artifacts/so101-keyboard-typing-benchmark/configs/mjwarp-anchorbench-19k.env.yaml \
  --robot-id my_typing_robot --out-dir output/my_typing_robot
./so101 deploy output/my_typing_robot NVIDIA
```

For workflow 1, supply your selected checkpoint and its `params/env.yaml` plus
the mapping verified for your robot/scene: `--joint-calibration FILE` for a measured
fit, or an explicit `--encoder-convention lerobot|benchmark`. New training does
not establish a physical joint mapping, so unknown artifact pairs cannot silently
export with zero offsets. See [hardware preparation](docs/release/HARDWARE_PREPARATION.md).

Export and the dry run check your calibration without connecting to hardware;
new bundles also reject changes to that calibration after export. The last
command is a software dry run, not a physical alignment test.
[The hardware guide](docs/release/HARDWARE_PREPARATION.md) provides the explicit
operator-run execution command and explains the joint offsets. Keep your own
robot's LeRobot calibration and verify physical/model alignment before motion.

## Documentation and validation

- [Detailed quickstart](docs/release/QUICKSTART.md)
- [Training](docs/release/TRAINING.md) and [task/physics configuration](docs/release/TASK_CONFIGURATION.md)
- [Hardware preparation](docs/release/HARDWARE_PREPARATION.md) and [reference fixture](FIXTURE_SETUP.md)
- [Software validation status](docs/release/STATUS.md) and [independent reproduction record](docs/release/REPRODUCTION.md)
- [Benchmark artifacts](https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark) and [runtime provenance](runtime/README.md)
- [Development and release checks](docs/DEVELOPMENT.md); [dated investigation records](docs/experiments/README.md)

Software execution and deployment dry runs have been tested locally. Physical
reproduction on another user's SO-101 remains unverified. Development-training
measurements are retained separately in the validation archive.
