# SO-101 keyboard typing: train and deploy

Train an SO-101 typing policy in Newton for your keyboard placement, evaluate it,
and deploy using the geometry saved during training and your robot's LeRobot
calibration. An established AnchorBench ~19k checkpoint is available as an
**optional shortcut for the documented reference fixture**.

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

Choose the nominal keyboard position/orientation relative to the robot in
simulation. Your physical fixture must match that trained geometry at deployment.
The task JSON supports nominal XYZ/yaw and optional XY/yaw placement randomization.
Start with the [bird’s-eye measurement guide](FIXTURE_SETUP.md) to translate ruler
measurements into simulation coordinates. See [task configuration](docs/release/TASK_CONFIGURATION.md)
for the JSON schema and randomization options.

```bash
./so101 fixture --out-dir output/reference-fixture
# Follow FIXTURE_SETUP.md; fill the blank template with your measured Q/P/Z coordinates.
cp output/reference-fixture/measurement-template.json output/my-measurements.json
# Edit output/my-measurements.json, then convert it:
./so101 fixture --measurements output/my-measurements.json --out-dir output/my-fixture
./so101 probe --task-config output/my-fixture/task.json

# Startup check; this budget is not intended to produce a deployable policy.
./so101 train --actuator anchorbench --stage p1a \
  --task-config output/my-fixture/task.json --num-envs 64 --iterations 10
```

For simulation-only exploration, you can use `configs/tasks/keyboard-pose.json`
directly instead of measuring hardware. Draw it with `./so101 fixture --task-config
configs/tasks/keyboard-pose.json --out-dir output/example-fixture`.

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
jaw. Simulation playback needs no physical robot. **Physical use requires matching
its trained fixture geometry**, not placing the keyboard wherever convenient.

| Reference-fixture requirement | Trained value |
| --- | --- |
| Keyboard CAD origin relative to robot model base | `(262.565, 12.135, 1.781) mm` |
| CAD roll, pitch, yaw | `(-5.000, 0.051, -88.606) degrees` |
| Physical measurement reference | Base datum D and Q/P/Z rim centres in [fixture guide](FIXTURE_SETUP.md) |
| Keyboard and jaw | Reference Logitech MX geometry; fixed typing jaw, moving jaw closed |
| Mat under keyboard | `4.44 mm` |

CAD angles are not physical top-down yaw. Follow [FIXTURE_SETUP.md](FIXTURE_SETUP.md)
for the authoritative quaternion, bird’s-eye drawing, signed measurements and height datum.
The matching environment YAML is the checkpoint's source of truth. If your setup
cannot match it, use workflow 1 to train for your intended placement.

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

```bash
./so101 setup-hardware
source .venv-hardware/bin/activate
lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/YOUR_CONTROLLER --robot.id=my_typing_robot

./so101 prepare-deployment --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/matching/env.yaml --robot-id my_typing_robot \
  --encoder-convention lerobot --out-dir output/my_typing_robot
./so101 deploy output/my_typing_robot NVIDIA
```

For workflow 2, supply the downloaded `model_19000.pt` and
`configs/mjwarp-anchorbench-19k.env.yaml` paths above. For workflow 1, supply your
own selected checkpoint and its `params/env.yaml`.

Export and the dry run both check that a readable LeRobot calibration file exists
for your robot ID, without connecting to hardware. The last command is a software dry run. Match the physical fixture to the saved
simulation geometry and check model/encoder alignment before enabling motion.
[The hardware guide](docs/release/HARDWARE_PREPARATION.md) provides the explicit
operator-run execution command. `benchmark` encoder offsets are only for
reproducing the historical robot's separately fitted convention.

## Documentation and validation

- [Detailed quickstart](docs/release/QUICKSTART.md)
- [Training](docs/release/TRAINING.md) and [task/physics configuration](docs/release/TASK_CONFIGURATION.md)
- [Hardware preparation](docs/release/HARDWARE_PREPARATION.md) and [reference fixture](FIXTURE_SETUP.md)
- [Software validation status](docs/release/STATUS.md) and [independent reproduction record](docs/release/REPRODUCTION.md)
- [Benchmark artifacts](https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark) and [runtime provenance](runtime/README.md)

Software execution and deployment dry runs have been tested locally. Physical
reproduction on another user's SO-101 remains unverified. Development-training
measurements are retained separately in the validation archive.
