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
This NVIDIA-dev repository has Internal visibility: cloning requires access through
NVIDIA's GitHub enterprise. Runtime and artifact downloads are publicly available.

| Budget | Planning guidance |
| --- | --- |
| Disk | Built image: about 32.6 GB (30.4 GiB). Budget at least 70 GB free for image/build layers, caches and runs; this is headroom guidance, not a measured peak. |
| First install | Download/build and first kernel compilation depend on bandwidth, CPU and cache state; allow tens of minutes to hours. This is an estimate, not a measured cold-install time. |
| Optional ~19k playback | No training required; evaluation follows installation and kernel startup. |
| September calibrated-scene example, 500 + 2,000 updates | Measured 2 h 38 min 21 s including evaluations/videos on an RTX 5090; completed one physical NEWTON trial. Your scene and seed need their own evaluation. |
| Example 4,000 + 16,000 updates, 4096 environments | Roughly 20.4 hours of training-loop time on an RTX 5090, extrapolated from 36.72 s / 10 updates. Excludes startup and evaluation; not a measured full run. |

See [training budgets](docs/release/TRAINING.md) for early evaluation and timing logs.
ZIP/tarball source downloads are supported: run commands from the extracted folder;
provenance records content hashes with no Git revision when `.git` is absent.

```bash
git clone https://github.com/NVIDIA-dev/so101-keyboard-typing-internal.git
cd so101-keyboard-typing-internal
./so101 build
```

## Workflow 1: train for your setup

Follow this order for physical typing:

1. [Install the hardware environment and calibrate your robot](docs/release/HARDWARE_PREPARATION.md#install-and-calibrate) with LeRobot.
2. [Measure and freeze the physical fixture](FIXTURE_SETUP.md), saving the starting layout as `output/my-fixture/task.json`.
3. [Record and fit the scene](docs/release/REAL2SIM.md): hand-guide key presses with torque off, fit joint offsets and keyboard pose, then pass the reset-per-key contact checks. Keep the fitted `output/my-scene/task.json` and `mapping-report.json` together.
4. [Train and evaluate](docs/release/TRAINING.md) using that fitted scene, then complete the alignment and deployment steps below.

The recordings calibrate the model; they are not policy-training demonstrations.
The suspected encoder/model or contact-geometry discrepancy remains unresolved;
the fit is an effective correction, not proof of geometrically identical scenes.

The nominal keyboard position/orientation is relative to the simulated robot.
Keep the physical fixture fixed after calibration; train and deploy with its saved
scene and joint mapping.
For a calibrated physical setup, use `output/my-scene/task.json` from the guide.
The task JSON supports nominal XYZ/full orientation and optional XY/yaw placement randomization.
Use [task configuration](docs/release/TASK_CONFIGURATION.md) to choose a simulated
layout. The [fixture helper](FIXTURE_SETUP.md) draws that model and can fit a
candidate layout from measurements; it does not establish real-to-sim alignment.

```bash
./so101 fixture --task-config output/my-scene/task.json --out-dir output/my-scene-view
./so101 probe --task-config output/my-scene/task.json

# Startup check; this budget is not intended to produce a deployable policy.
./so101 train --actuator anchorbench --stage p1a \
  --task-config output/my-scene/task.json --num-envs 64 --iterations 10
```

Without hardware, use the [simulation-only startup check](docs/release/QUICKSTART.md#simulation-only-startup-check)
or the reference checkpoint below. Editing an example task JSON alone does not
validate physical contact alignment.

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

## Workflow 2: optionally try a released checkpoint

The default download is the established AnchorBench ~19k checkpoint, trained for
a specific keyboard pose and fixed jaw. Simulation playback needs no physical robot. The historical setup used the
Logitech MX keyboard, closed moving jaw, fixed typing tip and a 4.44 mm keyboard
mat. Its 89/100 physical six-letter result was measured on the original benchmark
robot/fixture; it is not a new-robot success guarantee.

The scene's joint mapping and keyboard pose were calibrated from recorded IRL
contacts and then frozen in the matching environment YAML. The [historical
provenance record](docs/fixtures/historical_offset_provenance_20260718.md)
documents the measured contacts, the rejected diagnostic fits, the locked
contract, and its 78/78 direct replay gate. The [historical model drawing](docs/fixtures/reference-birdseye.svg)
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

### September calibrated-scene example

**Latest fitted keyboard pose — September 17, 2026**, used by `model_2498.pt`:

| Quantity | Value |
| --- | --- |
| Keyboard CAD origin XYZ, relative to the simulated robot base | `[259.567686, 26.287386, 10.914638]` mm |
| Asset orientation, roll / pitch / yaw | `[0, -0.5, -86.921376]` degrees |

The [task JSON](configs/tasks/replay-calibrated-20260917.json) stores the full-precision
position and XYZW quaternion; the [fitted bird's-eye view](docs/fixtures/replay-calibrated-birdseye-20260917.svg)
shows this scene. These are effective model coordinates, not ruler distances to
keyboard edges or the physical keyboard's yaw. The real fixture stayed fixed while
this pose and the joint mapping were fitted together; keep that pair together.

The additional `model_2498.pt` release completed one physical **NEWTON** trial on
the newly calibrated setup. Download its exact weights, matched scene/environment,
setup-specific mapping and sanitized evidence without replacing the historical 19k:

```bash
./so101 download --calibrated
bundle=.artifacts/so101-keyboard-typing-benchmark/releases/calibrated-20260917
./so101 video --actuator anchorbench --stage transit15 \
  --checkpoint "$bundle/model_2498.pt" \
  --env-config "$bundle/params/env.yaml" --target NEWTON
```

Keep the complete `params/` directory: `public_task.json` restores the fitted scene.
This command produces a simulation video. For another physical setup, use your own
calibration and scene fitting; the reference mapping is evidence, not a transferable
robot calibration. See the [physical result and limitations](docs/experiments/physical_typing_success_20260917.md).

## Both workflows: qualify alignment and deploy

**You perform standard LeRobot calibration on your own robot.** Calibration
establishes encoder coordinates; it does not measure the keyboard pose. Do not
copy another robot's calibration or assume its additional fitted offsets apply.
Workflow 1 already completed [hardware setup and calibration](docs/release/HARDWARE_PREPARATION.md#install-and-calibrate)
before recording; reuse that calibration at deployment. For workflow 2, complete
it before physical alignment. Do not recalibrate between validation and export.

Complete the [alignment workflow](docs/release/ALIGNMENT.md) for your frozen
scene. It produces `output/my-fixture/qualified/qualification.json` from reserved
trajectory replay and physical Q/G/P checks. Then export and dry-run your policy:

```bash
# Workflow 1: your evaluated checkpoint and its matching trained scene.
./so101 prepare-deployment \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml \
  --robot-id my_typing_robot \
  --alignment output/my-fixture/qualified/qualification.json \
  --out-dir output/my_typing_robot
./so101 deploy output/my_typing_robot NVIDIA
```

For workflow 2, substitute the downloaded 19k checkpoint and environment paths
above, retaining qualification for that historical scene.
Training does not establish a physical joint mapping. For a software-only dry
run, you may omit `--alignment`: the released pair selects its historical mapping;
new training requires an explicit mapping as described in
[hardware preparation](docs/release/HARDWARE_PREPARATION.md).

Export and the dry run check your calibration without connecting to hardware;
new bundles also reject changes to that calibration after export. Physical
execution in the standard workflow requires the qualified alignment. For an
operator-supervised candidate test, see [experimental deployment](docs/release/ALIGNMENT.md#optional-supervised-experiments).
The final command above checks the
software contract without moving the robot.
[The hardware guide](docs/release/HARDWARE_PREPARATION.md) provides the explicit
operator-run execution command and explains the joint offsets. Keep your own
robot's LeRobot calibration and verify physical/model alignment before motion.

## Documentation and validation

- [September 17 physical NEWTON result and how it was achieved](docs/experiments/physical_typing_success_20260917.md)
- [Detailed quickstart](docs/release/QUICKSTART.md)
- [Training](docs/release/TRAINING.md) and [task/physics configuration](docs/release/TASK_CONFIGURATION.md)
- [Hardware preparation](docs/release/HARDWARE_PREPARATION.md), [alignment qualification](docs/release/ALIGNMENT.md) and [reference fixture](FIXTURE_SETUP.md)
- [Software validation status](docs/release/STATUS.md) and [independent reproduction record](docs/release/REPRODUCTION.md)
- [Benchmark artifacts](https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark) and [runtime provenance](runtime/README.md)
- [Development and release checks](docs/DEVELOPMENT.md); [dated investigation records](docs/experiments/README.md)

Software execution and deployment dry runs have been tested locally. A policy
trained after recorded-contact scene fitting completed one physical NEWTON trial
on the new development robot. Repeated reliability and reproduction on another
user's machine/SO-101 remain unverified. Earlier development-training measurements
are retained separately in the validation archive.
