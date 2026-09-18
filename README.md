# SO-101 keyboard typing: setup, training and deployment

Train an SO-101 typing policy in Newton for your keyboard placement, evaluate it,
and export it for deployment with your robot's calibration and a verified joint
mapping. The September fitted-scene policy completed one physical **NEWTON**
trial; the historical ~19k policy achieved 89/100 six-letter trials on its original
fixture. Both are available for simulation playback. For your own physical setup,
start with workflow 1.

This repository covers physical setup and scene calibration, policy training and
evaluation, and hardware deployment.

[Watch the historical typing timelapse](https://huggingface.co/datasets/nvidia/Anchor-Lab/blob/main/media/typing_abc_common7500_timelapse_hud_3x.mp4)
in the [Anchor-Lab media collection](https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/main/media).

## Install

Requirements: Linux x86-64, an NVIDIA GPU, Docker with NVIDIA Container Toolkit,
host Python 3, and `curl` or `wget` for artifact downloads. Hardware setup
additionally requires `uv`; install it before
`./so101 setup-hardware` ([uv installation](https://docs.astral.sh/uv/getting-started/installation/)).
The pinned container supplies Isaac Lab, Newton and RSL-RL.
The tested GPU is an RTX 5090 with 32 GB. Runtime dependencies and benchmark
artifacts are publicly available. No additional source checkout or cluster
account is required.
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
4. [Train and evaluate](docs/release/TRAINING.md) using that fitted scene. The measured September recipe used 500 two-letter updates, then 2,000 six-letter updates.
5. [Qualify alignment](docs/release/ALIGNMENT.md) using fresh recordings excluded from fitting and physical Q/G/P checks with the same mapping.
6. [Export, dry-run and deploy](#both-workflows-qualify-alignment-and-deploy) the evaluated checkpoint with its matching scene and alignment evidence.

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
checkpoint and its matching environment. After the six-letter continuation, use
those paths below. For a two-letter checkpoint, use `--stage p1a` and `--target HE`:

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

Choose a matched policy/scene bundle. Neither download needs hardware for simulation:

| Download | Policy and scene | Physical evidence |
| --- | --- | --- |
| `./so101 download` | Historical `model_19000.pt`, July fitted scene, mat fixture | 89/100 six-letter trials on the original benchmark setup |
| `./so101 download --calibrated` | September `model_2498.pt`, newly fitted scene, no-mat fixture | One successful NEWTON trial on the new development robot |

Their keyboard poses and joint mappings differ. Use the matching environment and
keep the September bundle's complete `params/` directory. For physical use, your
robot needs its own calibration and alignment evidence for the selected scene.
If your fitted scene differs, train a policy for it using workflow 1.

The default download is the established AnchorBench ~19k checkpoint, trained for
a specific keyboard pose and fixed jaw. Simulation playback needs no physical robot. The historical setup used the
Logitech MX keyboard, closed moving jaw, fixed typing tip and a 4.44 mm keyboard
mat. Its 89/100 physical six-letter result was measured on the original benchmark
robot/fixture; it is not a new-robot success guarantee.

The scene's joint mapping and keyboard pose were calibrated from recorded IRL
contacts and then frozen in the matching environment YAML. The [reference-policy
contract](docs/fixtures/reference_policy.md) records the required fixture and
mapping. The [historical model drawing](docs/fixtures/reference-birdseye.svg)
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

**Physical layout used for this checkpoint:** the clamped SO-101 plastic base and
Logitech MX Keys rested directly on the same flat table, with no keyboard mat.
The front base face was about **130 mm (13 cm)** from the nearest keyboard case
edge, with the edges parallel. The base's short right-side straight edge aligned
with the bottom-right corner of the right Ctrl key beside the left-arrow key.
See the [physical setup instructions and reference drawing](docs/fixtures/physical_reference_20260915.md#physical-setup-used-for-the-september-newton-checkpoint)
for the measurement directions and lateral cross-checks. This is the real fixture
that remained frozen while the simulation below was fitted.

**Latest fitted simulation keyboard pose — September 17, 2026**, used by `model_2498.pt`:

| Quantity | Value |
| --- | --- |
| Keyboard CAD origin XYZ, relative to the simulated robot base | `[259.567686, 26.287386, 10.914638]` mm |
| Asset orientation, roll / pitch / yaw | `[0, -0.5, -86.921376]` degrees |

The [task JSON](configs/tasks/replay-calibrated-20260917.json) stores the full-precision
position and XYZW quaternion; the [fitted bird's-eye view](docs/fixtures/replay-calibrated-birdseye-20260917.svg)
shows this scene. These are effective model coordinates, not ruler distances to
keyboard edges or the physical keyboard's yaw. **The real robot base and keyboard
remained frozen at their measured placement.** We fitted the simulated keyboard
pose and encoder-to-model joint mapping to reproduce the key contacts in the
recorded real motions; keep that fitted pose/mapping pair together.

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
robot calibration. See the [physical result and limitations](docs/release/RESULTS.md).

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

For either released policy, substitute its checkpoint and matching environment
paths above, and qualify your robot's mapping against that policy's trained scene.
The historical 19k route is detailed in the alignment guide. The September
reference fit is evidence for its original robot, not qualification for yours.

Training does not establish a physical joint mapping. For software-only exports
without `--alignment`, automatic mapping recognizes **only the exact historical
19k checkpoint/environment pair**. The September checkpoint and your own training
require an explicit mapping as described in
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

- [Detailed quickstart](docs/release/QUICKSTART.md)
- [Recorded-motion scene calibration](docs/release/REAL2SIM.md)
- [Training](docs/release/TRAINING.md) and [task/physics configuration](docs/release/TASK_CONFIGURATION.md)
- [Hardware preparation](docs/release/HARDWARE_PREPARATION.md), [alignment qualification](docs/release/ALIGNMENT.md) and [reference fixture](FIXTURE_SETUP.md)
- [Measured results and tested configuration](docs/release/RESULTS.md)
- [Benchmark artifacts](https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark) and [runtime provenance](runtime/README.md)
- [Development and release checks](docs/DEVELOPMENT.md)
