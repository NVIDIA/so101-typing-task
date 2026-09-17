# Calibrate and deploy your SO-101

Use the standard [LeRobot SO-101 follower calibration](https://huggingface.co/docs/lerobot/so101).
The robot ID used for calibration must be the same ID used for deployment.
Calibration is performed by the user on their own robot. It establishes encoder
coordinates; it does not locate the keyboard or prove agreement with the simulator.
Physical deployment needs both a joint mapping and a matching keyboard layout.
For a new physical layout, perform this setup before the
[pre-training measured-trajectory checks](../../FIXTURE_SETUP.md#before-policy-training-verify-the-real-scene).
Reuse the validated calibration at deployment; recalibration requires rechecking
the mapping.

## Install and calibrate

The tested deployment environment uses Linux x86-64, Python 3.12, CPU PyTorch,
LeRobot with Feetech support, and evdev. It does not require Isaac Sim or a GPU.
Install `uv` and a C compiler/Python headers if your system needs to build evdev.
The setup script creates its own `.venv-hardware` and uses `runtime/hardware.lock`.

```bash
./so101 setup-hardware
source .venv-hardware/bin/activate
lerobot-find-port
lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/YOUR_CONTROLLER --robot.id=my_typing_robot
```

The operator follows LeRobot's neutral-pose and joint-range instructions. This is
an interactive hardware procedure; keep the calibration sweep clear before
placing and freezing the keyboard for recording. Identify the stable device paths
with `ls -l /dev/serial/by-id/ /dev/input/by-id/`; select the typing keyboard's
`*-event-kbd` entry for recording and deployment. Give your user access to the serial controller
and keyboard evdev device using your system's `dialout`/`input` groups or device
rules; do not run the entire pipeline as root.

## Choose the physical layout

[FIXTURE_SETUP.md](../../FIXTURE_SETUP.md) describes the supported Logitech MX
keyboard, closed moving jaw and fixed typing tip. The historical 19k fixture used
a mat; your own measured and calibrated layout need not. Its labelled bird's-eye
views distinguish measured-reference placement from fitted simulation scenes.
Fitted dimensions are not physical ruler instructions. The height schematic
explains the measurement convention.
`./so101 fixture` fits a candidate XYZ/full-orientation pose and preserves the
asset's built-in key geometry; it does not verify real-to-sim contact alignment.
You can change the nominal keyboard position/yaw with a task JSON, retrain, and
export that policy's geometry. You can also train over bounded XY/yaw placement
variation and reset-joint noise. See [TASK_CONFIGURATION.md](TASK_CONFIGURATION.md).

The released pretrained checkpoints correspond to the original setup. Moving
that physical keyboard does not update their training distribution. For your own
policy, calibrate the scene using [recorded motion](REAL2SIM.md), train with its
saved task JSON, and keep the physical fixture unchanged. The fitted scene is an
effective model; its coordinates are not instructions to move the real keyboard.
Any placement variation needs separate evaluation.

For an FK sanity check, compute the fixed typing-tip position with
`scripts/so101_homing/so101_kinematics.py` and compare it with the selected
letter's `letter_xyz_b_m` in the matching environment. The two values share the
robot-base frame when the task configuration, base datum, fixed tip and
encoder-to-model mapping are the same. Agreement is necessary for a physical
press but does not certify contact or key release.

## Qualify alignment for physical execution

Follow [the alignment workflow](ALIGNMENT.md) to freeze a direct, historical or
measured/fitted mapping, pass reserved measured-motion replay, and collect physical
Q/G/P press, release and clearance evidence. `qualification.json` binds these
checks to the robot calibration, fixture, model and closed jaw. The qualification
is representative alignment evidence, not a policy success guarantee.

Export with `--alignment output/my-fixture/qualified/qualification.json` to make
a physical-capable bundle. The original mapping options below still support
software export and dry runs; they do not bypass the qualification requirement.
Existing bundles without this evidence must be re-exported before physical use.
For an explicit supervised experiment before qualification, export with
`--alignment-candidate` and use `deploy --experimental --execute`; see
[the experimental command and retained checks](ALIGNMENT.md#optional-supervised-experiments).

## Export a deployment bundle

```bash
# Your evaluated six-letter policy after qualifying its physical alignment.
./so101 prepare-deployment \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml \
  --robot-id my_typing_robot \
  --alignment output/my-fixture/qualified/qualification.json \
  --out-dir output/my_typing_robot
./so101 deploy output/my_typing_robot NVIDIA
```

For the optional 19k policy, substitute its downloaded checkpoint/environment
paths from the [README](../../README.md#workflow-2-optionally-try-a-released-checkpoint)
and qualification for its original scene.

Use a two-letter target such as `HE` for P1A policies; Transit15 policies use six
letters. Export and the dry run both check the local LeRobot calibration for the selected
robot ID. Missing or malformed calibration is an error before any robot connection.
The second command is a software dry run and does not open the robot.
The bundle contains the checkpoint, its environment YAML, optional task/physics
JSONs, nominal A-Z geometry, normalization, robot ID and a generated reset target.
Its hashes and geometry are checked before the runner can connect to hardware.
Qualified bundles copy and revalidate the replay/probe evidence at deployment.
New bundles record the LeRobot calibration hash and reject subsequent changes.
Recalibration requires checking the mapping and exporting a new bundle. Legacy
bundles still load under their explicitly recorded mapping; re-export to add this
check. Calibration-file checks do not establish physical alignment.

To check CPU inference against saved simulator actions, record a rollout with
`./so101 video ... --trace-policy-actions 400`, then run:

```bash
.venv-hardware/bin/python -m scripts.so101_homing.check_actor_parity \
  --checkpoint output/my_typing_robot/checkpoint.pt \
  --report /absolute/path/to/video/evaluation.json
```

This verifies the checkpoint hash, observation normalization and deterministic
Beta actor output against the recorded simulation trace. It does not connect to
hardware or establish encoder alignment.

### Joint offsets: matching the simulated scene to IRL

The dated convention and fixed-jaw mapping are recorded in
[encoder/model mapping (2026-09-15)](../experiments/encoder_model_mapping_20260915.md).

The July 18 scene was calibrated against timestamped powered physical key
presses: A–Z, three times each. The final joint-angle mapping and keyboard pose
form one frozen real-to-sim scene contract, validated by direct Newton replay of
the measured trajectories (78/78 contacts). The archived diagnostic least-squares
candidates were rejected on held-out geometry; they are not the source of the
released vector. See the [historical offset provenance](../fixtures/historical_offset_provenance_20260718.md)
for the full evidence chain. The offsets are separate from LeRobot motor
calibration. Deployment applies the same mapping in both directions:

```text
Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
Offsets (degrees): [0, +1.9336272481, -10.7735791533, +1.3647648917, 0]
Read:    q_model = q_lerobot + offset
Command: q_lerobot_target = q_model_target - offset
```

For software-only exports without `--alignment`, the default
`--encoder-convention auto` recognizes only the exact released 19k
checkpoint/environment hashes and chooses `benchmark`. `lerobot` adds no offsets;
it is an explicit choice for a verified direct mapping or a controlled comparison.
Omitting the fitted offsets changes the benchmark policy's commanded physical poses.
Keep your robot's own LeRobot calibration and check physical/model alignment;
these fitted values are not universal SO-101 constants.

### Your trained policy or measured joint mapping

New training does not measure your robot's encoder-to-model correspondence.
For physical use, pass `--alignment` and supply your checkpoint and its matching
`params/env.yaml`. For software-only export, unknown pairs are rejected in `auto`
mode; explicitly select `--encoder-convention lerobot`, `benchmark`, or a measured
joint fit. These choices do not qualify the mapping for hardware.

For independently measured offsets, the existing offline fitter accepts your
robot ID, ordered five-joint encoder/model angles at three or more poses, and a
measurement reference for each pose:

```bash
.venv-hardware/bin/python -m scripts.so101_homing.joint_calibration \
  output/pose-measurements.json --out output/joint-fit.json
./so101 prepare-deployment --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --robot-id my_typing_robot \
  --joint-calibration output/joint-fit.json --out-dir output/my-trained-policy
```

See [the fitter's input fields](../../scripts/so101_homing/joint_calibration.py).
This is an advanced measurement path, not automatic physical calibration. Export
copies and hashes the fit evidence and uses the offsets for both feedback and
commands, including the reset. It does not change the trained keyboard map.

`rest_pose.json` is a software-derived reset target, not a claimed physical
measurement. The runner measures actual joint arrival before starting the policy.
The policy's q_reset reference remains fixed even when training randomizes reset
arrival. The typing contact is the fixed jaw; the moving jaw must be closed.

## Operator-run deployment

Use a bundle exported with qualified `--alignment` evidence:

```bash
./so101 deploy output/my_typing_robot NVIDIA --execute \
  --port /dev/serial/by-id/YOUR_CONTROLLER \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD_EVENT_DEVICE
```

`--execute` opts into robot commands and the existing operator confirmation.
Check the reset path and tip/keyboard alignment first. Keep the stop control/power
accessible. Begin with controlled validation and stop on unexpected motion or a
wrong key. The runner enforces measured reset arrival and strict press/release
judging. Logs are written to the bundle's `runs/` directory.

Software preparation and dry runs have been tested. [Second-robot evidence and
remaining release work](STATUS.md#physical-deployment-follow-up) distinguish
restored reaching/contact from complete typing reproduction.
