# Calibrate and deploy your SO-101

Use the standard [LeRobot SO-101 follower calibration](https://huggingface.co/docs/lerobot/so101).
The robot ID used for calibration must be the same ID used for deployment.
Calibration is performed by the user on their own robot. It establishes encoder
coordinates; it does not locate the keyboard or prove agreement with the simulator.
Physical deployment needs both a joint mapping and a matching keyboard layout.
For a new physical layout, perform this setup before the
[pre-training measured-trajectory checks](REAL2SIM.md).
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

[FIXTURE_SETUP.md](../../FIXTURE_SETUP.md) describes the September setup: Logitech
MX Keys and the clamped plastic robot base directly on the same flat tabletop,
with no mat or riser, the moving jaw closed and the fixed typing tip unchanged.
Start with about 130 mm between the base front and nearest keyboard case edge,
keep those edges parallel, and align the base's short right straight edge to the
bottom-right corner of right Ctrl. The
[measured physical reference](../fixtures/physical_reference_20260915.md) shows
these datums separately from the fitted simulation scene. Check and save your
own placement; fitted model coordinates are not physical ruler instructions.
`./so101 fixture --measurements FILE --out-dir NEW_DIR` fits a candidate
XYZ/full-orientation pose and preserves the
asset's built-in key geometry; it does not verify real-to-sim contact alignment.
You can change the nominal keyboard position/yaw with a task JSON, retrain, and
export that policy's geometry. You can also train over bounded XY/yaw placement
variation and reset-joint noise. See [TASK_CONFIGURATION.md](TASK_CONFIGURATION.md).

The September checkpoint corresponds to its saved fitted scene. Moving the
physical keyboard does not update the policy's frozen map. For your own
policy, calibrate the scene using [recorded motion](REAL2SIM.md), train with its
saved task JSON, and keep the physical fixture unchanged. The fitted scene is an
effective model; its coordinates are not instructions to move the real keyboard.
Any placement variation needs separate evaluation.

For an FK sanity check, compute the fixed typing-tip position with
`source/so101_typing/hardware/so101_kinematics.py` and compare it with the selected
letter's `letter_xyz_b_m` in the matching environment. The two values share the
robot-base frame when the task configuration, base datum, fixed tip and
encoder-to-model mapping are the same. Agreement is necessary for a physical
press but does not certify contact or key release.

## Qualify alignment for physical execution

Follow [the alignment workflow](ALIGNMENT.md) to freeze your fitted scene and
joint mapping, pass reserved measured-motion replay, and collect physical
Q/G/P press, release and clearance evidence. `qualification.json` binds these
checks to the robot calibration, fixture, model and closed jaw. The qualification
is representative alignment evidence, not a policy success guarantee.

Export with `--alignment output/my-fixture/qualified/qualification.json` to make
a physical-capable bundle. A fitted mapping alone is insufficient; keep the
reserved replay and physical contact evidence with the export.
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
./so101 deploy output/my_typing_robot NEWTON
```

For the released September policy, substitute its downloaded checkpoint/environment
paths from the [README](../../README.md#workflow-2-optionally-try-a-released-checkpoint)
and your robot's qualification for that exact trained scene. The `--alignment`
file supplies the explicit mapping; the downloaded offsets do not calibrate your
robot automatically.

Use a two-letter target such as `HE` for P1A policies; Transit15 policies use six
letters. Export and the dry run both check the local LeRobot calibration for the selected
robot ID. Missing or malformed calibration is an error before any robot connection.
The second command is a software dry run and does not open the robot.
The bundle contains the checkpoint, its environment YAML, optional task/physics
JSONs, nominal A-Z geometry, normalization, robot ID and a generated reset target.
Its hashes and geometry are checked before the runner can connect to hardware.
Released checkpoint hashes are also checked against their matching environment.
For your own checkpoints, keep the original `params/` directory: the checkpoint
format does not embed the scene, so export cannot infer a missing training pairing.
Qualified bundles copy and revalidate the replay/probe evidence at deployment.
New bundles record the LeRobot calibration hash and reject subsequent changes.
Recalibration requires checking the mapping and exporting a new bundle.
Calibration-file checks do not establish physical alignment.

To check CPU inference against saved simulator actions, record a rollout with
`./so101 video ... --trace-policy-actions 400`, then run:

```bash
.venv-hardware/bin/python -m so101_typing.hardware.check_actor_parity \
  --checkpoint output/my_typing_robot/checkpoint.pt \
  --report /absolute/path/to/video/evaluation.json
```

This verifies the checkpoint hash, observation normalization and deterministic
Beta actor output against the recorded simulation trace. It does not connect to
hardware or establish encoder alignment.

### Joint offsets: matching the simulated scene to IRL

The [record-and-fit workflow](REAL2SIM.md) selects a joint mapping and an effective
keyboard pose together. Preserve both with the trained policy. These offsets
are separate from LeRobot motor calibration: they convert measured encoder
coordinates to the model's joint coordinates, and convert model goals back to
hardware coordinates. Deployment applies the same mapping in both directions:

```text
Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
Read:    q_model = q_lerobot + offset
Command: q_lerobot_target = q_model_target - offset
```

Omitting fitted offsets changes the commanded physical poses. Keep your robot's
own LeRobot calibration and its matched scene/mapping report; the released values
are not universal SO-101 constants. Qualified export uses the mapping frozen in
your `--alignment` evidence.

### Your trained policy or measured joint mapping

New training does not measure your robot's encoder-to-model correspondence.
For physical use, pass `--alignment` and supply your checkpoint and its matching
`params/env.yaml`. The September checkpoint and new training require an explicit
mapping; leaving a software-only export in `auto` mode does not infer one.

For independently measured offsets, the existing offline fitter accepts your
robot ID, ordered five-joint encoder/model angles at three or more poses, and a
measurement reference for each pose:

```bash
.venv-hardware/bin/python -m so101_typing.hardware.joint_calibration \
  output/pose-measurements.json --out output/joint-fit.json
./so101 prepare-deployment --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --robot-id my_typing_robot \
  --joint-calibration output/joint-fit.json --out-dir output/my-trained-policy
```

See [the fitter's input fields](../../source/so101_typing/hardware/joint_calibration.py).
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
./so101 deploy output/my_typing_robot NEWTON --execute \
  --port /dev/serial/by-id/YOUR_CONTROLLER \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD_EVENT_DEVICE
```

`--execute` opts into robot commands and the existing operator confirmation.
Check the reset path and tip/keyboard alignment first. Keep the stop control/power
accessible. Begin with controlled validation and stop on unexpected motion or a
wrong key. The runner enforces measured reset arrival and strict press/release
judging. Logs are written to the bundle's `runs/` directory.

During policy execution, press **Esc on the selected typing keyboard** to request
a controlled stop. The runner then attempts its configured return-to-rest motion
before releasing keyboard capture. Motor torque remains enabled after exit.
For an immediate motion stop, switch off motor power; be ready to support the arm.

Keep the robot calibration, fixed jaw and physical fixture unchanged after
alignment. Recalibration, moving the fixture or replacing the jaw requires new
alignment checks before deployment.
