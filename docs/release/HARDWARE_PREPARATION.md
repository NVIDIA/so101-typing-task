# Calibrate and deploy your SO-101

Use the standard [LeRobot SO-101 follower calibration](https://huggingface.co/docs/lerobot/so101).
The robot ID used for calibration must be the same ID used for deployment.
Calibration is performed by the user on their own robot. It establishes encoder
coordinates; it does not locate the keyboard. A separate multi-pose calibration
fit is not a mandatory setup step.

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
an interactive hardware procedure. Give your user access to the serial controller
and keyboard evdev device using your system's `dialout`/`input` groups or device
rules; do not run the entire pipeline as root.

## Choose the physical layout

[FIXTURE_SETUP.md](../../FIXTURE_SETUP.md) describes the published Logitech MX
keyboard, closed jaw, fixed typing tip and reference mat. Its bird’s-eye and
height drawings define physical measurement datums; `./so101 fixture` converts
Q/P/Z measurements to a nominal task pose without accessing hardware. A clamp
plate changes the height datum if it raises the plastic base.
You can change the nominal keyboard position/yaw with a task JSON, retrain, and
export that policy's geometry. You can also train over bounded XY/yaw placement
variation and reset-joint noise. See [TASK_CONFIGURATION.md](TASK_CONFIGURATION.md).

The released pretrained checkpoints correspond to the original setup. Moving
that physical keyboard does not update their training distribution. For your own
trained policy, match the physical layout to its nominal task configuration and
check that any placement variation is within the distribution you evaluated.

## Export a deployment bundle

```bash
./so101 prepare-deployment \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml \
  --robot-id my_typing_robot --encoder-convention lerobot \
  --out-dir output/my_typing_robot
./so101 deploy output/my_typing_robot NVIDIA
```

Use a two-letter target such as `HE` for P1A policies; Transit15 policies use six
letters. Export and the dry run both check the local LeRobot calibration for the selected
robot ID. Missing or malformed calibration is an error before any robot connection.
The second command is a software dry run and does not open the robot.
The bundle contains the checkpoint, its environment YAML, optional task/physics
JSONs, nominal A-Z geometry, normalization, robot ID and a generated reset target.
Its hashes and geometry are checked before the runner can connect to hardware.
Calibration is checked again on each deployment, including bundles copied to
another machine. This checks the saved calibration file, not physical alignment.

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

`lerobot` means standard calibrated arm degrees converted to simulator radians,
with no additional fitted offsets. `benchmark` reproduces the additional
encoder-to-model corrections used for the published robot. Those corrections are
separate from LeRobot's motor calibration and are not a universal SO-101 property.
Compare physical/model neutral poses and key alignment before a first rollout;
if they disagree, correct the convention or setup rather than copying another
robot's calibration. The offline `joint_calibration` fitter is an optional
advanced diagnostic, not a prerequisite for this workflow.

`rest_pose.json` is a software-derived reset target, not a claimed physical
measurement. The runner measures actual joint arrival before starting the policy.
The policy's q_reset reference remains fixed even when training randomizes reset
arrival. The typing contact is the fixed jaw; the moving jaw must be closed.

## Operator-run deployment

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

Software preparation and dry runs have been tested. This new-user workflow has
not yet been validated on a second physical SO-101.
