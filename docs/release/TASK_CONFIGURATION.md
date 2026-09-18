# Change keyboard placement and randomization

Fresh simulation defaults to the fitted September scene. The physical-reference
drawing is separate; neither example replaces fitting your own robot's scene.
Choose independent task and physics configuration files:

```bash
./so101 train --actuator usd --task-config configs/tasks/keyboard-pose.json \
  --physics-config configs/physics/mjwarp-soft-drives.json --iterations 10
./so101 train --actuator anchorbench \
  --task-config configs/tasks/keyboard-domain-randomization.json --iterations 10
```

These short runs test setup. Increase the training budget for a learned policy.
Use `./so101 probe --task-config FILE` to check reset stability, commanded action
limits, and key-map alignment over several resets before training.

## Physical measurements to a candidate model pose

Use the [bird’s-eye fixture guide](../../FIXTURE_SETUP.md) to identify the plastic
base datum and measure Q/P/Z keycap rim centres in millimetres. The host-only
`./so101 fixture --measurements FILE --out-dir NEW_DIR` command fits the supported
XYZ translation and full orientation and writes `task.json`, a drawing and a
measurement report. Measured tilt replaces the reference scene's fitted tilt;
the asset's own key geometry stays intact. This produces
a candidate simulation layout, not a validated physical setup. It does not
measure the robot, calibrate encoders or validate contact alignment.
`./so101 fixture --task-config FILE --out-dir NEW_DIR` draws an existing nominal
configuration. Both commands work before the Docker image is built.

## Nominal pose

`keyboard.position_m` is the keyboard CAD origin's XYZ position in the robot-base
frame, in metres. `keyboard.yaw_offset_deg` rotates around robot-base Z relative
to the benchmark's orientation, about the keyboard CAD origin. It is not an
absolute CAD yaw. The keyboard mesh and nominal 26-key target map are transformed
together. A nominal height change also translates the support plane by that height.

For a complete orientation, use `keyboard.rotation_xyzw`: an absolute unit
quaternion `[x, y, z, w]` in the robot-base frame, mutually exclusive with
`yaw_offset_deg`. The fixture measurement helper writes this field automatically.
It replaces the old scene rotation, including its fitted roll/pitch. The native
MX Keys asset already includes the deck slope; do not add that slope again as
a scene rotation. With an absolute rotation, the horizontal support plane is
set to the transformed case's lowest point.

`configs/tasks/keyboard-pose.json` demonstrates a 10 mm X shift and 2 degree yaw
change from the fitted September scene, saved as an absolute orientation.
Task files describe overrides to the underlying asset reference, not patches on
the launcher's default example; start from your fitted task and retain its
`keyboard` section when adding randomization. The policy keeps the existing
22-element observation and five arm actions.
Its nominal target coordinates change; normalization and the joint reset reference
stay fixed. Nominal roll/pitch are supported through the absolute quaternion;
vertical and tilt randomization are not exposed.

## Domain randomization

The reset distribution accepts uniform bounds for `keyboard_x_m`, `keyboard_y_m`,
and `keyboard_yaw_deg`, plus a nonnegative `reset_joint_noise_rad`. XY translations
and yaw are expressed in the robot-base/world axes about the nominal fixture.
The randomization example retains the fitted September nominal pose and uses
+/-3 mm, +/-1 degree, and 0.005 radian reset noise.
These are example ranges, not a claim of achieved physical robustness.

The actor sees the nominal key coordinates throughout an episode. It does not
receive the sampled keyboard perturbation or live simulated key coordinates.
Rewards/contact judging can use live geometry, as in the original task. A separate
validation map is computed from the sampled reset transform and known key geometry,
then checked against simulated keys at every reset. This preserves the geometry
check without treating deliberate randomization as a map error.

The robot's observation/action reference q_reset remains fixed; the actual reset
arrival is randomized around it. Reset audits verify the actor map remains fixed.
Evaluate on enough seeds and episodes before claiming coverage of a distribution.

## Measure a fixed policy's placement tolerance

Use an explicit perturbation file to vary actual reset placement while preserving
the trained nominal actor map and policy weights:

```bash
./so101 evaluate --actuator anchorbench --stage transit15 \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --num-envs 1024 --seed 2307 \
  --perturbation-config configs/evaluation/placement-1mm.json
```

Match `--actuator` and `--stage` to your checkpoint; the example assumes an
AnchorBench six-letter policy.

The training environment contract is validated before applying this separate
experiment. The perturbation file accepts only `domain_randomization`, replaces
the reset distribution, and cannot change nominal keyboard geometry or physics.
Reports label this `fixed_policy_reset_robustness` and save the exact ranges.
The supplied zero, +/-1 mm with +/-0.25 degree yaw, and +/-3 mm with +/-1 degree
yaw files use zero robot reset noise. Compare the same checkpoint, seed, episode
count and target-bank hash against the zero-offset control. This tests combined
XY/yaw tolerance; it does not isolate each axis or measure hardware robustness.

Measured results are in [RESULTS.md](RESULTS.md). Changing actuator physics is a
separate experiment; the matched evaluator continues to reject physics-contract
mismatches.

## Reproduce and deploy

Training saves `params/public_task.json` with `params/env.yaml`. Evaluation and
resume automatically discover it beside the configuration/checkpoint. Keep it
with `public_physics.json` when both were used. The task JSON hash becomes part of
the checkpoint's environment contract. A changed/missing config is rejected during
evaluation. Evaluation retains the requested DR distribution; benchmark checkpoints
without task JSON retain the original deterministic fixture evaluation.

`prepare-deployment` exports the nominal map and normalization from the trained
environment. Hardware uses that nominal map; no simulator-only sampled pose is
available to it. See [the hardware guide](HARDWARE_PREPARATION.md).
