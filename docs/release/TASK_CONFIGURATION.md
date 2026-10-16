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

## Measure your fixture

Use the [September placement guide](../../FIXTURE_SETUP.md) for the reference
layout. For a setup arranged to that reference, save a starting estimate:

```bash
./so101 fixture --out-dir output/my-fixture
```

This writes `task.json`, `birdseye.svg`, predicted landmarks and a blank measurement
form. Check the drawing against your physical placement before recording contacts.

The current fixture drawing and FK probe use the pinned Workshop SO-101 USD
with the robot root rotated 90 degrees toward the keyboard. The September
reference and its published AnchorBench policy belong to the earlier robot
asset. Use your newly measured Q/P/Z coordinates and record fresh contacts for
this model; an old fitted pose or checkpoint does not establish alignment.

For a **different placement**, instead measure released Q, P and Z keycap rim
centres. The helper uses the repository's MX Keys keyboard and `base_so101_v2`
plastic base; a different keyboard, base or jaw needs matching asset geometry.

Datum **D** is the intersection of the plastic base's keyboard-facing flat face,
left/right symmetry plane and flat underside plane. Use a straightedge and square
to establish these lines. Looking from behind the robot toward the keyboard:

| Axis | Measurement from D, in millimetres |
| --- | --- |
| X | Forward toward the keyboard, perpendicular to the base front face |
| Y | Left is positive; right is negative |
| Z | Up from the plastic base underside plane |

![Base underside height datum](../fixtures/height-datum.svg)

Measure the centre of each keycap outline at its **released top rim**, not the
recessed bowl or printed letter. If a clamp plate raises the base, subtract that
height from tabletop-to-keycap measurements. The converter handles D's CAD offset;
enter ruler measurements rather than robot-origin coordinates.

```bash
./so101 fixture --out-dir output/measurement-form
cp output/measurement-form/measurement-template.json output/my-measurements.json
```

Edit `output/my-measurements.json`, replacing every `null` with the measured
Q/P/Z X/Y/Z coordinates in millimetres. Then fit and inspect the layout:

```bash
./so101 fixture --measurements output/my-measurements.json --out-dir output/my-fixture
./so101 scene --task-config output/my-fixture/task.json
```

Use a new output directory for each fit. The helper fits XYZ translation and full
orientation while preserving the asset's native key slope. A maximum landmark
residual above 1 mm rejects the fit; check units, signs and rim locations. This
checks measurement consistency, not physical typing accuracy.

Both approaches produce a **starting reference**. Continue with
[recorded-contact scene calibration](REAL2SIM.md) before training. Keep the measured
reference separate from the resulting fitted `output/my-scene/task.json` and its
mapping report. Draw the fitted scene with:

```bash
./so101 fixture --task-config output/my-scene/task.json --out-dir output/my-scene-view
```

The fixture helper runs on host Python before Docker is built; `scene` needs the
simulator image. Drawings show nominal model geometry, omit the arm and clamp,
and are not printable alignment templates. The
[geometry manifest](../fixtures/model_geometry.json) pins the source assets.

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

The [archived placement reports](https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark/tree/44d8ad43e2619e0d6e89528fea07f887de93678e/evaluation)
(`public-transit15-placement-{zero,1mm,3mm}.json`, included by `./so101 download --all`)
cover the earlier `model_549.pt`, not the September policy. Changing actuator
physics is a separate experiment; the matched evaluator continues to reject
physics-contract mismatches.

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
