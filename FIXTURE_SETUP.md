# Measure the keyboard fixture

Use this guide for the reference Logitech MX Keys geometry and the stock
`base_so101_v2` plastic SO-101 base. A clamp is fine if the base is rigid and its
mounting plane stays level. Measure the **plastic base**, not the clamp, table
edge, servo cover, or cable bracket. A different base, keyboard, typing jaw or
keyboard tilt needs geometry validation before these measurements are usable.

![Bird’s-eye model drawing with robot axes, base datum and Q/P/Z landmarks](docs/fixtures/reference-birdseye.svg)

[Open the full-size drawing](docs/fixtures/reference-birdseye.svg). It is a model
projection, not a printable alignment template or a reachability map. The grid
spacing is 50 mm. The arm and clamp are omitted so the base datum is visible.

## 1. Establish the robot measurement frame

Look down at the base. Its broad plastic outline must match the drawing. The
controller/cable bracket is behind the base; the straight front face is toward
the keyboard. If your base does not match, do not substitute the clamp outline.

Define **D** as the intersection of three planes: the frontmost flat face of the
plastic base, its left/right symmetry plane, and its flat underside plane. This
is a constructed measurement point; no hole or mark at D is required. Use a
straightedge against the front face, find its midpoint, and extend perpendicular
lines with a square onto paper/tape. Keep these lines fixed to the base.

| Measurement | Direction from D | How to measure |
| --- | --- | --- |
| X | Forward from the base toward the keyboard | Perpendicular distance from the front-face line; not a diagonal ruler distance |
| Y | Left when standing behind the robot looking toward the keyboard | Signed lateral distance from the base centreline; right is negative |
| Z | Up from the plastic base underside plane | Keycap rim height minus base-underside height, both measured from the same level reference |

The exact CAD offset of D is `(64.635455, 0, -2.400258) mm` in the robot-base
frame. The converter applies this offset. Do not enter these CAD coordinates
as your ruler readings. If the plastic underside is flush on the table, that
table is the height zero. If a clamp plate raises the base, measure and subtract
that height. Do not assume that clamping implies a particular height.

![Side schematic showing the base underside height datum](docs/fixtures/height-datum.svg)

## 2. Measure Q, P and Z on the released keyboard

Secure the keyboard in the intended typing position, including the mat and feet
configuration. Use the fixed typing jaw with the moving jaw closed at deployment.
Identify the centre of each **keycap outline** on Q, P and Z. For height, use the
plane of the cap's top rim at that centre, not the recessed bowl or printed letter.
A straightedge across the rim helps establish this plane. Do not depress the keys.

Project each centre vertically onto the measurement plane (a square or plumb
reference helps) and record its X, Y and Z in millimetres. These are mechanical
landmarks for placement; they are not the policy's key-body target coordinates.
Do not use perspective distances from an uncalibrated photograph.

Create the drawing and a blank measurement form; this needs only host Python 3,
with no Docker build, GPU, calibration or robot connection:

```bash
./so101 fixture --out-dir output/reference-fixture
cp output/reference-fixture/measurement-template.json output/my-measurements.json
# Replace every null with your measured X, Y and Z in millimetres.
```

The form contains exactly three entries:

```json
{
  "Q": [null, null, null],
  "P": [null, null, null],
  "Z": [null, null, null]
}
```

The reference model predicts the following readings. **These are model values,
not measurements of your robot**; do not copy them into a measurement record.

| Key | X from D (mm) | Y from D (mm) | Z above D (mm) |
| --- | ---: | ---: | ---: |
| Q | 199.110 | 184.581 | 8.131 |
| P | 203.272 | 13.541 | 7.980 |
| Z | 161.205 | 169.519 | 7.117 |

The small reference heights are intentional consequences of the saved effective
model transform. The keyboard sits low relative to the robot base in that
fixture. The historical `4.44 mm` mat and `125.28 mm` edge gap do **not** specify
all three-dimensional mounting relationships. That historical gap also has no
uniquely specified housing datum, so it is not the measurement input used here.
If your flat-table arrangement produces different heights, record those values
and train for that geometry; do not force the old checkpoint to fit the hardware.

## 3. Convert measurements into the training pose

```bash
./so101 fixture --measurements output/my-measurements.json \
  --out-dir output/my-fixture
# Open output/my-fixture/birdseye.svg and compare it to the actual layout.
./so101 probe --task-config output/my-fixture/task.json
./so101 train --actuator anchorbench --stage p1a \
  --task-config output/my-fixture/task.json --num-envs 64 --iterations 10
```

The last command is a startup check. Continue with the
[P1A-to-six-letter training recipe](docs/release/TRAINING.md), retaining this task
configuration. The tool fits XYZ translation and yaw while preserving the
reference keyboard roll/pitch. It rotates about robot Z, converts millimetres to
metres and writes the existing task JSON schema. No policy is modified.

`measurement-report.json` records the inputs, source geometry hashes, pose and
per-key fit residuals. A maximum residual above **1 mm** rejects the fit before
creating output. This is a measurement-consistency check, **not** evidence that
a policy tolerates 1 mm placement error. Check axis signs, units, landmark
locations, base shape and keyboard tilt on rejection. Unsupported roll/pitch or
a different board cannot be fixed by merely translating the CAD origin.

A successful three-point fit still does not prove the other keys, jaw, robot
calibration or contacts match. Check additional keys across the board against
the model and inspect the simulation probe before training or physical use.

To draw an existing nominal task pose instead of fitting measurements:

```bash
./so101 fixture --task-config output/my-fixture/task.json \
  --out-dir output/my-fixture-review
```

Drawings and `expected-landmarks.json` show the nominal pose only, not randomized
reset samples. Output directories must be new, protecting existing measurements.

## 4. Keep the two deployment workflows distinct

- **Train for your fixture:** use the measured pose before training, evaluate the
  resulting checkpoint and export its matching `params/env.yaml`. Keep the
  keyboard fixed in the measured position. Re-measure if the base or board moves.
- **Use the optional ~19k checkpoint:** physically reproduce its saved geometry.
  Use the reference drawing and expected measurements for comparison. A new
  measured pose does not reposition that checkpoint's trained map; train a new
  policy if the reference fixture cannot be reproduced.

In both cases, perform standard LeRobot calibration for **your** SO-101.
Calibration determines encoder coordinates; it does not register the keyboard
or compensate for a raised clamp plate. Export with `--encoder-convention lerobot`
unless deliberately reproducing the historical robot's separate fitted offsets.
Follow [hardware preparation](docs/release/HARDWARE_PREPARATION.md) for calibration,
export, the offline deployment dry run and the operator-controlled execution gate.
Start physical validation under supervision after checking geometry and reset
alignment. The helper never opens a serial device or commands the robot.

## Geometry provenance and limits

The drawing comes from the pinned robot USD used by the Docker runtime and the
repository's MX Keys asset. [model_geometry.json](docs/fixtures/model_geometry.json)
contains their hashes, the reference pose, base outline and keycap rim centres.
The helper checks the local keyboard/configuration hashes before using it.
`python3 scripts/build_fixture_geometry.py --output FILE` regenerates the asset
inside the pinned simulator container (numpy and USD/pxr required there only).

The authoritative reference pose is `(262.5651584, 12.1353200, 1.7813943) mm`,
with quaternion XYZW `(-0.0309082351, 0.0307819167, -0.6977733512, 0.7149891643)`.
Its CAD yaw is not a physical squareness reading. The helper applies the same
robot-Z yaw convention as [task configuration](docs/release/TASK_CONFIGURATION.md).

This is a software measurement aid derived from an effective calibrated model.
The physical correspondence of these datums and keycap surfaces still needs
verification on the recipient's hardware. It is not an independently qualified
physical jig, automated camera registration, or an A-Z contact validation probe.
