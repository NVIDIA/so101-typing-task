# Place the keyboard and verify the scene

Start with the September setup: a clamped SO-101 and Logitech MX Keys on the
same flat table, without a mat. Measure the plastic robot base, excluding the
clamp, motor cover and cable bracket. Use the fixed typing jaw with the moving
jaw fully closed.

| Placement check | September reference |
| --- | --- |
| Support | Plastic base and keyboard directly on the same flat tabletop |
| Forward gap | About **130 mm** from the keyboard-facing flat base face to the nearest long keyboard case edge |
| Orientation | The long keyboard edge is parallel to the base's front/back edges |
| Side alignment | Looking from behind the robot, project the short straight right base edge toward the keyboard: it lines up with the bottom-right corner of the **right Ctrl key**, beside the left arrow |

![September physical placement reference](docs/fixtures/physical-reference-birdseye-20260915.svg)

The [measurement record](docs/fixtures/physical_reference_20260915.md) includes
asset dimensions and additional datums. Its precise drawing dimensions are
model projections; the physical front gap was measured as approximately 130 mm.
They are not demonstrated placement tolerances.

## Keep physical placement and simulation calibration separate

The policy has no vision of the keyboard. First establish its position relative
to the robot base, then keep the fixture fixed while checking recorded contacts.
A ruler match alone does not establish that encoder angles produce matching
jaw poses in simulation.

The September workflow fitted an effective simulation keyboard pose and joint
mapping from real recordings. **The real keyboard stayed fixed.** Use the
physical-reference drawing above for placement, and the fitted scene below for
the released policy's simulation. Do not move the real keyboard by the fitted
model correction.

| View | Use |
| --- | --- |
| [Physical reference](docs/fixtures/physical-reference-birdseye-20260915.svg) | Ruler checks and the starting layout; default `./so101 fixture` view |
| [Fitted training scene](docs/fixtures/replay-calibrated-birdseye-20260917.svg) | September checkpoint's effective simulation scene; default fresh `train`, `probe` and `scene` layout |

For a setup arranged to the reference, draw and save a starting estimate:

```bash
./so101 fixture --out-dir output/my-fixture
```

This creates `task.json`, `birdseye.svg`, predicted landmarks and a blank
measurement form. Check your actual placement against the drawing. Copying this
reference does not calibrate your robot: complete your own LeRobot calibration,
then use the [record → fit → train guide](docs/release/REAL2SIM.md).

Hand-guided recordings contain measured joint angles and physical key events.
They calibrate the scene and encoder/model mapping; they are **not policy
training demonstrations**. Resolve replay contact failures before PPO training.
Keep the measured reference task and the fitted task as separate files.

## A different placement: measure Q, P and Z

For a different layout, fit a starting pose from three released-key landmarks.
The helper supports the repository's MX Keys geometry and `base_so101_v2` plastic
base. A different keyboard, base or typing jaw needs matching asset geometry.

Define datum **D** at the intersection of the plastic base's frontmost flat
face, left/right symmetry plane and flat underside plane. A straightedge against
the front face and a square establish these reference lines; D need not be a
visible hole or mark.

| Axis | Measurement from D |
| --- | --- |
| X | Forward toward the keyboard, perpendicular to the base front face |
| Y | Left when looking from behind the robot; right is negative |
| Z | Up from the plastic base underside plane |

If the base underside rests on the table, tabletop height is zero. If a clamp
plate raises it, subtract that height from keycap heights. The converter handles
D's CAD offset; enter ruler measurements, not robot-origin coordinates.

![Base underside height datum](docs/fixtures/height-datum.svg)

Measure the centre of the **keycap outline at its top rim** for Q, P and Z,
with keys released. Use the rim plane, not the recessed bowl or printed letter.
A straightedge can extend that plane to a ruler. Record X/Y/Z in millimetres;
do not use perspective distances from an uncalibrated photograph.

```bash
./so101 fixture --out-dir output/measurement-form
cp output/measurement-form/measurement-template.json output/my-measurements.json
# Replace every null with your measured X, Y and Z in millimetres.
```

```json
{
  "Q": [null, null, null],
  "P": [null, null, null],
  "Z": [null, null, null]
}
```

Fit and inspect the measured reference:

```bash
./so101 fixture --measurements output/my-measurements.json --out-dir output/my-fixture
./so101 scene --task-config output/my-fixture/task.json
```

Use a new output directory for each fit. The helper fits XYZ translation and
full orientation of the existing keyboard asset, preserving its native key
slope. It saves metres and an absolute `rotation_xyzw` quaternion. A maximum
landmark residual above **1 mm** rejects the fit before creating output; this
checks measurement consistency, not policy robustness. Check units, axis signs,
landmark locations and asset dimensions if the fit fails.

## Record, fit, train and deploy

Use `output/my-fixture/task.json` as the measured starting reference in the
[real-to-sim guide](docs/release/REAL2SIM.md). Record several hand-guided typing
trajectories, fit the scene and joint mapping, and check intended first contacts
in per-key replay before training. Reserve separate recordings for validation.
Train with the resulting fitted `task.json`, keeping its mapping report paired
with the checkpoint.

For any fitted scene, generate its own labelled view:

```bash
./so101 fixture --task-config output/my-scene/task.json --out-dir output/my-scene-view
```

Joint offsets change the arm pose, not the base or keyboard outline. A fitted
keyboard pose does change that outline. Drawings omit the arm and clamp, show
the nominal pose only, and are not printable alignment templates or reachability
maps. The [September comparison](docs/fixtures/physical_reference_20260915.md#physical-reference-versus-fitted-training-scene)
records both saved poses and their relationship.

Follow [alignment validation](docs/release/ALIGNMENT.md) and
[hardware preparation](docs/release/HARDWARE_PREPARATION.md) for reserved-recording
checks, physical Q/G/P probes, export and deployment. Your robot needs its own
LeRobot calibration and verified joint mapping. A released checkpoint also
needs alignment to its frozen training scene; a new measurement file does not
reposition an already trained policy's map.

## Geometry source

The drawings use the pinned robot USD and repository MX Keys asset.
[model_geometry.json](docs/fixtures/model_geometry.json) records their hashes,
base outline and keycap landmarks. The helper checks source hashes before
applying the selected task pose. `scripts/build_fixture_geometry.py` regenerates
this geometry inside the pinned simulator container. See
[task configuration](docs/release/TASK_CONFIGURATION.md) for pose and
randomization options.
