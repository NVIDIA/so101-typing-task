# Frozen physical-reference placement — 2026-09-15

This preserves the datum-based starting placement for the clamped SO-101 and
Logitech MX Keys setup. Keep the physical base and keyboard on the same tabletop
and keep this reference record unchanged. The September 17 contact fit saved a
separate effective simulation pose; it did not move the real keyboard or replace
these measurements. A physical setup change requires a new dated measurement.

The tracked pose record is [`physical_reference_20260915.json`](physical_reference_20260915.json).
The portable physical-reference configuration is
[`configs/tasks/physical-reference-20260915.json`](../../configs/tasks/physical-reference-20260915.json).
Pass it explicitly with `--task-config`; the earlier photo `candidate-task.json`
uses a different position and is not this frozen reference.
The generated reference task is
`ctrl-line-aligned.json` (local artifact: `output/fixture-photos-20260915/manual-recordings/20260915T220312.112879Z/xy-yaw-offset-fit/placement-sweep/ctrl-line-aligned.json`).
It uses the existing keyboard asset, tabletop height and native keyboard slope,
parallel keyboard/base edges, and the physical landmark that the short straight
segment on the robot base projects through the bottom-right corner of the right
Ctrl key. The keyboard pose is:

```text
position_m       = (0.2606351862725319, 0.0140614168264038, 0.0076366833485381)
rotation_xyzw   = (0.0, 0.0, -0.7074538725855443, 0.7067595193301018)
```

This is a frozen **physical-reference candidate**, not a claim that the current
encoder/FK mapping or contact replay is validated. The closest base-front to
keyboard-case gap is approximately 129.2 mm edge to edge. The bird’s-eye and
datum sheets are beside the task file in `placement-sweep/`.

The placement is independent of the optional historical joint convention. The
same task must be used for direct (zero-added-offset) and benchmark-offset
diagnostics so that placement and encoder mapping are not changed together.

Physical-reference drawing: [physical-reference-birdseye-20260915.svg](physical-reference-birdseye-20260915.svg).
Its rounded dimensions describe this starting reference, not the later fitted scene.

## Physical reference versus fitted training scene

Comparison documented **September 17, 2026**. The current training scene uses
[`replay-calibrated-20260917.json`](../../configs/tasks/replay-calibrated-20260917.json),
matching the selected `refined/task.json` in the [calibration record](../experiments/scene_calibration_20260917.md).
Its [bird's-eye drawing](replay-calibrated-birdseye-20260917.svg) is generated from
that task and the same asset geometry.

| Keyboard asset pose in robot-base coordinates | Physical-reference model | Fitted training scene |
| --- | ---: | ---: |
| Origin X, mm | 260.635 | 259.568 |
| Origin Y, mm | 14.061 | 26.287 |
| Origin Z, mm | 7.637 | 10.915 |
| Roll / pitch / yaw, degrees | 0 / 0 / -90.056 | 0 / -0.500 / -86.921 |

The model-origin change is **-1.068 mm X, +12.226 mm Y, +3.278 mm Z**, with
**+3.135 degrees yaw** and **-0.500 degrees pitch**. These are effective model
corrections, not instructions to shift or raise the physical keyboard. Rotation
means the change is not the same displacement at every key or case edge.

The saved joint offsets describe encoder/model arm correspondence; they do not
move the base outline or keyboard in a bird's-eye drawing. Use the measured
reference for physical placement checks and the fitted task for this policy's
simulation. Keep each fitted task paired with its own mapping report.

## Original physical-reference dimension checks

Measurement cross-check, 2026-09-16 (frozen placement unchanged): dimensions
recomputed from the case hull and plastic `base_so101_v2` mesh, with asset hashes
checked against the runtime. Looking from behind the robot toward the keyboard:

| Measurement | mm |
| --- | ---: |
| Keyboard case width × depth × height, asset axes | 430.20 × 133.63 × 20.07 |
| Keyboard case footprint in the frozen scene | 430.25 × 133.62 |
| Plastic robot base width × depth × full mesh height | 110.93 × 87.00 × 72.00 |
| Broad plastic foot: underside to flat top (upright excluded) | 17.50 |
| Front base edge to near case edge, along robot centreline | 129.17 |
| Keyboard left end beyond widest left base edge, lateral projection | 173.73 |
| Keyboard right end beyond widest right base edge, lateral projection | 145.59 |

Base height includes its upright plastic structure, excluding the clamp and
motor. A direct mesh check on 2026-09-16 measured the broad foot at 17.50 mm
(front region X > 30 mm and side regions |Y| > 30 mm); the 72.00 mm maximum
belongs to the narrow upright. The user's approximate 20 mm foot measurement
is a comparison to 17.50 mm, not 72.00 mm; exact physical agreement is not
established by the approximate measurement. See the
side/front mesh projections (local artifact: `output/frozen-layout-measurements-20260916/base-height-check.png`).
Both case and base undersides are at the same model Z plane. The base's
short right straight projects to the right Ctrl edge within 0.02 mm. Earlier
datum sheets label the far-right corner as the bottom-right corner; the actual
near-right corner has this same lateral alignment to within 0.02 mm.
These are simulation dimensions for physical ruler checks, not proof of an IRL match.
The regenerated dimensioned image (local artifact: `output/frozen-layout-measurements-20260916/birdseye-measurements.png`)
and numbers (local artifact: `output/frozen-layout-measurements-20260916/measurements.json`)
use the unchanged frozen task hash.

For a stationary Newton view that stays open until the window closes:
`./scripts/view_frozen_scene.sh` shows the September 15 reference. To inspect the
current fitted example instead, use
`./so101 scene --task-config configs/tasks/replay-calibrated-20260917.json`.
Both render without advancing physics or running a policy.
