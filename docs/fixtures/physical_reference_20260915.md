# Frozen physical-reference placement — 2026-09-15

This preserves the datum-based starting placement for the clamped SO-101 and
Logitech MX Keys setup. Keep the physical base and keyboard on the same tabletop
and keep this reference record unchanged. The September 17 contact fit saved a
separate effective simulation pose; it did not move the real keyboard or replace
these measurements. A physical setup change requires a new dated measurement.

## Physical setup used for the September NEWTON checkpoint

This is the real fixture used for the September 16 recordings and the successful
September 17 `model_2498.pt` trial. To reproduce its placement, stand behind the
robot looking toward the keyboard and measure the **plastic base**, excluding
the clamp, motor cover and cable bracket:

| Physical setup item | Placement used |
| --- | --- |
| Hardware | SO-101 with the same fixed typing jaw; moving jaw fully closed; Logitech MX Keys keyboard |
| Support plane | Plastic base and keyboard directly on the same flat tabletop; base clamped securely; no keyboard mat or added riser |
| Forward gap | About **130 mm (13 cm)** from the keyboard-facing flat face of the plastic base to the nearest long keyboard **case edge**, measured perpendicular to the edge along the base centreline |
| Edge orientation | Nearest long keyboard edge parallel to the base's front/back edges: approximately zero physical edge yaw |
| Left/right alignment | Extend the short straight segment on the base's right side forward toward the keyboard; it lines up with the bottom-right corner of the **right Ctrl key**, next to the left-arrow key |

Use the [physical-reference drawing](physical-reference-birdseye-20260915.svg)
to cross-check the layout. Its lateral projections are about **173.7 mm** from
the widest left base edge to the keyboard's left end and **145.6 mm** on the
right. These side distances, and the more precise **129.17 mm** front gap below,
are computed from the datum-based reference model. The operator reported the
physical front gap as approximately **130 mm**; the model decimals are not
independent ruler measurements or a demonstrated placement tolerance.

Keep this real fixture fixed. The September 17 fit adjusted the **simulation**
keyboard pose and encoder-to-model joint mapping to reproduce recorded contacts;
do not move the real keyboard by that model correction. Another robot still
needs its own LeRobot calibration and [alignment checks](../release/ALIGNMENT.md).
A different fitted training scene requires its own policy.

## Preserved datum-based reference model

The tracked pose record is [`physical_reference_20260915.json`](physical_reference_20260915.json).
The portable physical-reference configuration is
[`configs/tasks/physical-reference-20260915.json`](../../configs/tasks/physical-reference-20260915.json).
This is the default `./so101 fixture` drawing. Pass it explicitly with
`--task-config` to view the physical reference in simulation; bare `./so101 scene`
shows the separately fitted training scene.
It uses the existing keyboard asset, tabletop height and native keyboard slope,
parallel keyboard/base edges, and the physical landmark that the short straight
segment on the robot base projects through the bottom-right corner of the right
Ctrl key. This reference model's keyboard CAD-origin pose is:

```text
position_m       = (0.2606351862725319, 0.0140614168264038, 0.0076366833485381)
rotation_xyzw   = (0.0, 0.0, -0.7074538725855443, 0.7067595193301018)
```

This is a frozen **physical-reference candidate**, not a claim that the current
encoder/FK mapping or contact replay is validated. The closest base-front to
keyboard-case gap is approximately 129.2 mm edge to edge.

Physical-reference drawing: [physical-reference-birdseye-20260915.svg](physical-reference-birdseye-20260915.svg).
Its rounded dimensions describe this starting reference, not the later fitted scene.

## Physical reference versus fitted training scene

The September trained policy uses
[`replay-calibrated-20260917.json`](../../configs/tasks/replay-calibrated-20260917.json),
matching the fitted task in the [released calibrated bundle](../../README.md#september-calibrated-scene-example).
Its [bird's-eye drawing](replay-calibrated-birdseye-20260917.svg) is generated from
that task and the same asset geometry.

| Model quantity in robot-base coordinates | Physical-reference model | Fitted training scene |
| --- | ---: | ---: |
| Origin X, mm | 260.635 | 259.568 |
| Origin Y, mm | 14.061 | 26.287 |
| Origin Z, mm | 7.637 | 10.915 |
| Roll / pitch / yaw, degrees | 0 / 0 / -90.056 | 0 / -0.500 / -86.921 |
| Front base-to-case gap along base centreline, mm | 129.172 | 129.437 |

The model-origin change is **-1.068 mm X, +12.226 mm Y, +3.278 mm Z**, with
**+3.135 degrees yaw** and **-0.500 degrees pitch**. These are effective model
corrections, not instructions to shift or raise the physical keyboard. Rotation
means the change is not the same displacement at every key or case edge.
The front gaps are computed from the case-hull projection, not new physical
measurements. Their small difference does not make the lateral shift or rotation
negligible. See the [joint-mapping convention](../release/REAL2SIM.md#what-the-fitted-offsets-mean).

The saved joint offsets describe encoder/model arm correspondence; they do not
move the base outline or keyboard in a bird's-eye drawing. Use the measured
reference for physical placement checks and the fitted task for this policy's
simulation. Keep each fitted task paired with its own mapping report.

## Original physical-reference dimension checks

Dimensions computed from the case hull and plastic `base_so101_v2` mesh, with asset hashes
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
motor. The broad foot measures 17.50 mm
(front region X > 30 mm and side regions |Y| > 30 mm); the 72.00 mm maximum
belongs to the narrow upright. The user's approximate 20 mm foot measurement
is a comparison to 17.50 mm, not 72.00 mm; exact physical agreement is not
established by the approximate measurement.
Both case and base undersides are at the same model Z plane. The base's
short right straight projects to the right Ctrl edge within 0.02 mm.
These are simulation dimensions for physical ruler checks, not proof of an IRL match.

For a stationary Newton view that stays open until the window closes:
`./scripts/view_frozen_scene.sh` shows the September 15 reference. To inspect the
current fitted example instead, use
`./so101 scene --task-config configs/tasks/replay-calibrated-20260917.json`.
Both render without advancing physics or running a policy.
