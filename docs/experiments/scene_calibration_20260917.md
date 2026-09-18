# Recorded-contact scene calibration — September 17, 2026

`model_2498.pt` typed physical
**NEWTON** with exact release/clearance success using the selected pose and
offsets below. See the [complete result and workflow](physical_typing_success_20260917.md).
The earlier fitting/replay limitations in this record remain unchanged.

## Replay result

**Passed the original type of scene-validation test: reset between keys, then
replay the measured approach and check the first key contact.** The selected
joint/keyboard fit achieved **60/60 powered + 33/33 hand-guided = 93/93** correct
first contacts across the September 16 recordings, including all 26 letters.
Simultaneous extra actuation fails this test. Both recordings were used for
fitting, so this is reproduction of collected contacts, not held-out validation.

The July workflow also reset between keys. Continuous replay was added later
as an extra diagnostic; it is **not a prerequisite for reproducing that original
per-key scene-validation result**. Its failures do not change this 93/93 result.
This result does not by itself establish physical policy performance or clean
transit/release behavior.

### Selected effective keyboard pose

Coordinates describe the keyboard asset origin in the robot-base frame, not an
edge-to-edge ruler distance. This is a fitted simulation scene; the measured
physical-reference task and physical keyboard remain unchanged.

| Parameter | Value |
| --- | ---: |
| X | 259.567686 mm |
| Y | 26.287386 mm |
| Z | 10.914638 mm |
| Roll | 0 degrees |
| Pitch | -0.5 degrees |
| Yaw | -86.921376 degrees |

Exact quaternion XYZW:
`[-0.0030013315867738564, -0.003167092771073995, -0.6878501661003925, 0.7258395900801692]`.

### Selected encoder-to-model offsets

| Joint | Offset (degrees) |
| --- | ---: |
| Shoulder pan | 0 |
| Shoulder lift | +1.3357291219509193 |
| Elbow flex | -10.150392639441748 |
| Wrist flex | +1.4143882688448561 |
| Wrist roll | 0 |

All five offsets were available during contact refinement; pan and wrist roll
remained zero in the selected solution. Apply `q_model = q_LeRobot + offset`
when reading encoders and `q_LeRobot_goal = q_model_goal - offset` when sending
model-angle goals. These are fitted model-coordinate corrections for this robot
and scene; the LeRobot calibration file was not changed.

The [task JSON](../../configs/tasks/replay-calibrated-20260917.json) preserves the
selected scene. The downloaded calibrated release includes `calibration/task.json`,
`mapping-report.json` and `contact-result.json`; keep that pose, mapping and report
together. The original local candidate is named `refined/`. The separate
`continuous-refinement/` candidate was not selected for the successful policy.

### Recurring trend across historical and September fits

Values are additive degrees in
`q_model = q_LeRobot + offset`; the inverse subtraction is used for hardware goals.

| Mapping and evidence | Pan | Lift | Elbow | Wrist flex | Wrist roll |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical locked scene: 78/78 per-key replay, later 89/100 physical policy trials | 0 | +1.933627 | -10.773579 | +1.364765 | 0 |
| September fixed-keyboard four-joint diagnostic | +3.126509 | +0.748687 | -9.612755 | +0.540195 | 0 (fixed) |
| Selected September joint-and-keyboard fit: 93/93 per-key replay | 0 | +1.335729 | -10.150393 | +1.414388 | 0 |

The common trend is a negative elbow correction near -10 degrees, accompanied
by smaller positive lift and wrist-flex corrections. In the earlier powered-data
analysis, applying the historical elbow correction alone reduced mean absolute
Z error from **20.01 to 7.94 mm** on 22 presses excluded from that fit. This is
measured-pose FK evidence of a substantial vertical contribution, not physical
policy success. Full historical offsets reduced that subset's Z error further,
but left substantial horizontal error.

The numerical agreement motivates investigating a shared encoder/model
correspondence issue; contact fitting can also absorb geometry error, so the
mechanical cause remains unresolved. Pan, keyboard pose and the other joint
corrections can compensate for one another. The earlier four-joint diagnostic
failed the original July replay at **0/78** when transferred unchanged, despite
its similar elbow correction. Keep the fitted pose and offsets together and
validate each robot/calibration/fixture combination.

Sources: [historical locked contract](../fixtures/historical_offset_provenance_20260718.md#locked-contract),
[September four-joint fit and transfer](encoder_model_mapping_20260915.md#fully-new-four-joint-fit-and-transfer--2026-09-16),
and the selected local `refined/mapping-report.json` and `contact-result.json`
identified above. The selected September fit used fresh recording-based estimates
and contact refinement; the historical constants were not optimization inputs.

## Inputs and fitting method

The selected data comprised 60 powered leader/follower presses and 33 torque-off
hand-guided presses, covering A–Z. Both recordings entered the final contact
refinement; neither is held-out validation of the selected fit. Powered replay
uses measured follower positions, not leader requests or commanded goals.

The initial fitter associates measured joint states with physical key-down events,
uses fixed-jaw FK and equally weights letters. Key-centre location, USB event
timing and nominal key depression are modeling assumptions. It first compares a
conservative three-offset/rigid-pose fit with an all-joint/full-pose fit. Geometry,
keyboard scale, source recordings and the robot's LeRobot calibration stay fixed.

Newton refinement then searches actual simulated first contacts, resetting
between keys and rejecting simultaneous extra actuation. All five arm offsets and
keyboard XYZ/roll/pitch/yaw can change. Pan/yaw correlations mean the result is an
effective scene correspondence, not independently measured mechanical joint zeros.
The selected candidate reproduced 60/60 powered and 33/33 hand-guided contacts.

Use [the short workflow](../release/REAL2SIM.md) to collect and fit your own data.
Different searches may produce different passing candidates; each candidate's
scene, mapping and contact report must remain paired.

## Continuous replay remains a separate limitation

The selected candidate did not reproduce uninterrupted recorded down/up sequences.
The manual recording produced 38 rather than 33 down presses, including unwanted
9, U, Z and Space contacts; the powered recording also had errors. Isolated first
contacts therefore do not establish clean recorded release/transit behavior.

A separate continuous-sequence search changed wrist flex to about +0.914 degrees
and still failed both recordings. It is not the +1.414-degree candidate used for
93/93 first contacts and the successful physical policy. Do not mix those mappings.

Replay also exhibits millimetre-scale within-step robot drift, and some powered
USB events share a polling batch without an intervening encoder sample. Preserve
those timing and simulation limitations when interpreting errors. They do not
turn the failed sequences into passes or explain away the manual recording's
extra contacts.

## Training and physical result

The selected scene was frozen for 500 P1A updates followed by 2,000 Transit15
updates, using AnchorBench, Newton MJWarp, CUDA graphs, 4,096 environments and seed
1307. Its saved environment matches the selected task. Offsets are kept for the
hardware coordinate conversion; they are not added inside PPO.

The resulting `model_2498.pt` passed 2,048/2,048 strict six-letter simulation
episodes and completed one physical NEWTON trial with the matching mapping.
See the [physical result](physical_typing_success_20260917.md) for measured runtime,
deployment details and remaining qualification.
