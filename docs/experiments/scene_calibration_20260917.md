# Recorded-contact scene calibration — September 17, 2026

**Update, 14:36 PDT:** training completed and `model_2498.pt` typed physical
**NEWTON** with exact release/clearance success using the selected pose and
offsets below. See the [complete result and workflow](physical_typing_success_20260917.md).
The earlier fitting/replay limitations in this record remain unchanged.

## Current September 16 replay result — documented September 17

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

The selected local evidence is `output/scene-calibration-20260917/refined/`:
`task.json` holds the exact keyboard pose, `mapping-report.json` holds the
`joint_scene` offsets/provenance, and `contact-result.json` records the 93/93
per-key result. Use this candidate when referring to the September 16 replay
success. The later `continuous-refinement/` candidate has a different wrist-flex
offset and is not the candidate that established 93/93.

### Recurring trend across historical and September fits

Comparison documented **September 17, 2026**. Values are additive degrees in
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

## Earlier numerical fits and additional diagnostics

Offline fitting and Newton MJWarp replay on September 17, approximately
09:50–10:00 PDT. No robot motion or LeRobot calibration change. This implements
the hardware-to-simulation calibration stage described in
[advanced scene diagnostics](scene_diagnostics.md#fit-joint-offsets-and-an-effective-keyboard-pose).
It does **not** yet produce a qualified calibration for physical typing.

## Inputs and method

The measured September 15 keyboard/base task remains unchanged. Fresh parameters
are estimated from September 16 powered follower measurements in
`mapping-collection-20260916/20260916T220434.751544Z/recording.json`:
60 complete presses, all 26 letters. The separate manual, torque-off capture
`mapping-validation-20260916/20260916T231850.429607Z/recording.json` supplies
33 validation presses. Both paths are under the local
`output/fixture-photos-20260915/` directory. These sessions were inspected in
earlier experiments; they are regression evidence, not pristine unseen data.

`scripts/calibrate_scene.sh` selects the measured sample nearest every recorded
key-down and fits three joint offsets plus a rigid keyboard XYZ/yaw. Pan and
wrist roll stay zero to avoid the pan/keyboard-yaw ambiguity. The fixed jaw tip,
robot geometry and keyboard scale stay fixed. Six deterministic optimization
starts use only fitting data, with equal total weight per letter. No historical
offset or pose constants are optimization inputs. All original bytes are retained.

USB receipt timestamps, centre-of-key contact and half-travel key-down are
explicit assumptions. The fitter does not infer motor homing registers, identify
the unique mechanical cause, or claim the effective scene is a ruler measurement.

## Initial numerical-fit result — superseded by contact refinement

Joint offsets in pan/lift/elbow/wrist-flex/wrist-roll order:
`[0, +0.835729, -10.150393, +0.914388, 0]` degrees.
Use `q_model = q_LeRobot + offset`; hardware goals use the inverse subtraction.

Effective keyboard origin: `[260.567686, 26.287386, 9.914638]` mm in robot-base
coordinates; yaw `-86.421376` degrees with roll/pitch zero. Relative to the
measured reference, this is approximately `[-0.068, +12.226, +2.278]` mm and
`+3.635` degrees yaw. It is a separate candidate, not a changed physical datum.
No fitted parameter reached its bound.

| Check | Fit recording | Separate manual recording |
| --- | ---: | ---: |
| Mean contact-position residual | 3.703 mm | 5.177 mm |
| Physical down/up edges | 120 | 66 |
| Newton down/up edges | 118 | 70 |
| Exact full-sequence match | Fail | Fail |

The manual replay includes all 33 intended down presses in order, with two
additional presses: `9` at 24.95 s and `U` at 33.20 s. These extras invalidate
the result. The fitting replay also has misses, repeated contacts and extra
keys; similar total event counts do not imply success. Maximum tip drift during
a physics step was 2.92 mm (fit) and 2.12 mm (manual), separately from the displayed
prescribed-pose check. This remains a geometry/contact diagnostic, not actuator
tracking validation.

An optional effective-tilt fit reduced the fitting mean only to 3.648 mm and
validation mean to 5.039 mm. Its fitting replay produced 124 edges versus 120
and failed; the case extends 1.54 mm below the reference support plane. It was
not selected or validated for deployment.

Local evidence: `output/scene-calibration-20260917/`. The `calibration/` directory
is the end-to-end command's bundle, with task, mapping, original recording copies,
logs, traces and replay summary. Earlier `event-fit/` and `event-fit-tilt/`
directories preserve the comparison runs. Raw hardware recordings are not shipped
with the source repo.

## Interpretation of the initial numerical fit

Fresh fitting independently returns an elbow correction near the historical
negative ten-degree trend. That supports a repeatable correspondence correction,
not a universal offset or a proven encoder-zero cause. The remaining work is
contact-level diagnosis/refinement on fitting data, followed by a frozen replay
on separate data and physical Q/G/P commands. The historical 78/78 result used
per-key resets and first-contact checks; it is not equivalent to the continuous
down/up test here. Do not train for physical transfer on this candidate yet.

## Joint and pose contact refinement, September 17, 10:05–10:25 PDT

The user requested optimizing keyboard layout and offsets together, including
shoulder pan and wrist roll. The numerical fitter now supports all five offsets
and full keyboard XYZ/roll/pitch/yaw. With both pan and yaw free, a weak offset
prior chooses among equivalent coordinate representations; it does not identify
unique mechanical zero angles. Both September recordings were used for fitting.

The all-free numerical candidate did not improve contact replay sufficiently.
The new Newton first-contact search then compared that candidate and the fresh
three-joint fit. It adjusted all five joint offsets and the six keyboard pose
parameters using actual simulated key actuation, retaining the source encoder
trajectories. After 24 evaluations it achieved **60/60 powered + 33/33 manual =
93/93 correct first-key contacts**, rejecting simultaneous extra actuation.
Each press is reset independently and scored through the first actuation;
this is comparable in purpose to the original July gate, not identical capture
timing or evidence of clean complete replay.

The passing first-contact candidate has offsets
`[0, 1.335729, -10.150393, 1.414388, 0]` degrees,
keyboard position `[259.567686, 26.287386, 10.914638]` mm, and keyboard RPY
`[0, -0.5, -86.921376]` degrees. Pan and wrist roll were adjustable but remained
zero in this selected candidate. The case remains about 1.62 mm above the
measured reference support plane. Evidence is under
`output/scene-calibration-20260917/refined/`.

Uninterrupted replay of that same candidate still failed: the manual recording
produced 38 rather than 33 down presses, including unwanted `9`, `U`, `Z` and
`SPACE` contacts. The powered recording also has missing/repeated/wrong contacts.
Passing isolated approaches therefore does not settle transit/release geometry.

A second search scored the full raw down/up sequences directly, evaluating
50 candidate replays in two rounds. It changed wrist flex to `0.914388` degrees
and retained the other parameters above. Its sequence edit distances are 25
(powered) and 6 (manual); zero would be an exact match. The second round did not
improve the best score. All five offsets and six pose components were explored.
Evidence and independently replayable outputs are under
`output/scene-calibration-20260917/continuous-refinement/`. This candidate remains
diagnostic and must not be described as making every full trajectory work.

Independent single-environment replays confirmed failure: 114 simulated edges
versus 120 physical edges for the powered recording, and 72 versus 66 for the
manual recording. The manual sequence retains every intended press but adds
`9`, `U` and `SPACE`. Each candidate directory has a `replay-summary.json` linking
the unchanged full reports with hashes. The 93/93 first-contact result belongs
to `refined/`; it must not be silently attributed to the different continuous
candidate's wrist-flex offset.

The simplified `./scripts/real2sim.sh fit` entry point was subsequently run
end-to-end on these same two September 16 captures. It again passed 93/93
reset-per-key contacts and saved a training task plus deployment mapping in
`output/scene-calibration-20260917/lean-workflow-check/`. This checks the packaged
workflow on existing data; no new hardware recording or policy training was run.
That rerun produced another passing candidate (including elbow -9.650421 degrees);
it does not overwrite the selected `refined/` values above. Each fit's mapping
must stay paired with its own task and contact report.
The [short user guide](../release/REAL2SIM.md) is the recommended entry point.

Some physical up/down edges in the powered recording share a USB polling batch:
three release-to-repress intervals are about 1.5 microseconds in the saved
receipt timestamps, with no intervening encoder sample. No edges or samples
were deleted to obtain a better score. This timing limitation must be retained
when interpreting full-sequence errors; it does not explain away the manual
recording's extra keys.

## Training launch record — September 17, 2026, 10:45:44 PDT

Exact start: **2026-09-17 10:45:44 PDT / 17:45:44 UTC**. A fresh AnchorBench
policy began training with Newton MJWarp and CUDA graphs on the RTX 5090, using
4096 environments and seed 1307. The saved environment was checked against the
selected `refined/task.json`; its keyboard position and quaternion match exactly.
The joint offsets are preserved for hardware conversion, not added inside PPO.

Budget: 500 P1A two-letter updates, then a strict 1024-episode evaluation.
At least 90% success advances automatically to 2000 additional Transit15
six-letter updates, followed by evaluations with seeds 2307 and 3307 and rollout
video. A failed evaluation stops the pipeline for inspection.

Historical check **2026-09-17 10:52:28 PDT / 17:52:28 UTC**: P1A was running and
the log had reached iteration 94/500 at about 3.82 seconds per iteration. The
pipeline subsequently finished at **13:24:04 PDT**, with 2,048/2,048 strict
six-letter evaluation successes. The [physical success record](physical_typing_success_20260917.md)
documents the later NEWTON trial and separate background continuation.

Local run directory: `output/traj-calibrated-anchorbench-mjwarp-20260917/`.
It contains `RUN.md`, immutable code/image provenance in `provenance.json`, the
selected calibration copies, and `run/status.json`. The initial training log is
`run/01-p1a-train.launcher.log`; the tmux session is
`so101-traj-calibrated-20260917`. This run uses the documented 93/93 `refined/`
candidate, not the later lean-workflow or continuous-search candidates.
