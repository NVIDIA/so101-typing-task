# Historical SO-101 offset provenance — 2026-07-18

This document is the provenance record for the joint values used by the
released physical-calibrated scene. It describes the original fixed SO-101 and
MX Keys bench. It is a model-frame contract for that bench, not a replacement
for LeRobot calibration and not a universal calibration for another robot.

## Locked contract

The arm joint order is `shoulder_pan`, `shoulder_lift`, `elbow_flex`,
`wrist_flex`, `wrist_roll`. The released scene used these model-frame offsets:

```text
[0.0, 1.9336272481, -10.7735791533, 1.3647648917, 0.0] degrees
```

At the physical boundary the convention is:

```text
q_model = q_lerobot + joint_zero_offset
q_lerobot_goal = q_model_goal - joint_zero_offset
```

The same dated contract fixed the keyboard pose at
`[0.2625651584, 0.0121353200, 0.0017813943]` m with RPY
`[-5.0, 0.0506258799, -88.6058757454]` degrees. The keyboard pose and joint
mapping are one frozen real-to-sim scene definition; neither is a per-key
correction.

## Effective scene versus physical placement — clarification 2026-09-17

The historical scene underlying the 89% physical benchmark was an **effective
contact alignment**, not an independently verified literal reconstruction of the
tabletop. The original July calibration document explicitly says the effective
rotation includes an "aggregate physical-to-model frame correction" and does not
claim that the physical case was tilted by 5 degrees. Keyboard pose and joint
offsets were selected together during that workflow; offsets alone were not the
entire correction. The exact final fitter-output provenance remains limited as
described below.

| Simulator quantity | Historical July scene | September frozen physical-reference candidate |
| --- | --- | --- |
| Keyboard asset origin XYZ in robot-base frame, mm | 262.565 / 12.135 / 1.781 | 260.635 / 14.061 / 7.637 |
| Asset transform roll / pitch / yaw, degrees | -5.000 / +0.051 / -88.606 | 0 / 0 / -90.056 |
| Near case edge to front of plastic base, mm | 125.28 | 129.17 |

XYZ locates the asset origin, not its underside or keycap surface; the asset
already contains keyboard geometry and native slope. The historical effective
pose can absorb model mismatch. Its successful replay does not separately prove
correct physical placement, correct joint zeros, or correct jaw geometry.

An earlier September experiment instead held the measured placement fixed and
fitted four joint offsets. On September 17 the operator reported that this policy
could press keys but dragged between keys, subsequently hit the space bar, and
also failed with zero offsets. **Offsets alone on the current frozen placement
have not demonstrated reliable physical typing.** These failures do not isolate
encoder zero, placement, geometry, or tracking as the cause.

This led to the [joint-and-scene fitting work](../experiments/scene_calibration_20260917.md)
using the recorded real trajectories, followed by the
[first physical NEWTON success](../experiments/physical_typing_success_20260917.md).
Preserve the measured scene as the physical reference; the adjusted effective
scene is a separate dated candidate with its departures from measured datums
disclosed. Fresh reserved-data qualification remains outstanding.

## Original recorded-trajectory workflow

Recovered from the July collection/fitting/replay scripts and saved reports;
reviewed 2026-09-17. Real contact recordings calibrated the simulator before PPO.

1. **Collect measured motion and key events.** The final collection contained
   three powered repetitions per A-Z key. Associate the actual encoder reading
   with the monotonic key-down timestamp; preserve approach traces for replay.
2. **Compute the fixed typing jaw's contact point.** Apply candidate joint
   corrections before FK, using the unchanged `gripper_link` site
   `[-0.0106769064, -0.0005212274, -0.1041385575]` m. The gripper origin is not
   the typing point. Link dimensions, jaw geometry and keyboard scale stay fixed.
3. **Fit a shared mapping and rigid keyboard placement.** The diagnostic fitter
   fixes pan and wrist-roll offsets at zero, estimates lift/elbow/wrist-flex
   offsets, and solves keyboard XYZ/yaw against the contact points. It starts
   offsets at zero and uses a fit/held-out key split. Fixing pan removes the
   ambiguity between rotating the arm and rotating/translating the keyboard.
4. **Check candidates using actual Newton contacts.** Prescribe measured arm
   poses, let the simulated keyboard respond, and compare the first actuated
   key. The saved sequence was 19/26 for the flat diagnostic candidate, then
   23/26 for revised tilt/offsets. Translating that revised keyboard by
   +2 mm X, +1 mm Y and +1.5 mm Z gave 26/26 in each of three repetitions.
   These are separate saved candidates; the missing final optimizer provenance
   below prevents claiming one preserved fitting command generated all of them.
5. **Freeze the replay-validated scene before training.** Save its joint
   conversion, effective keyboard pose, reset and A-Z target coordinates as one
   model definition. Training uses that scene; deployment adds the same offsets
   on reads and subtracts them on writes.

The historical first-contact gate and its limitations are detailed below. A new
setup also needs complete press/release/transit replay and physical command
checks; a first-contact pass does not establish those results.

## Historical measurements and evidence

1. The original fixed-jaw SO-101 pressed the powered A–Z fixture. Link
   dimensions, keyboard scale and pitch, key centres, and the fixed jaw contact
   site were kept unchanged.
2. The contact capture was repeated three times with edge-synchronised key
   events: 26 contacts per repetition, 78 contacts total. The dataset is
   identified by SHA-256
   `dca2c7c51d4fabf5512cbeac59e617e289ab6deb0bf0855be5cd21c8c616b2d7` and the
   recorded trace by
   `e1bb5b645fddfddca6e02e54c4e0e9c447d3acdaaea406f22b37d51e1d7ea01c`.
3. A diagnostic rigid-contact fitter held shoulder-pan and wrist-roll as
   gauges, optimized shoulder-lift/elbow-flex/wrist-flex offsets, and fitted
   one rigid keyboard translation plus yaw. It used robust least squares,
   retained the existing robot and keyboard geometry, and did not fit key-by-key
   offsets or rescale the keyboard.
4. The persisted diagnostic candidates were rejected. The elbow reached its
   bound and the held-out geometry gates failed (the repeated fit reported
   held-out XY RMS/max `3.69/6.64` mm and Z RMS/max `1.69/2.87` mm). Widening
   the elbow bound moved the candidate to about `-10.77` degrees but still
   failed. These reports are diagnostic evidence, not the released result.
5. The final dated constants above were then locked in the physical-calibrated
   environment configuration. Newton replay of 78 separate measured key probes
   passed the **first-key-actuation gate**: `26/26` for each repeat, `78/78`
   aggregate. Each probe resets the keyboard and stops at its first simulated
   key-down; it requires no initially pressed key and only the expected key at
   that first actuation. It does not test release, clearance, subsequent contacts,
   or a continuous A–Z sequence. The archived script reports analytical FK versus
   the loaded SO-101 gripper origin differences of `0.045–0.103` mm after settling.

The original gate uses four physics steps per nominal 25 Hz sample, zero robot
joint velocities, 20 settling steps per key, and a plane 40 mm below the keyboard
root to isolate keyboard contact. The NPZ stores measured joint positions without
per-sample timestamps. These conditions differ from the new robot's continuous,
timestamped torque-off recording replays and their full down/up sequence check.

## Provenance boundary

The archive preserves the measured data, fitting method, rejected reports, final
constants, and the `78/78` first-key replay gate. It does **not** preserve a
promoted fitter output whose numeric vector is exactly the locked vector. Do not
say that `fit_powered_contact_calibration.py` directly generated the released
numbers: its saved candidates were rejected and differ from them. The accurate
description is that the values are the frozen model-frame correction selected
and locked during the July real-to-sim calibration workflow, then validated by
direct trajectory replay.

The historical source trail is preserved by commits `989e9ec50`, `aa042a76b`,
`97d3cd009`, `8c5d1e96e`, and `b76a2d4ec`. The original detailed record is
`docs/so101_physical_calibration_20260718.md`; the diagnostic implementation is
`scripts/so101_homing/fit_powered_contact_calibration.py`.

## September 16 recordings: application status as of 2026-09-17

**The September 16 recordings now pass 93/93 reset-per-key first-contact probes
after fresh joint/keyboard contact refinement.** This reproduces the original
type of scene-validation criterion. Continuous replay is a later additional
diagnostic and does not pass; it was not required by the original July gate.
The [selected pose, offsets and evidence](../experiments/scene_calibration_20260917.md)
identify the passing `refined/` candidate. Both recordings were used for fitting;
physical deployment is not established by this result.
A fresh policy trained in the
selected September scene completed [physical NEWTON typing](../experiments/physical_typing_success_20260917.md)
with its own fitted mapping. That single-trial result is separate from both
the 93/93 fitting check and the historical 89/100 benchmark.
The [cross-fit comparison](../experiments/scene_calibration_20260917.md#recurring-trend-across-historical-and-september-fits)
records the recurring roughly -10-degree elbow correction and smaller positive
lift/wrist-flex corrections, including the limits on transferring those values.

Earlier experiments include offsets-only fits, pose fits with historical offsets,
and new joint-plus-pose fits followed by continuous Newton replay. The new
four-joint/XY-yaw candidate produced 70 simulated edges for 66 physical edges;
it did not pass. It also freed pan and yaw together and averaged angles across
presses, differing from the original fitting procedure. The full-pose diagnostic
was an offline fit, not a passing contact replay.

The [initial calibration script](../../scripts/calibrate_scene.sh) fitted fresh
lift/elbow/wrist-flex offsets and XYZ/yaw with pan and roll fixed, using every
key-down's nearest encoder sample. No historical values are used. Its separate
manual replay reproduces all 33 intended presses in order but adds `9` and `U`;
this initial numerical fit was not a pass. The later contact refinement above
established the reset-per-key result.
The current recordings' timestamps and capture modes must remain explicit;
they cannot be relabelled as the original powered, kernel-edge-synchronised data.
See the [current investigation](../experiments/mapping_plan_20260916.md#original-workflow-recovered-from-archived-source).
Exact inputs and results are in the [September calibration report](../experiments/scene_calibration_20260917.md).

## Historical replay reproduction

This result uses the **original July robot recordings + original July keyboard
layout + historical joint offsets** listed above.

The unmodified archived replay script, hash-verified original dataset/NPZ,
historical keyboard pose and historical offsets were rerun in the current pinned
Newton MJWarp runtime. Each repetition passed 26/26 first-key probes: **78/78**.

| Original July A–Z recording | Correct first-key contacts in simulation |
| --- | ---: |
| Repetition 1 | 26/26 |
| Repetition 2 | 26/26 |
| Repetition 3 | 26/26 |
| Total | **78/78** |

The original per-key resets, timing assumptions and isolated-plane gate above
were retained. The current frozen scene and ongoing training were unchanged.

Commands, hashes, timestamps and all three reports are linked from the local
rerun report (local artifact: `output/fixture-photos-20260915/historical-az-replay-20260916/README.md`).
This confirms reproduction of the historical gate; it does not qualify transfer
to the new robot or replace the current full-event-sequence validation.

## Extended contact comparison

The original raw reports also preserve measured retraction. Replaying these on
the historical layout with historical offsets gives 78/78 expected releases,
but **77/78 clean down/up cycles at 25 Hz**: repetition 2 adds J while retracting
from U. A stricter check that also rejects 100 Hz key bounce passes 72/78.
Only 8/78 additionally sustain 15 mm clearance; the recorded hover returns were
not collected with that later clearance requirement. Physical key-up was not
recorded, so these are simulation results.

Under the same extended simulation checks, 35 probes from recordings excluded
from the new mapping fit on the current frozen layout give 32/35 correct first contacts,
31/35 clean 25 Hz down/up cycles, and 30/35 passing the stricter bounce and
clearance checks. Different capture methods and trajectories prevent treating
this as an isolated comparison of mapping quality. The current mapping remains
diagnostic. See the comparison report and plots (local artifact: `output/fixture-photos-20260915/matched-contact-comparison-20260916/README.md`)
for exact commands, short-approach baselines, and continuous-replay limitations.
