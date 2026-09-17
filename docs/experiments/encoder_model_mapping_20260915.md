# SO-101 encoder to model mapping — 2026-09-15

> Dated investigation record. Commands and results describe this workstation at the time; follow [the portable workflow](../../FIXTURE_SETUP.md) for a new setup. Raw recordings and `output/` evidence are local artifacts, not included in the source release.

There are two different calibrations in this project:

1. **LeRobot motor calibration** is the per-robot Feetech encoder homing and
   range calibration. LeRobot applies it when it returns normalized joint
   positions. A user's calibration file is therefore required for that robot,
   but its `homing_offset` values must not be added again as model angles.
2. **Scene joint-zero mapping** is the optional correspondence between
   those normalized LeRobot angles and the USD/Newton model. It is a separate
   transformation from keyboard placement, but depends on the zero convention
   established by this robot's motor calibration file. Bind measurements to its
   calibration hash and revalidate the mapping after recalibration.

The installed LeRobot arm `DEGREES` path computes
`q_deg = (Present_Position - (range_min + range_max)/2) * 360/4095`.
The servo has already applied its homing offset to `Present_Position`; software
does not add it again. This is an encoder-count scale about the calibrated range
midpoint, not a conversion of the measured range to ±180 degrees. That midpoint
is not guaranteed to be the CAD joint zero. The read-only audit and constrained
fit protocol are in [mapping_plan_20260916.md](mapping_plan_20260916.md).

The base convention in the recorder and FK tools is direct conversion:

```text
q_model_arm_rad = radians(q_lerobot_arm_deg)
q_model_gripper_rad = -10 deg + (gripper_percent / 100) * 110 deg
```

The fixed typing jaw is passive. Its pose is the fixed site
`FIXED_JAW_TYPING_TIP_SITE_POS`; the moving jaw is commanded closed for typing.
The gripper percentage conversion only supplies the model gripper joint and does
not change the fixed typing tip geometry.

The historical values from the 2026-07-18 physical-calibrated profile are:

| joint | model = LeRobot + offset (deg) |
| --- | ---: |
| shoulder_pan | 0.0000000000 |
| shoulder_lift | 1.9336272481 |
| elbow_flex | -10.7735791533 |
| wrist_flex | 1.3647648917 |
| wrist_roll | 0.0000000000 |

Yes, these are the same angles used by the earlier 19k benchmark checkpoint.
They are applied only when the explicit benchmark/physical-calibrated profile is
selected. Hardware commands use the inverse:

```text
q_lerobot_goal_deg = degrees(q_model_goal_rad) - offset_deg
```

The current direct recorder stores zero-added-offset model angles. The benchmark
replay adds the five historical values above to a copy of that recording. Recent
replays show that the historical values reduce the vertical error in this setup,
but leave a substantial horizontal residual; they are consequently a historical
diagnostic, not a universal calibration for a new SO-101. A new robot should use
its own LeRobot calibration and a separately measured joint correspondence before
promoting a nonzero scene mapping. Do not copy another robot's calibration file.

The sign must be interpreted at the correct boundary. A direct FK calculation of
raw encoder readings can place the *model estimate* below the key, while sending
model goals directly as raw LeRobot commands can leave the *physical arm* above
the key. The deployment adapter must apply the inverse conversion on writes; do
not infer physical motion direction from an unconverted FK residual.

## Shoulder-pan correction — 2026-09-16, documented 15:20 PDT / 22:20 UTC

The historical correction changes three motors: shoulder lift, elbow flex and
wrist flex. Shoulder pan and wrist roll have zero offsets. The sixth motor moves
the gripper jaw and does not move the fixed typing tip in this model.

For the September development follower, with its existing calibration and
[`physical-reference-20260915.json`](../../configs/tasks/physical-reference-20260915.json)
frozen, fitting **only shoulder pan** gives **+3.128876938 degrees**. All other
offsets retain their historical values. The candidate in arm-joint order is:

```text
                    pan          lift           elbow           wrist flex     roll
offset_deg = [3.128876938, 1.9336272481, -10.7735791533, 1.3647648917, 0.0]
```

The sign follows `q_model = q_lerobot + offset`; any future physical command
adapter must use the inverse already described above. This candidate is saved
only as diagnostic evidence; the historical deployment profile is unchanged.

Fit used the first 27 presses of capture `20260916T220434.751544Z`. Every retained
press improved horizontally, including the later presses and two earlier powered
captures evaluated with the same duration/motion criteria:

| Data | Presses | Historical mean XY error | With pan correction | Mean absolute Z error, unchanged |
| --- | ---: | ---: | ---: | ---: |
| Fit portion | 27 | 13.66 mm | 3.49 mm | 1.89 mm |
| Later portion of the same capture | 22 | 13.93 mm | 2.99 mm | 0.82 mm |
| Earlier capture `20260916T212058.654332Z` | 9 | 12.21 mm | 3.28 mm | 2.47 mm |
| Earlier capture `20260916T212205.548019Z` | 5 | 15.34 mm | 4.45 mm | 1.78 mm |

Leaving each fit key out gives pan estimates from +3.09475 to +3.15931 degrees.
Changing the sample window gives +3.11314 to +3.11473 degrees. This supports a
consistent correction in the recorded typing workspace. Pan rotates the jaw
around a vertical shoulder axis, correcting horizontal direction while preserving
Z; it cannot remove arbitrary radial reach error. The later capture's maximum
remaining XY error is 7.76 mm, despite its 2.99 mm mean.

These are measured-encoder FK comparisons against modeled fully depressed key
centres. Contact location and depth are assumptions. The data had already been
inspected before this pan follow-up, so these checks are exploratory rather than
blind validation. A fresh recording with the candidate frozen, followed by
supervised physical FK presses, is the next verification step. The result does
not establish a universal offset or its mechanical cause.

Exact input/calibration/model hashes, split, optimizer settings, per-press errors
and reproducible code are linked from the
pan analysis (local artifact: `output/fixture-photos-20260915/mapping-collection-20260916/analysis-20260916T220434/pan-results/README.md`).

Historical-setup transfer check, **2026-09-16**: replayed the original 78 July
measured probes on their original keyboard pose in the same pinned Newton MJWarp
runtime for both mappings. Historical offsets passed **78/78** first contacts;
adding the current **+3.128876938° pan** passed **0/78** (69 wrong first contacts,
9 with no first contact). Mean horizontal FK error at recorded key-down poses
increased from **2.76 to 13.55 mm**. The candidate was used unchanged, without
refitting. Exact run timestamps, commands and per-probe evidence are in the
transfer report (local artifact: `output/fixture-photos-20260915/pan-transfer-july-20260916/README.md`).

The added pan value therefore does not transfer to the historical capture setup.
A correct joint-coordinate mapping should remain fixed for a given robot,
calibration and model when only the keyboard moves. This comparison also changes
the capture robot/calibration context; it does not isolate keyboard-placement
transfer on the new robot or identify the physical cause. Keep the added pan as
a candidate for the current calibration/fixture, and preserve the historical
profile. This simulation gate checks the first contact only, with per-key resets;
release, clearance and physical deployment were not tested.

## Fully new four-joint fit and transfer — 2026-09-16

The operator clarified that transfer should use all four newly fitted offsets.
The earlier experiment above held three historical values fixed and added pan;
it is a different experiment. A new joint fit frees pan, lift, elbow and wrist
flex together on the original 27 current-robot fit presses. No July recordings
enter optimization; wrist-roll offset remains zero and keyboard/model geometry
stays frozen. The resulting vector, in pan/lift/elbow/wrist flex/roll order, is:

```text
[3.1265085073, 0.7486865699, -9.6127550477, 0.5401949659, 0.0] degrees
```

### How the four values were assigned

The offsets were assigned by one joint numerical fit, not manually tuned one at
a time and not copied from the LeRobot calibration file. The fit used the first
27 accepted key presses from powered leader/follower capture
`20260916T220434.751544Z`. Each sample stores the follower's measured
`Present_Position` after LeRobot calibration; leader goals were not treated as
measured follower poses. These measurements are scene-calibration evidence and
were never supplied to PPO as training demonstrations.

For each retained press, the model target was the centre of that key at an
assumed full 1.8 mm depression. The keyboard transform, robot USD, link geometry,
fixed-jaw site and wrist-roll offset were held fixed. The optimizer minimized
the robust XYZ residual across all selected presses:

```text
residual = FK(q_lerobot + offset) - depressed_key_centre
free offsets = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex]
bounds = [-15 deg, +15 deg] for each free offset
wrist_roll offset = 0 deg, fixed
loss = soft-L1, 2 mm scale
```

The fit was started independently from zero and from the historical offset
vector; both starts converged to the same result without reaching a bound. The
mapping is applied additively on measured angles and inversely on commands:

```text
q_model = q_lerobot + offset
q_lerobot_goal = q_model_goal - offset
```

Shoulder pan is comparatively well constrained by lateral coverage across the
keyboard. Lift, elbow and wrist flex are coupled through the jaw position and can
partly compensate for one another. The fitted numbers are therefore one effective
constant mapping for this robot/calibration/model contract, not four independently
measured mechanical zero errors. The large negative elbow value is nevertheless
repeatable: the historical mapping used -10.77 degrees and this fit found
-9.61 degrees.

Relationship summary documented **2026-09-16, 16:03 PDT / 23:03 UTC**:
both the historical and newly fitted mappings have a large negative elbow
correction and smaller positive lift/wrist-flex corrections. The elbow correction
alone reduces much of the height discrepancy; the current robot also benefits
from about **+3.13 degrees of shoulder pan** to correct horizontal direction.
These are additive angle corrections, not a scale factor between real and sim.

Both optimizer starts converge to the same vector with no bound hits. On the
22 later presses excluded from fitting:

| Mapping | Mean XY error | Mean absolute Z error |
| --- | ---: | ---: |
| No offsets | 34.66 mm | 20.01 mm |
| Historical elbow only | 16.41 mm | 7.94 mm |
| Full historical offsets | 13.93 mm | 0.82 mm |
| Newly fitted four offsets | 3.13 mm | 1.52 mm |

The new fit improves overall position, but the historical mapping gives lower
height error on this subset. The two earlier current-robot recordings give new-fit
XY errors of **3.38 mm** (9 presses) and **4.33 mm** (5 presses).

The joint values are partly coupled: lift, elbow and wrist flex can compensate
for one another at the jaw, so tip-only contact data do not uniquely identify
each mechanical zero. Leaving each fit key out gives pan estimates of
**+3.09 to +3.16 degrees**, while lift spans +0.29 to +1.18, elbow -10.30 to
-8.94, and wrist flex -0.01 to +1.07 degrees. These refit ranges show greater
stability for pan; they are not confidence intervals. Wrist roll was held at zero,
so this experiment does not establish its correct offset.

Transferred unchanged to the original July recordings and layout, this full
four-joint vector gives **0/78** correct first contacts: **16 wrong contacts and
62 misses**. Mean horizontal FK error is **13.48 mm**, versus the verified
historical baseline's **78/78** and **2.76 mm**. The baseline is reused from the
immediately preceding run; runtime image, worker, trace suite and complete replay
contract match except for the offset vector. The full new mapping also fails to
transfer to the historical capture context. This remains a cross-context check,
not isolated keyboard-placement transfer on one calibrated robot.

The supported conclusion is a repeatable empirical improvement for the current
recordings, not a universal SO-101 conversion or uniquely measured joint zeros.
Current-data errors above are offline FK comparisons of measured follower angles
against assumed key-centre positions at full 1.8 mm depression, not simulated
keypress counts or physical typing success rates. The 22 excluded presses come
from the same capture; earlier recordings had already been inspected. The
candidate remains diagnostic and has not been installed or physically validated.

Fresh torque-off replay check — **2026-09-16**: capture
`20260916T231850.429607Z` retained 22 quality-qualified presses from 33 physical
presses. Offline FK means were 37.87 mm XY / 18.51 mm absolute Z for zero,
13.32 / 3.14 mm for historical offsets, and **4.33 / 1.72 mm** for the current
four-joint candidate. Dynamic Newton replay of the exact measured poses failed
badly for direct mode and improved but still had many extra/neighboring contacts
with historical offsets.

The unchanged four-joint vector produced 33 simulated downs for 33 physical
downs and matched **32/33 intended contacts**. O also contacted `9`, while N was
missed; the other A-Z contacts and all repeated Q/P/M/Z/F/H/K contacts matched.
The approximately 2.5 micrometre FK error is measured **after reapplying the
recorded robot pose**, not throughout the contact integration step. The four-joint
replay reports up to 2.24 mm tip drift within a physics step (direct: 9.26 mm;
historical: 1.48 mm). This corrects the earlier claim that the small displayed-pose
error ruled out replay effects. During the O and N windows, maximum tip drift is
0.95 and 0.90 mm respectively. N reaches 44.4% simulated key travel and `9` reaches
54.9%, against the 50% down threshold; these are threshold-near outcomes that
require contact/replay checks before attributing them solely to joint zeros.
The sequence is still not an exact replay pass, and the four-joint mapping remains
diagnostic rather than qualified for deployment.

### Root-cause check: keyboard height versus arm correspondence — 2026-09-16

Recomputing the 22 retained torque-off presses with zero offsets puts the fixed-jaw
site 3.99–12.58 mm below the model tabletop/base-underside plane at every press.
Its mean residual relative to the assumed depressed key centres is
`[-36.81, -5.97, -18.51]` mm. A height-only keyboard translation would have to move
the keyboard **down by 18.51 mm** just to centre the Z residual, putting its case
below the common support plane and leaving 37.87 mm mean XY error unchanged.
Keyboard height alone therefore cannot explain the main discrepancy under the
measured common-tabletop setup. This is an analytical comparison; the scene was
not moved or refitted.

The same pattern occurs with powered measured-follower data, so command tracking
alone cannot explain it. Encoder/model zero conventions remain a leading
hypothesis: LeRobot degrees are measured about calibrated encoder-range midpoints,
which need not coincide with USD joint zeros. However, contact fitting can also
absorb arm-frame or tip-geometry errors; useful offsets do not independently prove
their mechanical origin. Smaller keycap/collision-height errors remain possible.
Independent physical link landmarks at several measured poses would distinguish
an arm-coordinate discrepancy from keyboard geometry; hold the scene fixed while
making that comparison.

### Concrete suspect: two definitions of arm zero — 2026-09-16 17:25 PDT

The installed LeRobot degree normalization subtracts the midpoint of each motor's
calibrated `range_min`/`range_max`, then scales encoder counts to degrees:
`q_LeRobot_deg = (position - (range_min + range_max)/2) * 360/4095`.
The motor's homing offset is already reflected in its reported position; it must
not be added again here. USD articulation zero instead follows the asset's joint
frames. Neither the range-midpoint calibration nor matching the keyboard datums
establishes that these two zeros represent the same physical link alignment.
That is a concrete coordinate-convention suspect, not proof of faulty hardware
or proof that the contact-fitted offsets are the true mechanical zeros.

View the distinction, using the frozen scene and the existing four-joint report:

```bash
./scripts/view_zero_conventions.sh --zero-mapping /absolute/path/to/four-joint-report.json
```

The Newton side panel has three buttons:

- **USD arm zero:** all five model arm coordinates are zero.
- **LeRobot zero: direct:** also all five model arm coordinates zero. This
  renders the assumption of equal conventions, not an independent physical pose.
- **LeRobot zero: fitted prediction:** `q_model = q_LeRobot + b`, so at LeRobot
  zero the model displays `[3.1265, 0.7487, -9.6128, 0.5402, 0]` degrees. This
  extrapolates the contact fit to zero; it is not a measured physical zero.

The gripper stays closed at model -10 degrees in every mode (LeRobot gripper 0%
is not model 0 degrees). The viewer prescribes static model poses, takes no physics
steps, never accesses the robot, and leaves the frozen keyboard and base unchanged.
It stays open until closed. `--zero-pose lerobot-fit` selects the fitted prediction
at startup; the normal `view_frozen_scene.sh` still shows the original reset pose.
To establish the physical zero alignment, compare independently observed link
poses against encoder readings; switching these model views alone cannot do that.

Viewer verification on 2026-09-16: switched through all three buttons in the live
Newton window. The report (local artifact: `output/zero-conventions-20260916/fit-toggle/scene.json`)
records zero physics steps and unchanged keyboard/base; the 11 CLI tests passed.

Four-joint fit and current-data checks (local artifact: `output/fixture-photos-20260915/mapping-collection-20260916/analysis-20260916T220434/four-joint-results/README.md`)
and full-vector July replay (local artifact: `output/fixture-photos-20260915/four-joint-transfer-july-20260916/README.md`)
preserve the exact values, input hashes, commands, timestamps and per-probe results.

Implementation sources: `scripts/so101_homing/so101_kinematics.py` performs the
direct degree/radian and gripper conversion; `scripts/so101_homing/fixed_cartesian_policy.py`
defines the dated optional offsets and their forward/inverse use; and
`scripts/so101_homing/joint_calibration.py` fits a new constant mapping only from
independently measured poses.

The current recording-based investigation is summarized in
[encoder/model mapping diagnostic (2026-09-15)](encoder_model_mapping_diagnostic_20260915.md).
The three-run Q/G/P pre-contact result and the simulator-versus-hardware boundary
are recorded in
[new-robot pre-contact probe (2026-09-16)](new_robot_precontact_probe_20260916.md).
The complete July evidence chain is in
[historical offset provenance (2026-07-18)](../fixtures/historical_offset_provenance_20260718.md);
it distinguishes the rejected diagnostic fits from the final locked contract.
