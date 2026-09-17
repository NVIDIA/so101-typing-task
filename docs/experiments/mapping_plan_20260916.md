# Frozen-scene mapping investigation — 2026-09-16 UTC

> Dated investigation record. Commands and results describe this workstation at the time; follow [the portable workflow](../../FIXTURE_SETUP.md) for a new setup. Raw recordings and `output/` evidence are local artifacts, not included in the source release.

## Agreed pre-training workflow — 2026-09-16

This section records the earlier offsets-only plan. The
[September 17 revision](#trajectory-based-scene-alignment--2026-09-17) below
expands the investigation to separately versioned effective scene candidates;
the measured placement remains the reference.

Use [the repository workflow](../../FIXTURE_SETUP.md#before-policy-training-verify-the-real-scene)
as the sequence for physical typing. Our datum measurements and bird's-eye
cross-checks establish the current placement reference; measured-trajectory
replay then tests the encoder/model correspondence. **Torque-off key-press
recordings are scene-validation evidence, not policy training data.** They
preserve actual measured follower motion rather than motor goals.

| Order | Current state and next work | Completion evidence |
| --- | --- | --- |
| 1. Freeze the measured scene | Keep `configs/tasks/physical-reference-20260915.json`, base/keyboard datums, USD, fixed jaw and current follower calibration unchanged. Placement has been physically cross-checked. | Dated datum sheet and hashes; no keyboard adjustment during offset fitting. |
| 2. Preserve and extend real recordings | Torque-off A–Z/QGP, powered leader/follower recordings and a fresh torque-off A–Z-plus-repeat validation capture now exist. | Original timestamps, follower angles, USB edges and torque/calibration identities; separate sessions remain separate. |
| 3. Freeze the four-joint candidate | The fit is `[+3.12651, +0.74869, -9.61276, +0.54019, 0]` degrees, from powered measured-follower data. It was applied unchanged to the fresh torque-off repeat. | The new recording did not enter fitting. Offline FK errors and sampling exclusions are recorded below. |
| 4. Validate measured-pose replay | The frozen candidate matched 32/33 intended contacts; O also contacted `9`, and N was missed. | This is strong transfer evidence but fails the exact replay gate. Resolve and revalidate these contacts without refitting on the same validation recording. |
| 5. Qualify physical commands | Add an explicit, provenance-checked candidate mapping to the policy-free physical probe; currently it supports only direct/historical modes. Then test separated Q/G/P hover/press/release one key at a time. | Inverse conversion on writes, unchanged model trajectory, actual joint tracking and physical USB/clearance evidence. No candidate is installed yet. |
| 6. Train against the qualified scene | After these checks, use the same scene for AnchorBench/Newton MJWarp training, early evaluation and export with the verified mapping. | Linked scene/mapping validation and policy evaluation evidence. Existing exploratory training is not retroactive proof that these checks passed. |

The negative elbow trend is reproducible: historical **−10.77°**, latest full
four-joint fit **−9.61°**; historical elbow alone reduces reserved mean absolute
height error from **20.01 to 7.94 mm**. Lift/elbow/wrist values partly compensate
for one another, while current pan is stable near **+3.13°**. This supports an
empirical mapping to validate, not a universal angle correction or a uniquely
identified mechanical fault. The latest coefficients came from powered measured
follower data; do not relabel them as a torque-off fit. See
[the measured relationship](encoder_model_mapping_20260915.md#fully-new-four-joint-fit-and-transfer--2026-09-16).

Next operator-run capture, with the arm supported before torque is disabled:

```bash
./scripts/scene_check.sh record \
  --robot-id so101_5B79015117 \
  --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79015117-if00 \
  --keyboard-device /dev/input/by-id/usb-Logitech_USB_Receiver-event-kbd \
  --task-config configs/tasks/physical-reference-20260915.json \
  --seconds 300 --output-root output/fixture-photos-20260915/mapping-validation-20260916
```

Keep the moving jaw closed; gently bottom one key at a time, hold about 0.5 s,
release and lift. Cover A–Z in any order. Reserve this fresh recording to test the
already frozen candidate, rather than using it to refit that candidate. The
recorder saves measured states and leaves torque off.

Fresh torque-off validation capture completed **2026-09-16 16:18 PDT** at
`output/fixture-photos-20260915/mapping-validation-20260916/20260916T231850.429607Z/recording.json`.
It contains 3,969 samples over 79.9 seconds and 33 physical presses (A–Z plus
repeats). The predeclared stable-window rule retained 22 presses and excluded
11 presses shorter than 200 ms or without the required settled window. The
capture uses robot `so101_5B79015117`, calibration SHA-256
`b4ef349bf420b14887f69d01f82f1d880ef0a5c9f26f52900fb51445bc4891ae`, and frozen
task SHA-256 `3690b63ec8711698bbe3af39f4ae9236c1c5715e0fd9880bcce6a36121d62feb`.

Offline FK scoring of the retained 22 presses gives:

| Mapping | Mean XY | Mean absolute Z | Mean 3D |
| --- | ---: | ---: | ---: |
| Direct, zero offsets | 37.87 mm | 18.51 mm | 42.25 mm |
| Historical | 13.32 mm | 3.14 mm | 13.73 mm |
| Current four-joint candidate | 4.33 mm | 1.72 mm | 4.79 mm |

All three mappings were replayed dynamically in Newton with the exact measured
poses. Direct replay produced many neighboring-key and modifier contacts;
historical replay reduced the errors but still produced extra/neighboring
contacts. The frozen four-joint candidate produced 33 simulated key-downs for
33 physical key-downs and matched 32 intended contacts. The O press also contacted
`9`, and N was missed; every other A-Z press and all repeated Q/P/M/Z/F/H/K
presses matched. The overall sequence therefore remains a failure, despite the
large improvement. No offsets were refitted from this validation recording.

The **2.5 micrometre** maximum FK error describes the pose after it is reapplied
for display. It does not rule out contact-step artifacts: the four-joint replay
reports up to **2.24 mm** tip drift during physics integration, with about
0.95/0.90 mm in the O/N windows. N reaches 44.4% key travel and `9` reaches 54.9%
against the 50% down threshold. The remaining failures require a contact/replay
audit as well as the static mapping comparison. The four-joint candidate remains
diagnostic and is not installed or qualified for physical commands.

The subsequent [root-cause check](encoder_model_mapping_20260915.md#root-cause-check-keyboard-height-versus-arm-correspondence--2026-09-16)
finds every retained direct-FK jaw position below the common tabletop plane.
A height-only keyboard adjustment cannot explain the combined vertical and
horizontal discrepancy; independent physical arm landmarks remain necessary to
identify why the encoder/model correction is needed.

Evidence files are `scene-report.json`, `benchmark-offsets/scene-report.json`,
and `placement-3690b63ec871-mapping-b070d9cca51b-fit4_frozen_20260916/scene-report.json`
beside the recording. The source recording SHA-256 is
`ce055ea12a597981968a98e651fea7ec6052f179e76ad2d6ee3512d6b525f2b8`.

## Experiment record

### Proposed automatic-calibration comparison — 2026-09-16 PDT

Test whether manual endpoint measurement contributes to the encoder/model
discrepancy. Automatic mechanical-stop measurement can improve repeatability;
it does not guarantee zero variability or agreement between the physical range
midpoint and USD zero. Keep the current USD, fixed jaw and datum placement as
the comparison reference. Initial hardware attempts are recorded below; no
complete endpoint result or replacement calibration has been accepted.

The reviewed [LeRobot PR #3282](https://github.com/huggingface/lerobot/pull/3282)
is still open at this check. Its pinned implementation is
`1c8e185e3694def2469d7d312c78dab217e336da`; the
local source audit (local artifact: `output/external-reviews/servotools-20260916/lerobot-pr3282/README.md`)
records all-motor initialization, controller-setting changes, missing restoration
and stop-detection limitations. Omitting `--save` does not prevent motor writes.

1. Preserve the existing calibration, mapping and recording identities; a
   file-only baseline is under `output/auto-calibration-comparison/`. Before
   motion, snapshot the live calibration and every controller register the
   adapted routine will change. A calibration JSON alone is not a full backup.
2. Prepare the pinned routine in an isolated environment. Restore controller
   gains/modes/limits on failure; preserve original actuator settings for the
   comparison. Reject timeout/communication-error endpoints. Ensure the full
   sweep is clear; if anything in the fixture must move, restore and recheck
   datums before collecting typing evidence.
3. Measure endpoints in three supervised runs. Compare endpoints, physical
   midpoints and angle spans in a common encoder frame that accounts for
   changed homing offsets and wraparound. Report per-joint repeatability in
   degrees; repeating a midpoint is not evidence of its CAD alignment.
4. Under the selected automatic calibration, collect fresh measured Q/G/P
   contacts, then a reserved broader key set. Replay with zero extra offsets
   first, using the frozen scene and the same FK/contact metrics as before.
   Do not reinterpret old normalized trajectories as newly calibrated angles,
   or reuse the old fitted offsets without converting their coordinate basis.
5. If direct replay improves, quantify the part explained by calibration.
   If repeated automatic calibration is stable but the discrepancy remains,
   manual endpoint variation alone is insufficient; test the remaining
   mechanical-reference-to-USD correspondence and geometry. Keep calibration
   repeatability, FK error and dynamic contact outcomes as separate results.

**Prepared first test, 2026-09-16 PDT:** `scripts/run_auto_calibration_probe.sh`
is a bounded, single-joint experiment using the installed SDK, rather than an
installation of the full upstream automatic-calibration CLI. Start with
`elbow_flex`; `--joint` also supports shoulder pan/lift and wrist flex. It does
not calibrate wrist roll/gripper, automatically unfold the arm, or install a new
calibration. An operator must first arrange a collision-free sweep for the
selected joint; other arm joints hold their fresh measured starting positions.

```bash
./scripts/run_auto_calibration_probe.sh             # offline plan; no hardware access
./scripts/run_auto_calibration_probe.sh --snapshot  # read-only controller backup
./scripts/run_auto_calibration_probe.sh --check-mode # supported stationary encoder check
./scripts/run_auto_calibration_probe.sh --execute   # supervised elbow experiment
```

Three automatic min/max repeats use raw speed 100, torque limit at most 200,
55-second directional timeouts, and a 12-degree search/acceptance window around
each existing endpoint. The 12-degree window includes the approximately
10-degree elbow correction being investigated; it is a bounded check of the
existing manual calibration, not an independent first-time range finder.
Stable stops outside that window, controller faults, communication failures,
time gaps, and holding-joint drift reject the run. Both endpoint spreads must be
at most 1 degree; this repeatability threshold is not a typing-accuracy claim.

Homing, PID and Phase remain unchanged. The selected motor temporarily changes
operating mode, position limits and torque limit; all touched configuration
registers are backed up and verified after restoration. On successful completion
it backs away from the endpoint into the original limits. **Shutdown holds fresh
measured arm goals with torque ON, even if torque started OFF; it does not replay
stale goals or drop the arm to reproduce the original torque state.** A joint
that cannot be restored/held is left torque OFF where communication permits;
this is reported as a failure requiring operator support. Gripper is untouched.

The original calibration is preserved. Candidate JSON is written only after
complete, repeatable measurements and successful restoration, outside LeRobot's
active calibration directory. Its comparison explicitly computes
`q_auto - q_manual = (old_midpoint - new_midpoint) * 360/4095`, since homing stays
fixed. This determines whether the endpoint change explains any of the fitted
correction, without silently changing old recordings' conventions.

The follower's read-only diagnostic and complete controller backup passed on
2026-09-16 at 17:38/17:44 PDT. Torque was OFF and calibration/register values
matched. Fifteen offline tests cover successful repetitions/backoff, early
obstruction, faults, communication failure, timeout, Ctrl-C with a failed stop
packet, failed restoration, coordinate conversion, delayed mode feedback,
camera freshness and repeatability. Optional `--camera-state` requires a live
RGB heartbeat; it detects lost frames, not obstacles. The operator must supervise.

**Hardware paused at the operator's request, 2026-09-16 18:09 PDT
(2026-09-17 01:09 UTC).** No probe process remains active and camera recording
is closed. The last cleanup reported restored controller configuration and arm
torque ON; pausing issued no further motor commands. The original calibration
file is unchanged and no candidate was installed. Resume requires a newly
confirmed clear sweep and physical support during the stationary mode check.

The first attempt (`output/auto-calibration-comparison/20260917T005500.316033Z-elbow_flex/`)
rejected its sweep because the probe initially compared velocity-mode readings
against position-mode bounds without converting coordinates. This is a probe
bug, not evidence of bad manual calibration. Subsequent stationary mode-0/1/0
samples show `q_position_ticks = q_velocity_ticks + 267` for this elbow, whose
unchanged homing register is -267. Three repeated checks passed at 18:05 PDT;
immediate feedback after another mode change also returned the old frame once.
The probe now waits 20 ms after switching mode and validates the convention
before each sweep. This controller-mode conversion is separate from the fitted
approximately -10-degree LeRobot-to-USD mapping; ordinary position-mode
recordings must not receive an extra homing correction.

The corrected 18:06 attempt rejected an early stop after only two encoder
counts of upward movement at torque limit 200. That does not identify an
endpoint or prove whether load, startup delay or obstruction caused the stop.
Load/current logging and a 0.75-second startup allowance were added; all other
motion guards stay active. The final 18:08 attempt stopped during its stationary
check after 13 counts (1.14 degrees) of drift while elbow torque briefly
released. It never reached the instrumented sweep. **No complete min/max pair
or three-repeat calibration comparison exists yet.** Reports and raw checks are
under `output/auto-calibration-comparison/20260917T010512.482455Z-elbow_flex/`,
`20260917T010535.442407Z-elbow_flex/`, `20260917T010659.071111Z-elbow_flex/`, and
`20260917T010828.039824Z-elbow_flex/`.

Latest corrected experiment, 2026-09-16: jointly fitting all four offsets on the
27 current fit presses gives pan/lift/elbow/wrist flex
`[3.1265085073, 0.7486865699, -9.6127550477, 0.5401949659]` degrees; wrist roll stays
zero. The later 22 current-recording presses have 3.13 mm mean XY and 1.52 mm mean
absolute Z error. Transferring the full vector unchanged to July recordings on
their historical layout gives **0/78** correct first contacts (16 wrong, 62
misses), versus historical **78/78**. Mean July XY error is 13.48 versus 2.76 mm.
This is separate from the earlier historical-plus-pan experiment below. See the
[full four-joint result](encoder_model_mapping_20260915.md#fully-new-four-joint-fit-and-transfer--2026-09-16)
and replay evidence (local artifact: `output/fixture-photos-20260915/four-joint-transfer-july-20260916/README.md`).

Earlier historical-plus-pan transfer check, 2026-09-16: original July recordings on the
original layout pass **78/78** first contacts with historical offsets and **0/78**
after adding the current +3.12888° pan (69 wrong contacts, 9 misses). Horizontal
FK error increases from 2.76 to 13.55 mm. This checks a different historical
robot/calibration context as well as layout; same-robot layout transfer remains
untested. Preserve the historical profile. See the
transfer report (local artifact: `output/fixture-photos-20260915/pan-transfer-july-20260916/README.md`)
and [mapping interpretation](encoder_model_mapping_20260915.md).

Latest follow-up (2026-09-16, documented 15:20 PDT): adding a fitted **+3.12888°
shoulder-pan offset** to the fixed historical correction reduces mean XY error
on the later 22 presses from **13.93 to 2.99 mm**, preserving **0.82 mm** mean
absolute Z error. All 63 retained presses across three captures improve in XY.
The candidate remains diagnostic; fresh-recording and physical FK confirmation
are pending. The full mapping, sign convention, stability, residual limits and
evidence links are documented in
[the shoulder-pan result](encoder_model_mapping_20260915.md#shoulder-pan-correction--2026-09-16-documented-1520-pdt--2220-utc).

Three-joint baseline (2026-09-16, 15:13 PDT): the operator chose free-order powered key
collection. Capture `20260916T220434.751544Z` contains 60 complete presses across
all 26 letters. Before fitting, the first complete alphabet (27 presses) was
assigned to fit; later presses were reserved. Fixed duration/motion criteria kept
22 reserved presses and excluded 11 short presses. This is a same-recording split.

On those 22 reserved presses, mean absolute Z / mean XY error is 20.01 / 34.66 mm
with zero offsets, 7.94 / 16.41 mm with the historical elbow alone, and 0.82 /
13.93 mm with the full historical correction. All 22 improve over zero offsets.
The fitted lift/elbow/wrist correction is `[1.89593, -11.41455, 1.76487]` degrees,
close to historical `[1.93363, -10.77358, 1.36476]`, but its reserved errors are
1.40 / 14.12 mm. A fitted elbow-only correction of -8.66169 degrees gives 1.92 /
13.92 mm. Neither fit establishes an advantage over the historical correction.
Two earlier powered captures also improve strongly over zero offsets using the
same quality rules (9 and 5 retained presses); they are previously inspected data.

The historical correction is supported as a useful empirical baseline. Roughly
14 mm of horizontal residual remains, and these contact-centre comparisons do
not independently identify mechanical zeros or qualify physical deployment. No
mapping was installed. Reproducible code, pre-fit split, per-press CSV, sensitivity
checks and plot are in
`analysis-20260916T220434/results` (local artifact: `output/fixture-photos-20260915/mapping-collection-20260916/analysis-20260916T220434/results/README.md`).
The [recording protocol](leader_replay_20260916.md#fixed-keyboard-mapping-collection--2026-09-16-1502-pdt)
remains available for a fresh independent repeat; exact letter order is optional.

## Current engineering plan — 2026-09-16, after physical placement cross-check

The operator has checked the keyboard/base dimensions and relative placement.
Accept the frozen scene as the placement reference for this investigation. The
plastic foot is 17.50 mm high in the USD; 72.00 mm includes the upright. Keep the
keyboard, USD, fixed-jaw site and existing LeRobot calibration unchanged while
identifying the remaining discrepancy. The earlier joint fits below remain
diagnostics, not approved calibration. Physical contact still requires validation.

The question is whether measured encoder angles and USD joint angles describe
the same physical arm pose. Separate three transformations:

1. Controller Present_Position -> LeRobot joint degrees (software conversion).
2. LeRobot joint degrees -> physical/model joint coordinates (zero convention).
3. Model joint coordinates -> link and tip pose (forward kinematics).

For the installed arm degree mode, `q_L = (Present_Position - midpoint) *
360/4095`, where `midpoint=(range_min+range_max)/2`. Present_Position already
incorporates controller homing. This is not range-normalized percentage scaling.
The minimal candidate is `q_model = q_L + b`; the command inverse is
`q_L_goal = q_model_goal - b`. A physical correction must explain independently
measured link poses and predict new poses, not just improve a key-contact fit.

| Order | Experiment and evidence | Why it discriminates |
| --- | --- | --- |
| 1 | Preserve robot ID, calibration hash, live read-only register snapshot, raw counts, torque state and installed conversion source; verify read/write inverse and joint order. Reuse completed audits, refresh only live state. | Separates configuration changes and software conversion errors from physical zero error. Register agreement alone is not physical-angle agreement. |
| 2 | One stationary side-on photo paired with the existing read-only 10-second encoder capture. Mark corresponding base, shoulder, elbow, wrist and fixed-jaw landmarks in the USD and image. Compare direct and historical predictions using the same camera/base registration. | Establishes which link first disagrees independently of keyboard contact and motor tracking. This first image is a coarse diagnosis, not a five-joint calibration. |
| 3 | Acquire six distinct static poses: three for identification, three reserved before analysis for validation. Vary lift, elbow and wrist flex independently across comfortable typing-adjacent and higher poses; keep pan/roll fixed and record their values. Revisit selected poses from opposite directions, with torque/load/support state recorded. Add poses if sensitivity analysis shows insufficient information. | Independent link orientations separate joint errors that tip-only typing data let trade off. Repeats expose backlash or support/load dependence that a constant zero cannot explain. |
| 4 | Compare zero correction, historical correction and the smallest physically supported constant correction. Use the same fixed geometry and camera calibration. Quantify landmark uncertainty and parameter sensitivity; reject weakly constrained parameters. | Tests whether historical offsets represent a consistent coordinate conversion. A lower residual alone does not establish the physical cause or universal values. |
| 5 | Freeze the candidate and predict all reserved poses without retuning. Report per-link angles and tip XYZ errors, worst case and repeatability. Use prior A-Z/QGP recordings only as secondary checks; they have already influenced the investigation. | Tests prediction beyond the identification poses and avoids recycling the same failures as an independent test. |
| 6 | Once static correspondence passes, perform slow, supervised, policy-free hover/press/release tests at Q/G/P, one key at a time. Compare model goal, converted encoder goal, measured encoder response and physical tip/contact outcome. | Separates coordinate mapping from motor tracking and contact dynamics. Exact key down/up, no extra keys and physical clearance are separate acceptance evidence. |

Phone metrology must not silently introduce another fitted offset. Hold the camera
fixed and normal to the lift/elbow motion plane, use a rigid scale/reference in
that plane, and account for lens distortion/perspective if quantitative accuracy
is needed. Use joint centres and identified rigid-link landmarks, not arbitrary
plastic edges. Physical link angles must be converted through the USD's fixed
joint-frame transforms; a link's angle to the table is not automatically its USD
joint coordinate. A side view cannot independently calibrate pan and roll. Use a
separate view or reference for those joints only if their discrepancies warrant it.
If phone uncertainty prevents a decision, use a defined angle-gauge/jig reference
or calibrated multiple views before fitting further parameters.

Decision rules:

- Constant per-joint differences across poses support a zero-convention mapping.
  They do not alone distinguish CAD convention from motor/assembly indexing.
- Error that grows with angle calls for an explicit sign/scale/axis investigation;
  do not fit a free scale simply because it reduces a residual.
- Error that changes with approach direction or torque/load points to backlash,
  compliance or tracking. Preserve those conditions rather than average them into
  a supposedly universal zero.
- Agreement through wrist pose with a remaining tip-only mismatch would reopen
  the jaw-site check. Until that pattern appears, jaw geometry is secondary.

Choose tolerances before the held-out test, from actual key clearance/travel and
measurement uncertainty. Use the model's local Jacobian (`delta_p ~= J delta_q`)
to translate angular uncertainty into millimetres at each test pose. For scale,
0.5 degrees across a 300 mm lever arm is about 2.6 mm: the existing 0.5-degree
fitter residual limit is not proof of adequate accuracy for a 1.8 mm key stroke.
A phone image may identify a large zero error without resolving final contact
tolerance. Likewise, continuous Newton replay has recorded within-step robot
drift, so it is not the primary measurement of static correspondence.

The release deliverable is one documented, reversible encoder/model conversion
with robot/calibration/model provenance and withheld-pose plus physical-FK
evidence. If a constant conversion is validated, apply it symmetrically on reads
and writes, including startup and limit handling. Model-coordinate policy
training can stay unchanged when only a correct coordinate conversion is added;
that does not establish actuator-dynamics or hardware typing performance.
Determine whether any component is universal only by checking another calibrated
robot independently. Do not copy the historical robot's calibration file.

The existing read-only capture is documented at
`output/model-comparison-20260916/physical-conventions/reference-capture/README.md`.
This plan does not command motion, rewrite calibration, change gains or launch a
policy. Next physical evidence: the first stationary side photo and paired counts.

### RealSense replaces the phone capture — 2026-09-16

The operator connected a D435i, serial `243222074872`, firmware `5.17.3.10`,
enumerated over USB 3.2. A live camera-only check captured 1280×720 RGB and
848×480 native depth at requested 30 Hz, plus depth aligned to color and SDK
intrinsics/extrinsics. No robot serial connection or motor command was made.
Evidence is in `output/realsense-reference-20260916/20260916T193827.843659Z/`.

The view includes the arm/base, keyboard, a tabletop ChArUco board and a jaw
marker. All three saved images detect nine markers using DICT_5X5_50: board
IDs 0–7 plus another ID 0 on the jaw. Separate the board and jaw regions before
pose estimation; a lookup by marker ID alone is ambiguous. Printed square/marker
sizes await confirmation. Camera stream availability and valid depth do not
establish millimetre accuracy.

Use these images in place of phone photos and pair captures with read-only
encoder logs using host time brackets and stationary intervals. The board can
provide a camera reference once dimensions are confirmed; it does not by itself
establish the robot-base frame. Establish camera-to-base and marker-to-link
transforms from independent references, and keep them fixed while evaluating
joint-zero corrections. Otherwise those transforms could absorb the very error
we are trying to measure. Do not change the already checked keyboard placement.

Target definitions subsequently located in the H1-2 workspace:
The separate local H1-2 calibration archive's README describes the
small 4×4 ChArUco board with 47 mm squares and 35 mm markers, DICT_5X5_50.
`calibration/h1_2_left_palm_marker_35mm.json` in that workspace specifies ID 0,
35 mm outer black-square size, and records an earlier operator size confirmation.
Do not use its H1-2 palm mounting convention or historical camera extrinsics on
the SO-101. The larger 7×5 AprilTag board in that workspace is a different target.

An offline check of all three current RealSense images finds all nine internal
corners using `CharucoBoard((4,4),0.047,0.035,DICT_5X5_50)` with
`setLegacyPattern(True)`; the non-legacy pattern finds none. Board reprojection
RMS is 0.27–0.30 pixels using current stream intrinsics. This confirms image/layout
consistency, not absolute physical print scale or robot-frame accuracy. The jaw
ID-0 marker is also detected. Its single-view planar pose ambiguity remains
unresolved. Saved definitions, source hashes and camera-frame diagnostic poses:
`output/realsense-reference-20260916/target-definitions-and-check.json`.

The operator clarified that the marker is by the jaw motor and that the accepted
base-to-keyboard measurements should supply the robot reference. Use the frozen
keyboard as the initial camera-registration reference: correspond visible key/case
landmarks to their asset coordinates, solve camera-from-keyboard, then compose
`T_camera_from_base = T_camera_from_keyboard * inverse(T_base_from_keyboard)`.
Validate on other visible landmarks and check depth consistency. This registration
must not use arm FK or re-fit the keyboard placement. Carry the original physical
placement uncertainty into the result. The tabletop board can subsequently track
camera stability; its pose relative to the base need not be measured separately.

A marker rigidly attached to the jaw-motor housing can track that housing; it
need not be at the contact tip. Identify its actual supporting rigid link from
the mount before choosing the corresponding model body. Proximity to the motor
does not specify the exact marker-to-link translation and rotation. For the
initial direct-versus-historical comparison, estimate one constant mounting
transform per mapping candidate on identification poses and test predictions on
reserved poses. Relative rigid-body motion also eliminates that constant mount:
`T_base_marker_j * inverse(T_base_marker_i)` equals the corresponding link-motion
transform. This supports an initial comparison without measuring the marker-to-tip
offset, but it cannot establish every absolute joint zero. A free mounting
transform exactly absorbs the last observed joint's constant zero correction;
additional stationary downstream joints can create further ambiguities. Use an
independent link-orientation reference or restrict the parameters before claiming
a uniquely identified joint calibration.

### First paired static observation — 2026-09-16 12:57 PDT / 19:57 UTC

After the operator powered the arm on, a read-only capture recorded 200 encoder
samples over ten seconds alongside D435i RGB-D frames. All five arm readings
were constant; live calibration registers matched the saved robot-specific file.
No motor commands, torque writes, calibration edits or placement changes occurred.
Capture: `output/realsense-reference-20260916/paired-20260916T195734.060148Z/`.

An independent coarse camera registration used 19 manually identified keyboard
rim centres and the frozen keyboard geometry, with no arm landmarks or encoder
fitting. Reprojection RMS was 1.47 pixels. Manual-point perturbation sensitivity
is several millimetres at arm height; this is suitable for a broad visual check,
not a calibrated physical-angle estimate. The tabletop board had moved since the
earlier camera check and was not used as the registration anchor.

Exact pinned USD visual meshes were posed offline at the recorded coordinates,
then at those coordinates plus historical offsets. Both use the same projection.
The direct fixed-tip prediction is 8.03 mm below the common base-underside plane;
the historical prediction is 26.78 mm above it, a 34.81 mm difference between
model predictions. In the camera overlay, the historical fixed-jaw tip appears
closer to the real tip. Other link discrepancies remain; this one folded pose
does not identify correct offsets or qualify deployment. No contact simulation
or physical typing was performed. The direct lift reading is -101.45 degrees,
outside the nominal -100-degree model limit; geometry was deliberately not clipped.

Comparison image and reproduction/provenance:
`output/realsense-reference-20260916/static-comparison-renderer/paired-20260916T195734.060148Z/`.
Next evidence is a distinct, less-folded stationary arm pose paired with encoders
and images; retain the same camera/keyboard reference and compare both mappings.

Upstream context: [LeRobot SO-101 calibration](https://huggingface.co/docs/lerobot/so101)
and [NVIDIA's calibration procedure](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/07-calibrating-so101.html)
describe robot calibration and endpoint collection. The equations above were
checked against the installed code, not inferred from an upstream version.

## Earlier investigation and evidence

Scope: `so101_5B79015117`, its own LeRobot calibration, and the fixed jaw.
Keep `configs/tasks/physical-reference-20260915.json` unchanged (SHA-256
`3690b63ec8711698bbe3af39f4ae9236c1c5715e0fd9880bcce6a36121d62feb`).
This investigation does not modify the running training snapshot or deployment
mapping. Fitting and geometry audits run on the CPU. Contact replays use separate
single-environment Newton processes; controller inspection is read-only.

| Step | Work | Completion evidence |
| --- | --- | --- |
| 1 | Audit installed LeRobot degree normalization, homing, read/write signs, and joint order. | Exact equations and source locations; distinguish JSON values from hardware-register evidence. |
| 2 | Compare analytical FK and fixed-jaw contact site with the loaded USD. | Check link transforms, joint axes, site frame and collision geometry; report physical geometry uncertainty separately. |
| 3 | Compare bounded constant-offset models with three, four and five free joints. | Fixed keyboard and tip; zero/historical baselines; fit-only multistarts and sensitivity/conditioning diagnostics. |
| 4 | Test each frozen candidate on separate captures and a different event-sampling window. | Per-recording/per-key errors; no parameter updates from validation data. |
| 5 | Check contact replay, then supervised hardware FK. | Correct down/up sequence without adjacent keys in simulation; physical one-key checks after a candidate is reviewed. |

## Data partition, fixed before the new fits

- Fit: `20260915T220312.112879Z`, the first complete A–Z capture (26 presses).
- Independent-repeat validation: `20260915T222019.603680Z` (26 presses).
- Additional validation: `20260916T003449.479156Z`, `20260916T003732.245749Z`
  and `20260916T003804.510719Z` (Q/G/P, nine presses total).

All captures have already been inspected in earlier diagnostics. These are data
excluded from the new parameter fits, not a previously unseen experimental test.
Recorded task metadata uses the older photo candidate; this analysis explicitly
replaces only that nominal keyboard transform with the frozen reference. Original
recordings remain unchanged.

Compare mean FK over each key-down/up interval against modeled key centres at an
assumed 1.8 mm depression. The user confirmed on September 15 PDT / September 16
UTC that all Q/G/P presses reached the mechanical bottom; this confirms the
intended press depth, not independently measured travel or tip location.
Repeat using the last encoder sample before receipt of
key-down to measure sensitivity to the contact-window assumption. Neither USB
timing nor a letter event independently measures the physical jaw contact point.

The three-joint fit adjusts lift/elbow/wrist-flex with pan/roll fixed at zero;
the four-joint fit also adjusts pan; the five-joint fit is an identifiability
diagnostic. Each free offset is bounded to ±15 degrees. No link lengths, tip
location, keyboard pose, scales or per-key corrections are fitted simultaneously.

A candidate must improve all independent captures without unexplained bound hits
or strong sensitivity to sampling/initialization before progressing to contact
replay. A low fit residual alone is insufficient, and these reports are not
deployment-calibration files. Final physical accuracy still requires an actual
robot check.

## Recording audit

All five captures use the same robot, calibration hash and robot USD hash. Samples
are approximately 20.2 ms apart. In the A–Z captures wrist roll spans only
0.35–0.44 degrees across key-contact poses; wrist flex spans 0.53 degrees in the
second pass. Thus these data weakly constrain some joint corrections. The first
two Q/G/P captures add greater wrist-flex variation and are useful separate tests.
Detailed evidence is in
`output/fixture-photos-20260915/mapping-audit-20260916/recording-audit.json`.

## Offline results

The installed degree formula and stored degree/radian pairs are consistent.
Raw encoder counts were not saved, so this is a software/quantization check,
not an independent measurement of the physical zero.
There is no range-percentage/degree scaling error or software double addition of
homing offsets. Motor-register agreement was not recorded at collection time;
the connection helper now checks `bus.is_calibrated` before any position/torque
write, including torque-off recording connections.

Independent reconstruction from the exact loaded USD agrees with analytical FK
over 1,000 random poses: maximum fixed-tip difference 0.00385 mm. The active tip
site is 0.260 mm inside the fixed-jaw collision surface. This verifies the
software geometry, not the physical robot's dimensions or zero pose.

Fitting four offsets (wrist roll fixed at zero) gives the following **diagnostic
candidate**, in pan/lift/elbow/wrist-flex/wrist-roll order:

```text
q_model_deg = q_lerobot_deg + [2.73003, -1.20417, -7.20540, -2.00686, 0]
```

| Capture | Historical offsets: mean 3D error | Four-joint candidate |
| --- | ---: | ---: |
| First A–Z, fit only | 12.51 mm | 2.54 mm |
| Second A–Z, excluded from fit | 12.75 mm | 2.70 mm |
| First Q/G/P, excluded from fit | 13.09 mm | 4.50 mm |
| Second Q/G/P, excluded from fit | 12.62 mm | 2.46 mm |
| Third Q/G/P, excluded from fit | 13.29 mm | 2.63 mm |

All fitted offsets are inside the prescribed bounds; both initializations agree.
The five-joint fit provides negligible accuracy gain, while its Jacobian
condition number increases from about 38 to 331. Using pre-down samples instead
changes the four-joint offsets by at most 0.54 degrees and preserves the improvement.
In 100 first-capture bootstrap refits, individual lift/elbow/wrist offsets still
vary substantially; 95% of the predicted held-out tip displacements from the
nominal fit are within 1.37 mm. This supports a useful local predictive candidate,
not a uniquely identified universal joint-zero calibration.

Reports, exact input hashes and timestamps are under
`output/fixture-photos-20260915/mapping-audit-20260916/`: `encoder/`, `geometry/`,
`interval-fit/mapping-report.json`, `pre-down-fit/mapping-report.json`, and
`fit4-bootstrap.json`. No deployment mapping has been changed. Contact replay and
physical validation remain separate gates.

## Contact replay decision

The four-joint candidate was frozen before replaying the four validation
recordings on the exact current keyboard placement. It is **not qualified for
deployment**:

| Recording excluded from fitting | Newton contact result |
| --- | --- |
| Second A–Z | 25/26 intended letters registered; Q missed; extra 9 during O. |
| First Q/G/P | Q and P registered; G missed. |
| Second Q/G/P | Q registered twice during the Q press; P registered; G missed. |
| Third Q/G/P | Exact Q/G/P down/up sequence. |

Only 1/4 complete validation recordings reproduced the exact event sequence.
During the second A–Z Q hold, the modeled tip stays above the released key rim,
and its simulated depression never reaches the down threshold.
The extra 9 overlaps O; it is not evidence that the target label was changed.
Do not promote the candidate based on the improved mean point error.

The existing read-only controller diagnostic was then run: the current calibration
file **matches all motor calibration registers**, with zero register-read errors.
No motor registers, torque state, or position commands were changed. This is a
current check, not retroactive proof of register state during the older captures.
Evidence: `encoder/current-register-check.json` and `contact-replay-summary.json`
under the audit output directory.

Next, obtain a correspondence that constrains wrist orientation as well as tip
position before adding more free parameters. For example, deliberately varied
wrist poses at the same marked contact point provide different information from
more contacts with nearly fixed wrist angles. A physical orientation reference
would further separate joint-zero error from contact-point/model assumptions.
The present data support a useful local position correction but do not uniquely
identify a physically validated joint mapping. Keep the frozen keyboard and
deployment mapping unchanged while collecting that evidence.

## Historical comparison — 2026-09-16 UTC

With each setup's own recordings and frozen keyboard pose, the extended
per-key replay gives **77/78 clean 25 Hz press/release cycles for the historical
mapping** and **31/35 for the current four-joint candidate**. Rejecting faster
bounce and requiring sustained 15 mm clearance gives 8/78 and 30/35 respectively;
the historical captured hover motions often do not reach 15 mm. These isolated
probes reset between keys and do not replace the continuous replay gate above.
No offsets or keyboard placement were changed for the comparison. The
dated report (local artifact: `output/fixture-photos-20260915/matched-contact-comparison-20260916/README.md`)
preserves the different acquisition methods, approach endpoints, event sampling
rates, exact commands and results.

## Contact diagnosis — September 15, 2026 PDT / September 16 UTC

The failed Q/G paths stay inside the intended cap outline but remain too shallow
in the model. For the first, second and third Q/G/P captures, mean modeled G tip
height relative to its default rim is +0.51, −0.63 and −1.86 mm; only the third
reproduces G. Instrumenting the actual robot pose before the display pose is
reapplied preserves this depth mismatch. Q in the second capture also produces
brief simulated bounce instead of the sustained physical hold.

Controlled replays with zero velocity and with the derivative of the interpolated
position do not resolve the Q/G failures. Their borderline bounce counts can
change, so detector rate and continuous contact duration must remain explicit.
Live simulator-clock checks confirm 10 ms per replay step. The earlier event
times and counts do not require a timing correction.

Force-bearing collision pairs identify the extra 9 during O as **moving-jaw
contact**. Closing only the model gripper from the recorded −9.4943° to −10°
reduces 9's peak depression fraction from 0.54749 to 0.33959 and removes its down
event in the complete A–Z replay. The same run still misses Q. This isolates a
gripper-closure sensitivity in simulation; it does not establish a physical
gripper calibration correction. All five arm commands and the keyboard stay fixed.

On September 16 the operator confirmed that the moving jaw was fully closed
throughout collection. All 10,487 samples across the five captures contain the
same 0.459770% gripper reading, converted to −9.494253° by the existing range
mapping. Thus recorded jaw motion does not explain the extra 9; check the closed
endpoint convention and physical jaw geometry. The nominal model endpoint is
−10°, a 0.505747° difference. This is separate from the five arm offsets and does
not resolve the Q/G depth mismatch. Evidence:
`output/fixture-photos-20260915/qgp-contact-diagnostic-20260916/closed-jaw-confirmation.json`.

The product replay now logs the actual pre-repose robot state, tip drift and
event-sampling convention. No mapping or collision geometry was changed. The
dated diagnostic report (local artifact: `output/fixture-photos-20260915/qgp-contact-diagnostic-20260916/README.md`)
contains the controlled results, contact pairs, plots, hashes and regression
evidence. Keep fit4 unqualified; next constrain the actual jaw contact point and
wrist orientation at the same marked key location, rather than fitting more
offsets to these already-inspected failures.

## Model conventions and physical reference — September 16, 2026

The current USD and official new-calibration URDF agree over all 10,487 recorded
poses to 0.000071 mm at the fixed tip. The upstream older model is a coordinate
convention variant after base/joint conversion. These comparisons do not establish
agreement with the physical robot. Across 61 recorded presses, direct versus
historical conversion gives mean horizontal errors of 36.30 versus 12.17 mm and
mean vertical errors of -19.80 versus -3.34 mm (modeled tip minus modeled pressed
key centre; unchanged frozen keyboard). Both A–Z captures and all three Q/G/P
captures now have continuous direct/historical contact replays; none reproduces
the exact physical sequence. Within-step robot drift limits interpretation of
contact results. Adjusting only gripper closure cannot move the fixed typing tip.

No calibration or deployment mapping was changed. The operator chose a side-on
phone view paired with read-only encoder capture to obtain an independent
physical pose reference. Detailed dated evidence, assumptions, plots and commands:
`output/model-comparison-20260916/physical-conventions/README.md`.


### Autonomous motion and separation of tracking error — 2026-09-16

The operator explicitly authorized autonomous SO-101 motion. A new local bounded
helper seeds raw measured arm goals before enabling torque, changes arm joints
only, and preserves calibration/PID/gripper settings. Camera capture runs alongside
motion and three-second stationary encoder windows. No policy or key presses are
part of this experiment. Nominal speed was reduced from 1 to 0.5 degrees/second
when observed tracking error approached the unchanged 3-degree abort threshold.

Tracking error is `q_measured - q_goal`, using the encoded raw goal. Static model
correspondence is the observed physical link pose versus `FK(q_measured)` (with
zero or historical conversion). These are different comparisons. The renderer
uses measured median angles; replacing them with targets would confound tracking
with coordinate mapping. The initial stationary reference was captured with all
motor torque disabled and before this experiment issued any goals.

The first three elbow targets of -4/-6/-8 degrees relative to start produced
measured changes of -1.758/-3.516/-5.451 degrees, respectively. Settled error against
the quantized motor goal was +2.286/+2.549/+2.549 degrees. All stationary arm encoder
ranges were zero; this does not rule out structural compliance. Sources and exact
metrics are in `output/realsense-reference-20260916/tracking-versus-mapping.json`.

The elbow-only captures support repeatable image/encoder direction correspondence
but cannot identify a unique offset with an unknown marker mount. Compare complete
measured states on a reverse approach, then add independent joint variations.
Camera/marker uncertainty and load-dependent effects remain separate hypotheses;
do not fit or install a correction from goal tracking error or from these images
alone.

### Reverse approach and independent wrist check — September 16, 2026, 13:24 PDT

The outward -4° and reverse -2° elbow commands settled at nearly the same measured
arm pose: raw pan/lift/elbow/flex/roll differences `[0,-1,+1,0,0]` counts. Their
stationary marker centres differ by 0.664 pixels. Tracking error was +2.286° on the
outward approach and +0.352° on return. This shows a history-dependent command to
encoder relationship; a constant joint-zero conversion cannot explain that gap.
It does not identify which of deadband, friction, load or backlash dominates.

An independent wrist-flex motion changed only that measured joint by -2.197802°;
the other five motor counts stayed identical. The marker moved 4.590 pixels, and
the two retained planar-pose branches estimate rotations of 2.269° and 2.157°.
This supports angle/motion correspondence but does not identify the offsets:
direct and historical mappings predict the same wrist-axis direction, and the
small excursion plus unknown marker mount leaves zero estimation weak.
The mappings' predicted relative translations differ by only 0.938 mm, below the
2.3–2.6 mm variation in estimated marker depth during stationary captures. Do not
use this wrist test to choose between those conversions.

Historical measured-angle FK still looks closer at the jaw. Camera/base
registration and structural uncertainty prevent treating the overlay as a
qualified correction. No offsets were fitted or installed. The initial torque-off
visual discrepancy cannot be attributed to command tracking, but is not by itself
proof of an encoder-zero error.

The arm returned within 0.352° of its starting readings; arm torque remains on and
gripper torque off. Calibration, PID gains and frozen placement are unchanged.
The dated measurement report (local artifact: `output/realsense-reference-20260916/README.md`)
indexes the tracking table, near-matched reverse poses, overlays and raw evidence.

### Powered versus hand-guided comparison — September 16, 2026

The next operator-led comparison uses a teleoperation leader while recording the
follower's actual encoder trajectory and physical key events. Keep the current
scene and calibration fixed; use no added offsets. Compare Q/G/P geometry with
the torque-off recordings before attributing a difference to loading/compliance
or fitting any correction. The [local procedure](leader_replay_20260916.md) keeps
leader requests and follower goals separate from measured replay positions.

### Trajectory-based scene alignment — 2026-09-17

The historical benchmark used both an effective keyboard transform and joint
offsets. It did not establish literal tabletop geometry independently. See the
[historical/current pose comparison](../fixtures/historical_offset_provenance_20260718.md#effective-scene-versus-physical-placement--clarification-2026-09-17).
The current offsets-only policy trials produced presses, dragging and a reported
space-bar contact; the operator also reported failure without offsets. The
30 mm clearance override is an unvalidated experiment, not a demonstrated fix.
These observations do not uniquely identify the underlying error.

Next work, not yet completed:

1. Preserve the measured September scene, raw recordings, calibration and model
   hashes. Keep torque-off and powered recordings identified separately. Replay
   measured follower angles with their timestamps, not leader goals or commanded
   positions. These recordings calibrate/validate the scene; they are not PPO or
   imitation-learning training data.
2. On fitting sessions, compare the unchanged scene with zero/historical/current
   offsets, a keyboard-pose-only fit, and a combined rigid keyboard-pose plus
   four-joint-offset fit. Keep wrist roll, asset scale and jaw geometry fixed for
   this comparison. Use measured datums and their uncertainty to constrain pose;
   report parameter bounds, correlated solutions and departures from measurements.
   A solution needing tilt/height outside those bounds is an effective-model
   diagnostic, not proof that the real keyboard was there. Do not overwrite the
   physical-reference task or silently sink the keyboard through its support.
3. Freeze each candidate and replay complete recordings excluded from fitting.
   Compare intended down/up sequence, extra contacts including space/non-letter
   keys, retraction/clearance and per-key residuals. Key events constrain contact;
   encoder-only recordings do not independently measure the free-space jaw path.
   Previously inspected validation sessions are regression evidence; reserve a
   fresh session for final validation after selecting the approach.
4. Check the chosen scene/mapping with policy-free physical Q/G/P commands and
   measured tracking. A measured-pose replay pass alone does not prove powered
   command tracking. Only then train/evaluate against that scene, or explicitly
   evaluate an existing checkpoint as transfer if its trained scene differs.

Offsets and keyboard pose may compensate for each other; this experiment seeks
a consistent, validated operational model, not a unique mechanical diagnosis.
No new scene fit, replay pass, physical qualification or deployment change is
claimed by this plan.

### First trajectory-fit result — 2026-09-17

The first offline comparison used the mapping-collection recording from
`20260916T220434.751544Z`. Its first occurrence of each A–Z pair was retained
in a derived fit input; the raw recording was not modified. The later
`mapping-validation` recording was kept as the comparison recording. Some of
those key presses contain only two samples, so this is a relaxed diagnostic,
not the normal three-sample contact gate.

With the measured September keyboard pose held fixed, XY/yaw fitting after
direct zero-offset replay produced approximately 19.9 mm fit error, 20.5 mm
held-out error and 20.4 mm on the later comparison recording. Applying the
historical joint offsets before the same pose fit produced 3.38 mm fit error,
3.83 mm held-out error and 5.65 mm on the later comparison recording. The
resulting historical-plus-pose candidate was approximately
`[259.1, 26.0, 7.64]` mm asset origin with a `-87.46°` yaw. It is a diagnostic
effective scene, not a new physical measurement or deployment mapping.

The corresponding reports and task candidates are retained under
`output/fixture-photos-20260915/trajectory-scene-alignment-20260917/`. Newton
contact replay was not completed because the local runtime image fingerprint
must be rebuilt after the current checkout changes. The residual reduction is
therefore evidence that the combined historical mapping plus effective pose
explains yesterday's measured trajectories better than either zero offsets or
the frozen placement alone; it is not yet a policy or physical success result.

After rebuilding the pinned runtime, the held-out `mapping-validation` replay
was run with both candidates. Direct zero-offset plus pose produced 266 simulated
key events for 66 physical events, with repeated numeric and letter contacts.
Historical offsets plus the fitted pose produced 72 simulated events for the same
66 physical events. It tracked the intended sequence much more closely, but still
added a `9` contact around the O transition and produced extra U/J activity.
`key_sequence_matches` therefore remained false. Reports are retained under
`output/fixture-photos-20260915/trajectory-scene-alignment-20260917/placement-*`.
The combined candidate is promising diagnostic evidence, not a validated scene,
policy fix or physical deployment mapping.

### Yesterday-only joint-plus-pose fit — rejected diagnostic — 2026-09-17

To answer the requested comparison directly, a second fit used only the
September 16 mapping collection and no historical offset or pose constants. It
jointly optimized four arm offsets plus keyboard XY/yaw, with wrist roll held at
zero. The result was offsets
`[+15.00, +0.40, -8.70, +0.43, 0]` degrees and a keyboard origin near
`[257.2, -32.7, 7.64]` mm. The pan parameter reached its bound and the lateral
pose moved about 46.8 mm in Y from the measured reference (+14.1 to -32.7 mm).
Its Newton replay produced
70 events versus 66 physical events and still added a `9` contact.

A fully free rigid-pose diagnostic drove wrist roll and keyboard yaw to their
bounds and worsened held-out residuals. Both results are rejected: they are
correlated fits to contact data, not evidence that the robot or keyboard was
physically positioned at those values. These failed fits do not establish that
yesterday's recordings are insufficient. The historical-method audit below
identifies differences in the recent fitting procedure that must be corrected
before drawing that conclusion.

### Original workflow recovered from archived source — 2026-09-17

The archived `fit_powered_contact_calibration.py` fixes shoulder pan and wrist
roll at zero, fits lift/elbow/wrist-flex offsets, and solves keyboard XYZ/yaw
against the fixed-jaw contact site. It uses measured joint states associated with
key-down edges, not average joint angles over the entire down/up interval. The
original collection contains three powered repetitions per A-Z key; the current
recordings have different collection/timestamp semantics that must be retained.

The recent fit freed both pan and keyboard yaw. These share a coordinate
ambiguity: rotating the arm about shoulder pan and transforming the keyboard
with it preserves relative tip/contact geometry. A read-only numerical check
on yesterday's sampled poses absorbed a 15-degree pan change with one rigid
transform to below 1e-9 mm residual. Therefore pan hitting a bound does not
itself diagnose bad data or require independent camera measurements. Fixing one
coordinate reference removes this redundant degree of freedom.

Original Newton reports recover the contact-validation progression:

| Archived candidate | First-key replay result |
| --- | --- |
| Flat keyboard with diagnostic fitted offsets | 19/26 |
| Revised effective tilt and offsets | 23/26 |
| Same revised offsets/orientation, keyboard translated +2 mm X, +1 mm Y, +1.5 mm Z | 26/26 in each of three repetitions |

The final translation is directly recoverable from the archived reports; the
exact optimizer output producing the final tilt/offset combination is still not
preserved in the known diagnostic fit reports. The gate resets per key and stops
at first actuation. It does not establish clean full-trajectory release/transit.

Next use yesterday's measurements to estimate new values with the original
coordinate ambiguity removed, correct fixed-jaw FK and explicit key-event
sampling. Refine a separate effective rigid keyboard pose against actual
simulated contacts. Evaluate both the original first-contact test and complete
press/release replays, including extra keys. Freeze that verified scene and its
target map before training for physical transfer. The historical method is the
reference; historical parameter values are not inputs to the new fit.

**Update, September 17 at 10:00 PDT:** that fresh event-associated fit and
continuous Newton replay are now implemented in `scripts/calibrate_scene.sh`.
The separate manual run includes all 33 intended presses but adds `9` and `U`,
so calibration remains unqualified. See the [dated result and remaining work](scene_calibration_20260917.md).
