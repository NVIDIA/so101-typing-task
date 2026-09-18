# Encoder/model investigation: findings and remaining work

The current operational workflow fits a joint mapping and effective simulated
keyboard pose from real contacts, then trains in that scene. See the
[selected calibration](scene_calibration_20260917.md) and
[physical NEWTON result](physical_typing_success_20260917.md).
This page retains the useful root-cause findings and next experiments. Raw captures
and detailed diagnostic reports remain local; earlier session notes are in Git history.

## What is established

- The real keyboard/base datums were checked and the physical fixture stayed fixed.
  Direct replay of measured joint angles still produced incorrect contacts.
- Historical offsets reduced much of the vertical discrepancy. A fresh four-joint
  fit improved horizontal agreement, but offsets alone did not yield exact replay
  or reliable physical typing on the measured-reference scene.
- Joint offsets and a separately fitted simulation pose reproduced 93/93 isolated
  first contacts. A policy trained in that scene completed NEWTON physically.
- These results support an effective correction. They do not identify a unique
  mechanical fault, establish universal offsets or prove identical real/sim geometry.

The recordings contain measured follower motion and physical key events. Powered
leader recordings preserve leader goals separately; those goals are not replayed
as if they were measured follower positions. Recordings calibrate the scene and
are not PPO demonstrations.

## Coordinate and geometry audit

Keep three relationships distinct:

1. Raw controller position to normalized LeRobot degrees.
2. LeRobot degrees to model joint coordinates.
3. Model coordinates to link and fixed-jaw pose through FK.

The inspected degree conversion is
`q_LeRobot = (Present_Position - (range_min + range_max)/2) * 360/4095`.
Present_Position already includes the servo's homing correction. The calibrated
range midpoint need not equal the USD joint zero. Apply a fitted correspondence
as `q_model = q_LeRobot + offset`, with inverse subtraction for hardware goals.
Do not add the motor homing value a second time.

Analytical fixed-jaw FK agrees with the loaded USD within 0.004 mm over 1,000
random poses. The USD and upstream new-calibration URDF also agree on the recorded
poses. This checks software geometry, not the physical link dimensions or zeros.
The nominal fixed-tip site lies about 0.26 mm inside its model collision surface.

Direct FK placed the jaw below the common tabletop at recorded physical contacts.
A keyboard-height-only shift would leave the large horizontal error and require
an implausible support-plane crossing. See the
[height versus arm-correspondence check](encoder_model_mapping_20260915.md#root-cause-check-keyboard-height-versus-arm-correspondence--2026-09-16).
Smaller keycap, collision or jaw-geometry errors remain possible.

## What the earlier fits showed

| Comparison | Finding | Limit |
| --- | --- | --- |
| Historical elbow-only mapping on 22 excluded powered presses | Mean absolute Z error fell from 20.01 to 7.94 mm | FK against assumed contact centres, not a policy success rate |
| Fresh four-joint fit on the fixed September scene | Improved lateral agreement; fresh torque-off replay matched 32/33 intended contacts but added 9 and missed N | Failed exact replay |
| Fresh four-joint vector on original July recordings/layout | 0/78 first contacts versus historical 78/78 | Different robot/calibration context; not isolated layout transfer |
| Five-joint contact-only fits | Little extra accuracy and poorly constrained wrist roll | Low residual does not identify unique joint zeros |
| Full measured-motion replay | Millimetre-scale robot drift can occur during physics integration | Displayed imposed-pose accuracy does not rule out contact-step effects |

The [mapping comparison](encoder_model_mapping_20260915.md) preserves numerical
results, fitting assumptions and transfer limits. The repeatable negative elbow
correction near -10 degrees is useful evidence, but lift/elbow/wrist corrections
can compensate for one another.

One extra simulated 9 contact was traced to the moving jaw. A small change in
its closed model angle removed that contact but did not fix shallow Q/G presses.
The operator reported the jaw closed during recording. This identifies a model
closure sensitivity, not a proven physical calibration correction.

## Original workflow recovered from archived source

The original method associated measured joint states with key-down events, fitted
lift/elbow/wrist-flex offsets and a keyboard pose, then refined actual simulated
contacts with a reset between keys. Its contact progression was 19/26, then 23/26,
then 26/26 across each of three repetitions after a small rigid translation.
The final historical scene and mapping were frozen before training.

Freeing shoulder pan and keyboard yaw together introduces a coordinate ambiguity:
rotating the arm and transforming the keyboard with it can preserve relative
contact geometry. Fix one reference or use an explicit prior; a pan bound hit
alone does not diagnose bad recordings. The selected September search allowed
all five offsets and the keyboard pose, while reporting that ambiguity.

The [historical provenance](../fixtures/historical_offset_provenance_20260718.md)
distinguishes the recovered final contract from rejected diagnostic fits. The
September calibration follows that method using fresh data and parameters; it
uses neither the old offsets nor the old keyboard pose as a prescribed solution.

## Independent physical pose checks

Camera observations and encoder measurements must constrain the same physical
pose without allowing camera registration to absorb joint errors. Establish the
camera/base reference independently, record measured joints at stationary poses,
and keep registration fixed across identification and reserved poses. An unknown
marker-to-link transform can absorb some joint-zero corrections; a marker near
the jaw motor is not by itself an absolute joint-angle reference.

The initial RealSense overlays made historical FK look closer at the jaw, but
registration uncertainty prevented a calibrated offset estimate. A small wrist
motion produced consistent angular motion in the image; the direct/historical
translation difference was below marker-depth variation, so it could not choose
between the mappings.

Tracking and mapping are separate: tracking compares measured encoders with motor
goals, while mapping compares physical link pose with FK of those measured angles.
Opposite-direction elbow approaches reached nearly the same measured pose with
different goal errors. That history dependence cannot be explained by a constant
zero offset alone; load, friction, deadband and backlash remain possible factors.

## Automatic-calibration experiment

This experiment is paused. No complete repeatable endpoint comparison was accepted,
no candidate calibration was installed, and the existing LeRobot calibration was
preserved. Standard user setup still uses LeRobot's manual calibration.

A probe initially mixed position-mode and velocity-mode encoder frames. The
corrected handling accounts for the motor's existing homing register when comparing
modes. This is separate from the fitted LeRobot-to-model offset; it must not be
added to ordinary position-mode recordings. Subsequent attempts stopped on early
motion or stationary drift, without an accepted endpoint pair.

The bounded tool is `scripts/run_auto_calibration_probe.sh`; consult `--help` for
explicit inputs and modes. Camera availability checks do not detect obstacles.
Any resumed sweep requires a clear range and operator supervision. Automatic
endpoint repeatability would still not prove that a range midpoint equals CAD zero.

## Remaining investigation

1. Keep the measured fixture, original recordings, model and calibration as fixed
   references. Preserve the selected effective scene separately.
2. Capture independently measured link poses with sufficient lift/elbow/wrist
   variation. Reserve poses before fitting and quantify measurement uncertainty.
3. Compare direct, historical and the smallest supported constant correction.
   Distinguish camera uncertainty, joint-coordinate error and load/history effects.
4. Validate the frozen mapping on reserved poses and recordings, then physical
   Q/G/P commands with measured tracking and actual key events.
5. Test another calibrated robot before claiming any correction is universal.

For operational typing, use the [record/fit/train guide](../release/REAL2SIM.md)
and [alignment workflow](../release/ALIGNMENT.md). Successful contact fitting and
one policy trial are useful results while this root-cause work remains incomplete.
