# Encoder/model mapping diagnostic — 2026-09-15

> Dated investigation record. Commands and results describe this workstation at the time; follow [the portable workflow](../../FIXTURE_SETUP.md) for a new setup. Raw recordings and `output/` evidence are local artifacts, not included in the source release.

This report uses the two saved torque-off A–Z recordings from the current robot
and the frozen `physical_reference_20260915` keyboard placement. For each key it
compares the mean recorded arm pose during the physical key-down/up interval to
the frozen model key centre, with the model's 1.8 mm press-depth assumption. It
fits a constant five-joint offset with nonlinear FK.

The combined 52-press fit was:

```text
shoulder_pan   +2.4984 deg
shoulder_lift  -0.5999 deg
elbow_flex     -8.1694 deg
wrist_flex     -1.1292 deg
wrist_roll     +6.0380 deg
mean residual   2.60 mm
```

The historical July convention is `[0, +1.9336, -10.7736, +1.3648, 0]` deg.
For the same frozen placement, the direct zero-offset and historical conventions
produce mean residuals of approximately 41.26 mm and 12.50 mm respectively.
Thus the historical values clearly move the predicted jaw toward the keys,
especially in Z, but they do not uniquely identify the new robot's calibration.

The per-recording fits are not identical:

| recording | pan | lift | elbow | wrist flex | wrist roll | mean residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260915T220312 | +2.64 | -1.24 | -7.14 | -1.99 | +1.65 | 2.54 mm |
| 20260915T222019 | +2.23 | +0.08 | -9.25 | -0.15 | +13.68 | 2.61 mm |

This instability is expected because a keyboard event does not independently
measure the jaw pose: contact timing, key compliance, the exact contact point,
and hand-guided posture are all folded into the fit. The result is therefore a
diagnostic of the current scene, not a valid replacement for encoder calibration
or a universal SO-101 mapping. In particular, do not install these fitted values
in the deployment profile and do not copy another robot's LeRobot calibration.

The actionable convention remains:

* LeRobot calibration normalizes this robot's raw Feetech encoders.
* Direct replay converts normalized arm degrees to model radians with zero added
  offset.
* The optional benchmark profile adds the historical July values above.
* A production nonzero mapping needs independently measured robot/model poses,
  supplied to `scripts/so101_homing/joint_calibration.py`, followed by a fixed-jaw
  contact probe on the frozen placement.

For the historical July derivation and its rejected diagnostic fit reports, see
[historical offset provenance (2026-07-18)](../fixtures/historical_offset_provenance_20260718.md).
