# Scene and encoder-mapping investigation

These are dated engineering records and advanced diagnostics. For normal setup use
the [short real-to-sim workflow](../release/REAL2SIM.md), with
[fixture measurements](../../FIXTURE_SETUP.md), then
[alignment qualification](../release/ALIGNMENT.md) and
[hardware preparation](../release/HARDWARE_PREPARATION.md).

The [current September 16 recording fit](scene_calibration_20260917.md) passes
**93/93 first-key contacts with a reset between keys**, the original type of
scene-validation test. That report lists the selected keyboard pose and joint
offsets. Both recordings were used for fitting. Continuous replay is an additional
diagnostic and still fails; it was not part of the original per-key criterion.
The measured September 15 reference is preserved separately from this effective
simulation scene. On September 17 at 14:36 PDT, a fresh policy trained on this
fit completed [physical NEWTON typing](physical_typing_success_20260917.md) in
7.10 seconds, including release and clearance. This is one successful sequence;
broader reliability and the full alignment qualification remain unmeasured.

Earlier fixed-placement results: the historical elbow
correction (-10.77 degrees) and earlier four-joint fit (-9.61 degrees at the
elbow) both improve correspondence, but useful fitted offsets do not uniquely
identify a mechanical cause or transfer automatically between robots.
The reserved torque-off recording matched 32/33 intended simulated contacts
with an additional `9` contact and a missed N; the candidate remains diagnostic.

As of **2026-09-16 18:09 PDT**, the automatic elbow endpoint experiment is paused
at the operator's request. No complete min/max comparison was accepted and no
new calibration was installed. The measured velocity/position-mode encoder
difference is a controller-mode detail, not a new model offset. Hardware work
requires resuming with the operator; repository checks do not move the robot.

| Record | Contents |
| --- | --- |
| [First physical NEWTON success](physical_typing_success_20260917.md) | End-to-end solution, exact policy/scene/mapping, training time and verified physical events |
| [Advanced scene diagnostics](scene_diagnostics.md) | Lower-level recording/replay, joint/pose fitting alternatives and single-key probes |
| [Selected September scene fit](scene_calibration_20260917.md) | Recorded-contact optimization, selected pose/offsets and limits of the 93/93 result |
| [Mapping plan](mapping_plan_20260916.md) | Ordered investigation, reserved-data results and paused automatic-calibration experiment |
| [Encoder/model correspondence](encoder_model_mapping_20260915.md) | Direct, historical and current mapping comparisons and zero-convention hypothesis |
| [Initial diagnostic](encoder_model_mapping_diagnostic_20260915.md) | Earlier recording and geometry audit |
| [Pre-contact Q/G/P checks](new_robot_precontact_probe_20260916.md) | Policy-free simulation and physical-test boundaries |
| [Powered leader recordings](leader_replay_20260916.md) | Recorder behavior, timing and measured-follower provenance |
| [Historical July provenance](../fixtures/historical_offset_provenance_20260718.md) | Original recordings, frozen scene and 78/78 replay evidence |

`output/...` paths identify local evidence, which is not shipped with the source.
Some commands in these records describe earlier workstation launchers. Shared
tools now require explicit inputs; use their `--help` and the portable setup
guide. Workstation-only shortcuts are retained in ignored `local/scripts/`.

Automatic calibration is experimental and separate from standard LeRobot setup.
`./scripts/run_auto_calibration_probe.sh --help` describes the bounded single-joint
probe. It requires an explicit robot ID and port, preserves the existing
calibration, and never installs a candidate. `--execute` is physical motion,
`--snapshot` is read-only, and `--check-mode` briefly changes torque/mode and
requires physical support. No one should run those modes as routine source tests.
