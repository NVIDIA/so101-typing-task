# Scene and encoder-mapping investigation

These are dated engineering records. For setup instructions start with
[fixture measurement and scene validation](../../FIXTURE_SETUP.md), then
[alignment qualification](../release/ALIGNMENT.md) and
[hardware preparation](../release/HARDWARE_PREPARATION.md).

The September 15 keyboard/base placement stays frozen. The historical elbow
correction (-10.77 degrees) and current four-joint fit (-9.61 degrees at the
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
