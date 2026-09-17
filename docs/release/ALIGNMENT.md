# Qualify the scene before physical policy deployment

This workflow records which robot calibration, keyboard placement and joint
mapping passed measured-motion replay and physical Q/G/P contact checks.
It does not qualify policy performance or prove every key in the workspace.
The current development robot's fitted mapping has **not** passed this gate.

All `alignment` commands are offline. Only the explicitly marked FK `--execute`
step moves hardware. Standard [LeRobot calibration](HARDWARE_PREPARATION.md#install-and-calibrate)
remains the setup procedure; automatic endpoint calibration is an optional,
unfinished experiment.

## 1. Freeze the scene and reserve a recording

Follow [fixture setup](../../FIXTURE_SETUP.md): measure the keyboard relative to
the plastic robot base, save `output/my-fixture/task.json`, and record measured
key presses with the moving jaw closed. Use a separate capture for validation.
Cover A–Z where possible; qualification requires at least Q, G and P, complete
down/up pairs, and an exact simulated key sequence with no extra contacts.
Do not adjust placement while fitting joint offsets.

For the released 19k checkpoint, `configs/tasks/reference-fixture.json` selects
its original nominal simulation scene. Use that file consistently in place of
`output/my-fixture/task.json` below when testing that fixture. It does not assert
that your physical setup already matches the historical model.

## 2. Prepare an unqualified mapping

After fitting with the [recording-based diagnostic](../../FIXTURE_SETUP.md#fit-four-joint-offsets-with-the-keyboard-frozen):

```bash
./so101 alignment prepare --robot-id my_typing_robot \
  --task-config output/my-fixture/task.json \
  --mapping-report output/my-fixture/mapping-audit/mapping-report.json \
  --mapping-candidate fit4 --out-dir output/my-fixture/alignment
```

Alternatively replace the two mapping-report arguments with exactly one of:

- `--encoder-convention lerobot` for the direct encoder/model comparison.
- `--encoder-convention benchmark` for an explicit historical-offset comparison.
- `--joint-calibration output/joint-fit.json` for independently measured joint correspondence.

The output `alignment.json` is **unqualified**. Preparing it freezes offsets and
identities; it does not approve physical deployment. The recording-based path
accepts the standard diagnostic `fit4` report and requires distinct fitting and
reserved recording hashes. Ad-hoc legacy reports need their provenance preserved;
do not relabel them as standard fit reports.

## 3. Replay reserved measured motion in simulation

Replay using the prepared candidate. This uses the same frozen offsets for
direct, historical, independently measured and recording-fitted mappings:

```bash
./scripts/scene_check.sh replay --recording /absolute/path/to/validation/recording.json \
  --alignment output/my-fixture/alignment/alignment.json
```

Retain the printed `scene-report.json` path. A completed replay is not necessarily
a pass: qualification recomputes the physical/simulated event comparison. A
missed, extra, wrong or unreleased key fails the gate. For `fit4`, this recording
must belong to the report's reserved validation set, never its fitting set.

## 4. Simulate and physically test Q, G and P separately

For each key, generate one model-coordinate trajectory and run it in Newton.
Example for Q; repeat with G and P and separate `fk-G`/`fk-P` output folders:

```bash
./scripts/run_fk_probe_f.sh --generate --letter Q \
  --task-config output/my-fixture/task.json \
  --trajectory output/my-fixture/fk-Q/trajectory.json
./scripts/run_fk_probe_f.sh --trajectory output/my-fixture/fk-Q/trajectory.json --sim
```

After the simulation passes, with the physical sweep clear and the operator
ready to stop power, run **one key at a time**:

```bash
./scripts/run_fk_probe_f.sh --trajectory output/my-fixture/fk-Q/trajectory.json \
  --alignment output/my-fixture/alignment/alignment.json \
  --robot-id my_typing_robot --port /dev/serial/by-id/YOUR_CONTROLLER \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd --execute
```

Omit `--execute` for preflight. The selected mapping adds offsets to measured
encoder angles and subtracts them from the unchanged simulated commands.
The probe checks the calibrated command range, startup/tracking, actual key
down/up, and measured moving-jaw closure (at most 2% of its calibrated range).
After key-up it requires two consecutive measured FK samples at least 15 mm
above the released rim. This clearance is model-derived, not independent vision.
The arm remains torqued on after stopping. Keep every printed hardware log.

Older probe logs lack the alignment identity and jaw telemetry and cannot qualify
this gate. Do not edit or backfill them; collect fresh evidence when hardware
testing is authorized. A failed attempt stays a failed attempt.

## 5. Qualify and package the evidence

Supply the exact files produced above. Each `--probe` takes a trajectory, its
simulation report and one successful physical log:

```bash
./so101 alignment qualify \
  --alignment output/my-fixture/alignment/alignment.json \
  --recording /absolute/path/to/validation/recording.json \
  --replay-report /absolute/path/to/matching/scene-report.json \
  --probe output/my-fixture/fk-Q/trajectory.json output/my-fixture/fk-Q/sim-report.json /absolute/path/to/Q-hardware.jsonl \
  --probe output/my-fixture/fk-G/trajectory.json output/my-fixture/fk-G/sim-report.json /absolute/path/to/G-hardware.jsonl \
  --probe output/my-fixture/fk-P/trajectory.json output/my-fixture/fk-P/sim-report.json /absolute/path/to/P-hardware.jsonl \
  --out-dir output/my-fixture/qualified
```

The new directory contains `qualification.json` and copied evidence with hashes.
Qualification recomputes the checks from the evidence; setting a `passed` flag
alone cannot satisfy it. These are provenance checks, not signed proof of
physical truth. Keep unsuccessful attempts alongside the originals for review.

## 6. Train/export for that scene

Train and evaluate with the same nominal task configuration. Export using:

```bash
./so101 prepare-deployment --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --robot-id my_typing_robot \
  --alignment output/my-fixture/qualified/qualification.json \
  --out-dir output/my_typing_robot
./so101 deploy output/my_typing_robot NVIDIA
```

For the released checkpoint, use its downloaded checkpoint/environment paths.
Export checks the trained nominal key map, keyboard pose, reset and offsets
against the qualified scene. It copies all qualification evidence into the
bundle, so later checks do not depend on the original recording directories.
The deployment runner rechecks calibration, geometry and evidence before motor
access. Existing bundles without qualification still support software dry runs;
physical execution requires a new export with `--alignment`.

Recalibration, moving the fixture or changing the jaw requires requalification.
Software cannot detect a physically moved keyboard or replaced tip from file
hashes; keep the fixture marked and fixed. Q/G/P alignment does not establish
full-keyboard accuracy, placement tolerance or successful six-letter policy typing.
