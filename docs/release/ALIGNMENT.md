# Qualify the scene before physical policy deployment

This workflow records which robot calibration, keyboard placement and joint
mapping passed measured-motion replay and physical Q/G/P contact checks.
It does not qualify policy performance or prove every key in the workspace.
The current development robot's fitted mapping has **not** passed this gate.

All `alignment` commands are offline. FK and policy `--execute` commands move
hardware. Standard [LeRobot calibration](HARDWARE_PREPARATION.md#install-and-calibrate)
remains the setup procedure; automatic endpoint calibration is an optional,
unfinished experiment.

## 1. Freeze the scene and reserve a recording

Follow [fixture setup](../../FIXTURE_SETUP.md): measure the keyboard relative to
the plastic robot base, save `output/my-fixture/task.json`, and record measured
key presses with the moving jaw closed. Use a separate capture for validation.
Cover A–Z where possible; qualification requires at least Q, G and P, complete
down/up pairs, and an exact simulated key sequence with no extra contacts.
Preserve the measured placement in each recording. The joint-and-scene fitter
saves any effective pose correction in a separate task JSON. After the
[real-to-sim fit](REAL2SIM.md), collect a fresh capture in a separate directory;
do not include it in another fit:

```bash
./scripts/real2sim.sh record \
  --robot-id my_typing_robot --port /dev/serial/by-id/YOUR_CONTROLLER \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd \
  --task-config output/my-fixture/task.json \
  --seconds 120 --output-root output/my-fixture/validation
```

Support the arm: this disables torque and leaves it off. Keep the jaw closed and
record at least Q, G and P, including release and clearance between presses.
Use the printed recording path below. Keep the fitted task and mapping unchanged.

For the released 19k checkpoint, `configs/tasks/reference-fixture.json` selects
its original nominal simulation scene. Use that file consistently in place of
both task paths (`output/my-fixture/task.json` and `output/my-scene/task.json`)
in the recording, replay and FK commands when testing that fixture. Prepare the
historical candidate shown in step 2 instead of fitting a new keyboard pose.
This file does not assert that your physical setup matches the historical model.
If matching your setup requires a different simulated keyboard pose, use workflow 1
and train for that fitted scene; the 19k policy's map stays frozen.

## 2. Prepare an unqualified mapping

Use the fitted scene and offsets from `real2sim.sh fit`, with the new reserved
capture. Preparation checks its robot/calibration/model identity, original
measured scene, closed jaw and complete Q/G/P presses, and rejects fitting data:

```bash
./so101 alignment prepare --robot-id my_typing_robot \
  --task-config output/my-scene/task.json \
  --mapping-report output/my-scene/mapping-report.json \
  --mapping-candidate joint_scene \
  --validation-recording /absolute/path/to/validation/recording.json \
  --out-dir output/my-fixture/alignment
```

Preparation copies the new reserved capture and freezes its hash without changing
the fit report or refitting. Qualification later checks its actual replay.
Advanced `fit4` and `fit3_pose` reports can instead use their matching task and
candidate name; omit `--validation-recording` if the report already reserves
the capture. Never reuse a capture that influenced the final fit as validation.

Alternatively replace the mapping-report, mapping-candidate and
validation-recording arguments with exactly one of:

- `--encoder-convention lerobot` for the direct encoder/model comparison.
- `--encoder-convention benchmark` for an explicit historical-offset comparison.
- `--joint-calibration output/joint-fit.json` for independently measured joint correspondence.

For the optional 19k policy, this explicit historical comparison replaces the
fitted-report preparation command above; choose one, using a new output directory:

```bash
./so101 alignment prepare --robot-id my_typing_robot \
  --task-config configs/tasks/reference-fixture.json \
  --encoder-convention benchmark --out-dir output/my-fixture/alignment
```

Use your own LeRobot calibration. The historical offsets remain a candidate until
your reserved replay and physical probes pass; they are not universal constants.

The output `alignment.json` is **unqualified**. Preparing it freezes offsets and
identities; it does not approve physical deployment. The recording-based path
accepts diagnostic `fit4`, `fit3_pose` or `joint_scene` reports and requires distinct fitting and
reserved recording hashes. Ad-hoc legacy reports need their provenance preserved;
do not relabel them as standard fit reports.
Contact searches that used every supplied recording need the fresh reserved
capture above. Reserving it is not a passing replay or physical result.

## 3. Replay reserved measured motion in simulation

Replay using the prepared candidate. This uses the same frozen offsets for
direct, historical, independently measured and recording-fitted mappings:

```bash
./scripts/scene_check.sh replay --recording /absolute/path/to/validation/recording.json \
  --alignment output/my-fixture/alignment/alignment.json \
  --replay-task-config output/my-scene/task.json --headless
```

Use the same fitted task for replay, FK generation and policy training. Remove
`--headless` to open the Newton viewer from an X11 desktop terminal.

Retain the printed `scene-report.json` path. A completed replay is not necessarily
a pass: qualification recomputes the physical/simulated event comparison. A
missed, extra, wrong or unreleased key fails the gate. For recording-fitted
mappings, this capture must belong to the candidate's reserved validation set,
never its fitting set.

## 4. Simulate and physically test Q, G and P separately

For each key, generate one model-coordinate trajectory and run it in Newton.
Example for Q; repeat with G and P and separate `fk-G`/`fk-P` output folders:

```bash
./scripts/run_fk_probe_f.sh --generate --letter Q \
  --task-config output/my-scene/task.json \
  --trajectory output/my-fixture/fk-Q/trajectory.json
./scripts/run_fk_probe_f.sh --trajectory output/my-fixture/fk-Q/trajectory.json --sim
```

`--sim` requires an X11 desktop (`DISPLAY` and display access). For a headless
machine, replace it with `--video`; both produce `sim-report.json` beside the
trajectory. Check that report for a pass before physical replay.

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

## 6. Export the policy trained for that scene

Use the evaluated policy trained with this same nominal task configuration:

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
standard physical execution requires a new export with `--alignment`. The explicit
[supervised experimental mode](#optional-supervised-experiments) accepts a prepared
candidate instead.

After reviewing the dry run, the operator runs the qualified policy explicitly:

```bash
./so101 deploy output/my_typing_robot NVIDIA --execute \
  --port /dev/serial/by-id/YOUR_CONTROLLER \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd
```

Follow the [hardware guide's operator checks and stop procedure](HARDWARE_PREPARATION.md#operator-run-deployment).

Recalibration, moving the fixture or changing the jaw requires requalification.
Software cannot detect a physically moved keyboard or replaced tip from file
hashes; keep the fixture marked and fixed. Q/G/P alignment does not establish
full-keyboard accuracy, placement tolerance or successful six-letter policy typing.

## Optional supervised experiments

These are controlled tests of an unqualified mapping, outside the standard
qualification and deployment sequence above.

To inspect a policy's software contract using this candidate before qualification,
use `prepare-deployment --alignment-candidate PATH/TO/alignment.json` with the
checkpoint, matching environment, robot ID and a new output directory. The bundle
records the selected offsets and supports `deploy` dry runs. Standard physical
deployment requires the qualification evidence above and a new export with `--alignment`.

An operator can instead run a supervised policy experiment with this candidate:

```bash
./so101 deploy output/my-candidate-bundle NEWTON --experimental --execute \
  --port /dev/serial/by-id/YOUR_CONTROLLER \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd
```

Omit `--execute` to dry-run the same experimental contract. This explicit mode
requires a prepared direct-LeRobot or candidate bundle with the robot calibration
and trained geometry checked. It retains operator confirmation, reset arrival,
joint limits, keyboard feedback, wrong-key stops and timeouts. Logs identify an
unqualified experiment;
the command neither changes calibration nor creates passing alignment evidence.

To compare the policy without any fitted or historical offsets, prepare the
same checkpoint with ``--encoder-convention lerobot`` and use the same explicit
``--experimental`` flag. The resulting log is labelled
``experimental_direct_lerobot``.

If a physical trial reaches a key but drags across neighboring keycaps, the
candidate launcher can require a larger **experimental** measured-FK clearance
before lateral advance:

```bash
./so101 deploy output/my-candidate-bundle NEWTON --experimental \
  --experimental-clearance-m 0.030 --execute \
  --port /dev/serial/by-id/YOUR_CONTROLLER \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd
```

This changes the release/clearance transition gate relative to training. The
checkpoint weights, observation format and joint mapping stay fixed, but phase
timing changes and can therefore change observations and subsequent actions.
It does not command a guaranteed vertical lift or guarantee clearance throughout
lateral travel. The current physical trials have not established this override
as a fix; it remains an unqualified experiment.
