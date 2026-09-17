# Advanced scene and joint-mapping diagnostics

For the normal recording → fitting → training workflow, use the
[short real-to-sim guide](../release/REAL2SIM.md) and `./scripts/real2sim.sh`.
The lower-level commands below support controlled comparisons and investigation;
they are not additional steps required by that workflow. Run them from the
repository root, keeping each candidate's scene, mapping and evidence together.
Physical qualification follows the [alignment guide](../release/ALIGNMENT.md).

## Hand-guided scene check

Run [hardware setup and your robot's LeRobot calibration](../release/HARDWARE_PREPARATION.md#install-and-calibrate)
first, and build the simulator with `./so101 build`. Record all six encoders and
USB key events with motor torque off, then replay the measured poses in Newton.
Use explicit device and task paths; replace the device placeholders with yours:

```bash
./scripts/scene_check.sh record \
  --robot-id my_typing_robot --port /dev/serial/by-id/YOUR_CONTROLLER \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd \
  --task-config output/my-fixture/task.json \
  --seconds 300 --output-root output/my-fixture/fit-recordings

# Use the exact Recording path printed above.
./scripts/scene_check.sh replay --recording /absolute/path/to/recording.json
```

Repeat the recording command with `--output-root output/my-fixture/validation-recordings`
for a separate validation capture; do not use it to choose or tune offsets.
Recording requires explicit robot ID, serial port, keyboard device and task JSON;
there are no defaults for a particular robot or workstation. Replay can use an
explicit `--recording`, or the latest completed recording under `--output-root`.

Support the arm before recording: torque turns off after a three-second notice.
Guide the fixed typing tip across the board, keeping the moving jaw closed.
Press one key at a time near its centre, gently bottom it and hold about 0.5 s,
then fully release and lift before the next key. Cover all rows and both ends;
A–Z order is optional. The command above captures up to 300 seconds at a requested 50 Hz; Esc or
Ctrl-C saves early. Torque stays off afterward. There are no position commands
or automatic return movements. Keep motor power on so the encoders can be read.

Replay opens the selected recording in Newton using its saved scene and your own
calibration convention, without added angle offsets. It prescribes the recorded
robot joint states at each physics step; the keyboard remains simulated. Console
output compares real USB edges with simulated key edges. `scene-report.json` and
`scene-trace.jsonl` are saved next to the recording; add `--video` for headless
video. This checks geometry visually and diagnostically, not actuator dynamics.
A mismatch is evidence to investigate; encoder-derived FK alone cannot establish
the physical tip's absolute position or identify which geometry is wrong.

For a simulation-only comparison with the historical benchmark joint convention,
run `./scripts/scene_check.sh replay --recording PATH --benchmark-offsets`. It adds
`[0, 1.9336272481, -10.7735791533, 1.3647648917, 0]` degrees to the five recorded
arm angles. The source recording and keyboard pose stay unchanged; comparison
reports, traces and optional video go under `benchmark-offsets/` beside the
recording. This explicit diagnostic does not establish a universal calibration.

### Fit joint offsets and an effective keyboard pose

This is the recorded-contact real-to-sim calibration stage, before PPO. Run
`./so101 setup-hardware` to install its pinned NumPy/SciPy dependencies and
`./so101 build` for Newton replay. After collecting separate A–Z fit and validation
recordings as above:

```bash
./scripts/calibrate_scene.sh \
  --task-config output/my-fixture/task.json \
  --fit-recording /absolute/path/to/fit/recording.json \
  --validation-recording /absolute/path/to/validation/recording.json \
  --out-dir output/my-fixture/scene-calibration --replay
```

The script uses measured follower angles nearest each key-down, including repeated
presses. It fits lift/elbow/wrist-flex offsets plus keyboard XYZ/yaw, holding pan
and wrist roll at zero. Freeing pan together with scene yaw/translation would let
one rigid rotation compensate for the other. No historical offset values are
used. `--fit-tilt` additionally fits effective roll/pitch; any disagreement with
the measured support plane is reported, not treated as a new physical measurement.

Outputs are `task.json`, `mapping-report.json`, byte-identical input recordings,
Newton reports/traces and `replay-summary.json`. The source measurements and
LeRobot calibration are unchanged. The report records the fixed jaw tip, input
hashes, bounds, all samples selected and separate validation errors. Key-down is
assumed to occur at half of the 1.8 mm travel; September recordings have USB
receipt timestamps, which adds timing uncertainty. Contact centres are another
assumption, so a low fit error alone is insufficient.

To also fit shoulder pan and wrist roll, add `--fit-shoulder-pan --fit-wrist-roll`.
Pan fitting normally holds the keyboard yaw correction fixed. Add
`--free-keyboard-yaw` to optimize both together: the weak offset prior selects
among equivalent coordinate choices, so neither fitted value is a unique
physical measurement. `--fit-tilt` also frees keyboard roll/pitch. These options
produce a `joint_scene` candidate rather than `fit3_pose`.

`--replay` checks the entire down/up sequence, including unintended contacts,
and exits unsuccessfully on mismatch. Passing it still requires the physical
Q/G/P checks in [alignment qualification](../release/ALIGNMENT.md). These
recordings calibrate the simulation; they are not policy-training data.

### Refine using actual simulated contacts

The original workflow adjusted the effective scene using Newton contacts after
the numerical fit. This offline command reproduces that step with all five arm
offsets and the keyboard's six pose parameters adjustable:

```bash
./scripts/refine_scene_contacts.sh \
  --seed-report output/my-fixture/scene-calibration/mapping-report.json \
  --recording /absolute/path/to/fit/recording.json \
  --recording /absolute/path/to/second-fit/recording.json \
  --out-dir output/my-fixture/contact-refinement
```

The default runs reset-per-press, first-contact probes. Add `--continuous` to
search using the complete recorded down/up sequences, including extra keys and
transit contacts. Both modes optimize on **every supplied recording**; those
recordings no longer count as reserved validation. Multiple `--seed-report`
arguments compare fresh fit candidates. Outputs include each trial's score,
the selected effective `task.json`, and `mapping-report.json` with candidate
`joint_scene`. Neither mode installs calibration or declares physical readiness.
Run the ordinary `scene_check.sh replay` with these two output files and
`--mapping-candidate joint_scene` to independently check the resulting complete
trajectory. Reserve a new recording for subsequent qualification.

### Fit four joint offsets with the keyboard frozen

This alternative answers a narrower question: can offsets alone align the
measured scene? It is retained for comparison with the joint-and-scene fitter.

The existing diagnostic tool fits one recording and scores separate recordings.
Use the `fit4` candidate (pan/lift/elbow/wrist flex); it also reports zero,
historical and other diagnostic families, which are not automatically selected.
The `uv run` command supplies the offline NumPy/SciPy dependencies in an isolated
environment, without changing the hardware environment.

```bash
uv run --no-project --python .venv-hardware/bin/python \
  --with numpy==2.2.6 --with scipy==1.17.0 \
  python -m scripts.so101_homing.audit_scene_mapping \
  --task-config output/my-fixture/task.json \
  --fit-recording /absolute/path/to/fit/recording.json \
  --validation-recording /absolute/path/to/validation/recording.json \
  --out-dir output/my-fixture/mapping-audit

./scripts/scene_check.sh replay \
  --recording /absolute/path/to/validation/recording.json \
  --replay-task-config output/my-fixture/task.json \
  --mapping-report output/my-fixture/mapping-audit/mapping-report.json \
  --mapping-candidate fit4
```

Use a new output directory for each analysis. This fitter averages FK over each
down/up interval and assumes centre contact at full 1.8 mm travel; it does not
implement the later stable-window selection used for the September powered-data
fit. Preserve the method with each report rather than comparing coefficients
without their sampling assumptions. The fit report remains diagnostic. Use
`./so101 alignment prepare --mapping-report ... --mapping-candidate fit4` to freeze
it as an unqualified candidate. Replay reserved motion and run the physical
probes with the same `--alignment` file. Only passing evidence packaged by
`./so101 alignment qualify` can authorize physical policy deployment; see
[the full commands](../release/ALIGNMENT.md).

### Earlier placement-fitting diagnostic

Use this only while placement is still being measured, before freezing datums.
It is not part of fitting joint offsets to the frozen scene.
To test a candidate keyboard placement against the same recording, use
`./scripts/scene_check.sh replay --replay-task-config PATH_TO_TASK_JSON`.
The source recording is preserved; results and the tested task snapshot go in a
separate `placement-...-direct/` directory. A fitted pose that reproduces some
contacts can still disagree with physical placement or the other keys.

For a single complete A-Z recording, the offline diagnostic fitter is
`python -m scripts.so101_homing.fit_scene_recording --recording PATH --out-dir NEW_DIRECTORY`
(use the hardware environment's Python). It fits 18 letters and reserves
W/T/I/S/H/L/C/M for validation. Replay `task-fit18.json` to preserve that split.
`task-all26.json` is a separate full-data fit, not a held-out result. Optional
`--validation-recording PATH` scores an earlier capture with matching metadata.
The fitter adjusts **only X, Y and yaw**. The recorded scene's keyboard height
and tilt remain fixed, preserving its tabletop placement and the asset's native
key slope. It cannot lower or tilt the keyboard to absorb encoder/model errors.
Fits assume centre contacts and 1.8 mm key travel, use zero joint offsets, and
report XY and remaining Z errors without automatically promoting the scene.
For an explicit historical-offset comparison, add `--benchmark-offsets` to both
the fitter and the replay command. Those offsets stay fixed; only X/Y/yaw are
fitted. The fit report records the convention, while the task JSON contains
keyboard placement only. Keep the resulting fit in a separate output directory.

## Shared F contact probe

Before training, generate a joint trajectory for your scene, verify that Newton
physically depresses and releases F, then replay the **same saved commands** on
your robot. The offline planner uses the fixed jaw's FK/IK; neither replay runs
a policy or recomputes IK. Run `./so101 setup-hardware` and `./so101 build` first.

```bash
.venv-hardware/bin/python -m scripts.so101_homing.run_fk_probe \
  --generate --task-config output/my-fixture/task.json \
  --trajectory output/my-fixture/fk-F/trajectory.json

.venv-hardware/bin/python -m scripts.so101_homing.run_fk_probe \
  --trajectory output/my-fixture/fk-F/trajectory.json --sim

.venv-hardware/bin/python -m scripts.so101_homing.run_fk_probe \
  --trajectory output/my-fixture/fk-F/trajectory.json \
  --robot-id my_typing_robot --port /dev/serial/by-id/YOUR_ROBOT \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd --execute
```

Use `--video` instead of `--sim` for a headless video. Let the simulation finish;
hardware preflight requires its passing report and matching trajectory hash.
Omit `--execute` for preflight without a motor connection. The default probe uses your
robot's own LeRobot calibration with **no additional joint angle offsets**.
`--benchmark-offsets` explicitly selects the historical inverse conversion for a
supervised comparison; it does not qualify that mapping for your robot. For a
new fit, use `--alignment output/my-fixture/alignment/alignment.json` so the
probe records its exact identity, closed-jaw measurements and sustained release
clearance. Those logs can enter the qualification workflow.

The robot first moves to the trajectory's shared start pose, then replays at
25 Hz: approach, hover 15 mm above F's released rim, descend 2 mm below that rim,
lift and return. Hardware execution is enabled by `--execute`, records measured joints
and USB key events, and requires F down followed by F up. The startup pose allows
up to 4 degrees of measured servo tracking error; larger error aborts before the
F trajectory. Esc/Ctrl-C stops
commands with torque left on; support the arm before switching off motor power.
Clear the sweep before starting. Do not use another robot's calibration file.

Reports, video and traces live beside the trajectory; physical logs are under
`hardware/`. One physical F press checks the local correspondence. It does not
verify the entire keyboard placement or qualify a learned policy. A computed FK
match within the simulator alone does not establish physical alignment.
