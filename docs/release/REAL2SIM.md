# Hand-guide the robot, calibrate the scene, then train

Use real key presses to fit the keyboard pose and encoder-to-model joint offsets.
The recordings calibrate the simulation; they are **not PPO demonstrations**.
Keep the physical base, keyboard and fixed typing jaw unchanged during this process.

First complete [LeRobot setup/calibration](HARDWARE_PREPARATION.md#install-and-calibrate)
for your robot and [measure the starting layout](../../FIXTURE_SETUP.md), saving
`output/my-fixture/task.json`. Use the supported keyboard and fixed-jaw geometry.
Run `./so101 setup-hardware` and `./so101 build` before the steps below.

## 1. Collect two or three hand-guided recordings

Support the arm before running this command: it disables torque. Motor power
stays on so the encoders can be read. Move the follower **with your hand**, with
the moving jaw closed; no leader arm is needed.

```bash
./scripts/real2sim.sh record \
  --robot-id my_typing_robot --port /dev/serial/by-id/YOUR_CONTROLLER \
  --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd \
  --task-config output/my-fixture/task.json \
  --seconds 300 --output-root output/my-fixture/recordings
```

Press each A–Z key near its centre, gently bottom it, hold about half a second,
then release and lift clear before moving to the next key. Pause briefly between
presses. Cover all letters in the first recording; repeated presses are welcome.
Esc or Ctrl-C saves early. **Torque stays off afterward; support the arm.**
Run the same command again for the next capture. Each gets its own timestamped
directory. The recorder saves measured joint angles and physical key events.

## 2. Fit the scene and check each recorded key approach

```bash
./scripts/real2sim.sh fit \
  --recordings output/my-fixture/recordings \
  --out-dir output/my-scene
```

This fits fresh offsets and keyboard pose, then refines them using actual Newton
contacts. All five arm offsets and keyboard XYZ/roll/pitch/yaw are available.
The check resets between keys and requires the correct first contact, matching
the original scene-calibration approach. Simultaneous extra contacts fail.
Continuous replay is an optional diagnostic, not part of this per-key criterion.

The command reports `PASSED: N/N` or fails with the remaining mismatches. A failed
fit is not a calibrated scene; inspect its evidence before training. Keep the
whole output directory:

| Output | Purpose |
| --- | --- |
| `task.json` | Fitted keyboard scene to use for training |
| `mapping-report.json` | Robot-specific joint offsets to keep for deployment |
| `scene-calibration.json` | Per-key result, input hashes and artifact hashes |
| `refined/` | Original recording copies, search history and contact evidence |

Use a new output directory for each fit. Every supplied recording is fitting
data; separate future captures can test whether the mapping generalizes. Collect
those in a different directory and reserve them with
`./so101 alignment prepare --validation-recording ...` as shown in the alignment
guide; keep the final fit unchanged while evaluating them.
The source recordings, measured reference and LeRobot calibration file are preserved.

## 3. Train in that saved scene

After the per-key check passes, use its `task.json` rather than the starting
measurement JSON. For example:

```bash
./so101 train --actuator anchorbench --stage p1a \
  --task-config output/my-scene/task.json \
  --num-envs 4096 --iterations 4000 --seed 1307
```

Choose the budget for your GPU, evaluate early, then follow the
[six-letter continuation and evaluation recipe](TRAINING.md). These recordings
are not fed into PPO. Simulation already uses model joint coordinates: retain the
offsets for encoder reads and inverse hardware commands, rather than adding them
again inside training. Keep the same physical fixture for deployment and perform
the separate [physical alignment checks](ALIGNMENT.md) before policy execution.

The [September 16 example](../experiments/scene_calibration_20260917.md) passed
93/93 per-key contacts. Its selected shoulder-pan and wrist-roll offsets were
both zero despite being adjustable. That is a result for that fit, not a rule
that those joints never need correction.
