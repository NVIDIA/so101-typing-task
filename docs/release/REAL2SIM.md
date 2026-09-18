# Hand-guide the robot, calibrate the scene, then train

Use real key presses to fit the keyboard pose and encoder-to-model joint offsets.
The recordings calibrate the simulation; they are **not PPO demonstrations**.
Keep the physical base, keyboard and fixed typing jaw unchanged during this process.

For this tabletop workflow, place the plastic robot base and keyboard directly
on the same flat table, with no mat or riser. Clamp the base and secure the keyboard.
Start with about 130 mm front gap, parallel base/keyboard edges and the base's
short right straight edge aligned to the bottom-right corner of right Ctrl, as shown in
the [September physical setup](../../FIXTURE_SETUP.md).
Use the documented base edges and vertices as datums to
measure the keyboard's relative position and orientation and build the starting
simulation scene. Preserve those measurements throughout calibration.

First complete [LeRobot setup/calibration](HARDWARE_PREPARATION.md#install-and-calibrate)
for your robot and [measure the starting layout](TASK_CONFIGURATION.md#measure-your-fixture), saving
`output/my-fixture/task.json`. Use the supported keyboard and fixed-jaw geometry.
Run `./so101 setup-hardware` and `./so101 build` before the steps below.

## Why recorded-motion calibration comes before training

The September workflow selected the keyboard pose and joint offsets together,
checked the recorded contacts, then trained a fresh policy in the frozen scene.
Deployment used the matching conversion on encoder reads and hardware commands.

The policy uses a fixed keyboard map without vision, so physical typing depends
on the real and simulated robot/keyboard relationship agreeing. Matching external
datums is the starting point; recorded key presses test that correspondence.

Measured placement and checked asset dimensions
appeared consistent, but replaying measured real joint trajectories still produced
different contacts in simulation. **We suspect a discrepancy in the encoder-to-model
mapping or simulated robot/contact geometry, but have not isolated its exact cause.**
Matching external datums and dimensions does not by itself validate the joint-angle
conventions, fixed-jaw pose or contact geometry.

The practical correction is to fit joint-angle offsets and an effective simulated
keyboard pose together from the recordings, then check actual simulated contacts.
Keep the physical fixture fixed and save this fitted scene separately from the
measured reference. These parameters reproduce observed contacts; they do not
prove a particular mechanical or calibration fault, establish that the physical
keyboard was misplaced, or show that the fitted scene is geometrically identical
to the real setup.

Before training for physical use, require every recorded key approach to reach
the intended first key in simulation, with a reset between keys and no simultaneous
extra key actuation. Save the passing scene, offsets and evidence, then train in that
scene. This is the pre-training criterion used here; it does not establish clean
continuous replay or successful physical policy deployment. The September fit
passed **93/93** such checks on its fitting recordings.

**Physical result:** on a second SO-101, a fresh policy trained in that
jointly fitted scene completed **NEWTON** in 7.10 seconds with matching releases
and model-FK clearance, using the fitted conversion on encoder reads and hardware goals.
This was one supervised experimental trial without completed alignment qualification;
see the [recorded evidence](calibrated_policy_result.json).

### What the fitted offsets mean

Offsets use `q_model = q_LeRobot + offset`; hardware goals use the inverse
subtraction. The September fit selected about +1.336° shoulder lift, -10.150° elbow
flex and +1.414° wrist flex, with zero pan and wrist-roll offsets. These values
belong to that robot and fitted scene. Fit and validate your own complete mapping;
copying the released offsets does not calibrate another robot.
Both the July and September fits selected about a -10° elbow correction, with
smaller positive shoulder-lift and wrist-flex corrections.

After fitting, regenerate the bird's-eye view from the resulting `task.json`.
The starting physical-reference drawing no longer represents the adjusted
simulation scene. See the
[physical-reference versus fitted-scene comparison](../fixtures/physical_reference_20260915.md#physical-reference-versus-fitted-training-scene).

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
  --num-envs 4096 --iterations 500 --seed 1307
```

This starts the measured September recipe; choose the budget for your GPU,
evaluate early, then follow the
[six-letter continuation and evaluation recipe](TRAINING.md). These recordings
are not fed into PPO. Simulation already uses model joint coordinates: retain the
offsets for encoder reads and inverse hardware commands, rather than adding them
again inside training. Keep the same physical fixture for deployment and perform
the separate [physical alignment checks](ALIGNMENT.md) before policy execution.
