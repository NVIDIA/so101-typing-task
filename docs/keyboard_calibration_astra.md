# Automatic keyboard placement calibration with GPT-6 Astra

Use GPT-6 Astra as an agent to inspect recordings, run the repository's Newton
replay tools, fit the keyboard pose, and report contact errors. This is a
repeatable agent workflow, rather than an existing one-command multimodal
calibration program. The optimization runs in code; the model coordinates the
work and uses video to investigate disagreements.

The scope here is **keyboard placement only**. Keep the SO-101 USD, joint-zero
mapping, wrist orientation, identified actuator parameters, action timing and
recorded trajectories fixed. Motor calibration and system identification are
separate steps. Fit against the same calibrated robot model used for training.

## Inputs

Provide paths to:

- The starting task JSON, preferably based on
  [manual fixture measurements](release/TASK_CONFIGURATION.md#measure-your-fixture).
- At least two recordings containing measured follower joint angles, joint/frame
  timestamps, debounced keyboard down/up events, and wrist video. A synchronized
  external camera is useful for checking the height and approach direction.
- The robot's encoder-to-model mapping and LeRobot calibration, plus the intended
  GTC Berlin dynamics configuration. These stay fixed throughout this fit.
- A separate held-out recording reserved for validation.

Collect varied keys across the keyboard, including different rows, left/right
edges, and spaces/digits/period for the phrase task. Include clear release/lift
motions. Several repetitions distinguish consistent placement errors from bounce,
servo tracking error or compliant key contacts. A recording of one narrow region
cannot constrain the whole keyboard pose reliably.

See [recording and scene calibration](release/REAL2SIM.md) for data collection.
That guide's `real2sim.sh fit` command jointly changes keyboard pose and joint
offsets. It is **not** a keyboard-only fitting command and should only be used
when that wider calibration scope has been explicitly requested.

## Agent procedure

1. Inspect the repository's current calibration and replay recipes. Check branch,
   asset/config hashes, camera mounting, units and joint-coordinate conventions.
   Detect the input recording format before choosing or adapting a replay helper.
2. Align joint samples, key edges and camera frames using their recorded clocks.
   Use the timestamp sidecars or retimed videos; raw MP4 frame index alone may
   not represent elapsed capture time. Preserve down and up edges and report
   any visual timing offset or uncertainty.
3. Replay the **measured** joint trajectories with the fixed mapping. Do not use
   a trained policy as the trajectory source for fitting: policy actions would
   change when the keyboard is moved. Preserve the recorder's interpolation and
   control timing across candidate poses.
4. Search only the keyboard rigid pose. Start with translation in millimetres
   around the measured estimate, then yaw; include roll/pitch only when tilt
   measurements and contact coverage support identifying them. Convert to
   metres and a unit `rotation_xyzw` quaternion when writing the task JSON.
   Choose and record explicit search bounds from the measurement uncertainty.
5. For each recorded approach, compare the intended physical key and its down/up
   times with simulated key travel, first contact, release and tip clearance.
   Penalize wrong or simultaneous keys, missed intended presses, and excessive
   travel. Use timing and image agreement as additional evidence. Do not change
   the robot trajectory, force a key event, or relax contact thresholds to make
   a candidate appear successful.
6. Use wrist/external frames to reject visibly inconsistent solutions. Video can
   reveal the wrong camera rotation, wrong row or vertical gap; it cannot provide
   reliable metric height without calibration or a known scale. Numerical pose
   fitting and contact replay remain the primary tests.
7. Evaluate the frozen best pose on the reserved recording, without fitting to
   it. Report correct intended contacts, wrong keys, missed presses, simultaneous
   keys and timing errors for both fitting and validation data. Check continuous
   replay as well as isolated per-key contacts. A per-key pass alone does not
   establish clean continuous typing.
8. Save a new task JSON and evidence under a dated `output/` directory. Preserve
   the starting pose and all source recordings. Show a before/after replay and
   the measured keyboard-pose delta before proposing the configuration for
   training. The physical keyboard stays in its measured location.

The existing `public_task.py` helper updates nominal keyboard placement and its
key map together. Use that path instead of moving only a visible USD mesh.
`keyboard.position_m` is the keyboard CAD origin in the repository's configured
frame; it is not a ruler measurement from the plastic-base datum. Preserve the
asset's native key layout and slope. The wrist's encoder gauge and the robot
root rotation must match the established mapping before fitting starts.

## Prompt to give GPT-6 Astra

Replace the paths and provide this task to the agent:

```text
Calibrate only the simulated keyboard placement in this repository using:
- starting task: <task.json>
- fitting recordings: <recording directories>
- held-out validation: <separate recording directory>
- frozen robot mapping: <mapping/deployment JSON>
- frozen robot asset and GTC Berlin dynamics: <paths or run environment.json>

First inspect the repo's calibration recipes and these recording formats.
Replay measured follower joint angles at the logged timestamps. Fit only the
keyboard pose to match debounced intended key contacts, releases and clearance.
Use synchronized wrist and external video to investigate mismatches.

Keep all robot geometry, joint offsets, wrist orientation, dynamics, action
limits, contact thresholds and measured trajectories unchanged. Do not run
hardware motion. Do not use a policy rollout as calibration data. The existing
joint-offset fitter must not change offsets for this keyboard-only task.

Run project commands through the documented Docker runtime. Record pose-search
bounds and units. Reserve the validation recording and never optimize on it.
Save initial/best task JSON, source/config hashes, search history, per-key and
continuous-replay metrics, and before/after videos under output/calibration/<date>.
Do not commit or push recordings, logs, videos, checkpoints or reports.
Report unresolved errors and show the resulting scene before using it to train.
```

If the recordings lack identifiable presses, sufficient key coverage, a known
joint mapping, or usable timing, ask for the missing evidence. Do not compensate
for an unresolved robot/camera-model error by inventing a keyboard pose.

## Use a validated fit for vision training

Train the state teacher and vision student with the same fitted scene. Then pass
the matching teacher and task to the simple vision trainer:

```bash
scripts/vision/train_simple.sh output/vision_ppo/my_calibrated_run \
  output/my-calibrated-state-run/model_N.pt \
  --task-config /repo/output/calibration/my-fit/task.json
```

The default target is `ISAAC LAB 3.0`. Changing scene placement after training
changes the evaluation conditions; preserve the original checkpoint settings
and label any placement experiment separately. Keep all generated calibration
artifacts local under ignored `output/`.
