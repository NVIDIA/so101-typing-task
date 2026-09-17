# First physical NEWTON success on the September fitted scene

On **September 17, 2026 at 14:36:50 PDT / 21:36:50 UTC**, the new development
SO-101 completed **NEWTON** using the learned typing policy. The hardware log
contains six correct key-down/up pairs, followed by the required clearance after
each letter, and ends with `exact_match: true` and `stop_reason: target_complete`.
Active policy time was **7.099 seconds**, over 177 control ticks.

This is one verified physical sequence, not a measured success rate. The
[machine-readable result](physical_typing_success_20260917.json) records the
checkpoint, scene, mapping, evaluation and original-log hashes without workstation
paths or device identifiers. `./so101 download --calibrated` downloads the exact
weights, environment, fitted scene, anonymized mapping report, evaluations and
sanitized hardware log. Original recordings and robot calibration remain local.
The historical 19k download remains separate. See the
[playback command](../../README.md#september-calibrated-scene-example).

## What made the workflow work

The practical solution was to **fit the joint conversion and effective simulated
keyboard pose together from recorded real contacts, verify those contacts in
simulation, train a fresh policy in that fitted scene, then deploy with the same
mapping applied in both directions**. The physical fixture stayed fixed.

The earlier offsets-only deployment on the measured-reference scene could reach
keys but dragged or hit the space bar. Matching external ruler datums was useful,
but did not make measured robot motion reproduce the same contacts in simulation.
Those attempts did not establish a working complete robot/scene correspondence.

We suspect an encoder-to-model convention or simulated robot/contact geometry
discrepancy. **Its underlying cause is still unresolved.** The fitted parameters
are an effective correction that now supports this physical policy result; they
do not prove that the physical keyboard was misplaced or that joint zeros alone
were the cause. The fit did not replace the robot's LeRobot calibration.

### Relative contribution of joint mapping and keyboard placement

Interpretation recorded **September 17, 2026**. Joint mapping appears to be a
major contributor to the vertical reach problem. In the earlier
[powered-recording FK comparison](scene_calibration_20260917.md#recurring-trend-across-historical-and-september-fits),
the historical elbow offset alone reduced mean absolute Z error from
**20.01 to 7.94 mm** on 22 presses excluded from that fit. This is recorded-pose
FK evidence. Separately, the operator observed that adding offsets restored
physical key reach; those earlier policy attempts still dragged or hit the space bar.

The [keyboard correction](../fixtures/physical_reference_20260915.md#physical-reference-versus-fitted-training-scene)
was **-1.068 mm X, +12.226 mm Y, +3.278 mm Z**, with **+3.135 degrees yaw** and
**-0.500 degrees pitch**, relative to the datum-based physical-reference model.
The computed front gap along the base centreline barely changed, from
**129.172 to 129.437 mm**. The lateral shift and rotation still change key contact
locations materially; rotation gives different displacements at different keys.
The real fixture stayed frozen throughout.

The evidence therefore supports a substantial joint-mapping contribution to
height/reach, while the successful NEWTON run validates the **combined fitted
scene, joint mapping and policy trained in that scene**. These comparisons do
not quantify each component's share of the final success or establish that
offsets alone suffice. The encoder-convention versus model/contact-geometry
cause remains unresolved.

## How we got here

1. **Measure and freeze the real fixture.** Keep the base and keyboard on the same
   flat tabletop, secure them, and use base edges/vertices as placement datums.
   Preserve the [physical placement and measured reference](../fixtures/physical_reference_20260915.md#physical-setup-used-for-the-september-newton-checkpoint)
   separately from any fitted simulation transform.
2. **Record real measured motion and key events.** The selected September 16 data
   comprises 60 powered leader/follower presses and 33 torque-off hand-guided
   presses, covering A–Z. The powered recording uses measured follower angles,
   not leader goals. The fixed typing jaw was used with the moving jaw closed.
   These recordings calibrated the scene; they were not demonstrations fed into PPO.
3. **Fit fresh joint offsets and a shared keyboard pose.** Associate recorded
   key contacts with model FK, fit candidate parameters, then refine against
   actual Newton keyboard actuation. All five arm offsets and keyboard
   XYZ/roll/pitch/yaw were available during contact refinement. Pan and wrist roll
   remained zero in the selected solution. Historical July constants were not
   optimization inputs.
4. **Replay before training.** The selected `refined/` candidate passed
   **60/60 powered + 33/33 manual = 93/93** first-key contacts, resetting between
   keys and rejecting simultaneous extra actuation. Both recordings were used
   in refinement; this is fitting reproduction, not held-out validation.
   Continuous measured-motion replay still fails and remains a separate limitation.
5. **Freeze the passing scene and train in model coordinates.** Train using the
   selected task JSON, preserving its matching mapping report. Offsets are not
   added inside PPO. Use AnchorBench, Newton MJWarp and CUDA graphs.
6. **Evaluate and export the exact checkpoint/environment pair.** Review simulated
   NVIDIA and NEWTON rollouts, check the CPU actor against saved simulation
   actions, and pin the robot calibration, fitted geometry and mapping in the
   deployment bundle. The calibration file matched the live motor registers.
7. **Separate FK checks from the policy trial.** Q was physically reachable using
   the fitted inverse conversion. The operator then explicitly ran the learned
   six-letter policy in supervised experimental mode and obtained NEWTON.

This follows the purpose of the [July workflow](../fixtures/historical_offset_provenance_20260718.md#original-recorded-trajectory-workflow):
record contacts, select a joint/scene correction, check reset-per-key replay, then
train and deploy the matched model. July passed 78/78 replay contacts and later
89/100 physical policy trials. The September result uses fresh parameters and a
new policy; it does not inherit that 89% rate.

## The exact model used for the successful run

| Item | Selected value |
| --- | --- |
| Checkpoint | `model_2498.pt` |
| SHA-256 | `6af850866096ec6b6ad6b202c8ac9417681efa0ebe71f4e8e8fdcf57d393f225` |
| Effective keyboard origin XYZ | `[259.567686, 26.287386, 10.914638]` mm in the robot-base frame |
| Effective keyboard RPY | `[0, -0.5, -86.921376]` degrees |
| Joint offsets: pan, lift, elbow, wrist flex, wrist roll | `[0, +1.3357291219509193, -10.150392639441748, +1.4143882688448561, 0]` degrees |
| Training | 500 fresh P1A updates, then 2,000 additional Transit15 updates |
| Runtime | AnchorBench, MJWarp, CUDA graphs, 4,096 environments, seed 1307, RTX 5090 |
| Deployment | Native deterministic Beta mean, 25 Hz, unchanged joint-rate limits |
| Release/clearance | Matching key-up, then two consecutive model-FK clearance ticks at 15 mm |

The [selected task example](../../configs/tasks/replay-calibrated-20260917.json)
and [scene-fit record](scene_calibration_20260917.md) retain the transform and its
provenance. Asset-origin coordinates are not ruler distances to keyboard edges.
The fitted scene is separate from the measured physical-reference drawing.

The conversion is applied only at the hardware boundary:

```text
measured q_model = q_LeRobot + fitted_offset
hardware q_LeRobot_goal = q_model_goal - fitted_offset
```

It affects the measured policy observation, model FK/clearance and commanded
joint angles. The approximately -10-degree elbow correction repeats the broad
historical trend, but these values remain specific to the recorded robot,
calibration, jaw and fixture. They are not universal SO-101 constants.

## Training and simulation evidence

Training and evaluation ran **10:45:44–13:24:04 PDT**, totaling **2 h 38 min 21 s**
including container startup, evaluations and videos. Measured training-stage wall
times were **32 min 7 s** for P1A and **2 h 4 min 2 s** for Transit15.
These are workstation timings, not a guarantee on another GPU.

| Check | Result |
| --- | --- |
| P1A strict evaluation | 1,024/1,024 |
| Six-letter strict evaluation, seed 2307 | 1,024/1,024; zero wrong keys |
| Six-letter strict evaluation, seed 3307 | 1,024/1,024; zero wrong keys |
| Six-letter scrape/phase/contact failure terminations | Zero in both evaluation batches |
| CPU actor parity | Eight saved actions; maximum absolute error `3.5763e-7`, tolerance `1e-5` |

There were 2,500 total updates. The filename 2498 reflects the saved iteration
index and resumed boundary, not a 19k run. Parallel sampling used 4,096 environments
and 64 rollout steps per update. Simulation key depression reached approximately
1.35 times nominal travel in the batches, so contact mechanics remain approximate.

Another 2,500-update continuation started at **13:55:15 PDT** on the frozen source
and image. It is separate: the successful physical bundle remained pinned to
`model_2498.pt`. Additional training did not produce this already-prepared result.

## What happened with the slow Q probe

The `--probe Q` command was a **policy-free FK trajectory**, not learned typing.
Both logged slow attempts reached Q but stopped on `down → up → down`; neither
completed the strict probe. In the second, Q was held for 4.034 seconds, then
released and re-pressed 23.09 ms later during lift. The operator reported that
the tip looked steady. Ordinary held-key repeat events were already ignored.

The separate `--quick-probe` comparison shortened descent from 3.84 to 0.96 s,
dwell from 0.40 to 0.12 s, and lift from 4.04 to 1.04 s. It retained the same
depth, mapping and path waypoints, stayed inside joint-rate limits, and passed
Q/G/P simulation. No completed physical Q/G/P qualification set is archived.
These FK timing changes were not applied to the learned policy.

The successful policy instead held individual keys for approximately **30–83 ms**,
using actual key events to switch phases. It ran with the original 15 mm clearance,
rate scale 1.0 and no clearance override. A faster FK test is a contact-timing
comparison, not proof of why the earlier extra Q event occurred.

## Reproduction and remaining scope

For a new robot, follow [record → fit → replay → train](../release/REAL2SIM.md),
then [alignment and deployment](../release/ALIGNMENT.md). Keep the scene, fitted
mapping, robot calibration and checkpoint/environment paired. A different fit
requires its own scene and validation; do not copy these offsets in isolation.

This run used the documented [supervised experimental mode](../release/ALIGNMENT.md#optional-supervised-experiments).
Its log correctly retains `alignment_qualified: false`: a fresh reserved recording
and complete physical Q/G/P qualification were not supplied. The physical NEWTON
result is real evidence, while repeated trials, broader letter coverage,
independent installation, placement robustness and cross-robot transfer remain
to be measured. Typing success still required the exact key sequence, matching
releases and all clearance transitions; formal alignment qualification was not claimed.

Local evidence is under `output/traj-calibrated-anchorbench-mjwarp-20260917/`.
`SUCCESS_20260917.md` contains the workstation command and full artifact locations;
`hardware-preparation/success-evidence.json` pins their hashes. The raw success log
is `20260917T213637.673785Z_hardware.jsonl`. Earlier failed Q logs are preserved.
The downloadable release's `manifest.json` distinguishes original hashes from
sanitized-file hashes; its `SHA256SUMS` verifies the published files. No weights,
scene coordinates, offsets or numeric event results were changed during packaging.
