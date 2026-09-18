# Physical and simulation results

## September calibrated-scene policy

The recorded-contact workflow produced one verified physical **NEWTON** sequence.
The physical fixture stayed fixed while the joint mapping and effective simulated
keyboard pose were fitted together. A fresh policy was trained in that scene and
deployed with the same mapping applied to encoder reads and hardware commands.
Follow [scene calibration](REAL2SIM.md), [training](TRAINING.md) and
[alignment/deployment](ALIGNMENT.md) to apply this workflow to your setup.

| Evidence | Result |
| --- | --- |
| Scene calibration | 93/93 correct first contacts with resets between keys; 60 powered and 33 hand-guided contacts, all used for fitting |
| Training | 500 P1A + 2,000 Transit15 updates; `model_2498.pt`, seed 1307 |
| Pipeline wall time, including evaluation/video | 2 h 38 min 21 s on RTX 5090, 4,096 environments, AnchorBench/MJWarp/CUDA graphs |
| Strict six-letter simulation | 2,048/2,048 over seeds 2307 and 3307; zero wrong keys |
| Physical trial | NEWTON in 7.10 s; six down/up pairs and six model-FK clearance transitions; exact match |
| CPU inference parity | Eight saved simulator actions matched within the declared `1e-5` tolerance |

This was one supervised experimental trial using native 25 Hz policy timing and
15 mm clearance for two ticks. It did not complete standard alignment qualification
and does not establish a success rate, physical A–Z coverage or other-robot transfer.
Continuous measured-motion replay produced extra/mismatched contacts despite the
isolated first-contact pass.

The [sanitized evidence summary](calibrated_policy_result.json) pins the scene,
offsets, checkpoint, evaluation and original-log hashes. Download the matching
weights, scene and sanitized evidence with `./so101 download --calibrated`; see
[playback](../../README.md#september-calibrated-scene-example). Raw recordings and
the operator's LeRobot calibration remain local. The historical 19k checkpoint
remains the default download.

### Scene and joint mapping

The selected offsets, in shoulder-pan / shoulder-lift / elbow-flex / wrist-flex /
wrist-roll order, are approximately **[0, +1.336, -10.150, +1.414, 0] degrees**.
Use the full-precision mapping report supplied with the bundle. Deployment adds
them on encoder reads and subtracts them from model goals; PPO uses model
coordinates and does not apply an additional conversion inside simulation.

The roughly -10-degree elbow correction recurs in the historical and September
fits. An earlier elbow-only FK comparison reduced mean absolute Z error from
20.01 to 7.94 mm on 22 powered presses excluded from that fit. This supports a
substantial joint-mapping contribution to vertical reach. Earlier offsets-only
physical attempts could reach keys but dragged or hit the space bar.

The successful result used the **combined fitted scene, mapping and newly trained
policy**. The simulation keyboard shifted -1.068 mm X, +12.226 mm Y and +3.278 mm Z,
with +3.135 degrees yaw and -0.500 degrees pitch relative to the measured-reference
model. These are model corrections, not instructions to move the real keyboard.
See the [physical-reference and fitted-scene comparison](../fixtures/physical_reference_20260915.md#physical-reference-versus-fitted-training-scene)
for drawings and measurement directions.

We suspect an encoder-to-model mapping or simulated robot/contact-geometry
discrepancy; its exact cause remains unresolved. The fit is an effective correction,
not proof of a mechanical fault, a misplaced physical keyboard or universal joint
zeros. These comparisons do not isolate each component's contribution to success.

Simulation key depression reached approximately 1.35 times nominal travel for this
policy. Strict typing success checks events, release and clearance; it does not
establish accurate contact mechanics or independently measured physical clearance.

## Other released results

Keep these checkpoints and their matched scenes separate from the September
calibrated bundle.

| Policy | Measured result | Scope |
| --- | --- | --- |
| Historical AnchorBench `model_19000.pt` | 89/100 physical six-letter trials; 78/78 reset-per-key replay contacts | Original robot, fitted scene and mat fixture; [required contract](../fixtures/reference_policy.md) |
| Earlier public P1A `model_500.pt` | 2,048/2,048 two-letter episodes; zero wrong keys | Simulation only; 501 updates, 29 min 17 s training-loop time |
| Earlier public Transit15 `model_549.pt` | 2,027/2,048 six-letter episodes; 3 wrong-key and 18 scrape failures | Simulation only; 50 further updates, 2 min 57 s training-loop time |

The earlier public training used seed 1307, 4,096 environments and AnchorBench/MJWarp
on RTX 5090. Times exclude startup/evaluation. Maximum simulated key depression was
1.579 times nominal travel for its six-letter policy. The checkpoint names are
iteration labels, not cumulative update counts.

### Placement tolerance example

For the earlier `model_549.pt` policy, keep the actor's map fixed and uniformly
perturb the actual keyboard at reset. The same 1,024-sequence bank and seed 2307
produced:

| Keyboard perturbation | Strict successes | Wrong-key episodes |
| --- | ---: | ---: |
| None | 1,013/1,024 (98.93%) | 1 |
| X/Y each ±1 mm; yaw ±0.25° | 1,011/1,024 (98.73%) | 5 |
| X/Y each ±3 mm; yaw ±1° | 862/1,024 (84.18%) | 156 |

This is one policy/seed, with zero robot-reset noise. It does not establish
physical tolerance, tolerance of the September policy or robustness to different
actuator physics. Use `--perturbation-config` as described in
[task configuration](TASK_CONFIGURATION.md) to measure your policy.

## Tested configuration and evidence

Software checks used Linux, an RTX 5090 with 32 GB and the pinned Docker runtime.
A clean source snapshot passed host tests, source/link checks and offline command
checks. The container built successfully. These checks used the development
workstation and its caches.

The released September bundle was downloaded anonymously, passed its checksums
and completed 64/64 strict simulation episodes plus a NEWTON video. A fresh
hardware Python environment passed export and an experimental deployment dry run
with the recorded calibration and mapping. The [artifact revision](artifact_revision.json)
pins the published files; [development checks](../DEVELOPMENT.md) provide the
software verification commands.

The [artifact dataset](https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark)
contains matched environments, evaluation reports, videos, timing records and
inference-parity evidence. The downloader pins a revision and checksums; its
`--all` option includes the supported older development checkpoints above.
