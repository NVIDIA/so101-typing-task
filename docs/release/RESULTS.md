# Physical and simulation results

## September calibrated-scene policy

On a second SO-101, the [record/fit/train workflow](REAL2SIM.md) produced one
verified physical **NEWTON** sequence. The real fixture stayed fixed while the
simulated keyboard pose and joint mapping were fitted together; a fresh policy
was trained in that scene and deployed with the same mapping.

| Evidence | Result |
| --- | --- |
| Scene calibration | 93/93 correct first contacts with resets between keys; all contacts used for fitting |
| Training | 500 P1A + 2,000 Transit15 updates; `model_2498.pt`, seed 1307 |
| Pipeline wall time, including evaluation/video | About 2 h 38 min on RTX 5090, 4,096 environments, AnchorBench/MJWarp/CUDA graphs |
| Strict six-letter simulation | 2,048/2,048 over seeds 2307 and 3307; zero wrong keys |
| Physical trial | NEWTON in 7.10 s; six down/up pairs and six model-FK clearance transitions |

The physical result is **one supervised experimental trial**, not a measured
success rate, physical A–Z test or transfer test of the same checkpoint between
robots. It used 25 Hz control and model-FK clearance of 15 mm for two ticks;
standard [alignment qualification](ALIGNMENT.md) was not completed.

The 93-contact pass measures fitting performance, not held-out validation.
Continuous measured-motion replay still produced extra/mismatched contacts.
Simulated key depression reached about 1.35 times nominal travel, so strict typing
success does not establish accurate contact mechanics or independently measured
physical clearance.

### Scene and joint mapping

Both the July and September fits selected an approximately **-10-degree elbow
correction**, with smaller positive shoulder-lift and wrist-flex corrections.
The successful September trial used the **combined fitted scene, mapping and new
policy**; the individual contributions were not isolated.

We suspect an encoder-to-model mapping or simulated robot/contact-geometry
discrepancy; its exact cause remains unresolved. The fitted parameters are
setup-specific corrections, not universal joint zeros or instructions to move the
real keyboard. See the [mapping convention](REAL2SIM.md#what-the-fitted-offsets-mean)
and [physical-reference versus fitted-scene comparison](../fixtures/physical_reference_20260915.md#physical-reference-versus-fitted-training-scene)
for the values and drawings.

## Historical reference

The optional July AnchorBench `model_19000.pt` recorded **89/100 physical six-letter
trials** and 78/78 reset-per-key replay contacts on its original robot and
mat-supported fixture. Its scene and mapping are separate from September's.
See the [historical checkpoint](../fixtures/reference_policy.md) for download and
playback instructions.

## Evidence

`./so101 download` fetches the September checkpoint, matching scene and sanitized
evidence with pinned checksums. The [result summary](calibrated_policy_result.json)
records the mapping, evaluations and source-log hashes; the
[artifact revision](artifact_revision.json) identifies the published files.
Raw recordings and the operator's LeRobot calibration remain local.

The [artifact dataset](https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark)
contains evaluation reports, videos, timing, inference-parity checks and earlier
development results. For software verification commands, see
[development checks](../DEVELOPMENT.md).
