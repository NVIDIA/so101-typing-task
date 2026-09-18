# Validation status

The software workflow is implemented: record real key presses, fit the simulation
scene and joint mapping, train/evaluate, export, and deploy. The September policy
completed one physical **NEWTON** sequence. Repeated reliability, full alignment
qualification and independent-user reproduction remain pending.

## Demonstrated results

| Evidence | Result and scope |
| --- | --- |
| September scene calibration | 93/93 correct first contacts with resets between keys; all supplied recordings used for fitting |
| September training | 500 P1A + 2,000 Transit15 updates with AnchorBench/MJWarp; about 2 h 38 min including evaluation/video on RTX 5090 |
| September six-letter simulation | 2,048/2,048 strict successes over two evaluation seeds; zero wrong keys |
| September physical typing | One exact NEWTON sequence with matching releases and clearance; supervised experimental mode |
| Historical benchmark | 78/78 per-key replay contacts and 89/100 physical six-letter policy trials on the original setup |

The September and historical policies use different fitted scenes and mappings.
See [results](RESULTS.md) and the [reference-policy contract](../fixtures/reference_policy.md).
Neither result establishes universal offsets or guarantees another robot's success.

## Software verification

Clean-source checks pass host tests, source/privacy checks, Python compilation,
documentation links and offline command checks. The runtime builds from pinned
public dependencies. Containers use the host UID/GID; source archives work without
Git metadata; calibration and checkpoint/scene mismatches produce preflight errors.
See [development checks](../DEVELOPMENT.md) for the repeatable commands.

The released September bundle was downloaded anonymously and passed checksums,
64/64 strict simulation episodes and a NEWTON simulation video. CPU inference
matched eight saved simulator actions within the declared `1e-5` tolerance.
A fresh hardware Python environment passed candidate preparation, export and
an experimental deployment dry run using the original calibration and mapping.
The [artifact revision](artifact_revision.json) pins the published files and evidence.

These checks reused the development workstation and caches. They establish local
software execution, not a cold installation, another GPU's compatibility or
independent physical reproduction. Historical run identities remain in
[validation_runs.json](validation_runs.json).

## Physical deployment follow-up

| Remaining work | Required evidence |
| --- | --- |
| Qualify the selected scene/mapping | A fresh recording excluded from fitting, passing replay and physical Q/G/P press/release/clearance checks |
| Measure physical reliability | More sequences and repeated trials, retaining every failure as well as successes |
| Reproduce independently | Another user/machine/SO-101 completes the [documented workflow](REPRODUCTION.md) |
| Measure robustness | Placement and physics variation tested for the selected policy; physical tolerance remains unmeasured |

The selected fit passes isolated first-contact checks, but continuous measured
replay still has extra/mismatched contacts. Its successful physical policy trial
used an unqualified candidate; it did not complete the standard alignment gate.
Follow [alignment qualification](ALIGNMENT.md) for standard physical deployment,
or its explicitly labelled supervised experimental route.

Joint offsets and the effective keyboard pose compensate for the observed
mismatch; the underlying cause remains unresolved. Simulation key travel also
exceeds nominal travel in some rollouts, so strict typing success does not prove
accurate contact mechanics.

Keep the robot calibration, fixed jaw and physical fixture unchanged after
validation. Recalibration, a fixture move or jaw replacement requires renewed
alignment checks; file hashes cannot detect physical changes.
