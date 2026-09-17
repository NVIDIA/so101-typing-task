# New-user release acceptance ledger

The configurable software pipeline is implemented and locally qualified through
six-letter simulation, placement evaluation, export and CPU inference parity.
See [measured results](RESULTS.md) and the [recipient record](REPRODUCTION.md).
Independent installation and complete second-robot typing remain pending.
All new simulation runs use Newton MJWarp. Operator-run hardware checks on
2026-09-15 restored reaching/contact after applying the historical joint mapping;
they did not complete the requested six-letter sequence.

| Requirement | Implemented and measured evidence | Remaining qualification |
| --- | --- | --- |
| Public runtime | Public image digest, source ancestor and runtime patch build successfully; GPU reset/action probe passes | Independent recipient installation and GPU/driver coverage |
| Training | AnchorBench and USD save checkpoints and environment YAML; custom task/physics training and resume smoke tests pass; 64-environment, 10-update reference and DR runs pass | 501 P1A plus 50 Transit15 updates measured; wider seed/GPU coverage remains |
| Released policy playback | 1021/1024 strict evaluation episodes passed in the fresh README walkthrough; recorded NVIDIA typing passed | Additional seeds, independent GPU/machine and interactive X11 viewer |
| Physics choices | JSON solver settings and actuator parameters saved/restored with contract hashes | Sparse VBD is not supported by this interface |
| Keyboard placement and DR | Nominal XYZ/full orientation updates mesh and map; per-reset XY/yaw and joint noise; 3-reset probe preserved actor nominal map | Fixed-policy XY/yaw tolerance measured; physical and cross-physics robustness pending |
| Staged recipe | Public P1A-to-Transit15 commands, output conventions and budget caveats documented | Short learned recipe qualified; optional 20,000-update budget not reproduced |
| Standard robot calibration | Pinned CPU-only LeRobot environment installs; official follower calibration command documented | Operator performs calibration on their own robot |
| Deployment bundle | Qualified scene/mapping and copied replay/probe evidence required for physical execution; calibration and trained geometry rechecked; legacy software dry runs preserved | Complete physical/model alignment and typing validation |
| Reference fixture | Model projections, Logitech MX keyboard, fixed typing jaw, closed moving jaw and 4.44 mm mat documented | Measured physical layout and contact correspondence; model dimensions alone are not ruler instructions |
| Repository | [NVIDIA-dev/so101-keyboard-typing](https://github.com/NVIDIA-dev/so101-keyboard-typing) contains both documented workflows and the pinned runtime | Code publication does not establish physical qualification |

Acceptance: a recipient can install, try a released checkpoint, train with the
supported physics/task configuration, evaluate, calibrate their own robot using
LeRobot, and deploy using the saved nominal geometry. Software tests alone do not
establish successful typing on another physical robot.

The benchmark fixture is the default reference, not a restriction on training.
Changed nominal placement and bounded pose randomization are supported. See
[TASK_CONFIGURATION.md](TASK_CONFIGURATION.md) and
[HARDWARE_PREPARATION.md](HARDWARE_PREPARATION.md).

## Physical deployment follow-up

On 2026-09-15, the operator reported that restoring the historical offsets with
the same 19k policy resolved the new arm's reach/contact difficulty. Two recorded
attempts completed N/V/I press, release and clearance, then pressed F while
targeting D. The logs end in `wrong_key`, not success. Placement, remaining joint
alignment, dynamics and contact differences have not been isolated.
Local evidence: `output/so101_5B79015117_benchmark/runs/20260915T182653.769109Z_hardware.jsonl`
and `20260915T190721.809010Z_hardware.jsonl` in the same directory. These used the
original mat-trained `model_19000.pt`; the local no-mat `model_19996.pt` is separate.

| Remaining release work | Completion evidence |
| --- | --- |
| Run the implemented alignment workflow on hardware | Candidate preparation, reserved replay, Q/G/P qualification and deployment checks are implemented; current physical evidence has not passed |
| Verify real/model alignment and fixture instructions | Measured physical poses and contacts across the keyboard, with the robot calibration and joint mapping recorded |
| Complete the new-robot typing test | Exact press/release/clearance sequences and all attempted results retained |
| Repeat the documented workflow independently | Another user/machine follows README and records installation, training/export and physical results |

The software cleanup makes mapping selection explicit and reproducible. It does
not complete these physical checks or establish a universal offset correction.

The [alignment workflow](ALIGNMENT.md) now freezes a candidate, checks reserved
measured-motion replay, verifies physical Q/G/P logs and packages hashed evidence.
Export compares the qualified scene with the trained key map, keyboard pose and
reset. Standard physical policy execution requires the qualification and rechecks the
current calibration/model and copied evidence before motor access. Software dry
runs remain available for older bundles. The same candidate offsets are used in
simulation replay, physical probe reads/writes, policy reset and deployment.

The current robot's fitted mapping still fails reserved contact replay, so it
remains unqualified. The automatic calibration experiment is paused. Passing
synthetic software tests does not create physical evidence. The remaining release
gate is the new SO-101 passing the documented alignment procedure and repeated
policy typing, followed by an independent recipient walkthrough. A physical
fixture move or jaw replacement still requires operator detection and renewed
qualification; file hashes cannot sense those changes.

## Evidence and limits

`validation_runs.json` records successful local run IDs, image identities and
selected results. Host checks passed: 251 tests, 2 skipped, and 13 subtests (2026-09-17). They cover
CLI/configuration validation, geometry
transforms, deployment identity/hash checks, mapping selection, calibration-file
drift, measured-fit export, and the existing homing contracts. The released 19k
pair was exported with automatic mapping and passed a CPU deployment dry run;
the local no-mat pair and an existing public-training checkpoint passed with the
explicitly selected historical mapping.
These checks did not connect to hardware.
The Newton container contract suite passes 37 tests.
Randomly initialized smoke checkpoints are not expected to type successfully;
their evaluation checks configuration restoration and execution only.

These tests use an isolated public-derived container on the existing RTX 5090
workstation. A 4096-environment, 10-update capacity test completed in 36.72 seconds of
training time after initialization; one GPU sample showed 17,560 MiB total usage.
This establishes capacity for startup, not a full training result or peak memory.
The hardware dependency environment was installed separately with
CPU-only PyTorch. Bundle dry runs do not open serial devices. The standard LeRobot
encoder convention is implemented; the additional historical benchmark joint
corrections are not assumed to apply to other robots.

## Archived software-validation policy — 2026-09-11

The new six-letter checkpoint passes 2027/2048 strict episodes, with 3 wrong-key
and 18 scrape failures. NVIDIA video passes; CPU inference matches 261 recorded
actions within 1.32e-6. That short validation run was stopped. Metadata cleanup preserves historical
weights and numerical results, with relative checksums and pinned downloads.
Independent installation and physical typing remain pending.
Contact qualification is also incomplete: peak simulated key travel is 1.579
times the nominal range for the new six-letter policy. Strict success preserves
the existing event contract and does not reject that overtravel.

A fresh Git clone and new Python environments passed all 10 software validation
steps: public download/audit, cached image build, GPU probe, one-update training,
64/64 released-policy episodes, hardware setup, export, dry run and CPU parity.
Docker/package caches were reused on the same workstation; this does not count
as independent-machine or physical reproduction.

## Onboarding fixes

New container runs use the host UID/GID and a writable user cache. A one-update
training run passed; generated files were verified host-owned, editable and
deletable without sudo. The asset in the image is readable by non-root users.
A source archive without Git metadata passed 4/4 evaluation episodes and records
content hashes instead of a Git revision. Missing-image errors create no output
directory and explain how to build the runtime.

Export and deployment dry runs now require the local LeRobot calibration file.
Tests verify missing-file rejection before export and after removal from an
already-exported bundle; a synthetic offline fixture tests the successful path.
No robot was connected. Older recorded dry runs predate this calibration gate.
Sparse VBD payloads are excluded from supported downloads. Checksum lists use
mode 0644. The README lists uv, disk space and estimated install/training time.

## Pre-release review — 2026-09-11

Both documented software workflows passed a fresh local execution review:
custom task/physics probe, one-update training, six-letter resume, evaluation,
USD training, and reference-checkpoint playback. The established AnchorBench
checkpoint passed 64/64 strict episodes and a recorded NVIDIA sequence. Its CPU
actor matched 227 simulation actions within 1.20e-6. The startup checkpoint's
0/4 successes are retained in the ledger; the tiny training budget only checks
execution and configuration restoration.

All 40 supported download payloads passed pinned checksums with mode 0644.
Source/artifact hygiene checks passed, including 243 reachable historical blobs
before this review commit. Local Markdown links resolved. A source archive with
no Git metadata passed its hygiene check and 4/4 reference-policy episodes.
Hardware dependencies imported successfully and passed their compatibility check.
Seven export/deployment checks used an isolated synthetic calibration, including
missing/removed calibration and invalid-target rejection; no hardware connected.

Review fixes cover source-archive supervisor/audit support, readable malformed
bundle errors, numeric calibration-field checks, and skipping export when a
six-letter quality gate fails. The robot USD now uses an immutable dataset
revision as well as its existing hash. The Hugging Face guide matches the two
public actuator choices and disables the inapplicable tabular dataset viewer.
Historical hardware-tool documentation directs new users to the public workflow.

The checks reused this workstation and caches. Independent-machine/GPU coverage,
interactive X11 playback and physical reproduction remain unqualified. Existing
contact-model limitations are unchanged; this review adds software evidence.

## Fresh README walkthrough — 2026-09-11

A new anonymous HTTPS clone of `d26f215` followed the README with an empty runtime
cache, no downloaded checkpoints and a new hardware Python environment. The
public image build reused Docker layers; hardware installation reused public
package caches on this same machine. The first probe took 122.54 seconds including
kernel compilation and passed with the README's example keyboard pose.

The exact 64-environment, 10-update startup command passed. A reduced 10-update,
64-environment six-letter continuation, 1024-episode evaluation and MP4 recording
also executed successfully and preserved the saved placement. These startup
weights achieved 0/1024 successes; this checks the software path, not learning.
The full 4000 + 16000-update example was not run.

The default download fetched and verified exactly three reference artifacts.
The established 19k checkpoint's NVIDIA recording passed; its 1024-episode
assessment passed 1021 (99.71%), with no wrong keys, two scrape failures and one
phase timeout. Both MP4s were readable H.264 files. The new CPU environment passed
imports, dependency compatibility and actor parity for the eight default trace
samples. Generated run files belonged to the host user.

The walkthrough reached the operator-calibration boundary. Export for the new
robot ID failed clearly before creating a bundle, as expected with no calibration.
No robot was connected and no synthetic calibration was supplied. Successful
export/dry-run with synthetic calibration is covered separately above; this
walkthrough does not claim actual user calibration or physical deployment.

One usability issue was corrected: completed runs now print host paths for the
saved checkpoint, matching environment, evaluation report and video, so users
can fill the next README command without interpreting Docker mount paths.

## Fixture measurement aid — 2026-09-15

Added an asset-derived bird’s-eye drawing, a side height schematic and the
host-only `so101 fixture` command. Q/P/Z rim-centre measurements relative to a
specified plastic-base datum produce an XYZ/full-orientation task JSON, a drawing
and a residual/provenance report. Measured tilt replaces the old scene rotation
without modifying the keyboard asset. Fixture tests cover reference recovery,
parity with the training transform, units/axis/dimension inconsistency, input and
output protection, source geometry, drift rejection and operation without Docker.
The 1 mm fit threshold is a measurement consistency limit, not measured policy
robustness. Physical identification of these datums and second-robot contact
validation remain pending. No calibration or robot motion was performed.

A tabletop candidate using an absolute orientation passed the MJWarp probe:
three resets, 50 zero-action steps and the bounded-action integration check.
Maximum simulated mesh-to-target-map error was 0.191 mm. This checks simulator
geometry consistency, not the accuracy of photo-based physical placement.

## Source consolidation — 2026-09-16 PDT

NVIDIA-dev is the canonical private development repository. The older public
checkout remains the September 11 snapshot. The shared recorders and hardware
probe wrappers require explicit robot/device/scene inputs; workstation shortcuts
are preserved locally and excluded from source and Docker. Dated investigations
are indexed separately from setup instructions. Their raw recordings are named
as local evidence, not presented as downloadable repository links.

The release check now scans new untracked files as well as tracked source, and
checks local Markdown links against the distributed file set. Host regression
checks pass with 224 tests, 2 skips and 13 subtests. This includes rejection of
missing device identities and publication checks before staging files.
Hardware testing remains paused; no new calibration or physical result is claimed.

An extracted source snapshot with no Git metadata, hardware environment, local
shortcuts or recordings passed the same host suite, Python compilation, source
hygiene/link checks, fixture drawing and training command planning. The rebuilt
public image passed a one-environment AnchorBench/Newton MJWarp reset/action
probe on the frozen September scene, including 50 zero-action simulation steps.
This is execution evidence, not new policy/contact or physical qualification.
Logs and the pre-cleanup backup are local artifacts under
`output/repo-consolidation/20260917T030340Z/`. Docker layers and host dependencies
were reused; independent installation remains pending.

### Alignment integration — 2026-09-16 22:00 PDT

`./so101 alignment prepare` now freezes a candidate for measured-motion replay
and supervised physical probes. `alignment qualify` verifies reserved replay
and Q/G/P press/release/clearance evidence, then copies it into a portable
qualification. Export and physical policy execution revalidate that evidence,
the current calibration/model and the trained nominal geometry. Older bundles
retain software dry runs but require re-export with qualification for motion.

The checkout and an extracted source copy in a fresh development environment
each passed **246 tests, 2 skips and 13 subtests**. Tests exercise nonzero forward
and inverse mapping, evidence packaging, and rejection of calibration/scene
drift, incomplete or changed logs, wrong contacts and unqualified execution.
The clean-environment run exposed a missing SciPy test dependency; the documented
development install and CI now pin it. Source hygiene, links and compilation pass.
The released 19k checkpoint also exported and passed the CPU software dry run
with its historical mapping; that unqualified bundle cannot execute physically.

The rebuilt AnchorBench/Newton MJWarp runtime passed the frozen-scene Q/G/P
command probes, each with exactly one simulated down/up. Peak contact forces
were 2.41/1.92/1.88 N; maximum FK/model disagreement was below 0.003 mm. These
are simulation results. The candidate-based direct replay also ran successfully
as software but failed contact correspondence: 66 physical edges versus 180
simulated edges. The existing four-joint fitted replay was rejected by the new
check because of its extra non-letter contact. No physical qualification was
created, and no hardware was accessed. Calibration and fixture hashes are
unchanged. Local reports are under
`output/alignment-integration/20260917T045600Z/`.

### Candidate export — 2026-09-17

`prepare-deployment --alignment-candidate alignment.json` now supports software
inspection of a fitted mapping before qualification. Its bundle pins the
candidate, checks the trained scene and calibration, and remains blocked from
physical policy execution. The normal qualified deployment path is unchanged.
The current four-joint candidate was prepared with the completed frozen-scene
`model_19998.pt`; its CPU dry run passed with the fitted coefficients. Host
regression checks pass with 247 tests, 2 skips and 13 subtests. No physical motion
or alignment qualification is claimed.

### Supervised policy experiments — 2026-09-17

An explicit `deploy --experimental --execute` now permits an operator-supervised
trial with an unqualified candidate bundle. It checks calibration, model, mapping
and checkpoint/environment identity and retains the existing motion and keyboard
checks. The run is logged as `experimental_unqualified_mapping`; release
qualification and its evidence are unchanged. The requested model_19998/fit4
NEWTON dry run passed. Host checks pass with 251 tests, 2 skips and 13 subtests.
No physical motion was performed during preparation.
