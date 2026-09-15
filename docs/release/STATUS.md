# New-user release acceptance ledger

The configurable software pipeline is implemented and locally qualified through
six-letter simulation, placement evaluation, export and CPU inference parity.
See [measured results](RESULTS.md) and the [recipient record](REPRODUCTION.md).
Independent installation and second-robot hardware reproduction remain pending.
All new simulation runs use Newton MJWarp; no hardware motion was executed.

| Requirement | Implemented and measured evidence | Remaining qualification |
| --- | --- | --- |
| Public runtime | Public image digest, source ancestor and runtime patch build successfully; GPU reset/action probe passes | Independent recipient installation and GPU/driver coverage |
| Training | AnchorBench and USD save checkpoints and environment YAML; custom task/physics training and resume smoke tests pass; 64-environment, 10-update reference and DR runs pass | 501 P1A plus 50 Transit15 updates measured; wider seed/GPU coverage remains |
| Released policy playback | 1021/1024 strict evaluation episodes passed in the fresh README walkthrough; recorded NVIDIA typing passed | Additional seeds, independent GPU/machine and interactive X11 viewer |
| Physics choices | JSON solver settings and actuator parameters saved/restored with contract hashes | Sparse VBD is not supported by this interface |
| Keyboard placement and DR | Nominal XYZ/yaw transform updates mesh and map; per-reset XY/yaw and joint noise; 3-reset probe preserved actor nominal map | Fixed-policy XY/yaw tolerance measured; physical and cross-physics robustness pending |
| Staged recipe | Public P1A-to-Transit15 commands, output conventions and budget caveats documented | Short learned recipe qualified; optional 20,000-update budget not reproduced |
| Standard robot calibration | Pinned CPU-only LeRobot environment installs; official follower calibration command documented | Operator performs calibration on their own robot |
| Deployment bundle | Saved nominal geometry, q_reset, normalization, IDs and artifact hashes exported; standard and moved-keyboard/custom-physics dry runs pass | Operator-run physical/model alignment and typing validation |
| Reference fixture | Published Logitech MX keyboard, fixed typing jaw, closed moving jaw and 4.44 mm mat documented | Users measure their actual layout; benchmark reproduction requires matching its geometry |
| Repository | [NVIDIA-dev/so101-keyboard-typing](https://github.com/NVIDIA-dev/so101-keyboard-typing) contains both documented workflows and the pinned runtime | Code publication does not establish physical qualification |

Acceptance: a recipient can install, try a released checkpoint, train with the
supported physics/task configuration, evaluate, calibrate their own robot using
LeRobot, and deploy using the saved nominal geometry. Software tests alone do not
establish successful typing on another physical robot.

The benchmark fixture is the default reference, not a restriction on training.
Changed nominal placement and bounded pose randomization are supported. See
[TASK_CONFIGURATION.md](TASK_CONFIGURATION.md) and
[HARDWARE_PREPARATION.md](HARDWARE_PREPARATION.md).

## Evidence and limits

`validation_runs.json` records successful local run IDs, image identities and
selected results. Host checks passed: 105 tests, 2 skipped, and 7 subtests (2026-09-15). They cover
CLI/configuration validation, geometry
transforms, deployment identity/hash checks, and the existing homing contracts.
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

## New policy evidence

The new six-letter checkpoint passes 2027/2048 strict episodes, with 3 wrong-key
and 18 scrape failures. NVIDIA video passes; CPU inference matches 261 recorded
actions within 1.32e-6. Training is stopped. Metadata cleanup preserves historical
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
specified plastic-base datum produce the existing XYZ/yaw task JSON, a drawing
and a residual/provenance report. Eight fixture tests cover reference recovery,
parity with the training transform, units/axis/height inconsistency, input and
output protection, source geometry, drift rejection and operation without Docker.
The 1 mm fit threshold is a measurement consistency limit, not measured policy
robustness. Physical identification of these datums and second-robot contact
validation remain pending. No calibration or robot motion was performed.
