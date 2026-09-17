# Fresh-user reproduction record

Follow the repository README in a fresh clone. Record the Git commit, dataset
revision, OS, GPU, driver, Docker version, image ID and robot model/ID. Preserve
the generated request, exit, evaluation and deployment logs with the result.

## Software acceptance

1. Build with `./so101 build` using only public dependencies.
2. For the optional reference checkpoint, run `./so101 download`; every file must
   pass its pinned checksum. Training from scratch does not require this download.
3. Run `./so101 probe` with your task JSON, or the optional reference-policy video command. Inspect the
   recording and strict evaluation report, including failure reasons.
4. For physical typing on your own setup, first install the hardware environment,
   calibrate your robot with LeRobot, measure/freeze the fixture, and follow
   [REAL2SIM.md](REAL2SIM.md) to record and fit the scene. Then train with that
   fitted task using [TRAINING.md](TRAINING.md), evaluate both seeds, and record
   initialization wall time separately from training-loop time. Simulation-only
   smoke checks may use an example task without hardware.
5. Complete [alignment qualification](ALIGNMENT.md) for physical use and export
   with that evidence, the same robot ID/calibration, and the evaluated checkpoint's
   environment. Dry-run without `--execute`. For the optional 19k policy, qualify
   its frozen historical scene; changing it requires training a new policy.
6. Compare the exported CPU actor to a recorded simulation trace using the
   parity command in [HARDWARE_PREPARATION.md](HARDWARE_PREPARATION.md).

For a quick installation check, use one training update before spending the
learning budget. A random startup checkpoint is not expected to type successfully.
The learned policy can also be tested with the placement configurations described
in [TASK_CONFIGURATION.md](TASK_CONFIGURATION.md).

## Operator and physical acceptance

The operator performs standard LeRobot follower calibration before recording,
checks the fixed jaw, freezes the fixture pose, then validates reserved replay
and physical Q/G/P contacts before standard policy execution. Preserve that
calibration throughout the workflow; do not repeat it merely because training ended.
Follow the hardware guide for the explicit execution command and stop procedure.
Record attempted sequences and every wrong key, stop or failed reset, not only
successful clips. Keep new-robot results separate from the historical benchmark.

## Result template

| Item | Recipient evidence |
| --- | --- |
| Machine, GPU/driver and robot identity | Pending independent recipient |
| Git commit, dataset revision, image ID | Pending independent recipient |
| Build/download/probe and sample playback | Pending independent recipient |
| Training budget, wall time and loop time | Pending independent recipient |
| Seeded strict evaluation and video review | Pending independent recipient |
| Export, dry run and CPU parity | Pending independent recipient |
| Calibration and measured alignment | Pending operator |
| Physical attempts, successes and failures | Pending operator |

A local clean-checkout test is useful software evidence but does not fill this
independent-recipient record or establish physical success on another SO-101.
The [September 17 release audit](STATUS.md#calibrated-policy-release-verification--2026-09-17-pdt)
records the latest clean-source installation, download, simulation and export
checks separately from that pending independent reproduction.
