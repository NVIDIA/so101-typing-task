# Development and release checks

`NVIDIA-dev/so101-keyboard-typing-internal` is the canonical development repository,
with Internal visibility for NVIDIA's GitHub enterprise.

## Where things belong

Keep this repository focused on **setup, training and deployment**. Scene
calibration, evaluation, alignment checks and concise release evidence support
those workflows. Keep research diaries, one-off investigations, benchmark-session
orchestration and media production in a separate workspace or ignored `local/`.

The normal user entry points are `./so101` for simulation/deployment and
`./scripts/real2sim.sh record|fit` for measured-motion scene calibration. Other
scripts implement or verify these workflows.

Document the current procedure, result and limitation. Keep project follow-up
lists, internal status reports, clock times and session chronology in ignored
local notes or run logs. User-facing guides explain how to run the workflow and
what the published evidence demonstrates. Include
elapsed runtimes when they help users plan, and keep dates or hashes that identify
a calibration or release. Link to one detailed result instead of repeating it
throughout the setup guides.

Use the [NVIDIA SO-101 workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
as a documentation reference: concise requirements, ordered installation, model
downloads and a command index. Keep task-specific detail in the linked guides;
document this repository's tested runtime, hardware and commands.

| Location | Purpose |
| --- | --- |
| `README.md` | Install, train/evaluate, optional reference checkpoint, export/deploy |
| `so101` | Main simulation, artifact, alignment and deployment CLI |
| [FIXTURE_SETUP.md](../FIXTURE_SETUP.md) | Physical datums, measurements, drawings and a starting scene |
| [scripts/real2sim.sh](../scripts/real2sim.sh) | `record`: torque-off encoder/key-event capture; `fit`: optimize scene/offsets and check per-key Newton contacts |
| `configs/` | Task, physics and evaluation examples |
| `source/so101_typing/tasks/` | Simulator tasks, robot/keyboard assets and actuator profiles |
| `source/so101_typing/hardware/` | Recording, scene fitting, alignment, FK probes and deployment |
| `docs/release/` | Supported commands, requirements and measured results |
| `docs/fixtures/` | Geometry, drawings, physical reference and released-policy fixture contracts |
| `scripts/` | Portable tools with explicit hardware/scene inputs |
| `tests/`, `tests/hardware/` | CLI, calibration and deployment checks without hardware |
| `tests/simulation/` | Task contract checks requiring the simulator runtime |
| `docker/`, `runtime/` | Pinned simulator build, dependency locks and runtime patch |
| `output/`, `.artifacts/`, `local/` | Ignored runs, downloaded weights and workstation shortcuts; excluded from Docker |

The Python project is installed editable by `./so101 setup-hardware` and inside
the simulator image. Dependencies stay pinned in the runtime files. Use the
checkout and supported launchers; this is not a standalone wheel distribution.

Simulator modules retain the `isaaclab_tasks.core.dexsuite.config.so101` namespace
through a relative compatibility link. Saved environment configurations and
recording hashes refer to that namespace and asset location. Import tasks through
Isaac Lab, not `so101_typing.tasks`, to avoid registering the same tasks twice.
The `scripts/so101_homing` compatibility link keeps older commands and recorded
kinematics hashes valid; current hardware imports use `so101_typing.hardware`.

Raw recordings, camera captures and original fit reports remain local. Selected
sanitized reports and physical-run evidence are published with the calibrated
checkpoint. Preserve original hashes and files; do not rewrite old evidence to
match new code.
Portable source must not depend on a particular user's home, serial number or
an existing `output/` directory. Keep machine shortcuts in `local/`.
Use generic device placeholders in instructions and synthetic identifiers in
tests. Model dimensions, benchmark transforms, joint limits and pinned dependency
hashes are intentional reproducibility constants. Git author metadata and older
revisions require a separate history review; a clean current tree is not an
anonymized Git history.

## Host checks (no robot connection)

Run these inside a separate Python 3.11+ development environment (3.12 recommended).
The pinned NumPy and SciPy versions require Python 3.11 or newer. Keep the pinned
hardware environment unchanged.

```bash
python3 -m pip install pytest==8.4.2 pytest-subtests==0.14.2 numpy==2.3.1 scipy==1.17.0 PyYAML==6.0.2
python3 -m pip install --no-deps --editable .
python3 scripts/check_public_release.py
python3 -m compileall -q scripts source
python3 -m pytest -q tests
git diff --check
```

Host pytest skips `tests/simulation/`. Run that directory explicitly with the
Isaac Lab Python runtime when checking simulator contracts.

The source check includes tracked **and new
untracked** files and rejects local Markdown links absent from the distributed
source. In an extracted archive it scans the archive without consulting a parent
Git checkout. To inspect downloaded metadata too, add
`--artifact-dir .artifacts/so101-keyboard-typing-benchmark`.

After source changes, rebuild with `./so101 build`; the launcher checks the
source fingerprint. Test an extracted source snapshot without `output/`,
`.artifacts/`, `local/` or a hardware environment, including the fixture drawing
and source checks. A cached build on this workstation is not an independent
installation. Keep running training jobs on their frozen source/image snapshots.

Record published measurements in [RESULTS.md](release/RESULTS.md), review the diff,
and inspect the exact source archive. Keep detailed check logs in ignored outputs.
Green host CI does not establish physical alignment, contact accuracy or
second-robot typing.
