# Development and release checks

`NVIDIA-dev/so101-keyboard-typing` is the canonical development repository and
remains private. The older `VigneshBhavan/so101-keyboard-task-public` repository
is the September 11 released snapshot; do not maintain divergent workflow fixes
in both checkouts. Publishing changes or changing visibility is a separate step.

## Where things belong

| Location | Purpose |
| --- | --- |
| `README.md` | Install, train/evaluate, optional reference checkpoint, export/deploy |
| [FIXTURE_SETUP.md](../FIXTURE_SETUP.md) | Measure placement, record/replay real motion, test physical correspondence |
| `docs/release/` | Supported commands, requirements, results and remaining qualification |
| `docs/fixtures/` | Geometry, drawings, dated fixture and historical checkpoint provenance |
| [docs/experiments/](experiments/README.md) | Dated investigations; not required setup instructions |
| `scripts/` | Portable tools with explicit hardware/scene inputs |
| `output/`, `.artifacts/`, `local/` | Ignored runs, downloaded weights and workstation shortcuts; excluded from Docker |

Raw recordings, camera captures and fit reports remain local. Investigation
notes name them as artifact identifiers, not downloadable source links. Preserve
their hashes and originals; do not rewrite old evidence to match new code.
Portable source must not depend on a particular user's home, serial number or
an existing `output/` directory. Keep machine shortcuts in `local/`.

## Host checks (no robot connection)

Run these inside a separate development Python environment. Keep the pinned
hardware environment unchanged.

```bash
python3 -m pip install pytest==8.4.2 pytest-subtests==0.14.2 numpy==2.3.1 PyYAML==6.0.2
python3 scripts/check_public_release.py
python3 -m compileall -q scripts source
python3 -m pytest -q tests scripts/so101_homing/test
git diff --check
```

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

Before publication, update [STATUS.md](release/STATUS.md) with measured checks,
review the diff, and inspect the exact source archive. Green host CI does not
establish physical alignment, contact accuracy or second-robot typing.
