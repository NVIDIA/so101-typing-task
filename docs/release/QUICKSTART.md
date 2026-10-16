# Simulation quickstart

Check the installation without connecting a robot. First follow the
[README requirements and installation](../../README.md#install), then run these
commands from the repository root:

```bash
./so101 build
```

First startup compiles Warp kernels; the launcher retains a user-owned runtime
cache. See the [requirements](../../README.md#requirements) for disk and GPU budgets.
Rebuild after runtime source changes.

## Simulation-only startup check

Use a separate example directory to keep these checks apart from your measured
fixture and fitted scene:

```bash
mkdir -p output/sim-example
cp configs/tasks/replay-calibrated-20260917.json output/sim-example/task.json
./so101 fixture --task-config output/sim-example/task.json --out-dir output/sim-example-view
./so101 probe --task-config output/sim-example/task.json
./so101 train --actuator gtc_berlin --stage p1a \
  --task-config output/sim-example/task.json --num-envs 64 --iterations 10
```

The ten-update run checks that training starts and saves outputs; it is too short
to expect successful typing. Runs go under `output/public/`. Add `--plan` to a
simulation command to inspect its launch without starting simulation or writing
outputs. Keep each checkpoint's complete `params/` directory for evaluation,
continuation and export.

For an interactive scene view, run this from an X11 desktop terminal:

```bash
./so101 scene --task-config output/sim-example/task.json
```

The Newton viewer stays open until you close it. For trained-policy playback,
follow the [released-checkpoint example](../../README.md#workflow-2-optionally-try-a-released-checkpoint).

For physical typing, follow the [main workflow](../../README.md#workflow-1-train-for-your-setup):
calibrate your robot, measure the fixture, record and fit the scene, then train
and deploy using the matching joint mapping. This simulation check does not
calibrate the physical setup.
