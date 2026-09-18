# Historical July checkpoint

The original AnchorBench `model_19000.pt` remains available for simulation and
benchmark reference. Its **89/100 physical six-letter trials** apply to the
original robot and mat-supported fixture. New users should follow the
[September setup and calibration workflow](../../README.md#workflow-1-train-for-your-setup).

Download the historical checkpoint, matching environment and evaluation report:

```bash
./so101 download --historical
ARTIFACTS="$PWD/.artifacts/so101-keyboard-typing-benchmark"
./so101 video --actuator anchorbench --stage transit15 \
  --checkpoint "$ARTIFACTS/checkpoints/mjwarp-anchorbench-19k/model_19000.pt" \
  --env-config "$ARTIFACTS/configs/mjwarp-anchorbench-19k.env.yaml" \
  --target NVIDIA
```

Use its matching environment without a September task override. The checkpoint
retains its original scene and joint-coordinate contract; its offsets are not a
calibration for another robot. See [historical results](../release/RESULTS.md)
for benchmark evidence and [the archived model drawing](reference-birdseye.svg)
for its effective simulation placement. That drawing is not a physical setup
recipe. The current [physical reference](physical_reference_20260915.md) and
[fitted September scene](../../configs/tasks/replay-calibrated-20260917.json)
are separate.
