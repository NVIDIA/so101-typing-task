# RTX replay and advanced options

The main command runs Newton physics with the selected policy observations,
then opens the captured episode in an interactive RTX viewer. It does not
connect to physical hardware. Vision policies use OVRTX wrist-camera images;
the ISAAC LAB 3.0 state policy uses simulator state without image observations.
This revision does not include the two selected checkpoints. Pass a checkpoint
from a local training run. Provenance for the withheld files is in
[models/README.md](../models/README.md).

```bash
./so101 replay --model /absolute/path/to/newton_ckpt.pt
./so101 replay --model /absolute/path/to/isaac_lab_ckpt.pt
# Save an MP4 without opening a window.
./so101 replay --model /absolute/path/to/newton_ckpt.pt --video
# Export the rollout without a viewer or video rendering.
./so101 replay --model /absolute/path/to/newton_ckpt.pt --no-viewer
```

Use `--gpu 1` for a second GPU, `--episodes N` for multiple episodes, or
`--output output/my-replay` for an explicit output directory. Default output
from `./so101 replay` goes beneath `output/tutorial/`.

## Lower-level launcher

The following script is the same replay workflow used by `./so101 replay`.
It also exposes seed, target, and randomization controls.

Pass a vision checkpoint from a training run, keeping `environment.json`,
`agent.json` and `run.json` beside it:

```bash
scripts/vision/evaluate_policy.sh /absolute/path/to/newton_ckpt.pt
```

The launcher loads the saved actor and environment, evaluates one deterministic
student episode with online wrist images, then opens the captured rollout
in an interactive Isaac Sim RTX viewport. It uses no teacher checkpoint. The
inference/export process and Kit viewer run separately to keep their USD runtimes
compatible. The episode is evaluated before the viewport opens; the viewer
loops its captured outcome, including failures. Orbit with Alt + left drag, pan with middle drag, zoom with the
wheel, and close the window to exit. The default overview uses a close, elevated
side view with the robot on the right and the keyboard on the left, saved in
[`evaluation_replay_camera.json`](../configs/vision/evaluation_replay_camera.json).

Target and randomization strength are read from the training run. Use
`--target NEWTON` for an A–Z checkpoint if needed, `--seed 14002` to change
the seed (default: 14001), or `--domain-strength 0` for a nominal-placement diagnostic. The target
vocabulary must match the trained actor. The default is one episode; optionally use `--episodes N` for multiple seeds. `evaluation.json` records each seed,
typed result, failures, and aggregate success percentage. A single episode is a visual check, not a reliable success-rate estimate. Each `episode_001/`,
`episode_002/`, etc. contains its own USD replay and typing timeline.

```bash
# Export evaluation.json and all episode replays without opening a window.
scripts/vision/evaluate_policy.sh /absolute/path/to/newton_ckpt.pt \
  --output output/vision_ppo/newton_eval --no-viewer
```

Evaluation defaults to physical GPU 0; `SO101_VIEW_GPU=1` selects GPU 1.
The default output is a timestamped directory under `output/vision_ppo/evaluations/`.
RTX display requires an X11 `DISPLAY` and readable `XAUTHORITY`. The implementation
is [evaluate_policy.py](../scripts/vision/evaluate_policy.py), with
[evaluate_policy.sh](../scripts/vision/evaluate_policy.sh) handling Docker and display
mounts. Use `--help` for options.

## State-policy viewer shortcuts

`view_recorded_tabletop.sh`, `view_recorded_tabletop_rtx.sh`, and
`view_frozen_scene.sh` launch the ISAAC LAB 3.0 state checkpoint through
`./so101 replay` when `models/isaac_lab_state/isaac_lab_ckpt.pt` is present
locally. That file is not in this revision. They accept its options, including `--output`, `--no-viewer`,
`--video`, and `--viewer-seconds`. Their former stationary-scene arguments no
longer apply. Use `SO101_VIEW_GPU=1` or `--gpu 1` for a second GPU.

`view_typing_grid_rtx.sh --rows 2 --cols 2` first evaluates the state checkpoint,
then copies the successful recorded trajectory into independently colored cells
and opens an animated RTX viewport. It does not run independent policies in
those cells. It requires successful ISAAC LAB 3.0 completion before building the
grid. Options include `--rows`, `--cols` (up to 64 cells), `--gpu`, `--output`,
`--no-viewer`, `--viewer-seconds`, and `--plan`. Every launch uses fresh output;
no old recordings are required. Low-level static scene inspection remains
available through `./so101 scene`.
