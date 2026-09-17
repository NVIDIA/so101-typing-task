# Training stages, outputs and comparisons

The public entry point starts from scratch unless `--checkpoint` is supplied.
P1A trains two-letter sequences. Transit15 trains six-letter sequences with the
15 mm transit-clearance contract. Observation/action sizes remain 22/5.

For a policy intended for hardware, first follow the
[three-step real-to-sim guide](REAL2SIM.md): collect torque-off key presses, fit
joint offsets and keyboard pose, and pass the reset-per-key contact check.
Train using its `output/my-scene/task.json`. The recordings calibrate the scene;
they are not PPO training data. Keep the scene fixed throughout training and
keep its mapping report for separate physical deployment checks. Short software
checks below can run earlier and do not establish physical correspondence.

## Short validation

```bash
./so101 train --actuator anchorbench --num-envs 64 --iterations 10 --seed 1307
```

One-iteration runs with four environments have been tested for both public actuator
profiles, task DR, changed nominal pose, custom solver settings and custom actuator
parameters. The 64-environment, 10-update AnchorBench quickstart and DR run also
pass, as does a short P1A-to-Transit15 resume. These runs test the pipeline, not
policy performance. First startup
compiles Warp kernels; the user-owned runtime cache reuses them on later runs.

## Train for your configured keyboard pose

After the real-to-sim fit produces `output/my-scene/task.json`, an example two-stage
budget is 4,000 P1A updates followed by 16,000 six-letter updates. Evaluate early
and choose the continuation budget from your results:

```bash
./so101 train --actuator anchorbench --stage p1a \
  --task-config output/my-scene/task.json \
  --num-envs 4096 --iterations 4000 --seed 1307

# Evaluate a saved P1A checkpoint before choosing a continuation budget.
./so101 evaluate --actuator anchorbench --stage p1a \
  --checkpoint /absolute/path/to/P1A/model_N.pt \
  --env-config /absolute/path/to/P1A/params/env.yaml --num-envs 1024 --seed 2307
./so101 video --actuator anchorbench --stage p1a \
  --checkpoint /absolute/path/to/P1A/model_N.pt \
  --env-config /absolute/path/to/P1A/params/env.yaml --target HE

# Continue from the evaluated model with the same profile/configuration.
./so101 train --actuator anchorbench --stage transit15 \
  --checkpoint /absolute/path/to/P1A/model_N.pt \
  --num-envs 4096 --iterations 16000 --seed 1307
```

This documents a runnable staged recipe, not a newly reproduced benchmark result.
The historical benchmark used qualified precursor checkpoints and specific
multi-GPU runs. The released artifact manifest records checkpoint identities. A new seed, new geometry or new precursor is a new experiment.
RSL-RL checkpoint indices are inherited on resume; use saved iteration metadata
and the request's additional-iteration budget rather than guessing update counts
from a filename. Do not select a checkpoint solely because it has the largest
number: evaluate it under the intended sequence bank and event criteria.

`--checkpoint` restores optimizer/policy state. Task/physics JSONs beside the
checkpoint under `params/` are inherited automatically; keep the same actuator
choice unless intentionally performing a separately labelled transfer experiment.
Use `--task-config` or `--physics-config` explicitly when changing the experiment.

The 4096-environment, full-budget recipe has not been rerun in the new public
container. Memory use and wall time must be measured for the recipient's GPU.
Reduce `--num-envs` if necessary, but record it: it changes samples per PPO update
and therefore the training budget. The current default is 64 environments, not a
claim that 64 reproduces the historical training setup.

## Evaluate and compare

```bash
./so101 evaluate --actuator anchorbench --stage transit15 \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --num-envs 1024 --seed 2307
./so101 video --actuator anchorbench --stage transit15 \
  --checkpoint /absolute/path/to/model_N.pt \
  --env-config /absolute/path/to/params/env.yaml --target NVIDIA
```

Repeat evaluation with `--seed 3307`. Review the videos for clean presses,
release and clearance; scalar success alone does not establish contact quality.

Use the same task, reward, observations, training sample budget, seeds, checkpoint
selection rule and evaluation sequences when comparing independently trained
actuator profiles. Keep solver comparisons separate. A fixed-policy physics
ablation is a different experiment and is not performed silently by this CLI.

Reports include exact typing, strict press/release/clearance success, wrong keys,
termination reasons and checkpoint/configuration hashes. Generated configurations
and logs live under `output/public/<timestamp>-train/rsl_rl/...`; evaluation results
live in the corresponding `*-evaluate` directory. Keep the complete `params/`
folder with each policy for replay and deployment export.

The launcher prints host paths after each run: the latest saved checkpoint and
matching environment for training, or the evaluation report and video when
produced. Use these paths in place of the examples' `/absolute/path/to/...`
placeholders. The latest checkpoint still needs evaluation before deployment.

## Supervise the staged run

The host-side supervisor runs the same public commands sequentially and saves
stage status, console logs, evaluation reports, videos and GPU samples. Run it
inside a persistent terminal such as tmux for a long experiment:

```bash
./so101 build
python3 scripts/run_public_training_pipeline.py \
  --task-config output/my-scene/task.json \
  --out-dir output/my-training-seed1307
```

Defaults are 4096 environments, 4000 P1A updates, then 16000 additional Transit15
updates. It evaluates P1A on 1024 episodes with seed 2307 before advancing. The
default promotion threshold is 90% strict success, an engineering gate rather
than a claim of hardware readiness. A failed gate saves its report and video,
stops before Transit15, and marks `status.json` for inspection.

Transit15 is evaluated on 1024 episodes each with seeds 2307 and 3307. The default
run ends after simulation checks. For physical deployment, follow
[alignment qualification and export](ALIGNMENT.md) using the fitted scene/mapping
and the same robot calibration used for recording.

The optional `--prepare-deployment --robot-id YOUR_CALIBRATED_ID` only creates a
bundle without alignment qualification and a software dry run. It requires explicit
`--encoder-convention lerobot`, `--encoder-convention benchmark`, or
`--joint-calibration FILE`; it does not accept the scene-fit mapping report or
qualified alignment. Use the separate alignment/export workflow
for those. Export is skipped if either six-letter evaluation fails the quality
gate. Recorded videos still need visual review; scalar gates do not certify
contact quality or sim-to-real transfer.

The output directory must be new. `status.json` records current stage and terminal
status; `gpu.csv` records total GPU memory/use every 10 seconds, including desktop
processes. Each stage has its own public CLI request and resolved training inputs.
For individual commands, `./so101 ... --output-dir NEW_DIRECTORY` selects an explicit
output directory instead of the default timestamped location.

The command above uses your fitted task; for simulation-only experiments substitute
your example task JSON. `--physics-config FILE` optionally supplies MJWarp
overrides. Without these options it uses the default reference configuration.
Both files are validated before creating the output directory, copied into its
`inputs/` directory, and used for both training stages. Editing the original files
later does not change the running experiment. `status.json` records the original
paths, snapshot paths and SHA-256 hashes. Evaluation and video use the matching
checkpoint's saved `params/` configuration automatically. `--image IMAGE` selects
the Docker image for every simulation stage. Short-run results
are retained in [development validation](RESULTS.md), not as recommended policies.

## Disk, time and ownership

The measured image size is approximately 32.6 GB (30.4 GiB). Plan for at least
70 GB free across Docker storage and your workspace for layers, caches and runs;
actual peak storage depends on build cache and checkpoint retention.
First download/build and kernel compilation may take tens of minutes to hours,
depending on connection, CPU and caches; a cold-install wall time is not measured.

At the measured 36.72 seconds per 10 updates with 4096 environments, 4000 updates
extrapolate to 4.08 hours and another 16000 to 16.32 hours: about 20.4 hours total
on the RTX 5090 before startup/evaluation. This is a planning estimate, not a
completed 20,000-update timing. Evaluate early rather than assuming that budget
is necessary. Requests and exit metadata record timestamps and wall seconds;
RSL-RL console output reports training-loop time separately.

Containers run with your host UID/GID, so new checkpoints and run directories
can be edited or deleted without sudo. The writable container home/cache is
stored under `XDG_CACHE_HOME/so101-typing` (default: `~/.cache/so101-typing`).
This host cache uses additional disk space as kernels are compiled. The older
`so101-typing-warp-cache` Docker volume is no longer used. Once you have stopped
using containers from older versions, you can reclaim that volume's disk space:

```bash
docker volume rm so101-typing-warp-cache
```

Existing root-owned runs from older versions are not automatically changed; their owner must repair
permissions or remove them once.
