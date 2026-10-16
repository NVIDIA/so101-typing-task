#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Foreground teacher warmup + PPO on one GPU; Ctrl-C stops the process.
set -euo pipefail
simple_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$simple_root"
if [[ $# -lt 2 || "$1" == "--help" ]]; then
  echo "Usage: $0 OUTPUT_DIR TEACHER_CHECKPOINT [train_simple.py options]"
  echo "Example: SO101_TRAIN_GPU=1 $0 output/vision_ppo/simple output/my-phrase-state-run/model_N.pt"
  exit 0
fi
simple_output=$(realpath -m "$1")
simple_teacher=$(realpath "$2")
shift 2
case "$simple_teacher" in
  "$simple_root"/*) simple_teacher="/repo/${simple_teacher#"$simple_root"/}" ;;
  *) echo 'TEACHER_CHECKPOINT must be inside this repository.' >&2; exit 2 ;;
esac
case "$simple_output" in
  "$simple_root"/output/*) ;;
  *) echo 'OUTPUT_DIR must be under this repository output/ directory.' >&2; exit 2 ;;
esac
if [[ -d "$simple_output" && -n "$(ls -A "$simple_output")" ]]; then
  echo 'Use an empty output directory for a new training run.' >&2
  exit 2
fi
mkdir -p "$simple_output"
if ! docker image inspect so101-typing-public:dev >/dev/null 2>&1; then
  ./so101 build
fi
docker build -f docker/vision-ppo.Dockerfile -t so101-vision-simple:dev .
docker run --cap-drop=ALL --security-opt=no-new-privileges --rm --init --gpus "device=${SO101_TRAIN_GPU:-0}" --ipc=private \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=so101 -e LOGNAME=so101 \
  --shm-size=16g --no-healthcheck \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMP_NUM_THREADS=8 \
  -v "$simple_root:/repo:ro" -v "$simple_output:$simple_output" \
  so101-vision-simple:dev /workspace/isaaclab/scripts/vision/train_simple.py \
  --headless --output "$simple_output" --teacher "$simple_teacher" "$@"
