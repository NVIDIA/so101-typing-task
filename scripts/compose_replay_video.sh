#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Compose an overview, OVRTX wrist movie and evaluation events inside the runtime.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if (( $# < 1 )); then echo "Usage: $0 replay-output-directory [--overview videos/file.mp4] [--wrist videos/file.mp4] [--output videos/file.mp4]" >&2; exit 2; fi
replay_output="$(realpath "$1")"
shift
[[ -f "$replay_output/evaluation.json" ]] || { echo 'Missing evaluation.json' >&2; exit 2; }
docker image inspect so101-typing-public:dev >/dev/null 2>&1 || { echo 'Run ./so101 build first' >&2; exit 2; }
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -v "$replay_output:/results" -v "$repo_root/scripts/rendering:/render-scripts:ro" \
  --entrypoint /workspace/isaaclab/_isaac_sim/python.sh so101-typing-public:dev \
  /render-scripts/compose_replay_video.py /results "$@"
