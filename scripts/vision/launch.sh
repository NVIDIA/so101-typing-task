#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Host launcher only; simulator/trainer/evaluator all run in the defined Docker image.
set -euo pipefail
vision_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$vision_root"
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 OUTPUT_DIR TEACHER_CHECKPOINT TASK_CONFIG [supervisor options]" >&2
  exit 2
fi
vision_output=$(realpath -m "$1")
vision_teacher=$(realpath "$2")
vision_task=$(realpath "$3")
shift 3
mkdir -p "$vision_output"
docker build -f docker/vision-ppo.Dockerfile -t so101-vision-ppo:dev . > "$vision_output/build.log" 2>&1
vision_unit="so101-vision-$(basename "$vision_output" | tr '_' '-')"
systemd-run --user --unit="$vision_unit" --collect \
  --property="WorkingDirectory=$vision_root" \
  --property="StandardOutput=append:$vision_output/supervisor.log" \
  --property="StandardError=append:$vision_output/supervisor.log" \
  python3 "$vision_root/scripts/vision/supervise.py" --output "$vision_output" \
  --teacher "$vision_teacher" --task-config "$vision_task" "$@"
echo "$vision_unit" > "$vision_output/supervisor.service"
echo "Status: $vision_output/supervisor.json"
