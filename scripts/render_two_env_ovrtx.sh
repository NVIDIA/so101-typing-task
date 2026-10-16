#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Render the shared two-bench USD scene through the repository's isolated OVRTX image.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if (( $# < 2 )); then echo "Usage: $0 output-directory output-mp4 [renderer-options]" >&2; exit 2; fi
render_output="$(realpath "$1")"
video_name="$2"
shift 2
[[ "$video_name" == *.mp4 && "$video_name" != */* ]] || { echo 'Output must be an mp4 basename' >&2; exit 2; }
[[ -f "$render_output/two-envs.usdc" && -f "$render_output/two-envs.json" ]] || { echo 'Missing shared replay scene' >&2; exit 2; }
if ! docker image inspect so101-tabletop-ovrtx:dev >/dev/null 2>&1; then
  docker build -f docker/tabletop-ovrtx.Dockerfile -t so101-tabletop-ovrtx:dev .
fi
mkdir -p "$render_output/.renderer-home" "$render_output/.renderer-mdl" "$render_output/videos"
cache_mounts=()
for cache in nv_shadercache texturecache DerivedDataCache; do
  mkdir -p "$render_output/.renderer-cache/$cache"
  cache_mounts+=(-v "$render_output/.renderer-cache/$cache:/opt/ovrtx/lib/python3.12/site-packages/ovrtx/bin/cache/$cache")
done
docker run --cap-drop=ALL --security-opt=no-new-privileges --rm --gpus "device=${SO101_RENDER_GPU:-1}" --shm-size=16g \
  --user "$(id -u):$(id -g)" -e HOME=/results/.renderer-home -e USER=so101 -e LOGNAME=so101 \
  -e OMNICLIENT_HUB_MODE=disabled -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONPATH=/render-scripts \
  -v "$render_output:/results" "${cache_mounts[@]}" \
  -v "$render_output/.renderer-mdl:/opt/ovrtx/lib/python3.12/site-packages/ovrtx/bin/mdl/omniverse_exts" \
  -v "$repo_root/scripts/rendering:/render-scripts:ro" \
  so101-tabletop-ovrtx:dev /render-scripts/render_two_env_ovrtx.py /results/two-envs.usdc \
  --output "/results/videos/$video_name" "$@"
