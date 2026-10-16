#!/usr/bin/env bash
# Render the Workshop wrist camera from a saved --export-replay rollout.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if (( $# < 1 )); then echo "Usage: $0 replay-output-directory [renderer-options]" >&2; exit 2; fi
render_output="$(realpath "$1")"
shift
[[ -f "$render_output/replay.replay.json" ]] || { echo 'Missing replay export' >&2; exit 2; }
if ! docker image inspect so101-tabletop-ovrtx:dev >/dev/null 2>&1; then
  docker build -f docker/tabletop-ovrtx.Dockerfile -t so101-tabletop-ovrtx:dev .
fi
mkdir -p "$render_output/.renderer-home" "$render_output/.renderer-mdl" "$render_output/videos"
cache_mounts=()
for cache in nv_shadercache texturecache DerivedDataCache; do
  mkdir -p "$render_output/.renderer-cache/$cache"
  cache_mounts+=(-v "$render_output/.renderer-cache/$cache:/opt/ovrtx/lib/python3.12/site-packages/ovrtx/bin/cache/$cache")
done
docker run --rm --gpus "device=${SO101_RENDER_GPU:-0}" --shm-size=16g \
  --user "$(id -u):$(id -g)" -e HOME=/results/.renderer-home -e USER=so101 -e LOGNAME=so101 \
  -e OMNICLIENT_HUB_MODE=disabled -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$render_output:/results" "${cache_mounts[@]}" \
  -v "$render_output/.renderer-mdl:/opt/ovrtx/lib/python3.12/site-packages/ovrtx/bin/mdl/omniverse_exts" \
  -v "$repo_root/scripts/rendering:/render-scripts:ro" \
  so101-tabletop-ovrtx:dev /render-scripts/render_replay_ovrtx.py /results/replay.usdc \
  --output /results/videos/wrist-camera-ovrtx.mp4 "$@" 2>&1 | tee "$render_output/wrist-render-console.log"
