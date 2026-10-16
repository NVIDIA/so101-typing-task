#!/usr/bin/env bash
# Render one 2x2 or 8x8 Newton replay grid through the isolated OVRTX image.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if (( $# < 2 )); then echo "Usage: $0 grid-directory output-mp4 [renderer-options]" >&2; exit 2; fi
grid_dir="$(realpath "$1")"
video_name="$2"
shift 2
[[ "$video_name" == *.mp4 && "$video_name" != */* ]] || { echo 'Output must be an mp4 basename' >&2; exit 2; }
[[ -f "$grid_dir/grid.usdc" && -f "$grid_dir/grid.json" ]] || { echo 'Missing grid scene' >&2; exit 2; }
if ! docker image inspect so101-tabletop-ovrtx:dev >/dev/null 2>&1; then
  docker build -f docker/tabletop-ovrtx.Dockerfile -t so101-tabletop-ovrtx:dev .
fi
grid_name="$(basename "$grid_dir")"
grid_parent="$(dirname "$grid_dir")"
mkdir -p "$grid_dir/.renderer-home" "$grid_dir/.renderer-mdl" "$grid_dir/videos"
cache_mounts=()
for cache in nv_shadercache texturecache DerivedDataCache; do
  mkdir -p "$grid_dir/.renderer-cache/$cache"
  cache_mounts+=(-v "$grid_dir/.renderer-cache/$cache:/opt/ovrtx/lib/python3.12/site-packages/ovrtx/bin/cache/$cache")
done
docker run --rm --gpus "device=${SO101_RENDER_GPU:-1}" --shm-size=16g \
  --user "$(id -u):$(id -g)" -e HOME="/results/$grid_name/.renderer-home" -e USER=so101 -e LOGNAME=so101 \
  -e OMNICLIENT_HUB_MODE=disabled -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONPATH=/render-scripts \
  -v "$grid_parent:/results" "${cache_mounts[@]}" \
  -v "$grid_dir/.renderer-mdl:/opt/ovrtx/lib/python3.12/site-packages/ovrtx/bin/mdl/omniverse_exts" \
  -v "$repo_root/scripts/rendering:/render-scripts:ro" \
  so101-tabletop-ovrtx:dev /render-scripts/render_two_env_ovrtx.py "/results/$grid_name/grid.usdc" \
  --output "/results/$grid_name/videos/$video_name" "$@"
