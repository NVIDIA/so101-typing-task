#!/usr/bin/env bash
# Export the calibrated stationary Newton scene, then render it with OVRTX.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if (( $# > 1 )); then echo "Usage: $0 [new-output-directory]" >&2; exit 2; fi
render_output="${1:-output/recorded_tabletop_scene/ovrtx-$(date -u +%Y%m%dT%H%M%SZ)}"
render_output="$(realpath -m "$render_output")"
if [[ -e "$render_output" ]]; then echo "Choose a new output directory: $render_output" >&2; exit 2; fi
if ! docker image inspect so101-tabletop-ovrtx:dev >/dev/null 2>&1; then
  docker build -f docker/tabletop-ovrtx.Dockerfile -t so101-tabletop-ovrtx:dev .
fi
./scripts/view_recorded_tabletop.sh --frames 6 --export-scene --output-dir "$render_output"
mkdir -p "$render_output/.renderer-home" "$render_output/.renderer-mdl"
cache_mounts=()
for cache in nv_shadercache texturecache DerivedDataCache; do
  mkdir -p "$render_output/.renderer-cache/$cache"
  cache_mounts+=(-v "$render_output/.renderer-cache/$cache:/opt/ovrtx/lib/python3.12/site-packages/ovrtx/bin/cache/$cache")
done
docker run --rm --gpus "device=${SO101_RENDER_GPU:-0}" --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/results/.renderer-home -e USER=so101 -e LOGNAME=so101 \
  -e OMNICLIENT_HUB_MODE=disabled -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$render_output:/results" "${cache_mounts[@]}" \
  -v "$render_output/.renderer-mdl:/opt/ovrtx/lib/python3.12/site-packages/ovrtx/bin/mdl/omniverse_exts" \
  -v "$repo_root/scripts/rendering:/render-scripts:ro" \
  so101-tabletop-ovrtx:dev /render-scripts/render_tabletop_ovrtx.py /results/scene.usdc \
  --output /results/tabletop-rtx.png 2>&1 | tee "$render_output/render-console.log"
echo "RTX render: $render_output/tabletop-rtx.png"
