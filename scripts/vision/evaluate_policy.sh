#!/bin/bash
# Run one vision-policy episode, then open its exact replay in interactive Kit RTX.
set -euo pipefail
policy_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$policy_root"
if [[ $# -lt 1 || "$1" == "--help" ]]; then
  echo "Usage: $0 CHECKPOINT [--output DIR] [--no-viewer] [evaluation options]"
  echo "Example: SO101_VIEW_GPU=1 $0 output/my-run/model_4999.pt"
  exit 0
fi
policy_checkpoint=$(realpath "$1")
shift
[[ -f "$policy_checkpoint" ]] || { echo "Missing checkpoint: $policy_checkpoint" >&2; exit 2; }
case "$policy_checkpoint" in
  "$policy_root"/*) policy_checkpoint="/repo/${policy_checkpoint#"$policy_root"/}" ;;
  *) echo 'Checkpoint must be inside this repository.' >&2; exit 2 ;;
esac
policy_output="$policy_root/output/vision_ppo/evaluations/$(date -u +%Y%m%dT%H%M%S.%NZ)"
policy_view=1
policy_viewer_seconds=0
policy_extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) policy_output=$(realpath -m "${2:?--output needs a directory}"); shift 2 ;;
    --no-viewer) policy_view=0; shift ;;
    --viewer-seconds) policy_viewer_seconds="${2:?--viewer-seconds needs a value}"; shift 2 ;;
    *) policy_extra+=("$1"); shift ;;
  esac
done
case "$policy_output" in
  "$policy_root"/output/*) ;;
  *) echo 'Output must be under the repository output/ directory.' >&2; exit 2 ;;
esac
if (( policy_view )); then
  [[ -n "${DISPLAY:-}" ]] || { echo 'DISPLAY is required for the RTX viewer; use --no-viewer for export only.' >&2; exit 2; }
  policy_authority="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
  [[ -f "$policy_authority" ]] || { echo "Missing Xauthority: $policy_authority" >&2; exit 2; }
fi
mkdir -p "$policy_output"
if ! docker image inspect so101-typing-public:dev >/dev/null 2>&1; then ./so101 build; fi
policy_image="${SO101_EVAL_IMAGE:-so101-vision-simple:dev}"
if [[ -z "${SO101_EVAL_IMAGE:-}" ]]; then
  docker build -f docker/vision-ppo.Dockerfile -t "$policy_image" .
else
  docker image inspect "$policy_image" >/dev/null
fi
docker run --rm --init --gpus "device=${SO101_VIEW_GPU:-0}" --ipc=private --shm-size=16g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=so101 -e LOGNAME=so101 \
  -e OMNICLIENT_HUB_MODE=disabled \
  --tmpfs /opt/vision/lib/python3.12/site-packages/ovrtx/bin/cache/nv_shadercache:rw,mode=1777 \
  --tmpfs /opt/vision/lib/python3.12/site-packages/ovrtx/bin/cache/texturecache:rw,mode=1777 \
  --tmpfs /opt/vision/lib/python3.12/site-packages/ovrtx/bin/cache/DerivedDataCache:rw,mode=1777 \
  --tmpfs /opt/vision/lib/python3.12/site-packages/ovrtx/bin/mdl/omniverse_exts:rw,mode=1777 \
  --no-healthcheck -e NVIDIA_DRIVER_CAPABILITIES=all -e OMP_NUM_THREADS=8 \
  -v "$policy_root:/repo:ro" -v "$policy_output:$policy_output" \
  "$policy_image" /repo/scripts/vision/evaluate_policy.py \
  --checkpoint "$policy_checkpoint" --output "$policy_output" "${policy_extra[@]}"
echo "Evaluation and replay: $policy_output"
if (( ! policy_view )); then exit 0; fi
policy_home="$policy_root/output/.kit-policy-viewer"
for name in home cache data logs; do mkdir -p "$policy_home/$name"; done
exec docker run --rm --init --gpus "device=${SO101_VIEW_GPU:-0}" --ipc=private --shm-size=16g \
  --user "$(id -u):$(id -g)" --no-healthcheck \
  -e DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNICLIENT_HUB_MODE=disabled \
  -e HOME=/tmp/so101-home -e USER=so101 -e LOGNAME=so101 -e XAUTHORITY=/tmp/typing.xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v "$policy_authority:/tmp/typing.xauthority:ro" \
  -v "$policy_root:/repo:ro" -v "$policy_output:$policy_output:ro" \
  -v "$policy_home/home:/tmp/so101-home" -v "$policy_home/cache:/isaac-sim/kit/cache" \
  -v "$policy_home/data:/isaac-sim/kit/data" -v "$policy_home/logs:/isaac-sim/kit/logs" \
  so101-typing-public:dev -p /repo/scripts/vision/evaluate_policy.py \
  --replay "$policy_output/evaluation.json" --viewer-seconds "$policy_viewer_seconds"
