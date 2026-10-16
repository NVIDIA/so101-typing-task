#!/usr/bin/env bash
# SO-101 typing calibration: arm video, all servo telemetry, keyboard events.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec "${SO101_LEROBOT_PYTHON:-$repo_root/.venv-hardware/bin/python}" \
  -m so101_typing.hardware.typing_calibration_record \
  --execute --record-cameras --stop-on-return --telemetry-hz 5 \
  --output-root "$repo_root/output/calibration_video_gtc_berlin" "$@"
