#!/usr/bin/env bash
# Simulation only. Supply your diagnostic mapping report explicitly.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
  echo "Usage: $0 --zero-mapping PATH_TO_MAPPING_REPORT [--task-config PATH] [scene options]"
  exit 0
fi
exec "$repo_root/scripts/view_frozen_scene.sh" "$@"
