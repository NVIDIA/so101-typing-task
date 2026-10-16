#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mode="${1:---help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  record) exec "$repo_root/scripts/scene_check.sh" record "$@" ;;
  fit) exec "${SO101_LEROBOT_PYTHON:-$repo_root/.venv-hardware/bin/python}" -m so101_typing.hardware.real2sim "$@" ;;
  -h|--help) printf '%s\n' 'Usage: ./scripts/real2sim.sh record|fit [options]' 'record: torque-off hand-guided capture; see record --help' 'fit: fit scene and run reset-per-key Newton contact checks; see fit --help' ;;
  *) printf 'Unknown mode: %s (choose record or fit)\n' "$mode" >&2; exit 2 ;;
esac
