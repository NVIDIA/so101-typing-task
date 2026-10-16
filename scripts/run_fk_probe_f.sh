#!/usr/bin/env bash
# Explicit arguments only; see --help. No robot-specific defaults.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec "${SO101_LEROBOT_PYTHON:-$repo_root/.venv-hardware/bin/python}" \
  -m so101_typing.hardware.run_fk_probe "$@"
