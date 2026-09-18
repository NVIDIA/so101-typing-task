#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
if [[ ! -x "$root/.venv-hardware/bin/python" ]]; then
  uv venv --python 3.12 "$root/.venv-hardware"
fi
uv pip install --python "$root/.venv-hardware/bin/python" -r "$root/runtime/hardware.lock"
uv pip install --python "$root/.venv-hardware/bin/python" --no-deps --editable "$root"
"$root/.venv-hardware/bin/python" -c 'from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig; import torch, evdev, yaml; print("SO-101 hardware imports passed; no hardware connected")'
printf 'Hardware Python: %s\n' "$root/.venv-hardware/bin/python"
