#!/usr/bin/env bash
# Evaluate the ISAAC LAB 3.0 state policy, then replay that motion in an RTX grid.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec python3 -m scripts.tutorial.rtx_examples "$@"
