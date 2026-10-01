#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ "${1:-}" == replay ]]; then
  authority="/run/user/$(id -u)/gdm/Xauthority"
  if [[ -z "${XAUTHORITY:-}" && -f "$authority" ]]; then export XAUTHORITY="$authority"; fi
fi
exec "${SO101_LEROBOT_PYTHON:-$repo_root/.venv-hardware/bin/python}" \
  -m so101_typing.hardware.scene_check "$@"
