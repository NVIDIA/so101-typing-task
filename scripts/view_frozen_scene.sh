#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
authority="/run/user/$(id -u)/gdm/Xauthority"
if [[ -z "${XAUTHORITY:-}" && -f "$authority" ]]; then export XAUTHORITY="$authority"; fi
exec ./so101 scene --task-config configs/tasks/physical-reference-20260915.json "$@"
