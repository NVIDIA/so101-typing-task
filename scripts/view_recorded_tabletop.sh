#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Run the included ISAAC LAB 3.0 state policy and open its animated RTX replay.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec ./so101 replay --model models/isaac_lab_state/isaac_lab_ckpt.pt \
  --gpu "${SO101_VIEW_GPU:-0}" "$@"
