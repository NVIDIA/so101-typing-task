#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -eo pipefail
source /workspace/isaaclab/_isaac_sim/setup_python_env.sh
# Prefer OVRTX-compatible USD, retaining the simulator's bundled Python dependencies.
export PYTHONPATH="/opt/vision/lib/python3.12/site-packages:${PYTHONPATH:-}"
export OMNICLIENT_HUB_MODE=disabled
exec /opt/vision/bin/python "$@"
