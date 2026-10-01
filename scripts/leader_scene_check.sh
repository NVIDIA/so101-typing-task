#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Powered leader/follower scene recording. Explicit hardware identities only.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mode="${1:-help}"
if (( $# )); then shift; fi
python_bin="${SO101_LEROBOT_PYTHON:-$repo_root/.venv-hardware/bin/python}"
case "$mode" in
  help|--help|-h)
    echo "Usage: $0 {check|inspect|calibrate|record} --port FOLLOWER --leader-port LEADER --leader-id ID [recorder options]"
    echo "check/inspect/record also need --robot-id, --keyboard-device and --task-config."
    echo "Replay: $0 replay --recording PATH [replay options]"
    exit 0 ;;
  replay) exec "$repo_root/scripts/scene_check.sh" replay "$@" ;;
  check|inspect|calibrate|record) ;;
  *) echo "Unknown mode: $mode; use --help" >&2; exit 2 ;;
esac
leader_port="${SO101_LEADER_PORT:-}"
leader_id="${SO101_LEADER_ID:-}"
follower_port="${SO101_ROBOT_PORT:-}"
extra=()
while (( $# )); do
  case "$1" in
    --leader-port|--leader-id|--port)
      if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --leader-port) leader_port="$2" ;;
        --leader-id) leader_id="$2" ;;
        --port) follower_port="$2" ;;
      esac
      shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
if [[ -z "$leader_port" || -z "$leader_id" || -z "$follower_port" ]]; then
  echo "Pass --port, --leader-port and --leader-id explicitly; no serial device is selected automatically." >&2
  exit 2
fi
if [[ ! -e "$leader_port" || ! -e "$follower_port" || "$(readlink -f "$leader_port")" == "$(readlink -f "$follower_port")" ]]; then
  echo "Leader and follower must be existing, different serial devices." >&2; exit 2
fi
if [[ ! "$leader_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Leader ID must contain only letters, digits, underscores or hyphens." >&2; exit 2
fi
if [[ "$mode" == calibrate ]]; then
  if (( ${#extra[@]} )); then echo "Calibration accepts only leader/follower port and leader ID options." >&2; exit 2; fi
  exec "$(dirname "$python_bin")/lerobot-calibrate" --teleop.type=so101_leader --teleop.port="$leader_port" --teleop.id="$leader_id"
fi
flag="--$mode"
[[ "$mode" != record ]] || flag="--execute"
exec "$python_bin" -m so101_typing.hardware.teleop_scene_record \
  --leader-port "$leader_port" --leader-id "$leader_id" --port "$follower_port" "${extra[@]}" "$flag"
