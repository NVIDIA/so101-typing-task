# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable SO-101 names shared by the homing tools."""

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

DEFAULT_ROBOT_ID = "my_follower"
DEFAULT_KEYBOARD_DEVICE = "/dev/input/by-id/usb-Logitech_USB_Receiver-event-kbd"
HOMING_KEY_SEQUENCE = ("f", "y", "n")
DEFAULT_CAPTURE_PHRASE = "".join(HOMING_KEY_SEQUENCE)

# Fixed policy observation order. Registration preserves the physical position
# of each label in this order for the typing policy handoff.
POLICY_KEY_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("Backspace", "Space")
