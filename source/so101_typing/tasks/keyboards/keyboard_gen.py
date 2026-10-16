# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime implementation for procedural SO101 keyboard generation."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from isaaclab_tasks.core.dexsuite.config.so101.keyboards import keyboard_gen_cfg
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_geometry import resolve_keyboard
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_schema import ResolvedKeyboard

# Re-export: KeyboardSpawnerCfg.func resolves the string "{DIR}.keyboard_gen:spawn_keyboard" at
# spawn time, so this symbol must stay importable from this module (do not let linters strip it).
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_usd import spawn_keyboard  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg


class KeyboardAssetBank:
    """In-memory deterministic keyboard variant bank."""

    def __init__(self, cfg: keyboard_gen_cfg.KeyboardAssetBankCfg):
        self.cfg = cfg

    def make_spawner_cfgs(self) -> list[keyboard_gen_cfg.KeyboardSpawnerCfg]:
        """Create one :class:`KeyboardSpawnerCfg` per bank variant."""

        if self.cfg.count < 1:
            raise ValueError("KeyboardAssetBankCfg.count must be >= 1.")
        if not self.cfg.layout.families:
            raise ValueError("KeyboardAssetBankCfg.layout.families must contain at least one family.")
        spawners: list[keyboard_gen_cfg.KeyboardSpawnerCfg] = []
        for index in range(self.cfg.count):
            if self.cfg.layout.family == "random":
                family = self.cfg.layout.families[index % len(self.cfg.layout.families)]
            else:
                family = self.cfg.layout.family
            spawners.append(
                keyboard_gen_cfg.KeyboardSpawnerCfg(
                    layout=self.cfg.layout.replace(family=family),
                    style=self.cfg.style,
                    seed=self.cfg.seed,
                    variant_index=index,
                    topology_mode=self.cfg.topology_mode,
                    max_slots=self.cfg.max_slots,
                    bucket_sizes=self.cfg.bucket_sizes,
                    partition_mode=self.cfg.partition_mode,
                    partition_dof=self.cfg.partition_dof,
                    partition_tail_policy=self.cfg.partition_tail_policy,
                    use_tapered_keycaps=self.cfg.use_tapered_keycaps,
                    strict_validation=self.cfg.strict_validation,
                )
            )
        return spawners

    def resolve(self) -> list[ResolvedKeyboard]:
        """Resolve all bank variants to in-memory geometry."""

        return [resolve_keyboard(cfg) for cfg in self.make_spawner_cfgs()]


def keyboard_partition_prim_path(root_prim_path: str) -> str:
    """Return the IsaacLab articulation-view pattern for fixed-DOF keyboard parts."""

    return f"{root_prim_path.rstrip('/')}/parts/part_.*"


def make_keyboard_articulation_cfg(
    prim_path: str,
    spawn_cfg: keyboard_gen_cfg.KeyboardSpawnerCfg | None = None,
    *,
    actuator_name: str = "keys",
    joint_expr: str = "key_.*_joint",
    stiffness: float | None = None,
    damping: float | None = None,
    effort_limit: float | None = None,
    velocity_limit: float | None = None,
) -> ArticulationCfg:
    """Create an IsaacLab :class:`ArticulationCfg` for a single-articulation keyboard."""

    from isaaclab.actuators import ImplicitActuatorCfg  # noqa: PLC0415
    from isaaclab.assets import ArticulationCfg  # noqa: PLC0415

    if spawn_cfg is None:
        spawn_cfg = keyboard_gen_cfg.KeyboardSpawnerCfg()
    if spawn_cfg.partition_mode != "single":
        part_pattern = keyboard_partition_prim_path(prim_path)
        raise ValueError(
            "make_keyboard_articulation_cfg only supports partition_mode='single'. "
            "For fixed_dof, spawn the keyboard container separately and create "
            f"the articulation view at '{part_pattern}'."
        )
    resolved = resolve_keyboard(spawn_cfg)
    if stiffness is None:
        stiffness = resolved.style.stiffness
    if damping is None:
        damping = resolved.style.damping
    if effort_limit is None:
        effort_limit = resolved.style.max_force
    if velocity_limit is None:
        velocity_limit = 1.0
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={joint_expr: 0.0}, joint_vel={joint_expr: 0.0}),
        actuators={
            actuator_name: ImplicitActuatorCfg(
                joint_names_expr=[joint_expr],
                stiffness=stiffness,
                damping=damping,
                effort_limit=effort_limit,
                velocity_limit=velocity_limit,
            )
        },
    )


def summarize_keyboard(keyboard: ResolvedKeyboard) -> str:
    """Return a compact human-readable summary."""

    width, depth, height = keyboard.case_size
    warnings = [f"  - {msg}" for msg in keyboard.warnings] or ["  none"]
    lines = [
        f"family: {keyboard.family}",
        f"style: {keyboard.style_name}",
        f"base profile: {keyboard.style.base_profile}",
        f"deck tilt: {math.degrees(keyboard.style.deck_tilt):.1f} deg",
        f"seed: {keyboard.seed}",
        f"topology: {keyboard.topology_mode}",
        f"partition: {keyboard.partition_mode} x {keyboard.partition_count} @ {keyboard.partition_dof} dof",
        f"active keys: {keyboard.active_key_count}",
        f"slots: {keyboard.slot_count}",
        f"case size: {width:.3f} x {depth:.3f} x {height:.3f} m",
        f"travel: {keyboard.style.travel:.4f} m",
        "warnings:",
        *warnings,
    ]
    return "\n".join(lines)


def export_manifest_json(keyboard: ResolvedKeyboard, path: str) -> None:
    """Write a manifest JSON file for debugging."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(keyboard.manifest(), f, indent=2)
