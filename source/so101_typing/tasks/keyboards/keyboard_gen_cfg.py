# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cfg definitions for procedural SO101 keyboard generation.

This module intentionally contains only config classes and lightweight schema
exports. Runtime implementation lives in :mod:`keyboard_gen`; inspection and
profiling utilities live in separate modules.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_schema import (
    DEFAULT_BUCKET_SIZES,
    DEFAULT_FAMILIES,
    DEFAULT_MAX_SLOTS,
    DEFAULT_PARTITION_DOF,
    KeyboardFamily,
    PartitionMode,
    PartitionTailPolicy,
    TopologyMode,
)

MX_KEYS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mx_keys")
"""Default directory of the converted MX Keys asset (see the ``convert_mx_keys`` module)."""

if TYPE_CHECKING:
    from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_gen import KeyboardAssetBank
    from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_layouts import KeyboardLayoutSampler
    from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_styles import KeyboardStyleSampler


@configclass
class KeyboardLayoutSamplerCfg:
    """Configuration for selecting and building a logical keyboard layout."""

    class_type: type[KeyboardLayoutSampler] | str = "{DIR}.keyboard_layouts:KeyboardLayoutSampler"

    family: KeyboardFamily = "random"
    """Keyboard layout family. ``"random"`` samples from :attr:`families`."""

    families: tuple[str, ...] = DEFAULT_FAMILIES
    """Candidate families when :attr:`family` is ``"random"``."""


@configclass
class KeyboardStyleSamplerCfg:
    """Configuration for selecting a coherent keyboard visual/physics style."""

    class_type: type[KeyboardStyleSampler] | str = "{DIR}.keyboard_styles:KeyboardStyleSampler"

    name: str | None = None
    """Optional style archetype name. ``None`` samples a compatible style."""


@configclass
class KeyboardSpawnerCfg(SpawnerCfg):
    """Configuration for one procedurally generated keyboard.

    The cfg is pure data and import-safe. In ``partition_mode="single"`` it
    can be used directly as the ``spawn`` field of an
    :class:`isaaclab.assets.ArticulationCfg`. Fixed-DOF partitioning authors a
    container plus multiple articulation roots under ``parts/part_*``.
    """

    func: Callable | str = "{DIR}.keyboard_gen:spawn_keyboard"

    layout: KeyboardLayoutSamplerCfg = KeyboardLayoutSamplerCfg()
    """Logical keyboard layout selection cfg."""

    style: KeyboardStyleSamplerCfg = KeyboardStyleSamplerCfg()
    """Keyboard style selection cfg."""

    seed: int = 0
    """Seed for deterministic generation."""

    variant_index: int = 0
    """Optional variant offset used by asset-bank helpers."""

    topology_mode: TopologyMode = "exact"
    """Key-slot topology. Use ``exact`` for production; padded modes are ablation benchmarks."""

    max_slots: int = DEFAULT_MAX_SLOTS
    """Maximum key slots for the explicit ``global_padded`` ablation mode."""

    bucket_sizes: tuple[int, ...] = DEFAULT_BUCKET_SIZES
    """Canonical slot counts for the explicit ``bucketed`` padding ablation mode."""

    partition_mode: PartitionMode = "single"
    """Articulation partitioning mode: ``single`` or fixed-DOF parts under ``parts/part_*``."""

    partition_dof: int = DEFAULT_PARTITION_DOF
    """DOF per partition when ``partition_mode='fixed_dof'``."""

    partition_tail_policy: PartitionTailPolicy = "pad_tail"
    """Tail policy for fixed-DOF partitions: ``pad_tail`` or ``ragged``."""

    use_tapered_keycaps: bool = True
    """Use tapered visual meshes for keycaps. Collisions remain inset cuboids."""

    include_case_collision: bool = True
    """Whether the generated base/case should collide with the robot."""

    key_collision_margin: float = 0.0015
    """Inset applied to key collision boxes in meters."""

    case_collision_margin: float = 0.001
    """Inset applied to the case collision box in meters."""

    strict_validation: bool = True
    """Raise if the resolved keyboard has structural validation errors."""

    def resolved_seed(self) -> int:
        """Return the deterministic seed used for this concrete variant."""

        return int(self.seed) + int(self.variant_index)


@configclass
class MXKeysKeyboardSpawnerCfg(SpawnerCfg):
    """Configuration for the remodeled Logitech MX Keys keyboard (converted scan asset).

    Authors the same articulation structure as :class:`KeyboardSpawnerCfg` (same link/joint naming,
    drives, and per-body gprim counts) with the scan's segmented keycap and case meshes as visuals.
    The converted asset is produced offline by the ``convert_mx_keys`` module.
    """

    func: Callable | str = "{DIR}.keyboard_mx_keys:spawn_mx_keys_keyboard"

    asset_dir: str = MX_KEYS_DATA_DIR
    """Directory holding ``mx_keys_asset.npz`` and its ``textures/``."""

    partition_mode: PartitionMode = "single"
    """Articulation partitioning mode: ``single`` or fixed-DOF parts under ``parts/part_*``."""

    partition_dof: int = DEFAULT_PARTITION_DOF
    """DOF per partition when ``partition_mode='fixed_dof'`` (must divide 108)."""

    partition_tail_policy: PartitionTailPolicy = "pad_tail"
    """Tail policy for fixed-DOF partitions (the 108 slots always divide evenly)."""

    include_case_collision: bool = True
    """Whether the case collision slabs should collide with the robot."""

    key_collision_margin: float = 0.0015
    """Inset applied to key collision boxes in the cap plane [m]."""

    key_collision_thickness: float = 0.0035
    """Thickness of the key collision boxes [m]; the scan's caps are thin open shells, so the
    collider extends below the visual to give the fingertip a solid press target."""

    case_collision_margin: float = 0.002
    """Footprint inset of the lower case collision slab [m] (hides it inside the rounded shell)."""

    travel: float = 0.0018
    """Key press travel [m]; the physical MX Keys travel is 1.8 mm."""

    rest_upper_limit: float = 1.0e-4
    """Upper prismatic joint limit [m] (rest pose sits at the drive target 0)."""

    key_mass: float = 0.008
    """Keycap mass [kg]."""

    base_mass: float = 0.81
    """Keyboard body mass [kg] (the physical MX Keys weighs 810 g)."""

    stiffness: float = 60.0
    """Key return-spring stiffness [N/m] (raised at authoring time to hold the cap against gravity)."""

    damping: float = 2.5
    """Key return-spring damping [N*s/m]."""

    max_force: float = 4.0
    """Key drive force limit [N]."""

    strict_validation: bool = True
    """Raise if the resolved keyboard has structural validation errors."""


@configclass
class KeyboardAssetBankCfg:
    """Configuration for a deterministic in-memory bank of keyboard variants."""

    class_type: type[KeyboardAssetBank] | str = "{DIR}.keyboard_gen:KeyboardAssetBank"

    count: int = 1
    """Number of variants in the bank."""

    seed: int = 0
    """Base seed for generated variants."""

    layout: KeyboardLayoutSamplerCfg = KeyboardLayoutSamplerCfg()
    """Logical keyboard layout selection cfg for generated variants."""

    style: KeyboardStyleSamplerCfg = KeyboardStyleSamplerCfg()
    """Keyboard style selection cfg for generated variants."""

    topology_mode: TopologyMode = "exact"
    """Topology mode passed to each :class:`KeyboardSpawnerCfg`; padded modes are ablation-only."""

    max_slots: int = DEFAULT_MAX_SLOTS
    """Maximum key slots for the explicit ``global_padded`` ablation mode."""

    bucket_sizes: tuple[int, ...] = DEFAULT_BUCKET_SIZES
    """Bucket sizes for the explicit ``bucketed`` padding ablation mode."""

    partition_mode: PartitionMode = "single"
    """Articulation partitioning mode passed to each :class:`KeyboardSpawnerCfg`."""

    partition_dof: int = DEFAULT_PARTITION_DOF
    """DOF per partition when ``partition_mode='fixed_dof'``."""

    partition_tail_policy: PartitionTailPolicy = "pad_tail"
    """Tail policy for fixed-DOF partitions."""

    use_tapered_keycaps: bool = True
    """Use tapered keycap visuals."""

    strict_validation: bool = True
    """Raise on validation errors while resolving variants."""
