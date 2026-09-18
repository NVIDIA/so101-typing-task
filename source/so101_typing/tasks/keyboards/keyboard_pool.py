# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO101 letter-typing keyboard pool: backend-specific spawner banks plus global slot metadata.

Bundles everything the typing env config consumes from the procedural keyboard generator so the env
config stays declarative (no bank-building, validation, or slot-derivation at module scope). The pool
is generated in BOTH partition modes because the two physics backends expose multi-articulation-per-env
data differently:

* PhysX (``fixed_dof``): 18 articulation roots/env under ``parts/part_*``. PhysX's ArticulationView
  flattens all per-env instances onto one axis, so every part's data is readable.
* Newton (``single``): one 108-DOF articulation/env. IsaacLab's Newton ``ArticulationData`` exposes only
  the first articulation per env (``[:, 0]``), so partitioned parts would be invisible; a single
  articulation keeps all 108 keys in that one instance.

Geometry is identical across modes - only the articulation root structure differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from .keyboard_gen import KeyboardAssetBank
from .keyboard_gen_cfg import KeyboardAssetBankCfg, KeyboardLayoutSamplerCfg, MXKeysKeyboardSpawnerCfg
from .keyboard_geometry import base_profile_visuals, resolve_keyboard

# Every SO101 typing keyboard is a full 108-key ``ansi_full`` board with no padding slots, so the whole
# heterogeneous pool can share one ArticulationView.
_EXPECTED_KEYBOARD_DOF = 108


@dataclass(frozen=True)
class TypingKeyboardPool:
    """Generated keyboard spawners plus the global key-slot metadata derived from them.

    Args:
        spawners_partitioned: ``fixed_dof`` spawner cfgs (PhysX: 18 articulation roots/env).
        spawners_single: ``single`` spawner cfgs (Newton: one 108-DOF articulation/env).
        typeable_slots: Global key slots that are alphabetic (sampled as typing targets).
        backspace_slot: Global key slot of the backspace key.
        active_slots: Global key slots backed by a real key (non-padding).
        numpad_slots: Global key slots belonging to the numeric keypad cluster (key name prefix
            ``num_``). Useful for restricting typing targets to the primary (non-numpad) key block.
        slot_labels: Per global slot legend label (``""`` for empty slots), for the debug visualization.
    """

    spawners_partitioned: tuple[SpawnerCfg, ...]
    spawners_single: tuple[SpawnerCfg, ...]
    typeable_slots: tuple[int, ...]
    backspace_slot: int
    active_slots: tuple[int, ...]
    numpad_slots: tuple[int, ...]
    slot_labels: tuple[str, ...]


def build_typing_keyboard_pool(
    count: int, seed: int = 0, partition_dof: int = 6, include_mx_keys: bool = True
) -> TypingKeyboardPool:
    """Build the typing keyboard pool in both partition modes and derive its global slot metadata.

    Generates ``count`` deterministic ``ansi_full`` variants, validates that every variant is a uniform
    108-key board with identical case-trim topology (required for the shared ArticulationView, and
    enforced strictly because Newton's shared view needs uniform per-env shape strides), then derives the
    global slot roles from a reference variant (all variants share the same logical layout).

    With ``include_mx_keys``, the remodeled Logitech MX Keys scan (see the ``convert_mx_keys`` module)
    joins the pool as one more variant. Its spawner authors the same articulation structure and per-body
    gprim counts as the generator, and its slot order is validated against the reference layout here so
    typing targets and labels stay correct on the scanned board.

    Args:
        count: Number of distinct procedural variants in the heterogeneous pool.
        seed: Base seed for deterministic variant sampling.
        partition_dof: DOF per articulation part in ``fixed_dof`` mode.
        include_mx_keys: Whether to append the converted MX Keys variant to both spawner banks.

    Returns:
        The fully-built :class:`TypingKeyboardPool`.

    Raises:
        ValueError: If any variant is not exactly 108 keys, variants disagree on case-trim count, or
            the MX Keys asset's slot labels do not match the reference layout.
    """
    bank_cfg = KeyboardAssetBankCfg(
        count=count,
        seed=seed,
        layout=KeyboardLayoutSamplerCfg(family="ansi_full"),
        topology_mode="exact",
        partition_dof=partition_dof,
    )
    spawners_partitioned = KeyboardAssetBank(bank_cfg.replace(partition_mode="fixed_dof")).make_spawner_cfgs()
    spawners_single = KeyboardAssetBank(bank_cfg.replace(partition_mode="single")).make_spawner_cfgs()

    # Validate uniform topology; geometry is partition-mode-independent, so one bank covers both.
    trim_counts: set[int] = set()
    for spawner in spawners_partitioned:
        resolved = resolve_keyboard(spawner)
        if resolved.active_key_count != _EXPECTED_KEYBOARD_DOF or resolved.slot_count != _EXPECTED_KEYBOARD_DOF:
            raise ValueError(
                f"Keyboard variant (family={resolved.family}, seed={resolved.seed}) resolved to "
                f"{resolved.active_key_count} active keys / {resolved.slot_count} slots; expected exactly "
                f"{_EXPECTED_KEYBOARD_DOF} with no padding."
            )
        trim_counts.add(len(base_profile_visuals(resolved)))
    if len(trim_counts) != 1:
        raise ValueError(
            f"Keyboard variants have non-uniform case-trim counts {sorted(trim_counts)}; the shared "
            "ArticulationView needs identical per-env shape topology (base_profile_visuals must pad to a "
            "constant count)."
        )

    ref = resolve_keyboard(spawners_partitioned[0])

    if include_mx_keys:
        from .keyboard_mx_keys import mx_keys_slot_labels  # noqa: PLC0415

        mx_partitioned = MXKeysKeyboardSpawnerCfg(partition_mode="fixed_dof", partition_dof=partition_dof)
        mx_single = MXKeysKeyboardSpawnerCfg(partition_mode="single", partition_dof=partition_dof)
        mx_labels = mx_keys_slot_labels(mx_partitioned.asset_dir)
        ref_labels = tuple(k.label for k in ref.keys)
        if mx_labels != ref_labels:
            mismatch = next(i for i, (a, b) in enumerate(zip(mx_labels, ref_labels)) if a != b)
            raise ValueError(
                f"MX Keys asset slot labels diverge from the reference layout at slot {mismatch} "
                f"({mx_labels[mismatch]!r} != {ref_labels[mismatch]!r}); regenerate the asset with "
                "convert_mx_keys."
            )
        spawners_partitioned.append(mx_partitioned)
        spawners_single.append(mx_single)

    label_by_slot = {k.slot: k.label for k in ref.keys}
    return TypingKeyboardPool(
        spawners_partitioned=tuple(spawners_partitioned),
        spawners_single=tuple(spawners_single),
        typeable_slots=tuple(k.slot for k in ref.keys if k.group == "alpha"),
        backspace_slot=next(k.slot for k in ref.keys if k.label.lower() == "backspace"),
        active_slots=tuple(k.slot for k in ref.keys if k.active),
        numpad_slots=tuple(k.slot for k in ref.keys if k.name.startswith("num_")),
        slot_labels=tuple(label_by_slot.get(slot, "") for slot in range(len(ref.keys))),
    )


# The concrete pool the SO101 letter-typing env config consumes: 32 diverse full 108-key procedural
# variants plus the remodeled Logitech MX Keys scan.
TYPING_KEYBOARD_POOL = build_typing_keyboard_pool(count=32)
