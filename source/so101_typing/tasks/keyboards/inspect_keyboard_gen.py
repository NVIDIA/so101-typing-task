# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""No-sim inspection utilities for procedural SO101 keyboards."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_geometry import resolve_keyboard
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_schema import (
    DEFAULT_FAMILIES,
    DEFAULT_MAX_SLOTS,
    DEFAULT_PARTITION_DOF,
    ResolvedKeyboard,
)
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_styles import compatible_styles_for_family
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_viser import draw_keyboard_grid_viser
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_gen import (
    KeyboardAssetBank,
    export_manifest_json,
    summarize_keyboard,
)
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_gen_cfg import (
    KeyboardAssetBankCfg,
    KeyboardLayoutSamplerCfg,
    KeyboardSpawnerCfg,
    KeyboardStyleSamplerCfg,
)



def run_viser_inspector(
    cfg: KeyboardSpawnerCfg,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    count: int = 1,
    spacing: float | None = None,
    show_press_guides: bool = False,
) -> None:
    """Launch a lightweight viser inspector for generated keyboards."""

    try:
        import viser  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "viser or one of its dependencies is unavailable. "
            "Install/fix viser or use the 'summary' CLI command."
        ) from exc

    keyboards = resolve_inspector_keyboards(cfg, count)
    server = viser.ViserServer(host=host, port=port)
    draw_keyboard_grid_viser(server, keyboards, spacing=spacing, show_press_guides=show_press_guides)
    print(f"[inspect_keyboard_gen] viser inspector running at http://{host}:{port}")
    print(f"[inspect_keyboard_gen] showing {len(keyboards)} keyboard variant(s)")
    print("[inspect_keyboard_gen] Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


def resolve_inspector_keyboards(cfg: KeyboardSpawnerCfg, count: int) -> tuple[ResolvedKeyboard, ...]:
    """Resolve deterministic visual-inspection variants without writing USD."""

    if count < 1:
        raise ValueError("inspect count must be >= 1.")
    return tuple(resolve_keyboard(inspector_variant_cfg(cfg, index)) for index in range(count))


def inspector_variant_cfg(cfg: KeyboardSpawnerCfg, index: int) -> KeyboardSpawnerCfg:
    """Create the cfg for one visual-inspection variant."""

    if cfg.layout.family == "random":
        families = cfg.layout.families or DEFAULT_FAMILIES
        family = families[index % len(families)]
        style_index = index // len(families)
    else:
        family = cfg.layout.family
        style_index = index
    style_name = cfg.style.name if cfg.style.name is not None else _inspector_style_for_family(family, style_index)
    return KeyboardSpawnerCfg(
        layout=cfg.layout.replace(family=family),
        style=cfg.style.replace(name=style_name),
        seed=cfg.seed,
        variant_index=cfg.variant_index + index,
        topology_mode=cfg.topology_mode,
        max_slots=cfg.max_slots,
        bucket_sizes=cfg.bucket_sizes,
        partition_mode=cfg.partition_mode,
        partition_dof=cfg.partition_dof,
        partition_tail_policy=cfg.partition_tail_policy,
        use_tapered_keycaps=cfg.use_tapered_keycaps,
        include_case_collision=cfg.include_case_collision,
        key_collision_margin=cfg.key_collision_margin,
        case_collision_margin=cfg.case_collision_margin,
        strict_validation=cfg.strict_validation,
    )


def _inspector_style_for_family(family: str, index: int) -> str:
    # The first random inspector cycle should already be visually diverse.
    preferred_by_family = {
        "ansi_60": ("mechanical_dark", "pastel_pop", "apple_chiclet", "retro", "terminal_green"),
        "ansi_tkl": ("cream_classic", "retro", "industrial_gray", "mechanical_dark", "logitech_slim"),
        "ansi_full": ("industrial_gray", "logitech_slim", "cream_classic", "apple_chiclet", "mechanical_dark"),
        "numpad": ("terminal_green", "mechanical_dark", "pastel_pop", "apple_chiclet", "retro"),
        "macro_pad": ("pastel_pop", "mono", "terminal_green", "mechanical_dark", "logitech_slim"),
        "ortholinear": ("mint_minimal", "laptop_chiclet", "mono", "mechanical_dark", "pastel_pop"),
        "split_columnar": ("terminal_green", "mint_minimal", "mechanical_dark", "logitech_slim", "mono"),
        "compact": ("apple_chiclet", "pastel_pop", "terminal_green", "laptop_chiclet", "mono"),
    }
    preferred_order = preferred_by_family.get(
        family,
        (
            "mechanical_dark",
            "retro",
            "industrial_gray",
            "pastel_pop",
            "logitech_slim",
            "apple_chiclet",
            "low_profile",
            "mono",
            "blank_conventional",
        ),
    )
    compatible = set(compatible_styles_for_family(family))
    ordered = [style for style in preferred_order if style in compatible]
    if not ordered:
        ordered = list(compatible_styles_for_family(family))
    return ordered[index % len(ordered)]


def _cfg_from_args(args: argparse.Namespace) -> KeyboardSpawnerCfg:
    return KeyboardSpawnerCfg(
        layout=KeyboardLayoutSamplerCfg(family=args.family),
        style=KeyboardStyleSamplerCfg(name=args.style),
        seed=args.seed,
        variant_index=args.variant_index,
        topology_mode=args.topology_mode,
        max_slots=args.max_slots,
        partition_mode=args.partition_mode,
        partition_dof=args.partition_dof,
        partition_tail_policy=args.partition_tail_policy,
        strict_validation=not args.allow_warnings,
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--family", default="random", choices=("random",) + DEFAULT_FAMILIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant-index", type=int, default=0)
    parser.add_argument("--style", default=None)
    parser.add_argument("--topology-mode", default="exact", choices=("exact", "global_padded", "bucketed"))
    parser.add_argument("--max-slots", type=int, default=DEFAULT_MAX_SLOTS)
    parser.add_argument("--partition-mode", default="single", choices=("single", "fixed_dof"))
    parser.add_argument("--partition-dof", type=int, default=DEFAULT_PARTITION_DOF)
    parser.add_argument("--partition-tail-policy", default="pad_tail", choices=("pad_tail", "ragged"))
    parser.add_argument("--allow-warnings", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for no-sim inspection and validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Print a generated keyboard summary.")
    _add_common_args(summary_parser)
    summary_parser.add_argument("--manifest-json", default=None, help="Optional path to write manifest JSON.")

    inspect_parser = subparsers.add_parser("inspect", help="Launch a viser inspector without writing USD.")
    _add_common_args(inspect_parser)
    inspect_parser.add_argument("--host", default="127.0.0.1")
    inspect_parser.add_argument("--port", type=int, default=8080)
    inspect_parser.add_argument("--count", type=int, default=12)
    inspect_parser.add_argument("--spacing", type=float, default=None)
    inspect_parser.add_argument("--show-press-guides", action="store_true")

    bank_parser = subparsers.add_parser("bank-summary", help="Print summaries for an in-memory variant bank.")
    bank_parser.add_argument("--count", type=int, default=8)
    bank_parser.add_argument("--seed", type=int, default=0)
    bank_parser.add_argument("--topology-mode", default="exact", choices=("exact", "global_padded", "bucketed"))
    bank_parser.add_argument("--max-slots", type=int, default=DEFAULT_MAX_SLOTS)
    bank_parser.add_argument("--partition-mode", default="single", choices=("single", "fixed_dof"))
    bank_parser.add_argument("--partition-dof", type=int, default=DEFAULT_PARTITION_DOF)
    bank_parser.add_argument("--partition-tail-policy", default="pad_tail", choices=("pad_tail", "ragged"))
    bank_parser.add_argument("--allow-warnings", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "summary":
        keyboard = resolve_keyboard(_cfg_from_args(args))
        print(summarize_keyboard(keyboard))
        if args.manifest_json:
            export_manifest_json(keyboard, args.manifest_json)
        return 0
    if args.command == "inspect":
        run_viser_inspector(
            _cfg_from_args(args),
            host=args.host,
            port=args.port,
            count=args.count,
            spacing=args.spacing,
            show_press_guides=args.show_press_guides,
        )
        return 0
    if args.command == "bank-summary":
        bank_cfg = KeyboardAssetBankCfg(
            count=args.count,
            seed=args.seed,
            layout=KeyboardLayoutSamplerCfg(),
            topology_mode=args.topology_mode,
            max_slots=args.max_slots,
            partition_mode=args.partition_mode,
            partition_dof=args.partition_dof,
            partition_tail_policy=args.partition_tail_policy,
            strict_validation=not args.allow_warnings,
        )
        for index, keyboard in enumerate(KeyboardAssetBank(bank_cfg).resolve()):
            print(f"--- variant {index} ---")
            print(summarize_keyboard(keyboard))
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
