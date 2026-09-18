# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""No-sim profiler for procedural SO101 keyboard generation.

Examples:
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/keyboards/profile_keyboard_gen.py --count 32
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101/keyboards/profile_keyboard_gen.py \
        --workload usd --count 8 --cprofile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_labels import label_mesh_data
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_viser import draw_keyboard_grid_viser
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.inspect_keyboard_gen import resolve_inspector_keyboards
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_gen import spawn_keyboard
from isaaclab_tasks.core.dexsuite.config.so101.keyboards.keyboard_gen_cfg import (
    KeyboardLayoutSamplerCfg,
    KeyboardSpawnerCfg,
    KeyboardStyleSamplerCfg,
)


@dataclass
class WorkloadResult:
    name: str
    elapsed_s: float
    details: dict[str, Any]


class _FakeScene:
    def __init__(self):
        self.calls: Counter[str] = Counter()
        self.mesh_vertices = 0
        self.mesh_faces = 0

    def add_frame(self, *args, **kwargs):
        self.calls["frame"] += 1

    def add_box(self, *args, **kwargs):
        self.calls["box"] += 1

    def add_spline_catmull_rom(self, *args, **kwargs):
        self.calls["spline"] += 1

    def add_mesh_simple(self, name, vertices, faces, **kwargs):
        self.calls["mesh"] += 1
        self.mesh_vertices += len(vertices)
        self.mesh_faces += len(faces)


class _FakeServer:
    def __init__(self):
        self.scene = _FakeScene()


def _make_cfg(args: argparse.Namespace) -> KeyboardSpawnerCfg:
    return KeyboardSpawnerCfg(
        layout=KeyboardLayoutSamplerCfg(family=args.family),
        style=KeyboardStyleSamplerCfg(name=args.style),
        seed=args.seed,
        topology_mode=args.topology_mode,
        max_slots=args.max_slots,
        partition_mode=args.partition_mode,
        partition_dof=args.partition_dof,
        partition_tail_policy=args.partition_tail_policy,
        strict_validation=not args.allow_warnings,
    )


def _time(name: str, fn: Callable[[], dict[str, Any]]) -> WorkloadResult:
    start = time.perf_counter()
    details = fn()
    return WorkloadResult(name=name, elapsed_s=time.perf_counter() - start, details=details)


def _workload_resolve(args: argparse.Namespace) -> dict[str, Any]:
    keyboards = resolve_inspector_keyboards(_make_cfg(args), args.count)
    return {
        "keyboards": len(keyboards),
        "active_keys": sum(keyboard.active_key_count for keyboard in keyboards),
        "slots": sum(keyboard.slot_count for keyboard in keyboards),
        "families": sorted({keyboard.family for keyboard in keyboards}),
        "styles": sorted({keyboard.style_name for keyboard in keyboards}),
    }


def _workload_labels(args: argparse.Namespace) -> dict[str, Any]:
    keyboards = resolve_inspector_keyboards(_make_cfg(args), args.count)
    label_count = 0
    vertices = 0
    faces = 0
    for keyboard in keyboards:
        for key in keyboard.active_keys:
            key_vertices, key_faces = label_mesh_data(key, keyboard.style)
            if key_vertices:
                label_count += 1
                vertices += len(key_vertices)
                faces += len(key_faces)
    return {
        "keyboards": len(keyboards),
        "active_keys": sum(keyboard.active_key_count for keyboard in keyboards),
        "labels": label_count,
        "label_vertices": vertices,
        "label_quads": faces,
    }


def _workload_inspector_data(args: argparse.Namespace) -> dict[str, Any]:
    keyboards = resolve_inspector_keyboards(_make_cfg(args), args.count)
    server = _FakeServer()
    draw_keyboard_grid_viser(server, keyboards, spacing=args.spacing, show_press_guides=args.show_press_guides)
    return {
        "keyboards": len(keyboards),
        "active_keys": sum(keyboard.active_key_count for keyboard in keyboards),
        "draw_calls": dict(server.scene.calls),
        "mesh_vertices": server.scene.mesh_vertices,
        "mesh_faces": server.scene.mesh_faces,
    }


def _workload_usd(args: argparse.Namespace) -> dict[str, Any]:
    from pxr import Usd  # noqa: PLC0415

    cfg = _make_cfg(args)
    authored = 0
    prims = 0
    for index in range(args.count):
        variant_cfg = cfg.replace(variant_index=cfg.variant_index + index)
        stage = Usd.Stage.CreateInMemory()
        spawn_keyboard(f"/World/Keyboard_{index:03d}", variant_cfg, stage=stage)
        authored += 1
        prims += sum(1 for _ in stage.Traverse())
    return {"keyboards": authored, "usd_prims": prims}


_WORKLOADS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "resolve": _workload_resolve,
    "labels": _workload_labels,
    "inspector-data": _workload_inspector_data,
    "usd": _workload_usd,
}


def _run_workloads(args: argparse.Namespace) -> list[WorkloadResult]:
    names = list(_WORKLOADS) if args.workload == "all" else [args.workload]
    results: list[WorkloadResult] = []
    for name in names:
        best: WorkloadResult | None = None
        for _ in range(args.repeat):
            result = _time(name, lambda name=name: _WORKLOADS[name](args))
            if best is None or result.elapsed_s < best.elapsed_s:
                best = result
        assert best is not None
        results.append(best)
    return results


def _print_results(results: Sequence[WorkloadResult]) -> None:
    for result in results:
        print(f"{result.name}: {result.elapsed_s * 1000.0:.2f} ms")
        for key, value in result.details.items():
            print(f"  {key}: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("all", *tuple(_WORKLOADS)), default="all")
    parser.add_argument("--family", default="random")
    parser.add_argument("--style", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--topology-mode", choices=("exact", "bucketed", "global_padded"), default="exact")
    parser.add_argument("--max-slots", type=int, default=108)
    parser.add_argument("--partition-mode", choices=("single", "fixed_dof"), default="single")
    parser.add_argument("--partition-dof", type=int, default=6)
    parser.add_argument("--partition-tail-policy", choices=("pad_tail", "ragged"), default="pad_tail")
    parser.add_argument("--allow-warnings", action="store_true")
    parser.add_argument("--spacing", type=float, default=None)
    parser.add_argument("--show-press-guides", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--cprofile-top", type=int, default=30)
    parser.add_argument("--pstats-out", default=None)
    args = parser.parse_args(argv)

    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    if args.cprofile or args.pstats_out:
        profiler = cProfile.Profile()
        profiler.enable()
        results = _run_workloads(args)
        profiler.disable()
        if args.pstats_out:
            profiler.dump_stats(args.pstats_out)
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(args.cprofile_top)
        _print_results(results)
        print("\n[cProfile]")
        print(stream.getvalue())
    else:
        _print_results(_run_workloads(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
