# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grid-bucket downsampling - a drop-in replacement for farthest-point sampling on inputs that are already
near-uniform in space. Used by the reset-curriculum buffer builder to downsample an oversampled pool of
candidate typing states to a spatially-uniform subset over a (weighted) feature space."""

from __future__ import annotations

import torch


def grid_bucket_downsample(
    pts: torch.Tensor,
    k: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Downsample ``pts`` to ~``k`` samples with uniform spatial coverage.

    Partitions the D-dimensional axis-aligned bounding box of ``pts`` into a grid with cell side
    ``(volume / k)^(1/D)`` and keeps one randomly-chosen candidate per non-empty cell. Gives Poisson-disk-like
    spacing at ``O(N log N)`` via a single sort - a drop-in replacement for farthest-point sampling on inputs
    that are already near-uniform, with none of the ``O(N * K)`` distance work.

    Works for any ``D >= 1``. Degenerate dimensions (e.g. all points on a single plane) collapse to a bucket
    size of zero and therefore stop contributing to the binning.

    Args:
        pts: Candidate positions, shape ``[N, D]``.
        k: Desired number of samples.
        generator: Optional device-local generator for randomized tie-breaking.

    Returns:
        Indices into ``pts`` of the chosen samples, shape ``[min(k, N)]``. When the bounding-box bucketing
        yields fewer than ``k`` non-empty cells, the shortfall is filled with random un-chosen points so the
        result is always exactly ``min(k, N)`` entries.
    """
    n, d = pts.shape
    if n <= k:
        return torch.arange(n, device=pts.device)

    mins = pts.amin(dim=0)
    maxs = pts.amax(dim=0)
    extents = (maxs - mins).clamp(min=1e-6)
    volume = extents.prod()

    # Oversample cells so empty buckets don't starve the sampler below ``k``. Cap at the available candidate
    # count, then random-subsample bucket survivors down to exactly ``k``.
    target_cells = min(n, max(int(k * 3.0), k + 1))
    cell_side = (volume / target_cells) ** (1.0 / d)

    bidx = ((pts - mins) / cell_side).long()  # [N, D]
    # Mixed-radix encode [N, D] cell indices into a single int64 per point.
    shape = bidx.amax(dim=0) + 1
    strides = torch.empty(d, dtype=torch.int64, device=pts.device)
    strides[-1] = 1
    for i in range(d - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    bucket_id = (bidx.to(torch.int64) * strides).sum(dim=-1)

    # Random integer priority - tie-break per bucket without fp noise.
    priority = torch.randint(0, 1 << 30, (n,), dtype=torch.int64, device=pts.device, generator=generator)
    key = bucket_id * (1 << 30) + priority
    perm = torch.argsort(key)
    bucket_sorted = bucket_id[perm]

    first_mask = torch.ones_like(bucket_sorted, dtype=torch.bool)
    first_mask[1:] = bucket_sorted[1:] != bucket_sorted[:-1]
    chosen = perm[first_mask]

    if chosen.numel() > k:
        sub = torch.randperm(chosen.numel(), device=pts.device, generator=generator)[:k]
        return chosen[sub]
    if chosen.numel() < k:
        # Grid collapsed below ``k`` cells - fill the remainder with random un-chosen points.
        chosen_mask = torch.zeros(n, dtype=torch.bool, device=pts.device)
        chosen_mask[chosen] = True
        remaining = (~chosen_mask).nonzero(as_tuple=False).squeeze(-1)
        need = k - chosen.numel()
        extra = remaining[torch.randperm(remaining.numel(), device=pts.device, generator=generator)[:need]]
        chosen = torch.cat([chosen, extra])
    return chosen
