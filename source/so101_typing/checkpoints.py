# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Safe loading for third-party RSL-RL checkpoint readers."""

import os


def load_runner_checkpoint(runner, path, **kwargs):
    """Load tensor checkpoints without allowing RSL-RL to enable pickle execution.

    RSL-RL 5.4.1 calls torch.load(weights_only=False). PyTorch's supported
    process-wide override takes precedence over that argument. Keep it enabled
    for subsequent loads and child processes; never retry with unsafe loading
    or automatically allowlist classes from an untrusted checkpoint.

    This restricts deserialization, not model behavior or resource consumption.
    TorchScript programs are executable artifacts and are a separate trust boundary.
    """
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
    return runner.load(path, **kwargs)
