# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Vision task identifiers shared by simulator configuration and CPU deployment."""

NEWTON_VISION_TASK_CONTRACT = 'vision_ppo_gtc_berlin_6letters_ovrtx_v1'
PHRASE_VISION_TASK_CONTRACT = 'vision_ppo_gtc_berlin_random13_38keys_ovrtx_v2'
LEGACY_PHRASE_VISION_TASK_CONTRACT = 'vision_ppo_gtc_berlin_isaac_lab_3p0_ovrtx_v1'


def supports_vision_task(contract: str, goal_count: int) -> bool:
    supported = {
        26: (NEWTON_VISION_TASK_CONTRACT,),
        38: (PHRASE_VISION_TASK_CONTRACT, LEGACY_PHRASE_VISION_TASK_CONTRACT),
    }.get(goal_count, ())
    # Public task application appends the frozen task geometry fingerprint.
    return isinstance(contract, str) and contract.split('_task_', 1)[0] in supported
