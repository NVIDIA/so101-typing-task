# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Vision actor / privileged critic configuration and teacher loading."""
import copy
import torch
from rsl_rl.models import MLPModel
from isaaclab_tasks.core.dexsuite.config.so101.agents.rsl_rl_ppo_cfg import (
    SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg,
)


def runner_config(seed=4811, learning_rate=1e-4):
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    cfg = handle_deprecated_rsl_rl_cfg(
        SO101PhysicalCalibrated20260718FixedCartesianP1DTransit15PPORunnerCfg().default, '5.4.1').to_dict()
    cfg.update(seed=seed, num_steps_per_env=32, save_interval=50, logger='tensorboard',
               obs_groups={'actor':['proprio','rgb'], 'critic':['policy']})
    cfg['actor'].update(class_name='CNNModel', cnn_cfg={
        'output_channels':[16,32,32], 'kernel_size':[5,3,3], 'stride':[2,2,2], 'activation':'relu'})
    cfg['algorithm'].update(learning_rate=learning_rate, schedule='fixed', num_learning_epochs=4)
    return cfg


def load_teacher(obs, path, device):
    config = copy.deepcopy(runner_config()['critic'])
    config.pop('class_name')
    config['distribution_cfg']={'class_name':'BetaDistribution','action_range':[-1.,1.]}
    teacher=MLPModel(obs, {'teacher':['policy']}, 'teacher',5,**config).to(device)
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    teacher.load_state_dict(checkpoint['actor_state_dict'])
    teacher.eval().requires_grad_(False)
    return teacher,checkpoint
