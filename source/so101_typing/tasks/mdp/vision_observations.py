# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Camera observations and reset randomization for vision typing."""
import torch
import torch.nn.functional as F

from . import observations as state_obs
from .public_events import reset_public_keyboard
from .commands.fixed_cartesian_typing_commands import FixedCartesianTypingCommand
from .commands.fixed_cartesian_typing_commands_cfg import FixedCartesianTypingCommandCfg
from isaaclab.utils.configclass import configclass


class VisionTypingCommand(FixedCartesianTypingCommand):
    @property
    def target_xyz_b(self):
        letter = self._slot_to_letter[self.target_key_slot()].clamp(min=0)
        if hasattr(self._env, '_public_expected_key_xyz_b'):
            from isaaclab.utils.math import quat_apply
            expected_b = self._env._public_expected_key_xyz_b[torch.arange(self.num_envs, device=self.device), letter]
            # The historical teacher's frozen map is in environment/world axes,
            # despite the legacy *_b name. Keep that convention, including under DR.
            robot = self._env.scene['robot']
            return (quat_apply(robot.data.root_quat_w.torch, expected_b)
                    + robot.data.root_pos_w.torch - self._env.scene.env_origins)
        return self._letter_xyz_b[letter]

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        word = getattr(self._env, '_vision_eval_word', None)
        if word:
            from ..phrase_targets import MAPPED_CHARACTERS
            ids = slice(None) if env_ids is None else env_ids
            alphabet = MAPPED_CHARACTERS if len(self.cfg.letter_slots) == len(MAPPED_CHARACTERS) else tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            slots = [self.cfg.letter_slots[alphabet.index(c)] for c in word]
            self.target[ids, :len(word)] = torch.tensor(slots, device=self.device)
            self.target_len[ids] = len(word)
        return result


@configclass
class VisionTypingCommandCfg(FixedCartesianTypingCommandCfg):
    class_type: type = VisionTypingCommand


def reset_vision_domain(env, env_ids, letter_xyz_keyboard_m):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    strength = getattr(env, '_vision_curriculum', 0.0)
    height_strength = getattr(env, '_vision_height_curriculum', strength)
    reset_public_keyboard(env, env_ids, (-.005*strength, .005*strength),
                          (-.005*strength, .005*strength), (-.035*strength, .035*strength),
                          letter_xyz_keyboard_m, z_range_m=(-.002*height_strength, .002*height_strength))
    if not hasattr(env, '_vision_style'):
        env._vision_style = torch.ones((env.num_envs, 6), device=env.device)
        env._vision_image_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._vision_delay = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    n = len(env_ids)
    rand = torch.rand((n, 6), device=env.device)
    # Per-episode exposure, RGB balance, contrast and camera gain.
    env._vision_style[env_ids] = .8 + .4*rand
    env._vision_style[env_ids, 0] = .65 + .7*rand[:, 0]
    env._vision_image_valid[env_ids] = False
    env._vision_delay[env_ids] = torch.rand(n, device=env.device) < .25*strength
    if not hasattr(env, '_vision_affine'):
        env._vision_affine = torch.eye(2, 3, device=env.device).expand(env.num_envs,-1,-1).clone()
    env._vision_affine[env_ids] = torch.eye(2, 3, device=env.device)
    env._vision_affine[env_ids, 0, 0] = 1 + .03*(2*torch.rand(n,device=env.device)-1)
    env._vision_affine[env_ids, 1, 1] = env._vision_affine[env_ids, 0, 0]
    env._vision_affine[env_ids, :, 2] = .025*(2*torch.rand((n,2),device=env.device)-1)
    if strength:
        robot = env.scene['robot']
        if not hasattr(env, '_vision_nominal_gains'):
            env._vision_nominal_gains = (robot.data.joint_stiffness.torch.clone(), robot.data.joint_damping.torch.clone())
        for key, nominal, setter in zip(('stiffness','damping'), env._vision_nominal_gains,
                                   (robot.write_joint_stiffness_to_sim_index, robot.write_joint_damping_to_sim_index)):
            gains = nominal[env_ids] * (1 + strength*.1*(2*torch.rand_like(nominal[env_ids])-1))
            setter(**{key:gains}, env_ids=env_ids)


def vision_proprio(env):
    command = env.command_manager.get_term('typing')
    q = state_obs.fixed_joint_pos_rel_reset(env, 'typing')
    v = state_obs.fixed_joint_vel(env, 'typing')
    noise = float(getattr(env, '_vision_sensor_noise', 1.0))
    q = q + noise*.003*(2*torch.rand_like(q)-1)
    v = v + noise*.02*(2*torch.rand_like(v)-1)
    goal = command._slot_to_letter[command.target_key_slot()].clamp(min=0)
    return torch.cat((q, v, state_obs.fixed_joint_command_rel_reset(env, 'action', 'typing'),
                      F.one_hot(goal, len(command.cfg.letter_slots)).float(), state_obs.fixed_typing_phase_onehot(env, 'typing'),
                      state_obs.fixed_phase_elapsed_s(env, 'typing')), dim=-1)


def wrist_rgb_history(env):
    step = int(env.common_step_counter)
    if getattr(env, '_vision_image_step', None) == step and env._vision_image_valid.all():
        return env._vision_image_observation
    pixels = env.scene['camera_wrist'].data.output['rgb']
    if hasattr(pixels, 'torch'):
        pixels = pixels.torch
    rgb = pixels[..., :3].permute(0, 3, 1, 2).float() / 255.0
    if not hasattr(env, '_vision_style'):
        env._vision_style = torch.ones((env.num_envs, 6), device=env.device)
        env._vision_image_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._vision_delay = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, '_vision_affine'):
        grid = F.affine_grid(env._vision_affine, rgb.shape, align_corners=False)
        rgb = F.grid_sample(rgb, grid, padding_mode='border', align_corners=False)
    style = env._vision_style
    rgb = ((rgb-.5)*style[:, 4, None, None, None]+.5)*style[:, 0, None, None, None]
    rgb = rgb*style[:, 1:4, None, None]
    rgb = rgb + .008*torch.randn_like(rgb)
    # Mild blur, exposure drift and one-frame latency cover camera/USB variation.
    blurred = F.avg_pool2d(rgb, 3, stride=1, padding=1)
    rgb = torch.lerp(rgb, blurred, .15).clamp_(0, 1)
    previous = getattr(env, '_vision_previous_rgb', rgb)
    valid = env._vision_image_valid[:, None, None, None]
    previous = torch.where(valid, previous, rgb)
    delayed = torch.where((env._vision_delay & env._vision_image_valid)[:, None, None, None], previous, rgb)
    env._vision_image_observation = torch.cat((previous, delayed), dim=1)
    env._vision_previous_rgb = rgb.clone()
    env._vision_image_step = step
    env._vision_image_valid[:] = True
    return env._vision_image_observation
