"""Reset-only fixture pose randomization; sampled pose is not an actor input."""
import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, subtract_frame_transforms


def reset_public_keyboard(env, env_ids, x_range_m, y_range_m, yaw_range_rad, letter_xyz_keyboard_m):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene['keyboard']
    pose = asset.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    ranges = torch.tensor([x_range_m, y_range_m, yaw_range_rad], device=env.device)
    sample = ranges[:, 0] + torch.rand((len(env_ids), 3), device=env.device)*(ranges[:, 1]-ranges[:, 0])
    pose[:, :2] += sample[:, :2]
    zeros = torch.zeros(len(env_ids), device=env.device)
    delta = quat_from_euler_xyz(zeros, zeros, sample[:, 2])
    pose[:, 3:7] = quat_mul(delta, pose[:, 3:7])
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(env_ids),6),device=env.device), env_ids=env_ids)
    local = torch.tensor(letter_xyz_keyboard_m, device=env.device).expand(len(env_ids), -1, -1)
    rotations = pose[:, None, 3:7].expand(-1,26,-1).reshape(-1,4)
    expected_w = quat_apply(rotations, local.reshape(-1,3)).reshape(-1,26,3)+pose[:,None,:3]
    robot_pose = env.scene['robot'].data.default_root_pose.torch[env_ids].clone()
    robot_pose[:,:3] += env.scene.env_origins[env_ids]
    expected_b, _ = subtract_frame_transforms(robot_pose[:,None,:3].expand(-1,26,-1).reshape(-1,3),
        robot_pose[:,None,3:7].expand(-1,26,-1).reshape(-1,4), expected_w.reshape(-1,3))
    if not hasattr(env, '_public_expected_key_xyz_b'):
        env._public_expected_key_xyz_b = torch.zeros((env.num_envs,26,3),device=env.device)
        env._public_keyboard_pose_samples = torch.zeros((env.num_envs,3),device=env.device)
    env._public_expected_key_xyz_b[env_ids] = expected_b.reshape(-1,26,3)
    env._public_keyboard_pose_samples[env_ids] = sample
