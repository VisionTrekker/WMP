# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import math
import random

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from rsl_rl.datasets.motion_loader import AMPLoader
import cv2

COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.],
    [0.183, -0.047, 0.],
    [-0.183, 0.047, 0.],
    [-0.183, -0.047, 0.]]) + COM_OFFSET


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): 使用 args 中的 num_envs 更新后的 env_cfg
            sim_params (gymapi.SimParams): 更新后的 仿真器 Isaac Gym 的 模拟参数
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg

        # 1. 确定 各地形的 起止索引
        # 0 ~ 0% 的机器人在 wave
        self.wave_start_idx = 0
        self.wave_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:1]))
        # 0 ~ 5% (5%) 的机器人在 rough slope
        self.slope_start_idx = self.wave_end_idx
        self.slope_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:2]))
        # 5 ~ 20% (15%) 的机器人在 stairs up
        self.stairup_start_idx = self.slope_end_idx
        self.stairup_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:3]))
        # 20 ~ 35% (15%) 的机器人在 stairs down
        self.stairdown_start_idx = self.stairup_end_idx
        self.stairdown_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:4]))
        # 35 ~ 35% 的机器人在 discrete
        self.discrete_start_idx = self.stairdown_end_idx
        self.discrete_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:5]))
        # 35 ~ 60% (25%) 的机器人在 gap
        self.gap_start_idx = self.discrete_end_idx
        self.gap_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:6]))
        # 60 ~ 85% (25%) 的机器人在 pit 坑
        self.pit_start_idx = self.gap_end_idx
        self.pit_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:7]))
        # 85 ~ 90% (5%) 的机器人在 tilt
        self.tilt_start_idx = self.pit_end_idx
        self.tilt_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:8]))
        # 90 ~ 95% (5%) 的机器人在 crawl
        self.crawl_start_idx = self.tilt_end_idx
        self.crawl_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:9]))
        # 95 ~ 100% (5%) 的机器人在 rough flat
        self.roughflat_start_idx = self.crawl_end_idx
        self.roughflat_end_idx = self.cfg.env.num_envs

        self.sim_params = sim_params    # 仿真器 Isaac Gym 的 模拟参数（gymapi.SimParams类型）
        self.height_samples = None
        # for debug
        self.debug_viz = True
        self.lookat_id = 8

        # 2. ------ 初始化 ------
        self.init_done = False
        # 2.1 初始化RL训练中每个episode的总步数（1000步）、域随机化触发步数间隔（750步）、观测项的缩放、奖励项的缩放、控制指令的范围
        self._parse_cfg(self.cfg)
        # 2.2 调用父类 BaseTask 的初始化：
        #   获取 env_cfg 中的 envs个数、obs维度等
        #   创建 envs, sim, viewer
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # 2.3 创建深度图缩放器：将深度图使用 bicubic 缩放到 (64,64)
        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[0], self.cfg.depth.resized[1]),
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        # 2.4 设置观察视角
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # 2.5 创建存储 仿真 state / obs / action 的 tensor
        self._init_buffers()
        # 2.6 将使用的 奖励函数 存放到 self.reward_functions 中，并为每个奖励函数 创建一个(num_env,)的tensor，存储在episode中每个env的奖励累计值
        self._prepare_reward_function()
        self.init_done = True
        # ------ 初始化完成 ------

        self.global_counter = 0  # 全局步数计数器
        self.total_env_steps_counter = 0  # 总 env steps
        # domain randomization 中的 动作延迟范围 [0, 2]步（/ 物理仿真步长(0.005 s/步)，即从 秒 ==> 步数）
        self.latency_range = [int((self.cfg.domain_rand.latency_range[0] + 1e-8) / self.sim_params.dt),
                                 int((self.cfg.domain_rand.latency_range[1] - 1e-8) / self.sim_params.dt) + 1]

        # 若开启 课程学习，则从 每个课程计划中 提取 初始权重（reward_curriculum_schedule每项为 [start, end, start_value, end_value]）
        if self.cfg.rewards.reward_curriculum:
            self.reward_curriculum_coef = [schedule[2] for schedule in self.cfg.rewards.reward_curriculum_schedule]

        # 若开启 参考数据 初始化状态，则创建 AMPLoader，用于加载 参考运动数据 以模仿学习
        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

    def reset(self):
        """ 重置所有机器人 """
        # 1. 重置所有 env 的状态（位置、速度等）
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # 2. 如果需要包含历史观测步骤，重置 观测历史缓冲区
        if self.cfg.env.include_history_steps is not None:  # 默认值为 None
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])

        # 3. 执行一步 0 action 来获取 初始观测（obs_buf, privileged_obs_buf）
        obs, privileged_obs, _, _, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """
        执行 env step： 根据 Policy输出的 actions 并执行一步仿真

        Args:
            actions (torch.Tensor): (num_envs, 12)

        Returns:
            obs_bug: (num_envs, 285)
            privileged_obs_buf: (num_envs, 30)
            rew_buf: 所有env的 奖励之和 (num_envs,)
            reset_buf： 所有env的 重置标记 (num_envs,)
            extras： 额外信息，包含
                - episode: 回合统计信息（各奖励项均值等）
                - depth: 深度图像（如果启用相机）
                - time_outs: 超时标志
            reset_env_ids： 需要重置的env的 ID (num_envs_,)
            terminal_amp_states： 需要重置的env的 AMP观测 (num_envs_, 30)
        """
        # 1. 更新 全局步数计数器
        self.global_counter += 1
        self.total_env_steps_counter += 1

        # 2. 处理 actions
        # (1) 裁剪到合理范围 [-6.0, 6.0]
        clip_actions = self.cfg.normalization.clip_actions  # 6.0
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # (2) 计算 action 域随机化延迟步数 0/1/2
        rng = self.latency_range  # [0, 2]步
        action_latency = random.randint(rng[0], rng[1])

        # 3. 渲染环境（非headless模式）
        self.render()

        # 4. 执行一个 control 步（包含 4 个物理仿真步）
        for _ in range(self.cfg.control.decimation):
            # 从 actions 计算 扭矩 (num_envs, 12)
            if (self.cfg.domain_rand.randomize_action_latency and _ < action_latency):
                # 如果启用延迟，则在前3次中可能使用 上一control步的actions，在最后1次使用当前action
                self.torques = self._compute_torques(self.last_actions).view(self.torques.shape)
            else:
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            # 扭矩域随机化（扭矩 * 0.8-1.2的随机系数）
            if(self.cfg.domain_rand.randomize_motor_strength):
                rng = self.cfg.domain_rand.motor_strength_range  # [0.8, 1.2]
                self.torques = self.torques * torch_rand_float(rng[0], rng[1], self.torques.shape, device=self.device)

            # 应用 该扭矩 到 仿真环境
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)  # 执行物理仿真
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)  # 获取仿真结果
            self.gym.refresh_dof_state_tensor(self.sim)  # 更新 关节状态

        # 5. 执行 4个物理仿真步后的 操作
        # (1) 计算 奖励
        # (2) 计算 新的观测（privileged_obs_buf 和 obs_buf，都是 (num_envs, 285)）
        # (3) 重置某些env 获取它们的AMP观测
        # (4) 更新 depth_buffer
        # (5) 更新上一control步的数据（action、关节位置、关节速度、扭矩、base的线速度和角速度）
        #   返回： 需要重置的env的 ID (num_envs_,) 以及这些 env 的 AMP观测 (num_envs_, 30)
        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # 5.1 裁剪 观测obs_buf 到 [-100., 100.]
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        # 5.2 获取 策略观测policy_obs
        if self.cfg.env.include_history_steps is not None:
            # 如果启用历史观测，则将当前观测插入到历史缓冲区中，并加载 多个历史观测 作为策略观测
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:  # 默认
            policy_obs = self.obs_buf

        # 5.3 裁剪 特权观测privileged_obs_buf 到 [-100., 100.]
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # 5.4 更新 深度图像
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # 使用倒数第2帧（避免最新帧可能不完整）
            # interpolation = torch.rand((self.cfg.depth.camera_num_envs, 1, 1), device=self.device)
            # self.extras["depth"] = self.depth_buffer[:, -1] * interpolation + self.depth_buffer[:, -2] * (1-interpolation)
        else:
            self.extras["depth"] = None

        return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states


    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        # depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        # self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        start_time = time()
        self.gym.start_access_image_tensors(self.sim)
        # for i in range(self.num_envs):
        for i in range(len(self.depth_index)):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                                self.envs[self.depth_index[i]],
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)

            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            # if(i == 0): print(torch.mean(depth_image)) # for debug, sometimes isaacgym will return all -inf depth image if not config properly

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
                                                 dim=0)
        self.gym.end_access_image_tensors(self.sim)
        print('acquiring depth image time:', time()-start_time)


    def get_observations(self):
        """ 获取策略观测 """
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:  # 默认值为 None
            policy_obs = self.obs_buf
        return policy_obs

    def post_physics_step(self):
        """
        计算 奖励、观测、重置某些env 获取它们的AMP观测、更新depth_buffer、更新上一control步的数据（action、关节位置、关节速度、扭矩、base的线速度和角速度）

        Returns:
            env_ids: 需要重置的env的 ID (num_envs_,)
            terminal_amp_states: 需要重置的 env 的 AMP观测 (num_envs_, 30)
        """
        # 1. 从 Isaac Gym 仿真器中刷新各种状态张量，确保数据是最新的
        self.gym.refresh_actor_root_state_tensor(self.sim)   # 刷新 根状态 张量
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新 净接触力 张量
        self.gym.refresh_force_sensor_tensor(self.sim)       # 刷新 力传感器 张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)   # 刷新 刚体状态 张量

        # 2. 增加 当前回合的 步数计数器 和 通用步数计数器
        self.episode_length_buf += 1   # 当前回合的步数 +1
        self.common_step_counter += 1  # 通用步数计数器 +1

        # 3. 准备计算所需的量，更新机器人的姿态、速度和重力投影信息
        self.base_quat[:] = self.root_states[:, 3:7]  # 更新机器人 base 的旋转四元数
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])   # 更新机器人 base 的 线速度
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 更新机器人 base 的 角速度
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 更新投影到机器人坐标系的 重力向量

        # 原代码在计算奖励前调用 _post_physics_step_callback()，这可能不合理。
        # 例如，当前动作遵循当前命令，而 _post_physics_step_callback() 可能会重新采样命令，导致奖励较低。
        self._post_physics_step_callback()  # 调用回调函数进行通用计算，如重新采样命令、计算地形高度等

        # 4. 检查环境是否需要 重置
        self.check_termination()

        # 5. 计算 奖励
        self.compute_reward()

        # 6. 重置某些 env
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # 获取需要重置的 env ID
        # 获取需要重置的 env 的 AMP观测 (num_envs_, 30)
        terminal_amp_states = self.get_amp_observations()[env_ids]

        self.reset_idx(env_ids)  # 重置这些 env

        # 7. 更新 depth缓冲区
        self.update_depth_buffer()

        # 8. 重置环境后，机器人的姿态、速度和高度等信息可能发生变化，需要重新计算
        self.base_quat[:] = self.root_states[:, 3:7]  # 重新更新机器人 base 的旋转四元数
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 重新更新机器人 base 的线速度
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 重新更新机器人 base 的角速度
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 重新更新投影到机器人坐标系的重力向量

        # self._post_physics_step_callback()

        # 9. 计算 观测（特权观测privileged_obs_buf 和 观测obs_buf，都是 (num_envs, 285)）
        self.compute_observations()

        # 10. 更新上一control步的 actions、关节位置、关节速度、扭矩、base的线速度和角速度
        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # 11. 如果存在查看器，并且启用了查看器同步和调试可视化，则显示深度图像
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # self._draw_debug_vis()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口
                if len(self.depth_buffer) > 0:  # 显示深度图像
                    cv2.imshow("Depth Image", self.depth_buffer[0, -1].cpu().numpy() + 0.5) # [-0.5, 0.5] ==> [0, 1]
                cv2.waitKey(1)  # 等待 1 毫秒，处理窗口事件

        return env_ids, terminal_amp_states

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        vel_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        self.vel_violate = ((vel_error > 1.5) & (self.commands[:, 0] < 0.)) | ((vel_error < -1.5) & (self.commands[:, 0] > 0.))
        self.vel_violate *= (self.terrain_levels > 3)

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.vel_violate

        self.fall = (self.root_states[:, 9] < -3.) | (self.projected_gravity[:, 2] > 0.)
        self.reset_buf |= self.fall

    def reset_idx(self, env_ids):
        """
        重置指定 env ID 的 机器人状态
            env_ids (list[int]): 需要重置的 env_ID 列表
        """
        # 如果env_ids为空则直接返回
        if len(env_ids) == 0:
            return

        # 1. 更新地形课程（根据机器人表现调整地形难度）
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # 2. 更新命令课程（调整速度命令范围）
        # 避免每步都更新，因为最大命令对所有环境是共享的
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # 3. 重置机器人状态（关节和根状态）
        if self.cfg.env.reference_state_initialization:
            # 使用AMP参考运动初始化状态
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)  # 重置 关节位置 和 速度
            self._reset_root_states_amp(env_ids, frames)  # 重置 根状态
        else:
            # 使用默认方式初始化状态
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        # 4. 为重置的 env 重新采样 运动 commands
        self._resample_commands(env_ids)

        # 5. 域随机化 PD增益
        if self.cfg.domain_rand.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]  # P增益 (域随机化后的 * [0.8, 1.2])
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]  # D增益 (域随机化后的 * [0.8, 1.2])

        # 6. 重置各种缓冲区
        self.last_actions[env_ids] = 0.      # 上一动作
        self.last_last_actions[env_ids] = 0  # 上上一动作
        self.latency_actions[env_ids] = 0.   # 延迟动作
        self.last_dof_pos[env_ids] = 0   # 上一关节位置
        self.last_dof_vel[env_ids] = 0.  # 上一关节速度
        self.last_torques[env_ids] = 0.  # 上一扭矩
        self.feet_air_time[env_ids] = 0.      # 四足空中时间
        self.episode_length_buf[env_ids] = 0  # 当前episode长度
        self.reset_buf[env_ids] = 1  # 重置标志

        # 7. 记录episode信息
        self.extras["episode"] = {}
        # 计算并记录各奖励项的平均值
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.  # 重置奖励累计

        # 8. 记录课程信息
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            # 记录当前命令范围
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]

            self.extras["episode"]["max_command_flat_x"] = self.command_ranges["flat_lin_vel_x"][1]
            self.extras["episode"]["max_command_flat_yaw"] = self.command_ranges["flat_ang_vel_yaw"][1]

            self.extras["episode"]["push_interval_s"] = self.cfg.domain_rand.push_interval_s

        # 9. 发送超时信息给算法
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # reward curriculum
            if self.cfg.rewards.reward_curriculum:
                for j in range(len(self.cfg.rewards.reward_curriculum_term)):
                    if(name == self.cfg.rewards.reward_curriculum_term[j]):
                        rew *= self.reward_curriculum_coef[j]
                # print('reward:', name, ' coef:', self.reward_curriculum_coef)
            self.rew_buf += rew  # 累计 所有env的 奖励之和
            self.episode_sums[name] += rew  # 累计 当前episode的 所有env的 奖励之和
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]  # (num_envs,)
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


    def compute_observations(self):
        """
        计算并构建 obs_buf
            主要功能：
            1. 构建特权观测(privileged_obs_buf)，包含机器人状态和环境信息
            2. 根据配置添加各种感知输入
            3. 添加噪声(可选)
            4. 生成策略观测(policy observations)
        """
        # 1. 构建 基础的 特权观测 (privileged_obs_buf)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,  # base线速度 * 1.0
                                    self.base_ang_vel  * self.obs_scales.ang_vel,  # base角速度 * 0.25
                                    self.projected_gravity,  # 重力投影
                                    self.commands[:, :3] * self.commands_scale,  # commands * [1.0, 1.0, 0.25]
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节相对默认位置的偏移 * 1.0
                                    self.dof_vel * self.obs_scales.dof_vel,  # 关节速度 * 0.005
                                    self.actions  # actions
                                    ),dim=-1)  # (num_envs, 48)

        # 2. 如果启用特权观测，添加额外感知输入
        if (self.cfg.env.privileged_obs):  # 默认，启用特权观测
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:  # 默认，启用 地形高度测量 (num_envs, 187)
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

            if self.cfg.domain_rand.randomize_friction:  # 默认，启用 域随机化 静摩擦力 (num_envs, 1)
                self.privileged_obs_buf= torch.cat((self.randomized_frictions, self.privileged_obs_buf), dim=-1)

            if self.cfg.domain_rand.randomize_restitution:  # 默认，启用 域随机化 弹性系数 (num_envs, 1)
                self.privileged_obs_buf = torch.cat((self.randomized_restitutions, self.privileged_obs_buf), dim=-1)

            if (self.cfg.domain_rand.randomize_base_mass):  # 默认，启用 域随机化 base质量 (num_envs, 1)
                self.privileged_obs_buf = torch.cat((self.randomized_added_masses ,self.privileged_obs_buf), dim=-1)

            if (self.cfg.domain_rand.randomize_com_pos):  # 默认，启用 域随机化 质心位置 (num_envs, 3)
                self.privileged_obs_buf = torch.cat((self.randomized_com_pos * self.obs_scales.com_pos ,self.privileged_obs_buf), dim=-1)

            if (self.cfg.domain_rand.randomize_gains):  # 默认，启用 域随机化 PD增益 (num_envs, 12) + (num_envs, 12)
                self.privileged_obs_buf = torch.cat(((self.randomized_p_gains / self.p_gains - 1) * self.obs_scales.pd_gains ,self.privileged_obs_buf), dim=-1)
                self.privileged_obs_buf = torch.cat(((self.randomized_d_gains / self.d_gains - 1) * self.obs_scales.pd_gains, self.privileged_obs_buf),
                                                    dim=-1)
            # 添加 力传感器数据 (num_envs, 12)
            contact_force = self.sensor_forces.flatten(1) * self.obs_scales.contact_force
            self.privileged_obs_buf = torch.cat((contact_force, self.privileged_obs_buf), dim=-1)

            # 添加 指定关节是否发生碰撞的标志 (num_envs, 8)
            contact_flag = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1  # 每个env的指定关节的 接触力的模 > 0.1，则视为发生碰撞
            self.privileged_obs_buf = torch.cat((contact_flag, self.privileged_obs_buf), dim=-1)

        # 3. 添加噪声
        if self.add_noise:  # 默认 False
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec


        # 4. 生成 obs_buf（从特权观测中移除部分信息）
        if self.num_obs == self.num_privileged_obs - 6:  # 移除 线速度 和 角速度
            self.obs_buf = self.privileged_obs_buf[:, 6:]
        elif self.num_obs == self.num_privileged_obs - 3:  # 移除 线速度
            self.obs_buf = self.privileged_obs_buf[:, 3:]
        else:  # 默认，不移除任何信息
            self.obs_buf = torch.clone(self.privileged_obs_buf)

    def get_amp_observations(self):
        joint_pos = self.dof_pos
        # foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        # z_pos = self.root_states[:, 2:3]
        # if (self.cfg.terrain.measure_heights):
        #     z_pos = z_pos - torch.mean(self.measured_heights, dim=-1, keepdim=True)
        # return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)

    def get_full_amp_observations(self):
        joint_pos = self.dof_pos
        foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        pos = self.root_states[:, :3]
        if (self.cfg.terrain.measure_heights):
            pos[:, 2:3] = pos[:, 2:3] - torch.mean(self.measured_heights, dim=-1, keepdim=True)
        rot = self.root_states[:, 3:7]
        foot_vel = torch.zeros_like(foot_pos)
        return torch.cat((pos, rot, joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, foot_vel), dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            # if env_id==0:
            #     # prepare friction randomization
            #     friction_range = self.cfg.domain_rand.friction_range
            #     num_buckets = 64
            #     bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
            #     friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
            #     self.friction_coeffs = friction_buckets[bucket_ids]
            #
            # for s in range(len(props)):
            #     props[s].friction = self.friction_coeffs[env_id]
            rng = self.cfg.domain_rand.friction_range
            self.randomized_frictions[env_id] = np.random.uniform(rng[0], rng[1])
            for s in range(len(props)):
                props[s].friction = self.randomized_frictions[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            rng = self.cfg.domain_rand.restitution_range
            self.randomized_restitutions[env_id] = np.random.uniform(rng[0], rng[1])
            for s in range(len(props)):
                props[s].restitution = self.randomized_restitutions[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """
        处理并存储 关节属性，包括：位置限制、速度限制、力矩限制

        Args:
            props (numpy.array): 每个关节的属性数组，包含 位置/速度/力矩
            env_id (int): 当前环境ID，用于判断是否需要初始化限制参数

        Returns:
            [numpy.array]: 原始属性（未修改）
        """
        # 只在第一个环境初始化时设置关节限制
        if env_id==0:
            # 初始化存储关节限制的张量
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)  # (num_dof, 2)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # (num_dof,)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)   # (num_dof,)

            # 遍历每个关节属性
            for i in range(len(props)):
                # 存储原始关节限制
                self.dof_pos_limits[i, 0] = props["lower"][i].item()  # 最小 位置 限制
                self.dof_pos_limits[i, 1] = props["upper"][i].item()  # 最大 位置 限制
                self.dof_vel_limits[i] = props["velocity"][i].item()  # 最大 速度 限制
                self.torque_limits[i] = props["effort"][i].item()  # 最大 力矩 限制

                # 计算软限制（比硬件限制更宽松的范围）
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  # 中间值
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]  # 范围
                # 根据配置设置软限制范围（self.cfg.rewards.soft_dof_pos_limit 通常为0.9）
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props  # 返回原始属性（未修改）

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            added_mass = np.random.uniform(rng[0], rng[1])
            self.randomized_added_masses[env_id] = added_mass
            props[0].mass += added_mass

        # randomize com position
        if self.cfg.domain_rand.randomize_com_pos:
            rng = self.cfg.domain_rand.com_x_pos_range
            com_x_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,0] = com_x_pos
            rng = self.cfg.domain_rand.com_y_pos_range
            com_y_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,1] = com_y_pos
            rng = self.cfg.domain_rand.com_z_pos_range
            com_z_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,2] = com_z_pos
            props[0].com +=  gymapi.Vec3(com_x_pos,com_y_pos,com_z_pos)

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                props[i].mass = props[i].mass * np.random.uniform(rng[0], rng[1])

        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:self.roughflat_start_idx, 1], forward[:self.roughflat_start_idx, 0])
            self.commands[:self.roughflat_start_idx, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:self.roughflat_start_idx, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.measured_forward_heights = self._get_forward_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        # if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
        #     self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)


        #resample commands for rough flat terrain
        flat_env_ids = env_ids[torch.where(env_ids >= self.roughflat_start_idx)]
        if(len(flat_env_ids) > 0):
            self.commands[flat_env_ids, 0] = torch_rand_float(self.command_ranges["flat_lin_vel_x"][0],
                                                         self.command_ranges["flat_lin_vel_x"][1], (len(flat_env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[flat_env_ids, 1] = torch_rand_float(self.command_ranges["flat_lin_vel_y"][0],
                                                         self.command_ranges["flat_lin_vel_y"][1], (len(flat_env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[flat_env_ids, 2] = torch_rand_float(self.command_ranges["flat_ang_vel_yaw"][0],
                                                         self.command_ranges["flat_ang_vel_yaw"][1], (len(flat_env_ids), 1),
                                                         device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # set heading command for tilt envs to zero
        self.commands[self.tilt_start_idx:self.tilt_end_idx, 3] = 0
        self.commands[self.pit_start_idx:self.pit_end_idx, 3] = 0
        # self.commands[self.gap_start_idx:self.gap_end_idx, 3] = 0

    def _compute_torques(self, actions):
        """
        从 actions 计算 扭矩
            actions 根据控制模式，actions可被为当作： PD控制器 的 位置 / 速度目标 / 缩放后的扭矩
            注意：扭矩维度必须与自由度(DOF)数量相同，即使某些DOF未被驱动

        Args:
            actions (torch.Tensor): (num_evns, 12)
        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        # 获取PD增益参数
        if self.cfg.domain_rand.randomize_gains:  # 默认，如果启用了域随机化，使用随机化后的增益参数
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        # 根据控制类型计算扭矩
        if control_type=="P":  # 位置控制模式
            # 扭矩 = P增益 * (目标位置 - 当前位置) + D增益 * 当前速度            其中，目标位置 = actions * action_scale + default_dof_pos
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":  # 速度控制模式
            # 扭矩 = P增益 * (目标速度 - 当前速度) + D增益 * 加速度              其中，加速度 = (当前速度 - 上一时刻速度) / 物理时间步长
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":  # 扭矩控制模式
            # 直接使用 缩放后的action 作为 扭矩
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)  # 裁剪到扭矩限制范围内

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # the base y position of tilt and gap envs can not deviate too far from the origin center
        tilt_env_ids = env_ids[torch.where(env_ids >= self.tilt_start_idx)]
        tilt_env_ids = tilt_env_ids[torch.where(tilt_env_ids < self.tilt_end_idx)]
        gap_env_ids = env_ids[torch.where(env_ids >= self.gap_start_idx)]
        gap_env_ids = gap_env_ids[torch.where(gap_env_ids < self.gap_end_idx)]
        tilt_and_gap_env_ids = torch.concatenate((tilt_env_ids, gap_env_ids))

        if self.custom_origins:
            self.root_states[tilt_and_gap_env_ids] = self.base_init_state
            self.root_states[tilt_and_gap_env_ids, :3] += self.env_origins[tilt_and_gap_env_ids]
            self.root_states[tilt_and_gap_env_ids, :1] += torch_rand_float(-1., 1., (len(tilt_and_gap_env_ids), 1), device=self.device) # x position within 1m of the center
            self.root_states[tilt_and_gap_env_ids, 1:2] += torch_rand_float(-0.0, 0.0, (len(tilt_and_gap_env_ids), 1),
                                                               device=self.device)
        else:
            self.root_states[tilt_and_gap_env_ids] = self.base_init_state
            self.root_states[tilt_and_gap_env_ids, :3] += self.env_origins[tilt_and_gap_env_ids]

        # the base y position of gap env can not deviate too far from the origin center
        # gap_env_ids = env_ids[torch.where(env_ids >= self.gap_start_idx)]
        # gap_env_ids = gap_env_ids[torch.where(gap_env_ids < self.gap_end_idx)]
        # if self.custom_origins:
        #     self.root_states[gap_env_ids] = self.base_init_state
        #     self.root_states[gap_env_ids, :3] += self.env_origins[gap_env_ids]
        #     self.root_states[gap_env_ids, :1] += torch_rand_float(-1., 1., (len(gap_env_ids), 1), device=self.device) # x position within 1m of the center
        #     self.root_states[gap_env_ids, 1:2] += torch_rand_float(-0.0, 0.0, (len(gap_env_ids), 1),
        #                                                        device=self.device)
        # else:
        #     self.root_states[gap_env_ids] = self.base_init_state
        #     self.root_states[gap_env_ids, :3] += self.env_origins[gap_env_ids]

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. (瞬时的)
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy  # 获取最大推力线速度 [1m/s]
        # 给 base 在x/y线速度方向上添加随机推力
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots. (持续的)
        """
        # [-30, 30] N
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance  # 给 base 添加随机扰动
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)


    def update_reward_curriculum(self, current_iter):
        for i in range(len(self.cfg.rewards.reward_curriculum_schedule)):
            percentage = (current_iter - self.cfg.rewards.reward_curriculum_schedule[i][0]) / \
                         (self.cfg.rewards.reward_curriculum_schedule[i][1] - self.cfg.rewards.reward_curriculum_schedule[i][0])
            percentage = max(min(percentage, 1), 0)
            self.reward_curriculum_coef[i] = (1 - percentage) * self.cfg.rewards.reward_curriculum_schedule[i][2] + \
                                          percentage * self.cfg.rewards.reward_curriculum_schedule[i][3]

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above a certain percentage of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / (self.max_episode_length * self.reward_scales["tracking_lin_vel"]) +
                torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / (self.max_episode_length * self.reward_scales["tracking_ang_vel"]) > 1.65
                ):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.05,
                                                          -self.cfg.commands.max_lin_vel_backward_x_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_lin_vel_forward_x_curriculum)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.05,
                                                          -self.cfg.commands.max_lin_vel_y_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_lin_vel_y_curriculum)

            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.025,
                                                          -self.cfg.commands.max_ang_vel_yaw_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.025, 0.,
                                                          self.cfg.commands.max_ang_vel_yaw_curriculum)

            self.command_ranges["flat_lin_vel_x"][0] = np.clip(self.command_ranges["flat_lin_vel_x"][0] - 0.05,
                                                          -self.cfg.commands.max_flat_lin_vel_backward_x_curriculum, 0.)
            self.command_ranges["flat_lin_vel_x"][1] = np.clip(self.command_ranges["flat_lin_vel_x"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_flat_lin_vel_forward_x_curriculum)
            self.command_ranges["flat_lin_vel_y"][0] = np.clip(self.command_ranges["flat_lin_vel_y"][0] - 0.05,
                                                          -self.cfg.commands.max_flat_lin_vel_y_curriculum, 0.)
            self.command_ranges["flat_lin_vel_y"][1] = np.clip(self.command_ranges["flat_lin_vel_y"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_flat_lin_vel_y_curriculum)

            self.command_ranges["flat_ang_vel_yaw"][0] = np.clip(self.command_ranges["flat_ang_vel_yaw"][0] - 0.1,
                                                          -self.cfg.commands.max_flat_ang_vel_yaw_curriculum, 0.)
            self.command_ranges["flat_ang_vel_yaw"][1] = np.clip(self.command_ranges["flat_ang_vel_yaw"][1] + 0.1, 0.,
                                                          self.cfg.commands.max_flat_ang_vel_yaw_curriculum)

            self.cfg.domain_rand.push_interval_s = max(self.cfg.domain_rand.push_interval_s - 0.5, self.cfg.domain_rand.min_push_interval_s)
            self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_start_dim = self.privileged_dim - 3 # last 3-dim is the linear vel
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise  # False
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[noise_start_dim:noise_start_dim+3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[noise_start_dim+3:noise_start_dim+6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[noise_start_dim+6:noise_start_dim+9] = noise_scales.gravity * noise_level
        noise_vec[noise_start_dim+9:noise_start_dim+12] = 0. # commands
        noise_vec[noise_start_dim+12:noise_start_dim+24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[noise_start_dim+24:noise_start_dim+36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[noise_start_dim+36:noise_start_dim+48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[noise_start_dim+48:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # 从Isaac Gym仿真器中获取各种 state tensor
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # 根状态
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # 关节状态
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # 接触力
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # 刚体状态

        # 刷新这些张量以确保数据最新
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 将获取的原始张量包装成PyTorch张量
        self.root_states = gymtorch.wrap_tensor(actor_root_state)  # 根状态 (num_envs, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)  # 关节状态
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]  # 当前 关节位置 (num_env, 12, 1)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 当前 关节速度 (num_env, 12, 1)
        self.base_quat = self.root_states[:, 3:7]  # base 的旋转四元数（世界坐标系）

        # 存储每个刚体在xyz方向的接触力，(num_envs, num_bodies 17, 3)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # 获取并处理 力传感器 数据
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]  # 只取前3个分量 (num_envs, 4, 3)

        # 刚体 state 和 线速度
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 0:3] # (num_env, num_bodies, 3)
        self.rigid_body_lin_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[...,7:10] # (num_env, num_bodies, 3)


        # initialize some data used later on
        # 初始化计数器、额外数据、重力向量等
        self.common_step_counter = 0  # 步数计数器
        self.extras = {}  # 额外数据字典
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))    # 机器人的前进方向（base坐标系）
        # 初始化 torques
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # (num_envs, 12)
        # 初始化 PD增益
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # P增益 (num_envs, 12)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # D增益 (num_envs, 12)
        # 初始化 actions。TODO 机器人的 actions 为输出的 四肢的关节角度（按腿的顺序：FL, FR, RL, RR），(num_envs, 12)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros_like(self.last_actions)
        # for latency
        self.latency_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                           requires_grad=False)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # (num_envs, 4): x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.forward_height_points = self._init_forward_height_points()
        self.measured_heights = 0
        self.measured_forward_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self.compute_randomized_gains(self.num_envs)

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.cfg.depth.camera_num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)

        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)  # (num_evns, 17, 3)

    def compute_randomized_gains(self, num_envs):
        # (num_envs, 12) 的 0.8 - 1.2 的随机数
        p_mult = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1],
                                  (num_envs, self.num_actions), device=self.device)
        # (num_envs, 12) 的 0.8 - 1.2 的随机数
        d_mult = torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1],
                                  (num_envs, self.num_actions), device=self.device)
        return p_mult * self.p_gains, d_mult * self.d_gains


    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(l_up**2 + l_low**2 +
                                2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)

    def foot_positions_in_base_frame(self, foot_angles):
        foot_positions = torch.zeros_like(foot_angles)
        for i in range(4):
            foot_positions[:, i * 3:i * 3 + 3].copy_(
                self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1)**(i)))
        foot_positions = foot_positions + HIP_OFFSETS.reshape(12,).to(self.device)
        return foot_positions

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # 从所有奖励函数中 去除 env_cfg 中 rewards.scales 为 0 的项
        # 非0的各奖励函数的 scales * self.dt (0.02)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)  # reward 名称列表，没有 _reward_前缀
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # 当前episode的 所有env的 奖励之和：为 每个scale非0的 奖励函数 创建一个 tensor (num_env, )
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border_size
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[1]
            camera_props.height = self.cfg.depth.original[0]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_y_angle = np.random.uniform(config.y_angle[0], config.y_angle[1])

            camera_z_angle = np.random.uniform(config.z_angle[0], config.z_angle[1])
            camera_x_angle = np.random.uniform(config.x_angle[0], config.x_angle[1])


            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(np.radians(camera_x_angle),
                                                           np.radians(camera_y_angle), np.radians(camera_z_angle))
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print("[legged_robot] dof_names: ", self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:  # ["thigh", "calf"]
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])



        # use the sensor to acquire contact force, may be more accurate
        sensor_pose = gymapi.Transform()
        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)


        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        # for domain randomization
        self.randomized_frictions = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)  # 域随机化的 静摩擦力 (num_envs, 1)
        self.randomized_restitutions = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)  # 域随机化的 弹性系数 (num_envs, 1)
        self.randomized_added_masses = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)  # 域随机化的 base质量 (num_envs, 1)
        self.randomized_com_pos = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)       # 域随机化的 质心位置 (num_envs, 3)

        if(self.cfg.depth.use_camera):
            # All robots of Tilt and Crawl needs depth camera
            self.cfg.depth.camera_num_envs = min(self.cfg.depth.camera_num_envs, self.num_envs)
            self.depth_index_without_crawl_tilt = np.random.choice(range(self.tilt_start_idx), self.cfg.depth.camera_num_envs
                                                             - (self.crawl_end_idx - self.tilt_start_idx), replace=False)
            self.depth_index_without_crawl_tilt = np.sort(self.depth_index_without_crawl_tilt).astype(np.int)
            self.depth_index = np.concatenate((self.depth_index_without_crawl_tilt, range(self.tilt_start_idx, self.crawl_end_idx))).astype(np.int)
            self.depth_index_inverse = -np.ones(self.num_envs, dtype=np.int)
            for i in range(len(self.depth_index)):
                self.depth_index_inverse[self.depth_index[i]] = i

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            if(self.cfg.depth.use_camera and i in self.depth_index):
                self.attach_camera(i, env_handle, anymal_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)  # (num_envs, 8)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            # self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_levels = torch.fmod(torch.arange(self.num_envs, device=self.device), max_init_level + 1)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # 单个 control 的 仿真时间（s）= 控制解算器间隔（步） * 物理仿真步长（s/步） = 0.02 s
        self.obs_scales = self.cfg.normalization.obs_scales # 各观测值的 缩放系数
        self.reward_scales = class_to_dict(self.cfg.rewards.scales) # 各奖励项的 缩放系数
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)   # 各 command 的 范围
        # 非网格地形，则禁用 课程学习
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False

        self.max_episode_length_s = self.cfg.env.episode_length_s   # RL训练中每个 episode 的最大持续时间（s）
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)  # RL训练中每个 episode 的 control 仿真步数 = 20 / 0.02 = 1000 步
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt) # 域随机化的 触发步数间隔 = 15 / 0.02 = 750 步

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_forward_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_forward_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_forward_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_forward_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_forward_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_forward_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_forward_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_forward_height_points), self.forward_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_forward_height_points), self.forward_height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def get_forward_map(self):
        # 1. 计算相对高度：机器人base高度 - 基准高度 - 前方地形高度
        #   - self.root_states[:, 2].unsqueeze(1): 机器人base的Z坐标 (num_envs, 1)
        #   - self.cfg.normalization.base_height: 基准高度标量 (0.5)
        #   - self.measured_forward_heights: 前方地形高度测量值 (num_envs, 525)
        return torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_forward_heights, -1,
                             1.) * self.obs_scales.height_measurements

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # 惩罚 base的Z轴线速度（防止跳跃）
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # 惩罚 base的XY轴角速度（roll, pitch, 防止翻滚）
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # 惩罚 base非水平姿态
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # 惩罚 base偏离目标高度
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # 惩罚 关节扭矩过大（防止关节过热或损坏）
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_torques_distribution(self):
        # 惩罚 关节扭矩分布不均
        return torch.var(torch.abs(self.torques), dim=1)

    def _reward_dof_vel(self):
        # 惩罚 关节速度
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # 惩罚 关节加速度
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_pos_dif(self):
        # 惩罚 关节位置 在 相邻step 之间的差异
        return torch.sum(torch.square(self.last_dof_pos - self.dof_pos), dim=1)

    def _reward_action_rate(self):
        # 惩罚 action 在 相邻step 之间的差异（使机器人运动更加平滑连续）
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # 惩罚 指定关节的碰撞
        # 当指定关节接触力的 模 > 0.1N，则判定发生碰撞，计为 1
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)  # 每个env所有指定关节接触力之和 (num_envs,)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # 惩罚 关节位置接近极限
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # 惩罚 关节速度接近极限
        # 裁剪至 max error = 每个关节 1 rad/s，以避免 巨大惩罚
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # 惩罚 扭矩接近极限
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # 奖励 跟踪 commands 中XY方向的 线速度
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

        # clipping tracking reward
        lin_vel = self.base_lin_vel[:, :2].clone()
        # 实际速度的 上限/下限 = commands中XY方向线速度 分别 ± lin_vel_clip（0.1）
        lin_vel_upper_bound = torch.where(self.commands[:, :2] < 0, 1e5, self.commands[:, :2] + self.cfg.rewards.lin_vel_clip)
        lin_vel_lower_bound = torch.where(self.commands[:, :2] > 0, -1e5, self.commands[:, :2] - self.cfg.rewards.lin_vel_clip)
        # 对实际速度进行裁剪，限制在（commands速度 ± lin_vel_clip范围内）
        clip_lin_vel = torch.clip(lin_vel, lin_vel_lower_bound, lin_vel_upper_bound)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - clip_lin_vel), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)    # 误差 ==> [0,1]之间的奖励值

    def _reward_tracking_ang_vel(self):
        # 奖励 跟踪 commands 中yaw方向的 角速度
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # 奖励 四足的步频接近0.5s (2Hz)
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact = self.sensor_forces[:, :, 2] > 1.  # 检测z轴力 > 1N的接触
        self.contact_filt = torch.logical_or(contact, self.last_contacts)   # 当前帧和上一帧的接触状态 或运算（过滤接触信号，解决PhysX引擎在复杂地形上接触检测不可靠的问题）
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt   # 只考虑从空中首次触地的情况
        self.feet_air_time += self.dt   # 累加 control时间步长（0.02s）

        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # 计算与目标时间0.5s的偏差奖励
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 # 速度commands很小时不奖励
        self.feet_air_time *= ~self.contact_filt    # 接触时重置计时
        return rew_airTime

    def _reward_stand_still(self):
        # 惩罚 commands 速度接近0时（<0.1 m/s）的 关节位置 与 默认关节位置的 偏差
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # 惩罚 四足接触力过大（需<100）
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    # ------------newly added reward functions----------------
    def _reward_action_magnitude(self):
        # 限制 action 中的 髋关节hip（0,3,6,9）动作幅度（防止 > 1.0）
        return torch.sum(torch.square(torch.maximum(torch.abs(self.actions[:,[0,3,6,9]]) - 1.0,torch.zeros_like(self.actions[:,[0,3,6,9]]))), dim=1)
        # return torch.sum(torch.square(self.actions[:, [0, 3, 6, 9]]), dim=1)


    def _reward_power(self):
        # 惩罚 关节功率消耗（扭矩 * 关节速度）
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_power_distribution(self):
        # 惩罚 关节功率消耗分布不均
        return torch.var(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_smoothness(self):
        # 惩罚 action 的二阶平滑性（使动作更加平缓）
        return torch.sum(torch.square(self.last_last_actions - 2*self.last_actions + self.actions), dim=1)

    def _reward_clearance(self):
        # foot_pos = self.rigid_body_pos[:, self.feet_indices,:]
        #
        # foot_pos -= self.root_states[:,:3].unsqueeze(1)
        # foot_pos = torch.reshape(foot_pos,(self.num_envs * 4,-1))
        # foot_pos_base = quat_rotate_inverse(self.base_quat.repeat(4, 1), foot_pos)
        # foot_pos_base = torch.reshape(foot_pos_base,(self.num_envs,4,-1))
        # foot_heights = foot_pos_base[:,:,2]

        if self.cfg.terrain.mesh_type == 'plane':
            foot_heights = self.rigid_body_pos[:, self.feet_indices, 2]
        else:
            points = self.rigid_body_pos[:, self.feet_indices,:]

            #measure ground height under the foot
            points += self.terrain.cfg.border_size
            points = (points / self.terrain.cfg.horizontal_scale).long()
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

            heights1 = self.height_samples[px, py]
            heights2 = self.height_samples[px + 1, py]
            heights3 = self.height_samples[px, py + 1]
            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)

            ground_heights = torch.reshape(heights, (self.num_envs, -1)) * self.terrain.cfg.vertical_scale
            foot_heights = self.rigid_body_pos[:, self.feet_indices, 2] - ground_heights

        foot_lateral_vel = torch.norm(self.rigid_body_lin_vel[:, self.feet_indices,:2], dim = -1)
        # return torch.sum(foot_lateral_vel * torch.maximum(-foot_heights + self.cfg.rewards.foot_height_target, torch.zeros_like(foot_heights)), dim = -1)
        return torch.sum(foot_lateral_vel * torch.square(foot_heights - self.cfg.rewards.foot_height_target), dim = -1)


    def _reward_dof_error(self):
        # 惩罚 关节位置与默认位置的偏差
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_hip_pos(self):
        # 惩罚 髋关节hip（0,3,6,9）位置与默认位置的偏差
        return torch.sum(torch.square(self.dof_pos[:, [0,3,6,9]] - self.default_dof_pos[:, [0,3,6,9]]), dim=1)

    def _reward_delta_torques(self):
        # 惩罚 关节扭矩的变化率
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_cheat(self):
        # 惩罚 绕过障碍物的行为（只考虑 rough flat 之前的地形）
        forward = quat_apply(self.base_quat, self.forward_vec)  # 将 机器人前进方向 从 base坐标系 ==> 世界坐标系
        # 计算 yaw角（rad），只考虑 rough flat 之前的地形
        heading = torch.atan2(forward[:self.roughflat_start_idx, 1], forward[:self.roughflat_start_idx, 0])
        # 判定是否绕行：yaw角 绝对值 > 1.0 rad（57 度）
        cheat = (heading > 1.0) | (heading < -1.0)
        cheat_penalty = torch.zeros(self.num_envs, device=self.device)
        cheat_penalty[:self.roughflat_start_idx] = cheat
        return cheat_penalty

    def _reward_feet_edge(self):
        # 惩罚 四足在 gap 和 pit 地形的边缘落脚
        # 将 四足的世界坐标 转为 地形网格坐标（添加边界偏移并除以水平缩放系数）
        feet_pos_xy = ((self.rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices,
                        :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        # 边界裁剪
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        # 边缘检测，bool 表示每个足部是否位于 edge
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        # 只考虑 实际发生接触的 足部
        self.feet_at_edge = self.contact_filt & feet_at_edge
        # 只在 地形难度等级 > 3 才惩罚
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        # 只在 gap 和 pit 地形才计算惩罚
        edge_reward = torch.zeros_like(rew)
        edge_reward[self.gap_start_idx:self.pit_end_idx] = rew[self.gap_start_idx:self.pit_end_idx]
        return edge_reward

    def _reward_feet_stumble(self):
        # 惩罚 四足在 gap 和 pit 地形的垂直表面打滑
        # 判定是否打滑： XY方向 足部接触力 与 Z轴接触力 之比 > 4
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
             4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        rew = rew * (self.terrain_levels > 3)   # 只在 地形难度等级 > 3 才惩罚

        rew = rew.float()
        # 只在 gap 和 pit 地形才计算惩罚
        stumble_reward = torch.zeros_like(rew)
        stumble_reward[self.gap_start_idx:self.pit_end_idx] = rew[self.gap_start_idx:self.pit_end_idx]
        return stumble_reward

    def _reward_stuck(self):
        # 惩罚 卡住
        # 判定是否卡住：base 的 X方向线速度 < 0.1 m/s，但 commands 的 X方向线速度 > 0.1 m/s
        return (torch.abs(self.base_lin_vel[:, 0]) < 0.1) * (torch.abs(self.commands[:, 0]) > 0.1)
