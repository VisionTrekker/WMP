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

import time
import os
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import AMPPPO, PPO
from rsl_rl.modules import ActorCritic, ActorCriticWMP, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.modules import DepthPredictor
import torch.optim as optim

from dreamer.models import *
import ruamel.yaml as yaml
import argparse
import pathlib
import sys
import collections
from dreamer import tools
import datetime
import uuid
class WMPRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 history_length=5,
                 ):

        self.cfg = train_cfg["runner"]  # runner 相关的配置
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.depth_predictor_cfg = train_cfg["depth_predictor"]
        self.device = device
        self.env = env
        self.history_length = history_length

        # 1. 计算 Actor-Critic 的 观测维度
        if self.env.num_privileged_obs is not None:
            # 默认，critic 的 观测维度 = 特权观测维度，共285维 (53 + 33 + 12 + 187)，前 53 维是特权信息（包含3维base的线速度），中间 45 维是本体感知，后 187 维是特权观测中的 heightmap
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:   # 默认，actor 的 观测维度 = 285
            num_actor_obs = self.env.num_obs

        # 2. 创建 世界模型（包含 Encoder、RSSM、Decoder、奖励预测器）
        self._build_world_model()

        # 3. 创建 深度预测器
        self.depth_predictor = DepthPredictor().to(self._world_model.device)
        self.depth_predictor_opt = optim.Adam(self.depth_predictor.parameters(), lr=self.depth_predictor_cfg["lr"],
                                              weight_decay=self.depth_predictor_cfg["weight_decay"])
        # 4. RL
        # 历史观测维度
        self.history_dim = history_length * (self.env.num_obs - self.env.privileged_dim - self.env.height_dim - 3) # 5 * (总观测维度285 - 特权观测维度53 - 高度图维度187 - 3维的command(线速度x2 + 角速度) = 5 * 42

        # 4.1 Actor-Critic 网络（包含 history_encoder、wm_feature_encoder、critic_wm_feature_encoder、actor、critic）
        # 处理 世界模型特征、历史观测特征，然后结合 command 或 特权观测信息 进行 action 预测 和 状态价值估计
        actor_critic = ActorCriticWMP(num_actor_obs=num_actor_obs,
                                          num_critic_obs=num_critic_obs,
                                          num_actions=self.env.num_actions,
                                          height_dim=self.env.height_dim,
                                          privileged_dim=self.env.privileged_dim,
                                          history_dim=self.history_dim,
                                          wm_feature_dim=self.wm_feature_dim,
                                          **self.policy_cfg).to(self.device)

        # 4.2 对抗运动先验 AMP：用于提高运动自然性
        # 从预定义的运动文件中加载参考运动数据（动捕数据）
        amp_data = AMPLoader(
            device, time_between_frames=self.env.dt, preload_transitions=True,
            num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"])
        # 数据标准化器
        amp_normalizer = Normalizer(amp_data.observation_dim)
        # 判别器：用于区分策略生成运动与参考运动
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            train_cfg['runner']['amp_reward_coef'],
            train_cfg['runner']['amp_discr_hidden_dims'], device,
            train_cfg['runner']['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()

        # 4.3 PPO 算法初始化
        alg_class = eval(self.cfg["algorithm_class_name"])  # AMPPPO
        # 动作标准差的最小值：先按关节排序，再按腿排序 (12,)
        min_std = (
                torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
                (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))

        # 实例化 PPO 算法：整合所有组件进行端到端训练，设置动作标准差的最小值 (min_std) 防止过早收敛
        self.alg: PPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device,
                                  min_std=min_std, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]  # 每个 iteration 迭代 24 steps
        self.save_interval = self.cfg["save_interval"]

        # 初始化 storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions], self.history_dim, self.wm_feature_dim)

        # 5. Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # 6. 重置所有机器人，即执行一步 0 action 来获取 初始观测（obs, privileged_obs）
        _, _ = self.env.reset()


    def _build_world_model(self):
        """ 创建世界模型 """
        print('Begin construct world model')
        # 从 dreamer/configs.yaml 读取 网络模型配置
        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent.parent.parent / "dreamer/configs.yaml").read_text()
        )

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        # 2. 加载 configs.yaml 中的 defaults 配置到 dict 中
        name_list = ["defaults"]
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        # 再解析为 parser args
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--sim_device", default='cuda:0')
        parser.add_argument("--wm_device", default='None')
        parser.add_argument("--terrain", default='climb')
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        self.wm_config = parser.parse_args()

        # 允许 世界模型 和 RL env 在不同的 device 上
        if (self.wm_config.wm_device != 'None'):
            self.wm_config.device = self.wm_config.wm_device

        self.wm_config.num_actions = self.wm_config.num_actions * self.env.cfg.depth.update_interval    # 12 * 5

        prop_dim = self.env.num_obs - self.env.privileged_dim - self.env.height_dim - self.env.num_actions  # 本体感知维度 33
        image_shape = self.env.cfg.depth.resized + (1,) # (64, 64, 1)
        obs_shape = {'prop': (prop_dim,), 'image': image_shape,}

        # 3. 创建 世界模型（包含 Encoder、RSSM、Decoder、奖励预测器）
        self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=self.env.cfg.depth.use_camera)
        self._world_model = self._world_model.to(self._world_model.device)
        print('Finish construct world model')
        # 世界模型特征（确定性状态）维度，512
        self.wm_feature_dim = self.wm_config.dyn_deter #+ self.wm_config.dyn_stoch * self.wm_config.dyn_discrete


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        执行强化学习训练循环
            1. 初始化训练所需的各种组件和缓冲区
            2. 执行环境交互和数据收集
            3. 执行策略更新
            4. 训练世界模型和深度预测器
            5. 记录训练指标和保存模型
        Args:
            num_learning_iterations (int): 要执行的训练迭代次数
            init_at_random_ep_len (bool): 是否随机初始化 episode 长度, True
        """
        # 1. 初始化 TensorBoard 日志记录器
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # 2. 随机初始化 episode 长度（如果启用）
        if init_at_random_ep_len:   # True
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # 3. 获取 初始观测 数据（WMPRunner 初始化时，已进行过一次 0 action的 仿真，因此可以获取到初始观测）
        obs = self.env.get_observations()  # 策略观测 (num_envs, 285)
        privileged_obs = self.env.get_privileged_observations()  # 特权观测 (num_envs, 285)
        amp_obs = self.env.get_amp_observations()  # AMP观测 (num_envs, 30)
        critic_obs = privileged_obs if privileged_obs is not None else obs  # critic 观测 = 特权观测

        # 4. 将观测数据转移到指定设备
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)

        # 5. 设置 actor_critic 和 discriminator 为训练模式
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        # 6. 初始化 训练统计缓冲区
        ep_infos = []  # 存储 episode 信息
        rewbuffer = deque(maxlen=100)  # 奖励缓冲区（最近100个episode）
        lenbuffer = deque(maxlen=100)  # 长度缓冲区（最近100个episode）
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)      # 当前 episode 累计奖励 (num_envs,)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前 episode 长度 (num_envs,)

        # 7. 计算 总迭代次数
        tot_iter = self.current_learning_iteration + num_learning_iterations

        # 8. 初始化 轨迹历史 (num_envs, 5, 42)
        self.trajectory_history = torch.zeros(size=(self.env.num_envs, self.history_length,
                                                    self.env.num_obs - self.env.privileged_dim - self.env.height_dim - 3),
                                              device=self.device)

        # 获取 初始观测（ 本体感知(去除command) + action ）, (num_envs, 42)
        obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                            obs[:, self.env.privileged_dim + 9:-self.env.height_dim]), dim=1)
        # 9. 滑动窗口 更新 轨迹历史：丢弃最旧的历史观测、加入新的观测 (num_envs, 5, 42)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        # 10. 初始化 世界模型输入
        sum_wm_dataset_size = 0       # 世界模型 数据集大小
        wm_latent = None  # 世界模型 潜在状态，{"stoch": (num_envs, 32, 32), "deter": (num_envs, 512), "logit": (num_envs, 32, 32)}
        wm_action = None
        # prev_state (dict): 世界模型前一时间步潜在状态，
        wm_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)  # 标记是否为 episode 开始 (num_envs,)

        # 11. 构建 世界模型观测 dict
        wm_obs = {
            "prop": obs[:, self.env.privileged_dim:self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),  # 本体感知特征 (num_envs, 33)
            "is_first": wm_is_first,  # 为 1 表明是 episode 开始 (num_envs,)
        }

        if(self.env.cfg.depth.use_camera):  # 如果使用深度相机，则初始化深度图像缓冲区 (num_envs, 64, 64, 1)
            wm_obs["image"] = torch.zeros(((self.env.num_envs,) + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)

        wm_metrics = None  # 世界模型 训练指标
        self.wm_update_interval = self.env.cfg.depth.update_interval  # 世界模型更新间隔，5 个 timestep, 即 0.1s

        # 12. 初始化 一些参数
        wm_action_history = torch.zeros(size=(self.env.num_envs, self.wm_update_interval, self.env.num_actions), device=self._world_model.device)  # 世界模型 action历史 缓冲区 (num_envs, 5, 12)
        wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)  # 世界模型 奖励 (num_envs,)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim))  # 世界模型 当前时间步的 确定性状态 (num_envs, 512)

        # 13. 初始化 世界模型数据集 wm_dataset 和 wm_buffer 各分量为 0 tensor
        self.init_wm_dataset()

        # 循环训练
        for it in range(self.current_learning_iteration, tot_iter):
            # 14. 更新 奖励课程（如果启用）
            if (self.env.cfg.rewards.reward_curriculum):
                self.env.update_reward_curriculum(it)

            start = time.time()

            # 15. 环境交互 和 数据收集阶段
            with torch.inference_mode():  # 禁用 梯度计算 以提高性能
                # 15.1 每个 iteration 迭代 24 steps
                for i in range(self.num_steps_per_env):
                    print(f"[iter {it}] env step {i}")
                    # (1) 每 5 steps 更新 1 次世界模型的 状态
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        wm_embed = self._world_model.encoder(wm_obs)  # 世界模型观测 编码特征 (num_envs, 5120)

                        # 世界模型 潜在状态更新： 前一时间步状态 + 前一时间步action + 世界模型观测特征，预测 当前时间步状态（后验随机状态、先验确定性状态、状态分布参数）
                        # {"stoch": (num_envs, 32, 32), "deter": (num_envs, 512), "logit": (num_envs, 32, 32)}
                        wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"])

                        wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)  # 当前时间步的 确定性状态
                        wm_is_first[:] = 0  # 将起始标记 全置为 0，意为都不是 episode 开始

                    history = self.trajectory_history.flatten(1).to(self.device)  # (num_envs, 5*42)
                    # (2) 执行 PPO算法:
                    #   执行 Policy: 历史观测特征 + command + 世界模型特征 ==actor==> actions
                    #   状态价值评估：特权观测 + 世界模型特征 ==critic==> values
                    #   存储 obs、critic_obs、amp_obs数据（在 env.step() 之前）
                    actions = self.alg.act(obs, critic_obs, amp_obs, history, wm_feature.to(self.env.device))

                    # (3) 执行 env 仿真
                    #   执行一个control步（包含 4 个物理仿真步）
                    #   计算 奖励
                    #   计算 新的观测（privileged_obs_buf 和 obs_buf，都是 (num_envs, 285)）
                    #   重置某些env 获取它们的AMP观测
                    #   更新 depth_buffer
                    #   更新上一control步的数据（action、关节位置、关节速度、扭矩、base的线速度和角速度）
                    # 返回： obs、privileged_obs、所有env的 奖励之和、所有env的 重置标记、额外信息、需要重置的env的 ID、需要重置的env的 AMP观测
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)

                    # (4) 获取 所有env的 新的 AMP观测
                    next_amp_obs = self.env.get_amp_observations()

                    # TODO 19. 处理观测数据
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(
                        self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # 20. 更新世界模型输入
                    wm_action_history = torch.concat(
                        (wm_action_history[:, 1:], actions.unsqueeze(1).to(self._world_model.device)), dim=1)
                    wm_obs = {
                        "prop": obs[:, self.env.privileged_dim:self.env.privileged_dim + self.env.cfg.env.prop_dim]
                                 .to(self._world_model.device),
                        "is_first": wm_is_first,
                    }

                    # 21. 处理环境重置（将缓冲区数据存入数据集）
                    reset_env_ids = reset_env_ids.cpu().numpy()
                    if (len(reset_env_ids) > 0):
                        for k, v in self.wm_dataset.items():
                            if(k == "image"):
                                for id in reset_env_ids:
                                    idx_in_buffer = np.where(self.env.depth_index == id)[0]
                                    if(len(idx_in_buffer) > 0):
                                        v[idx_in_buffer, :] = self.wm_buffer[k][idx_in_buffer].to(self._world_model.device)
                            else:
                                v[reset_env_ids, :] = self.wm_buffer[k][reset_env_ids].to(self._world_model.device)

                        self.wm_dataset_size[reset_env_ids] = self.wm_buffer_index[reset_env_ids]
                        self.wm_buffer_index[reset_env_ids] = 0
                        sum_wm_dataset_size = np.sum(self.wm_dataset_size)

                        wm_action_history[reset_env_ids, :] = 0
                        wm_is_first[reset_env_ids] = 1

                    # 22. 更新世界模型动作和奖励
                    wm_action = wm_action_history.flatten(1)
                    wm_reward += rewards.to(self._world_model.device)

                    # 23. 存储当前步骤到缓冲区（按世界模型更新间隔）
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        if (self.env.cfg.depth.use_camera):
                            # 处理深度图像预测
                            forward_heightmap = self.env.get_forward_map().to(self._world_model.device)
                            pred_depth_image = self.depth_predictor(forward_heightmap, wm_obs["prop"])
                            wm_obs["image"] = pred_depth_image
                            self.wm_buffer["forward_height_map"][range(self.env.num_envs), self.wm_buffer_index,:] = forward_heightmap[:].to('cpu')
                            wm_obs["image"][self.env.depth_index] = infos["depth"].unsqueeze(-1).to(self._world_model.device)
                            self.wm_buffer["image"][range(self.env.cfg.depth.camera_num_envs),
                            self.wm_buffer_index[self.env.depth_index], :] = wm_obs["image"][self.env.depth_index].to('cpu')

                        # 存储非重置环境的数据
                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
                        if (len(not_reset_env_ids) > 0):
                            for k, v in wm_obs.items():
                                if(k != "is_first" and k != "image"):
                                    self.wm_buffer[k][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = v[not_reset_env_ids].to('cpu')
                            self.wm_buffer["action"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids]] = wm_reward[not_reset_env_ids].to('cpu')
                            self.wm_buffer_index[not_reset_env_ids] += 1

                        wm_reward[:] = 0  # 重置奖励

                    # 24. 处理AMP奖励
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states
                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    # 25. 更新轨迹历史
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.trajectory_history[env_ids] = 0
                    obs_without_command = torch.concat((
                        obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                        obs[:, self.env.privileged_dim + 9:-self.env.height_dim]), dim=1)
                    self.trajectory_history = torch.concat(
                        (self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

                    # 26. 记录训练统计信息
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                # 15.2 记录数据收集时间
                stop = time.time()
                collection_time = stop - start

                # 15.3 学习阶段 - 计算回报
                start = stop
                self.alg.compute_returns(critic_obs, wm_feature.to(self.env.device))

            # 16. 执行策略更新
            mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            # 17. 记录训练指标
            if self.log_dir is not None:
                self.log(locals())

            # 18. 定期保存模型
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

            ep_infos.clear()  # 清空episode信息

            # 19. 训练世界模型和深度预测器（如果数据集足够大）
            start_time = time.time()
            if (sum_wm_dataset_size > self.wm_config.train_start_steps):
                # 定期训练深度预测器
                if(it % self.depth_predictor_cfg["training_interval"] == 0):
                    depth_mse_loss = self.train_depth_predictor()
                    self.writer.add_scalar('DepthPredictor/loss', depth_mse_loss, it)

                # 训练世界模型
                wm_metrics = self.train_world_model()
                for name, values in wm_metrics.items():
                    self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)
            print('training world model time:', time.time() - start_time)

            # 20. 首次迭代时复制配置文件到日志目录
            if(it == 0):
                robot_name = self.cfg["experiment_name"].split("_")[0]
                file_name = self.cfg["experiment_name"].split("_")[0] + "_" + self.cfg["experiment_name"].split("_")[1]
                print(f"------ cp ./legged_gym/envs/{robot_name}/{file_name}_config.py ------")
                os.system(f"cp ./legged_gym/envs/{robot_name}/{file_name}_config.py " + self.log_dir + "/")

        # 21. 更新当前训练迭代次数并保存最终模型
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def init_wm_dataset(self):
        self.wm_dataset = {
            # (num_envs, 1000/5 + 3, 33)
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device=self._world_model.device),
            # (num_envs, 1000/5 + 3, 12*5)
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device=self._world_model.device),
            # (num_envs, 1000/5 + 3)
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device=self._world_model.device),
        }
        if(self.env.cfg.depth.use_camera):
            # (1024, 1000/5 + 3, 64, 64, 1)
            self.wm_dataset["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)
            # (num_envs, 1000/5 + 3, 525)
            self.wm_dataset["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device=self._world_model.device)

        self.wm_dataset_size = np.zeros(self.env.num_envs)

        self.wm_buffer = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device='cpu'),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device='cpu'),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device='cpu'),
        }
        if(self.env.cfg.depth.use_camera):
            self.wm_buffer["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device='cpu')
            self.wm_buffer["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device='cpu')

        self.wm_buffer_index = np.zeros(self.env.num_envs)

    def train_depth_predictor(self):
        total_mse_loss = 0
        for _ in range(self.depth_predictor_cfg["training_iters"]):
            batch_idx = np.random.choice(self.env.depth_index_without_crawl_tilt, self.depth_predictor_cfg["batch_size"],
                                         replace=True)
            time_index = [np.random.randint(0, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            forward_heightmap = self.wm_dataset["forward_height_map"][batch_idx, time_index]
            prop = self.wm_dataset["prop"][batch_idx, time_index]
            depth_image = self.wm_dataset["image"][self.env.depth_index_inverse[batch_idx], time_index]

            predict_depth_image = self.depth_predictor(forward_heightmap, prop)
            depth_predict_loss = (depth_image - predict_depth_image).pow(2).mean() * self.depth_predictor_cfg[
                "loss_scale"]
            # Gradient step
            self.depth_predictor_opt.zero_grad()
            depth_predict_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1)
            self.depth_predictor_opt.step()
            total_mse_loss += depth_predict_loss.detach() / self.depth_predictor_cfg["loss_scale"]
        return float(total_mse_loss / self.depth_predictor_cfg["training_iters"])

    def train_world_model(self):
        wm_metrics = {}
        mets = {}
        for i in range(self.wm_config.train_steps_per_iter):
            p = self.wm_dataset_size / np.sum(self.wm_dataset_size)
            batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, replace=True,
                                         p=p)
            batch_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.batch_length)
            if (batch_length <= 1):
                continue  # an error occur about the predict loss if batch_length < 1
            batch_end_idx = [np.random.randint(batch_length, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            batch_data = {}
            for k, v in self.wm_dataset.items():
                if (k == "forward_height_map"):
                    continue
                value = []
                for idx, end_idx in zip(batch_idx, batch_end_idx):
                    if (k == "image"):
                        idx_in_buffer = np.where(self.env.depth_index == idx)[0]
                        if (len(idx_in_buffer) == 0):
                            # not in the buffer, use the predicted ones
                            tmp_forward_heightmap = self.wm_dataset["forward_height_map"][idx,
                                                    end_idx - batch_length: end_idx]
                            tmp_prop = self.wm_dataset["prop"][idx, end_idx - batch_length: end_idx]
                            pred_depth_image = self.depth_predictor(tmp_forward_heightmap, tmp_prop)
                            value.append(pred_depth_image)
                        else:
                            value.append(v[idx_in_buffer[0], end_idx - batch_length: end_idx])
                    else:
                        value.append(v[idx, end_idx - batch_length: end_idx])
                value = torch.stack(value)
                batch_data[k] = value
            is_first = torch.zeros((self.wm_config.batch_size, batch_length))
            is_first[:, 0] = 1
            batch_data["is_first"] = is_first
            post, context, mets = self._world_model._train(batch_data)
        wm_metrics.update(mets)
        return wm_metrics

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/vel_predict', locs['mean_vel_predict_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/AMP_mean_policy_pred', locs['mean_policy_pred'], locs['it'])
        self.writer.add_scalar('Loss/AMP_mean_expert_pred', locs['mean_expert_pred'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Vel predict loss:':>{pad}} {locs['mean_vel_predict_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(), # AMPPPO中的 Actor-Critic网络参数
            'optimizer_state_dict': self.alg.optimizer.state_dict(),# AMPPPO中的 optimizer 参数
            'world_model_dict': self._world_model.state_dict(),
            'wm_optimizer_state_dict': self._world_model._model_opt._opt.state_dict(),
            'depth_predictor': self.depth_predictor.state_dict(),
            # 'discriminator_state_dict': self.alg.discriminator.state_dict(),
            # 'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True, load_wm_optimizer = False):
        """
        加载参数：
            AMPPPO中的 Actor-Critic网络参数、optimizer 参数
            世界模型参数
            训练的迭代次数
        """
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
        self._world_model.load_state_dict(loaded_dict['world_model_dict'], strict=False)
        if(load_wm_optimizer):
            self._world_model._model_opt._opt.load_state_dict(loaded_dict['wm_optimizer_state_dict'])
        # self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=False)
        # self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
