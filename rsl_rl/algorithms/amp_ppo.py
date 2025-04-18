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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer


class AMPPPO:
    """AMP-PPO算法实现类，结合了PPO(Proximal Policy Optimization)和AMP(Adversarial Motion Priors)"""
    actor_critic: ActorCritic

    def __init__(self,
                 actor_critic,  # Actor-Critic网络，包含策略和价值函数
                 discriminator, # AMP判别器，用于区分专家数据和策略数据
                 amp_data,      # 参考运动捕捉数据(专家数据)
                 amp_normalizer,    # 数据标准化器
                 num_learning_epochs=1,  # 每次数据收集后的学习epoch数
                 num_mini_batches=1,     # 每个epoch中的mini-batch数量
                 clip_param=0.2,         # PPO的clip参数
                 gamma=0.998,            # 折扣因子
                 lam=0.95,               # GAE的lambda参数
                 value_loss_coef=1.0,    # 价值函数损失的权重
                 entropy_coef=0.0,       # 熵正则项的权重
                 vel_predict_coef=1.0,   # 速度预测损失的权重
                 learning_rate=1e-3,     # 学习率
                 max_grad_norm=1.0,      # 梯度裁剪的最大范数
                 use_clipped_value_loss=True,  # 是否使用clip的价值函数损失
                 schedule="fixed",       # 学习率调度方式
                 desired_kl=0.01,        # 期望的KL散度值
                 device='cpu',           # 计算设备
                 amp_replay_buffer_size=100000,  # AMP回放缓冲区大小
                 min_std=None,           # 动作标准差的最小值 (12,)
                 ):
        """初始化AMP-PPO算法"""

        # 初始化基本参数
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std  # 用于限制策略网络输出的最小标准差

        # 初始化AMP相关组件
        self.discriminator = discriminator  # 对抗判别器
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()  # 存储AMP过渡数据
        self.amp_storage = ReplayBuffer(  # AMP回放缓冲区
            discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data  # 专家数据
        self.amp_normalizer = amp_normalizer  # 数据标准化器

        # 初始化PPO组件
        self.actor_critic = actor_critic.to(device)  # 策略和价值网络
        self.storage = None  # 主回放缓冲区，稍后初始化

        # 优化器配置
        params = [
            {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
            {'params': self.discriminator.trunk.parameters(), 'weight_decay': 10e-4, 'name': 'amp_trunk'},
            {'params': self.discriminator.amp_linear.parameters(), 'weight_decay': 10e-2, 'name': 'amp_head'}]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.transition = RolloutStorage.Transition()  # 存储PPO过渡数据

        # 初始化PPO超参
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.vel_predict_coef = vel_predict_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape,
                     history_dim, wm_feature_dim):
        """初始化 回放缓冲区"""
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
            action_shape, history_dim=history_dim,
            wm_feature_dim=wm_feature_dim, device=self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, amp_obs, history, wm_feature):
        """
        根据 观察 生成 动作

        Args:
            obs: (num_envs, 285)
            critic_obs: (num_envs, 285)
            amp_obs: (num_envs, 30)
            history: 4个历史观测 + 当前观测 (num_envs, 5*42)
            wm_feature: 世界模型特征 ———— 当前时间步的 确定性状态 (num_envs, 512)

        Returns:
            action: (num_envs, 12)
        """
        if self.actor_critic.is_recurrent:  # 默认，False
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        # 存储 历史观测 和 世界模型确定性状态
        self.transition.history = history
        self.transition.wm_feature = wm_feature.detach()

        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()

        # 1. 执行 Policy: 历史观测特征 + command + 世界模型特征 ==actor==> actions
        self.transition.actions = self.actor_critic.act(aug_obs, history, wm_feature).detach()

        # self.actor_critic.eval()
        # 2. 状态价值评估：特权观测 + 世界模型特征 ==critic==> values
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs, wm_feature).detach()
        # self.actor_critic.train()

        # 3. 记录 action 高斯分布的 相关信息
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()  # action 高斯分布的 对数概率 之和 (num_envs,)
        self.transition.action_mean = self.actor_critic.action_mean.detach()  # action 高斯分布的 均值   (num_envs,)
        self.transition.action_sigma = self.actor_critic.action_std.detach()  # action 高斯分布的 标准差 (num_envs,)

        # 4. 存储 obs、critic_obs、amp_obs数据（在 env.step() 之前）
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        """处理环境步骤，存储过渡数据"""
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # 处理超时情况
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # 存储AMP数据
        not_done_idxs = (dones == False).nonzero().squeeze()
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)

        # 存储 主PPO数据
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, wm_feature):
        """计算回报(returns)"""
        aug_last_critic_obs = last_critic_obs.detach()
        # self.actor_critic.eval()
        last_values = self.actor_critic.evaluate(aug_last_critic_obs, wm_feature).detach()
        # self.actor_critic.train()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """执行 PPO 和 AMP 的 更新"""
        # 初始化损失统计
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_vel_predict_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0

        # 创建数据生成器
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # AMP数据生成器
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)

        # 遍历所有 mini-batch 进行训练
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            # 解包数据
            obs_batch, critic_obs_batch, actions_batch, history_batch, wm_feature_batch, \
            target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample

            # 计算新策略的动作概率和值函数
            aug_obs_batch = obs_batch.detach()
            self.actor_critic.act(aug_obs_batch, history_batch, wm_feature_batch, masks=masks_batch,
                                  hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, wm_feature_batch, masks=masks_batch,
                                                     hidden_states=hid_states_batch[1])

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # 自适应学习率调整(基于KL散度)
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    # 根据KL散度调整学习率
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # 计算 PPO 的 surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 计算 价值函数 损失
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 计算 线速度预测 损失
            predicted_linear_vel = self.actor_critic.get_linear_vel(aug_obs_batch, history_batch)
            target_linear_vel = aug_critic_obs_batch[:,
                            self.actor_critic.privileged_dim - 3: self.actor_critic.privileged_dim]
            vel_predict_loss = (predicted_linear_vel - target_linear_vel).pow(2).mean()

            # 计算 AMP判别器 损失
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            # 数据标准化
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            # 计算判别器输出
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

            # 计算对抗损失
            expert_loss = torch.nn.MSELoss()(
                expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(
                policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)

            # 计算梯度惩罚损失
            grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)

            # 计算总损失
            loss = (surrogate_loss +
                   self.vel_predict_coef * vel_predict_loss +
                   self.value_loss_coef * value_loss -
                   self.entropy_coef * entropy_batch.mean() +
                   amp_loss + grad_pen_loss)

            # 执行梯度下降
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # 限制 策略输出的最小标准差
            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            # 更新 标准化器
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            # 累计 损失统计
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            mean_vel_predict_loss += vel_predict_loss.mean().item()

        # 计算 平均损失
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_vel_predict_loss /= num_updates

        # 清空 回放缓冲区
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred
