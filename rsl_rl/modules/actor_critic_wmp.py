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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCriticWMP(nn.Module):
    is_recurrent = False    # 不是一个循环网络

    def __init__(self, num_actor_obs,   # 285 (53 + 33 + 12 + 187)
                 num_critic_obs,    # 285 (53 + 33 + 12 + 187)
                 num_actions,       # 12
                 encoder_hidden_dims=[256, 128],
                 wm_encoder_hidden_dims = [64, 32],  # [64, 64]
                 actor_hidden_dims=[256, 256, 256],  # [256, 128, 64]
                 critic_hidden_dims=[256, 256, 256], # [512, 256, 128]
                 activation='elu',
                 init_noise_std=1.0,
                 fixed_std=False,
                 latent_dim = 32,   # 32 + 3
                 height_dim=187,    # 187
                 privileged_dim=3 + 24, # 53
                 history_dim = 42*5,
                 wm_feature_dim = 1536, # 512
                 wm_latent_dim=16,      # 32
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCriticWMP, self).__init__()

        activation = get_activation(activation) # ELU

        self.latent_dim = latent_dim
        self.height_dim = height_dim
        self.privileged_dim = privileged_dim

        # latent vector + command + wm_latent = 35 + 3 + 32 = 70
        mlp_input_dim_a = latent_dim + 3 + wm_latent_dim
        # 285 + 32 = 317
        mlp_input_dim_c = num_critic_obs + wm_latent_dim

        # 1. history_encoder: 历史观测(42*5维) ==> 潜在特征(35维)
        #   Linear(42*5 -> 256) --> ELU --> Linear(256 -> 128) --> ELU --> Linear(128 -> 35)
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*encoder_layers)

        # 2. wm_feature_encoder: 世界模型特征(512维) ==> 潜在特征(32维)
        #   Linear(512 -> 64) --> ELU --> Linear(64 -> 32) --> ELU --> Linear(32 -> 32)
        wm_encoder_layers = []
        wm_encoder_layers.append(nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]))
        wm_encoder_layers.append(activation)
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
            else:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                wm_encoder_layers.append(activation)
        self.wm_feature_encoder = nn.Sequential(*wm_encoder_layers)

        # 3. critic_wm_feature_encoder: 世界模型特征(512维) ==> 潜在特征(32维)
        #   Linear(512 -> 64) --> ELU --> Linear(64 -> 32) --> ELU --> Linear(32 -> 32)
        critic_wm_encoder_layers = []
        critic_wm_encoder_layers.append(nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]))
        critic_wm_encoder_layers.append(activation)
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
            else:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                critic_wm_encoder_layers.append(activation)
        self.critic_wm_feature_encoder = nn.Sequential(*critic_wm_encoder_layers)

        # 4. actor (Policy): 历史观测潜在特征(35维) + command(3维) + 世界模型潜在特征(32维) ==> action(12维)
        #   Linear(35+3+32 -> 256) --> ELU --> Linear(256 -> 128) --> ELU --> Linear(128 -> 64) --> ELU --> Linear(64 -> 12)
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # 5. critic (Value function): 特权观测(285维) + 世界模型潜在特征(32维) ==> 状态价值估计(1维)
        #   Linear(285+32 -> 512) --> ELU --> Linear(512 -> 256) --> ELU --> Linear(256 -> 128) --> ELU --> Linear(128 -> 1)
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)



        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)  # (12,)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None    # action 分布
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations) # (num_envs, 12)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, history, wm_feature, **kwargs):
        """
        执行 Policy： 历史观测特征 + command + 世界模型特征 ==actor==> action，并根据 action 的分布采样得到 最终的 action
             observations 观测 (num_envs, 285)
             history      历史观测 (num_envs, 42*5)
             wm_feature   世界模型特征 (num_envs, 512)
        """
        latent_vector = self.history_encoder(history)   # 历史观测的潜在特征 (num_envs, 35)
        command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]  # command (num_envs, 3)
        wm_latent_vector = self.wm_feature_encoder(wm_feature)  # 世界模型潜在特征 (num_envs, 32)
        # 历史观测的潜在特征 + command + 世界模型潜在特征 (num_envs, 70)
        concat_observations = torch.concat((latent_vector, command, wm_latent_vector),
                                           dim=-1)
        # 执行 actor，更新 action 的分布
        self.update_distribution(concat_observations)
        return self.distribution.sample()   # 采样得到最终的 action (num_envs, 12)

    def get_latent_vector(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        return latent_vector

    def get_linear_vel(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        linear_vel = latent_vector[:,-3:]
        return linear_vel

    def get_actions_log_prob(self, actions):
        """
        计算 actions 的 对数概率 之和 (num_envs,)
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, wm_feature):
        """ 推理模式下的 action （直接返回 actor 的输出）"""
        latent_vector = self.history_encoder(history)
        command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.concat((latent_vector, command, wm_latent_vector),
                                           dim=-1)
        actions_mean = self.actor(concat_observations)
        return actions_mean

    def evaluate(self, critic_observations, wm_feature,  **kwargs):
        """
        状态价值评估： 特权观测 + 世界模型特征 ==critic==> value
        """
        wm_latent_vector = self.critic_wm_feature_encoder(wm_feature)   # 世界模型潜在特征 (num_envs, 32)
        # 特权观测 + 世界模型潜在特征 (num_envs, 285+32)
        concat_observations = torch.concat((critic_observations, wm_latent_vector),
                                           dim=-1)


        value = self.critic(concat_observations)  # (num_envs, 1)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None