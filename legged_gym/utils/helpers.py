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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    """
        args: 使用 args 中的 num_envs 更新后的 env_cfg
        cfg:  env_cfg 中包含的 sim 相关配置 的字典
    """
    # 初始化仿真器 Isaac Gym 的 模拟参数对象
    sim_params = gymapi.SimParams()

    # 根据args中的物理引擎类型设置不同参数
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # 设置PhysX物理引擎的 GPU使用 和 子场景数量
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes

    # 设置是否使用GPU流水线
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if "sim" in cfg:    # config中包含sim的相关参数，则解析并覆盖初始参数
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # 如果命令行指定了线程数，则覆盖PhysX的线程数设置
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:  # 加载最新的模型
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--terrain", "type": str, "default": "climb",
         "help": 'Only for play'},
        {"name": "--wm_device", "type": str, "default": "None", "help": 'World model device. Overrides config file in dreamer/config.yaml if provided'},

    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        # 创建actor_critic导出器
        class ActorWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.history_encoder = copy.deepcopy(model.history_encoder)
                self.wm_feature_encoder = copy.deepcopy(model.wm_feature_encoder)
                self.actor = copy.deepcopy(model.actor)

            def forward(self, observations, history, wm_feature):
                """ 推理模式下的 action （直接返回 actor 的输出）"""
                latent_vector = self.history_encoder(history)
                wm_latent_vector = self.wm_feature_encoder(wm_feature)

                command = observations[:, 53 + 6:53 + 9]
                concat_observations = torch.concat((latent_vector, command, wm_latent_vector),
                                                   dim=-1)
                actions_mean = self.actor(concat_observations)
                return actions_mean

        wrapped_actor = ActorWrapper(actor_critic)
        wrapped_actor.eval()

        obs = torch.randn((1, 285), device="cuda:0")
        history = torch.randn((1, 5*42), device="cuda:0")
        wm_feature = torch.randn((1, 512), device="cuda:0")

        traced_model = torch.jit.trace(wrapped_actor, (obs,history,wm_feature))
        model_path = os.path.join(path, 'policy.jit')
        traced_model.save(model_path)


def export_wm_as_jit(world_model, path):
    """导出世界模型为JIT格式
        Args:
            world_model: 要导出的世界模型
            path: 导出路径
        """
    os.makedirs(path, exist_ok=True)

    # 创建世界模型导出器
    class WorldModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = copy.deepcopy(model.encoder)
            self.dynamics = copy.deepcopy(model.dynamics)

        def forward(self, obs_dict, prev_stoch=None, prev_deter=None):
            if not isinstance(obs_dict, dict):
                raise ValueError("Input must be a dictionary")

            if "image" in obs_dict and obs_dict["image"] is not None:
                image = obs_dict["image"]
            else:
                image = None

            prop = obs_dict["prop"]
            is_first = obs_dict["is_first"]

            encoder_input = {
                "prop": prop.to("cuda"),
                "is_first": is_first.to("cuda"),
            }
            if image is not None:
                encoder_input["image"] = image.to("cuda")

            embed = self.encoder(encoder_input)
            latent, _ = self.dynamics.obs_step(None, None, embed, obs_dict["is_first"])
            return self.dynamics.get_deter_feat(latent)

    wrapped_model = WorldModelWrapper(world_model)
    wrapped_model.eval()

    dummy_input = {
        "prop": torch.randn((1, world_model.encoder.mlp_shapes["prop"][0]), device="cuda:0"),
        "is_first": torch.ones(1, device="cuda:0"),
    }
    if "image" in world_model.encoder.cnn_shapes:
        dummy_input["image"] = torch.randn((1, *world_model.encoder.cnn_shapes["image"]), device="cuda:0")

    traced_model = torch.jit.trace(wrapped_model, (dummy_input,))
    model_path = os.path.join(path, 'world_model.jit')
    traced_model.save(model_path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
