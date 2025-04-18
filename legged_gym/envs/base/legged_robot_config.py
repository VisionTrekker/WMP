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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 235
        privileged_obs = True  # True：在 观测中obs 添加 特权信息
        privileged_dim = 24 + 3  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        height_dim = 187  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        num_privileged_obs = None # 不为 None：step()则返回 priviledge_obs_buf（用于非对称训练的 critic obs）；为 None： 返回其他
        num_actions = 12    # 机器人 action 维度，actions 为输出的 四肢的关节角度（按腿的顺序：FL, FR, RL, RR）
        env_spacing = 3.  # 每个环境的间距（heightfields/trimeshes 地形时不使用）
        send_timeouts = True # 向算法发送 超时信息
        episode_length_s = 20 # RL训练中每个 episode 的最大持续时间（s）
        reference_state_initialization = False # 从 参考数据 初始化状态，则创建 AMPLoader，用于加载 参考运动数据 以模仿学习

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 50 # [m]  change 25 to 50
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 0 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [wave, rough slope, stairs up, stairs down, discrete, rough_flat]
        terrain_proportions = [0.1, 0.1, 0.30, 0.25, 0.15, 0.1]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class depth:
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class asset:
        file = ""
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.25, 1.75]
        randomize_restitution = True
        restitution_range = [0, 1]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]  # kg
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_pos_range = [-0.05, 0.05]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        randomize_motor_strength = True

        motor_strength_range = [0.9, 1.1]
        randomize_action_latency = True
        latency_range = [0.00, 0.02]

    class rewards:
        reward_curriculum = True
        reward_curriculum_term = ["lin_vel_z"]
        reward_curriculum_schedule = [0, 1000, 1, 0]  #from iter 0 to iter 1000, decrease from 1 to 0
        class scales:
            termination = -0.0  # Episode终止惩罚：未启用。设为负值（如-10.0）可在跌倒时给予额外惩罚
            tracking_lin_vel = 1.0  # 奖励 跟踪XY方向线速度：控制实际线速度 与 commands速度的匹配度。权重最大，主导前进/后退训练
            tracking_ang_vel = 0.5  # 奖励 跟踪yaw方向角速度：控制实际转向 与 commands角速度的匹配度。若机器人转弯不稳定，可降低权重（如0.3）
            lin_vel_z = -2.0    # 惩罚 base的Z轴线速度：防止跳跃。若机器人跳跃频繁，增大惩罚（如-5.0）
            ang_vel_xy = 0      # 惩罚 base的roll/pitch角速度：抑制机身俯仰和横滚。跌倒时增大（如-0.2），默认 -0.05
            orientation = -0.   # 惩罚 base非水平姿态：抑制机身偏离水平面。地面不平时可减小（如-0.1）
            torques = -0.00001  # 惩罚 关节扭矩过大：防止关节扭矩过大导致过热或损坏，设为负值（如-1e-4）, A1: -0.0001
            dof_vel = -0.       # 惩罚 关节速度：抑制关节高速转动。若关节速度太快，设为负值（如-0.01）
            dof_acc = -2.5e-7   # 惩罚 关节加速度：抑制关节速度突变。若步态抖动，增大惩罚（如-1e-6）
            base_height = -0.   # 惩罚 base偏离目标高度：当高度低于 base_height_target 时触发
            feet_air_time =  1.0    # 奖励 四足在每个action首次触地的空中时间，使其接近0.5s（对应2Hz的步频）
            collision = -1.     # 惩罚 选定身体部位发生碰撞
            feet_stumble = -0.0 # 惩罚 四足在 gap 和 pit 地形的垂直表面打滑。若无法足部容易接触gap的垂直面，则设为负值（如-0.1）
            action_rate = -0.01 # 惩罚  action 在 相邻step 之间的差异。调大（如-0.05）可使运动更连续
            stand_still = -0.   # 惩罚 commands 为 0 时的 关节位置 与 默认关节位置的偏差

        only_positive_rewards = True # 负奖励保留：为 True 时，负总奖励裁剪到 0，避免早期训练频繁终止。复杂任务建议保持False
        tracking_sigma = 0.15   # 跟踪奖励的高斯分布 标准差，tracking reward = exp(-error^2 / sigma)
        soft_dof_pos_limit = 1. # 关节位置 软限位：关节角度超过 URDF 限位 100% 时触发惩罚。调低（如0.9）可提前约束
        soft_dof_vel_limit = 1. # 关节速度 软限位：超过最大速度 100% 时惩罚，保护电机模型不过载
        soft_torque_limit = 1.  # 关节力矩 软限位：超过额定扭矩 100% 时惩罚，防止仿真数值发散
        base_height_target = 1. # 机身目标高度（要低于 init height）
        max_contact_force = 100. # 选定身体部位判定发生碰撞时的 接触力阈值

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            # privileged
            height_measurements = 5.0
            contact_force = 0.005
            com_pos = 20
            pd_gains = 5
        clip_observations = 100.
        clip_actions = 6.0

        base_height = 0.5

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]，观察视角的位置
        lookat = [11., 5, 3.]  # [m]，观察视角的焦点

    class sim:
        dt =  0.002
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32
        # height_latent_dim = 16  # the encoder in teacher policy encodes the heightmap into a height_latent_dim vector
        # privileged_latent_dim = 8  # the encoder in teacher policy encodes the privileged infomation into a privileged_latent_dim vector
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # 每个 iteration 迭代 24 steps
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = 'trot'
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt