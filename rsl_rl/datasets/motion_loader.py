# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for processing motion clips."""

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations

from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util


# 运动数据加载器类，用于处理运动捕捉数据
class AMPLoader:
    # 定义各运动参数的 维度大小
    POS_SIZE = 3  # 位置(x,y,z)
    ROT_SIZE = 4  # 旋转(四元数)
    JOINT_POS_SIZE = 12  # 关节位置
    TAR_TOE_POS_LOCAL_SIZE = 12  # 足端目标位置
    LINEAR_VEL_SIZE = 3  # 线速度
    ANGULAR_VEL_SIZE = 3  # 角速度
    JOINT_VEL_SIZE = 12  # 关节速度
    TAR_TOE_VEL_LOCAL_SIZE = 12  # 足端目标速度

    # 定义各运动参数在数据数组中的索引范围
    ROOT_POS_START_IDX = 0  # 根位置 起始索引
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE  # 根位置 结束索引

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX  # 根旋转
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX  # 关节位置
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX  # 足端目标位置
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX  # 线速度
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX  # 角速度
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX  # 关节速度
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX  # 足端目标速度
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE

    def __init__(
            self,
            device,
            time_between_frames,    # 帧间 时间间隔(秒)
            data_dir='',            # 数据目录路径
            preload_transitions=False,  # 是否预加载转换数据
            num_preload_transitions=1000000,    # 预加载的转换数据数量
            motion_files=glob.glob('datasets/motion_files2/*'), # 运动数据文件列表
            ):
        """
        运动数据加载器：加载专家数据集数据（提供了 来自狗的动捕数据的 AMP观测结果）
        """
        self.device = device
        self.time_between_frames = time_between_frames
        
        # 初始化存储各轨迹数据的列表
        self.trajectories = []      # 存储处理后的轨迹数据(不含根位置和旋转)
        self.trajectories_full = [] # 存储完整的轨迹数据
        self.trajectory_names = []  # 轨迹名称列表
        self.trajectory_idxs = []   # 轨迹索引列表
        self.trajectory_lens = []   # 轨迹长度(秒)
        self.trajectory_weights = []  # 轨迹权重
        self.trajectory_frame_durations = []  # 每帧持续时间
        self.trajectory_num_frames = []  # 每轨迹帧数

        # 加载每个运动文件
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                # 仅在使用真实动物数据时需要重排
                # motion_data = self.reorder_from_pybullet_to_isaac(motion_data)

                # 标准化和规范化 四元数
                for f_i in range(motion_data.shape[0]):
                    root_rot = AMPLoader.get_root_rot(motion_data[f_i])
                    # root_rot = pose3d.QuaternionNormalize(root_rot)
                    # root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[
                        f_i,
                        AMPLoader.POS_SIZE:
                            (AMPLoader.POS_SIZE +
                             AMPLoader.ROT_SIZE)] = root_rot
                
                # 移除前7个观测维度 (根位置 和 旋转)
                self.trajectories.append(torch.tensor(
                    motion_data[
                        :,
                        AMPLoader.ROOT_ROT_END_IDX:AMPLoader.JOINT_VEL_END_IDX
                    ], dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(
                        motion_data[:, :AMPLoader.JOINT_VEL_END_IDX],
                        dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(
                    float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
        
        # 轨迹权重用于对不同轨迹进行加权采样
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # 预加载 转换数据
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')

        # 将所有完整轨迹数据堆叠起来
        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def reorder_from_pybullet_to_isaac(self, motion_data):
        """
        关节顺序转换： PyBullet [FR,FL,RR,RL] ==> IsaacGym [FL,FR,RL,RR]
        """
        # 获取各运动参数
        root_pos = AMPLoader.get_root_pos_batch(motion_data)
        root_rot = AMPLoader.get_root_rot_batch(motion_data)

        jp_fr, jp_fl, jp_rr, jp_rl = np.split(
            AMPLoader.get_joint_pose_batch(motion_data), 4, axis=1)
        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])  # 重排关节位置

        fp_fr, fp_fl, fp_rr, fp_rl = np.split(
            AMPLoader.get_tar_toe_pos_local_batch(motion_data), 4, axis=1)
        foot_pos = np.hstack([fp_fl, fp_fr, fp_rl, fp_rr])

        lin_vel = AMPLoader.get_linear_vel_batch(motion_data)
        ang_vel = AMPLoader.get_angular_vel_batch(motion_data)

        jv_fr, jv_fl, jv_rr, jv_rl = np.split(
            AMPLoader.get_joint_vel_batch(motion_data), 4, axis=1)
        joint_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr])

        fv_fr, fv_fl, fv_rr, fv_rl = np.split(
            AMPLoader.get_tar_toe_vel_local_batch(motion_data), 4, axis=1)
        foot_vel = np.hstack([fv_fl, fv_fr, fv_rl, fv_rr])

        return np.hstack(
            [root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel,
             joint_vel, foot_vel])

    def weighted_traj_idx_sample(self):
        """根据 轨迹权重 随机采样一个 轨迹索引"""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """批量采样轨迹索引
        参数:
            size: 需要采样的数量
        返回:
            采样得到的轨迹索引数组
        """
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """在 指定轨迹中 随机采样一个 时间点
        参数:
            traj_idx: 轨迹索引
        返回:
            采样得到的时间点(秒)
        """
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """批量采样 轨迹时间点
        参数:
            traj_idxs: 轨迹索引数组
        返回:
            对应每个轨迹的采样时间点数组
        """
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        """线性插值
        参数:
            val0: 起始值
            val1: 结束值
            blend: 插值比例(0-1)
        返回:
            插值结果
        """
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """获取 AMP 观测的 完整轨迹数据
        参数:
            traj_idx: 轨迹索引
        返回:
            完整的轨迹数据张量
        """
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """获取 指定轨迹 在特定时间的帧数据(不含根位置和旋转)
        参数:
            traj_idx: 轨迹索引
            time: 时间点(秒)
        返回:
            插值后的帧数据
        """
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """批量获取帧数据(不含根位置和旋转)
        参数:
            traj_idxs: 轨迹索引数组
            times: 时间点数组
        返回:
            批量插值后的帧数据
        """
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """获取完整帧数据(包含根位置和旋转)
        参数:
            traj_idx: 轨迹索引
            time: 时间点(秒)
        返回:
            插值后的完整帧数据
        """
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """批量获取完整帧数据
        参数:
            traj_idxs: 轨迹索引数组
            times: 时间点数组
        返回:
            批量插值后的完整帧数据
        """
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """在两个运动帧(frame0和frame1)之间进行线性插值
            包括: 位置、旋转、关节状态、足端位置、线速度、角速度等所有运动参数

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = AMPLoader.get_root_pos(frame0), AMPLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = AMPLoader.get_root_rot(frame0), AMPLoader.get_root_rot(frame1)
        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = AMPLoader.get_tar_toe_pos_local(frame0), AMPLoader.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = AMPLoader.get_linear_vel(frame0), AMPLoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = AMPLoader.get_angular_vel(frame0), AMPLoader.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        # remove z, foot_pos
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]
                s = torch.cat([
                    s,
                    self.preloaded_s[idxs, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]], dim=-1)
                s_next = self.preloaded_s_next[idxs, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]
                s_next = torch.cat([
                    s_next,
                    self.preloaded_s_next[idxs, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]], dim=-1)
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames))
                
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1] - 12

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    def get_tar_toe_pos_local(pose):
        return pose[AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(poses):
        return poses[:, AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(poses):
        return poses[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  

    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]  

    def get_tar_toe_vel_local(pose):
        return pose[AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX]

    def get_tar_toe_vel_local_batch(poses):
        return poses[:, AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX]