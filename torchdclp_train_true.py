# -*- coding: utf-8 -*-
"""
Soft Actor-Critic强化学习主训练文件 (PyTorch版本)
=======================================================

本文件实现完整的SAC算法训练流程，包括：
1. SAC算法核心实现
2. 经验回放缓冲区管理
3. 课程学习和难度自适应
4. 多实验并行和性能评估
5. 完整的日志管理和可视化系统

调用关系：
- 使用core_config_pytorch中的神经网络架构
- 与stage_obs环境进行交互
- 管理ROS+Stage仿真环境
- 使用TrainingLogger进行日志管理和可视化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gym
import time
import torchcore_true as core  # 使用PyTorch版本的网络文件
from stage_obs_comment import StageWorld  # 使用新的环境文件
from training_logger import TrainingLogger  # 导入日志系统
import random
import rospy
import os
import signal
import subprocess
import sys
from collections import deque
import scipy.stats as stats
from scipy.stats import rankdata
from scipy.stats import truncnorm


class ReplayBuffer:
    """
    经验回放缓冲区
    
    作用：存储和采样智能体的历史经验，支持离线策略学习
    特点：FIFO结构，支持动态采样策略
    """

    def __init__(self, obs_dim, act_dim, size):
        """
        初始化经验缓冲区
        
        参数：
            obs_dim - 观测维度(548)
            act_dim - 动作维度(2)
            size - 缓冲区容量(2M)
        """
        # 预分配内存存储经验数据
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)  # 当前状态
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)  # 下一状态
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)  # 动作
        self.rews_buf = np.zeros(size, dtype=np.float32)             # 奖励
        self.done_buf = np.zeros(size, dtype=np.float32)             # 终止标志
        
        # 缓冲区管理变量
        self.ptr = 0        # 当前写入位置
        self.size = 0       # 当前数据量
        self.max_size = size # 最大容量

    def store(self, obs, act, rew, next_obs, done):
        """
        存储一条经验
        
        参数：
            obs - 当前状态
            act - 执行动作
            rew - 获得奖励
            next_obs - 下一状态
            done - 是否终止
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size  # 循环覆盖
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=128, start=0):
        """
        采样训练批次
        
        参数：
            batch_size - 批次大小
            start - 采样起始位置(用于偏向新经验的采样策略)
        
        返回：包含obs1, obs2, acts, rews, done的字典
        """
        idxs = np.random.randint(int(start), self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )


class StageWorldLogger(StageWorld):
    """扩展StageWorld类，添加日志功能"""
    
    def __init__(self, beam_num, logger=None):
        super().__init__(beam_num)
        self.logger = logger
        self.step_context = "training"  # 当前step调用的上下文
    
    def set_step_context(self, context):
        """设置step调用的上下文"""
        self.step_context = context
    
    def step(self):
        """重写step函数，添加上下文感知的日志"""
        # 调用原始step函数
        state, reward, terminate, reset, distance, robot_pose = super().step()
        
        # 根据上下文决定是否记录碰撞
        if terminate and reset == 0:  # 碰撞终止
            if self.logger and self.step_context == "initialization":
                self.logger.log_crash("initialization")
            # 训练时的碰撞在episode_end中统一处理，这里不重复输出
        
        return state, reward, terminate, reset, distance, robot_pose


def sac(
    actor_critic=core.MLPActorCritic,
    seed=5,
    steps_per_epoch=5000,
    epochs=10000,
    replay_size=int(2e6),
    gamma=0.99,
    polyak=0.995,
    lr1=1e-4,
    lr2=1e-4,
    alpha=0.01,
    batch_size=100,
    start_epoch=100,
    max_ep_len=400,
    MAX_EPISODE=10000,
    device='cpu',
):
    """
    Soft Actor-Critic算法实现 (PyTorch版本)
    
    SAC核心思想：最大化期望奖励和策略熵的加权和
    目标函数：J(π) = E[r(s,a) + α·H(π(·|s))]
    
    参数说明：
        actor_critic - 神经网络架构类
        seed - 随机种子
        steps_per_epoch - 每epoch步数
        epochs - 总epoch数
        replay_size - 经验回放容量
        gamma - 折扣因子
        polyak - 目标网络软更新系数
        lr1/lr2 - 价值/策略网络学习率
        alpha - 熵正则化系数
        batch_size - 训练批大小
        start_epoch - 开始使用学习策略的epoch
        max_ep_len - 最大episode长度
        MAX_EPISODE - 最大episode数
        device - 计算设备('cpu'或'cuda')
    """
    
    # ============= 基础参数设置 =============
    sac_id = 1             # 算法标识
    obs_dim = 540 + 8      # 观测维度：激光雷达540 + 目标2 + 速度2 + 动力学参数4 = 548
    act_dim = 2            # 动作维度：线速度+角速度
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ============= 构建神经网络 =============
    # 主网络：用于训练
    main_model = actor_critic(obs_dim, act_dim).to(device)
    
    # 目标网络：用于计算目标Q值，提高训练稳定性
    target_model = actor_critic(obs_dim, act_dim).to(device)
    
    # 初始化目标网络参数与主网络一致
    target_model.load_state_dict(main_model.state_dict())
    
    # 冻结目标网络参数（不参与梯度更新）
    for param in target_model.parameters():
        param.requires_grad = False

    # 经验回放缓冲区
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # 网络参数统计
    total_params = core.count_vars(main_model)
    policy_params = core.count_vars(main_model.policy)
    q1_params = core.count_vars(main_model.q1)
    q2_params = core.count_vars(main_model.q2)
    print(f'\n网络参数数量: total: {total_params}, policy: {policy_params}, '
          f'q1: {q1_params}, q2: {q2_params}\n')

    # ============= 优化器设置 =============
    # 策略网络优化器
    pi_optimizer = optim.Adam(main_model.policy.parameters(), lr=lr2)
    
    # 价值网络优化器
    q_optimizer = optim.Adam(
        list(main_model.q1.parameters()) + 
        list(main_model.q2.parameters()) + 
        list(main_model.cnn_dense.parameters()), 
        lr=lr1
    )
    
    # L2正则化
    l2_reg = 0.001

    def get_action(o, deterministic=False):
        """
        获取动作
        
        参数：
            o - 状态观测
            deterministic - 是否使用确定性策略(测试时为True)
        
        返回：动作向量
        """
        with torch.no_grad():
            o_tensor = torch.FloatTensor(o.reshape(1, -1)).to(device)
            
            # 调用完整网络获取所有输出
            # main_model(o_tensor) 只传入状态，返回5个值：mu, pi, logp_pi, q1, q2
            mu, pi, logp_pi, _, _ = main_model(o_tensor)
            
            if deterministic:
                # 确定性策略：使用tanh(mu)
                action = torch.tanh(mu)
                return action.cpu().numpy()[0]
            else:
                # 随机策略：使用采样的动作pi
                return pi.cpu().numpy()[0]

    def update_networks(batch):
        """
        更新神经网络
        
        参数：
            batch - 训练批次数据
        
        返回：
            pi_loss, q_loss - 策略损失和Q网络损失
        """
        # 转换为PyTorch张量
        obs1 = torch.FloatTensor(batch['obs1']).to(device)
        obs2 = torch.FloatTensor(batch['obs2']).to(device)
        acts = torch.FloatTensor(batch['acts']).to(device)
        rews = torch.FloatTensor(batch['rews']).to(device)
        done = torch.FloatTensor(batch['done']).to(device)

        # ============= 更新Q网络 =============
        with torch.no_grad():
            # 目标网络计算下一状态的Q值
            # target_model(obs2) 只传入状态，返回5个值：mu, pi, logp_pi, q1_pi, q2_pi
            _, pi_next, logp_pi_next, q1_pi_targ, q2_pi_targ = target_model(obs2)
            
            # 双Q网络取最小值，减少过估计
            min_q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            
            # 计算目标值：r + γ(1-done)[Q_min - α*log_π]
            backup = rews + gamma * (1 - done) * (min_q_pi_targ - alpha * logp_pi_next)

        # 当前Q值
        # main_model(obs1, acts) 传入状态和动作，返回7个值
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = main_model(obs1, acts)
        
        # Q损失
        q1_loss = F.mse_loss(q1, backup)
        q2_loss = F.mse_loss(q2, backup)
        q_loss = q1_loss + q2_loss

        # 更新Q网络
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # ============= 更新策略网络 =============
        # 冻结Q网络参数，只更新策略网络
        for param in main_model.q1.parameters():
            param.requires_grad = False
        for param in main_model.q2.parameters():
            param.requires_grad = False
        for param in main_model.cnn_dense.parameters():
            param.requires_grad = False

        # 重新计算策略网络输出（只传入状态，让网络重新采样动作）
        # main_model(obs1) 只传入状态，返回5个值：mu, pi, logp_pi, q1_pi, q2_pi
        mu, pi, logp_pi, q1_pi, q2_pi = main_model(obs1)
        
        # 策略损失：最大化 Q(s,π(s)) - α*log_π(s)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        # L2正则化
        l2_penalty = sum(param.pow(2.0).sum() for param in main_model.policy.parameters())
        
        pi_loss = torch.mean(alpha * logp_pi - min_q_pi) + l2_reg * l2_penalty

        # 更新策略网络
        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()

        # 解冻Q网络参数
        for param in main_model.q1.parameters():
            param.requires_grad = True
        for param in main_model.q2.parameters():
            param.requires_grad = True
        for param in main_model.cnn_dense.parameters():
            param.requires_grad = True

        # ============= 软更新目标网络 =============
        with torch.no_grad():
            for param_main, param_target in zip(main_model.parameters(), target_model.parameters()):
                param_target.data.mul_(polyak)
                param_target.data.add_((1 - polyak) * param_main.data)

        return pi_loss.item(), q_loss.item()

    # ============= TensorBoard设置 =============
    log_dir = f'./logssac{sac_id}'
    summary_writer = SummaryWriter(log_dir)

    # ============= 训练初始化 =============
    episode = 0              # episode计数
    T = 0                   # 总训练步数
    epi_thr = 0

    # 实验数据存储
    suc_record_all = np.zeros((5, 150, 9))
    suc_record_all_new = np.zeros((5, 150, 9))
    
    # 加载测试结果存储数组（如果不存在则创建新的）
    try:
        test_result_plot = np.load('test_result_plot1.npy')
    except FileNotFoundError:
        # 如果文件不存在，创建新的数组：[实验数, 尺寸组, 测试次数, 测试轮次, 指标数]
        test_result_plot = np.zeros((5, 5, 50, 101, 5))
        print("创建新的测试结果存储数组")
    
    env_record = np.zeros((5, 4))
    train_result = np.zeros((5, 12000))
    test_time = 0

    # ============= 多实验循环 =============
    for hyper_exp in range(1, 5):  # 运行4个独立实验
        print(f"\n========== 实验 {hyper_exp} 开始 ==========")
        
        # ============= 创建日志管理器 =============
        logger = TrainingLogger(experiment_id=hyper_exp)
        
        # ============= 创建带日志功能的环境 =============
        env = StageWorldLogger(540, logger=logger)
        
        # ============= ROS频率控制（在环境创建后） =============
        rate = rospy.Rate(10)   # ROS频率控制
        
        # 实验初始化
        goal_reach = 0
        past_env = [0]
        current_env = [0]
        
        # 课程学习变量
        suc_record = np.zeros((5, 50))    # 成功记录：5个机器人尺寸组×50次记录
        suc_record1 = np.zeros((5, 50))   # 无碰撞成功记录
        suc_record2 = np.zeros((5, 50))   # 超时记录
        suc_pointer = np.zeros(5)         # 记录指针
        mean_rate = np.zeros(5)           # 平均成功率
        env_list = np.zeros(5)            # 各组当前环境难度(0-6)
        p = np.zeros(9)
        p[0] = 1.0                        # 初始只使用环境0
        mean_rate[0] = 0.0
        best_value = 0
        
        # 设置随机种子
        seed = hyper_exp
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 重新初始化网络
        main_model = actor_critic(obs_dim, act_dim).to(device)
        target_model = actor_critic(obs_dim, act_dim).to(device)
        target_model.load_state_dict(main_model.state_dict())
        
        for param in target_model.parameters():
            param.requires_grad = False
            
        # 重新初始化优化器
        pi_optimizer = optim.Adam(main_model.policy.parameters(), lr=lr2)
        q_optimizer = optim.Adam(
            list(main_model.q1.parameters()) + 
            list(main_model.q2.parameters()) + 
            list(main_model.cnn_dense.parameters()), 
            lr=lr1
        )
        
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        
        # 重置实验变量
        episode = 0
        T = 0
        epi_thr = 0
        goal_reach = 0
        test_time = 0
        new_env = 0          # 当前解锁的最高环境等级
        succ_rate_test = 0
        b_test = True
        length_index = 0

        # ============= 主训练循环 =============
        while test_time < 101:  # 进行101次性能测试
            
            # 环境难度选择
            if new_env == 0:
                env_no = 0  # 初期只使用最简单环境
            else:
                env_no = new_env
            
            # 随机机器人尺寸(域随机化提高泛化性)
            length_index = np.random.choice(range(5))
            length1 = np.random.uniform(0.075, 0.6)    # 前方长度
            length2 = np.random.uniform(0.075, 0.6)    # 后方长度
            width = np.random.uniform(0.075, (length2 + length1) / 2.0)  # 宽度
            
            env_no = int(env_list[length_index])

            # 机器人尺寸约束检查
            if length_index == 0:
                while length1 + length2 + width * 2 >= 0.8:
                    length1 = np.random.uniform(0.075, 0.6)
                    length2 = np.random.uniform(0.075, 0.6)
                    width = np.random.uniform(0.075, (length2 + length1) / 2.0)
            else:
                while (length1 + length2 + width * 2 < (length_index + 1.0) * 0.4 or 
                       length1 + length2 + width * 2 >= (length_index + 2.0) * 0.4):
                    length1 = np.random.uniform(0.075, 0.6)
                    length2 = np.random.uniform(0.075, 0.6)
                    width = np.random.uniform(0.075, (length2 + length1) / 2.0)

            # ============= 设置训练阶段 =============
            logger.set_phase("TRAINING", 
                            env=int(env_list[length_index]), 
                            robot_size=[length1, length2, width])

            # Episode初始化
            T_step = 0
            goal_reach = 0

            # ============= 设置step上下文为初始化 =============
            env.set_step_context("initialization")

            # 重置环境
            if goal_reach == 1 and b_test:
                robot_size = env.Reset(env_no)
            else:
                robot_size = env.ResetWorld(env_no, length1, length2, width)
                b_test = True

            # 生成目标点(基于成功率的自适应目标生成)
            env.GenerateTargetPoint(mean_rate[length_index])
            o, r, d, goal_reach, r2gd, robot_pose = env.step()
            rate.sleep()

            # 确保有效的起始条件
            try_time = 0
            while r2gd < 0.3 and try_time < 100:  # 确保距离目标不太近
                try_time = try_time + 1
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()
            
            try_time = 0
            while d and try_time < 100:  # 确保起始时无碰撞
                try_time = try_time + 1
                robot_size = env.ResetWorld(env_no, length1, length2, width)
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()

            try_time = 0
            while r2gd < 0.3 and try_time < 1000:
                try_time = try_time + 1
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()

            # 设置episode参数
            max_ep_len = int(40 * 0.8**env_no / robot_size * 5)  # 根据环境和机器人大小调整最大长度
            
            # ============= 记录episode开始 =============
            logger.log_episode_start(
                episode=episode,
                env_no=env_no,
                robot_size=[length1, length2, width],
                distance=r2gd,
                success_rate=mean_rate[length_index],
                max_steps=max_ep_len
            )

            # ============= 设置step上下文为训练 =============
            env.set_step_context("training")

            # Episode执行
            reset = False
            return_epoch = 0      # episode总奖励
            total_vel = 0         # 总速度
            ep_len = 0            # episode长度
            d = False
            last_d = 0

            # ============= Episode主循环 =============
            while not reset:
                
                # 动作选择
                if episode > start_epoch:
                    a = get_action(o, deterministic=False)  # 使用学习的策略
                else:
                    a = env.PIDController()  # 预训练阶段使用PID控制器

                # 执行动作
                env.Control(a)
                rate.sleep()
                env.Control(a)  # 执行两次确保稳定
                rate.sleep()
                
                past_a = a
                o2, r, d, goal_reach, r2gd, robot_pose2 = env.step()
                
                return_epoch = return_epoch + r
                total_vel = total_vel + a[0]

                # 存储经验
                replay_buffer.store(o, a, r, o2, d)
                ep_len += 1

                o = o2
                last_d = d

                # Episode终止检查
                if d:
                    if episode > start_epoch:
                        # 更新成功记录
                        suc_record[length_index, int(suc_pointer[length_index])] = goal_reach
                        
                        if env.stop_counter < 1.0:  # 无碰撞
                            suc_record1[length_index, int(suc_pointer[length_index])] = goal_reach
                            suc_record2[length_index, int(suc_pointer[length_index])] = 0
                        else:  # 有碰撞
                            suc_record1[length_index, int(suc_pointer[length_index])] = 0.0
                            suc_record2[length_index, int(suc_pointer[length_index])] = 0
                        
                        suc_pointer[length_index] = (suc_pointer[length_index] + 1) % 50
                    
                    # ============= 记录episode结束（正确判断状态） =============
                    # 当d=True时，只有两种情况：成功(goal_reach=1)或碰撞(goal_reach=0)
                    # 在正式训练阶段才更新总训练步数T
                    logger.log_episode_end(
                        episode=episode,
                        reward=return_epoch,
                        steps=ep_len,
                        success=bool(goal_reach),
                        crash=(not bool(goal_reach)),  # d=True但没成功，就是碰撞
                        timeout=False,
                        update_training_steps=(episode > start_epoch)  # 只在正式训练时更新T
                    )
                    reset = True
                else:
                    if ep_len >= max_ep_len:  # 超时
                        if episode > start_epoch:
                            suc_record[length_index, int(suc_pointer[length_index])] = goal_reach
                            suc_record1[length_index, int(suc_pointer[length_index])] = goal_reach
                            suc_record2[length_index, int(suc_pointer[length_index])] = 1.0
                            suc_pointer[length_index] = (suc_pointer[length_index] + 1) % 50

                        # ============= 记录episode结束（超时） =============
                        logger.log_episode_end(
                            episode=episode,
                            reward=return_epoch,
                            steps=ep_len,
                            success=bool(goal_reach),
                            crash=False,
                            timeout=True,
                            update_training_steps=(episode > start_epoch)  # 只在正式训练时更新T
                        )
                        reset = True

                # ============= 网络训练 =============
                if episode > start_epoch:
                    if goal_reach == 1 or env.crash_stop or (ep_len >= max_ep_len):
                        if ep_len == 0:
                            ep_len = 1
                        
                        reset = True
                        average_vel = total_vel / ep_len

                        # 批量训练：每个episode步数对应一次参数更新
                        for j in range(ep_len):
                            T = T + 1
                            
                            # 动态经验采样：偏向使用新经验
                            start = np.minimum(
                                replay_buffer.size * (1.0 - (0.996 ** (j * 1.0 / ep_len * 1000.0))),
                                np.maximum(replay_buffer.size - 10000, 0),
                            )
                            
                            batch = replay_buffer.sample_batch(batch_size, start=start)
                            
                            # 更新网络
                            pi_loss, q_loss = update_networks(batch)
                            
                            # 记录到TensorBoard
                            if T % 100 == 0:
                                summary_writer.add_scalar('Loss/Policy', pi_loss, T)
                                summary_writer.add_scalar('Loss/Q_Network', q_loss, T)

                            # ============= 定期性能测试 =============
                            if T % 10000 == 0:  # 每10000步测试一次
                                # ============= 设置测试阶段 =============
                                logger.set_phase("TESTING", test_round=T//10000)
                                logger.start_test_round(T//10000)
                                
                                # ============= 设置step上下文为测试 =============
                                env.set_step_context("testing")
                                
                                # 存储每组的测试结果
                                group_results = {}
                                
                                for shape_no in range(5):  # 测试5种机器人尺寸
                                    logger.log_test_group_start(shape_no, int(env_list[shape_no]))
                                    
                                    # 存储该组的所有测试结果
                                    group_rewards = []
                                    group_success = []
                                    group_crashes = []
                                    
                                    for k in range(50):     # 每种尺寸测试50次
                                        total_vel_test = 0
                                        return_epoch_test = 0
                                        ep_len_test = 0
                                        
                                        rospy.sleep(2.0)
                                        
                                        # 设置测试环境
                                        velcity = env.set_robot_pose_test(k, int(env_list[shape_no]), shape_no)
                                        env.GenerateTargetPoint_test(k, int(env_list[shape_no]), shape_no)
                                        max_ep_len = int(40 * 0.8 ** int(env_list[shape_no]) / velcity * 5)
                                        o, r, d, goal_reach, r2gd, robot_pose = env.step()

                                        # 测试episode
                                        for i in range(1000):
                                            a = get_action(o, deterministic=True)  # 确定性策略测试
                                            
                                            env.Control(a)
                                            rate.sleep()
                                            env.Control(a)
                                            rate.sleep()
                                            
                                            o2, r, d, goal_reach, r2gd, robot_pose2 = env.step()
                                            return_epoch_test = return_epoch_test + r
                                            total_vel_test = total_vel_test + a[0]
                                            ep_len_test += 1
                                            o = o2

                                            if d or (ep_len_test >= max_ep_len):
                                                # 记录测试结果
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 0] = return_epoch_test    # 总奖励
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 1] = goal_reach           # 成功标志
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 2] = (ep_len_test * 1.0 / max_ep_len)  # 时间效率
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 3] = (1.0 * goal_reach - ep_len_test * 2.0 / max_ep_len)  # 综合指标
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 4] = env.crash_stop       # 碰撞标志

                                                # 收集统计数据
                                                group_rewards.append(return_epoch_test)
                                                group_success.append(goal_reach)
                                                group_crashes.append(env.crash_stop)
                                                
                                                break
                                    
                                    # 计算该组的统计结果
                                    group_result = {
                                        'success_rate': np.mean(group_success),
                                        'avg_reward': np.mean(group_rewards),
                                        'collision_rate': np.mean(group_crashes),
                                        'test_count': len(group_success)
                                    }
                                    group_results[shape_no] = group_result
                                    
                                    # 记录该组测试完成
                                    logger.log_test_group_end(shape_no, group_result)
                                    
                                    # 原有的成功率更新逻辑（保持不变）
                                    mean_rate[shape_no] = np.mean(
                                        test_result_plot[hyper_exp, shape_no, :, test_time, 1]
                                    )
                                    
                                    # 课程学习：成功率达到90%时解锁下一环境
                                    if (mean_rate[shape_no] >= 0.90 and int(env_list[shape_no]) < 7):  # 8个环境(0-7)
                                        env_list[shape_no] = env_list[shape_no] + 1
                                        succ_rate_test = 0
                                        mean_rate[shape_no] = 0.0
                                        suc_record[shape_no, :] = 0.0
                                        suc_pointer[shape_no] = 0.0
                                        logger.write_log(f"🎉尺寸组{shape_no}解锁环境{int(env_list[shape_no])}")
                                    
                                    np.save(f'test_result_plot{sac_id}.npy', test_result_plot)

                                # ============= 结束测试轮次 =============
                                logger.end_test_round(env_list)
                                test_time = test_time + 1
                                
                                # ============= 生成可视化图表 =============
                                logger.plot_learning_curves()
                                logger.plot_test_results_heatmap()
                                
                                # ============= 保存数据 =============
                                logger.save_training_data()
                                
                                # 保存模型
                                if (test_time) % 1 == 0:
                                    # 使用相对路径保存模型
                                    model_dir = './saved_models'
                                    os.makedirs(model_dir, exist_ok=True)  # 确保目录存在
                                    
                                    save_path = os.path.join(
                                        model_dir,
                                        f'configback540GMMdense{sac_id}lambda{hyper_exp}_{test_time}.pth'
                                    )
                                    
                                    # 保存模型状态字典
                                    torch.save({
                                        'main_model_state_dict': main_model.state_dict(),
                                        'target_model_state_dict': target_model.state_dict(),
                                        'pi_optimizer_state_dict': pi_optimizer.state_dict(),
                                        'q_optimizer_state_dict': q_optimizer.state_dict(),
                                        'test_time': test_time,
                                        'T': T,
                                        'episode': episode,
                                    }, save_path)
                                    
                                    logger.write_log(f"💾 模型已保存: {save_path}")
                                    rospy.sleep(3.0)
                                
                                # ============= 重新设置为训练阶段 =============
                                logger.set_phase("TRAINING")
                                env.set_step_context("training")
                                b_test = False

            episode = episode + 1
            epi_thr = epi_thr + 1

        # ============= 实验结束时生成总结报告 =============
        logger.generate_summary_report()
        logger.write_log(f"✅ 实验{hyper_exp}完成")

    # 关闭TensorBoard writer
    summary_writer.close()


def main():
    """主函数：启动SAC训练"""
    print("🚀 开始SAC训练...")
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    sac(actor_critic=core.MLPActorCritic, device=device)
    print("✅ 训练完成!")


if __name__ == '__main__':
    # ============= ROS环境初始化 =============
    random_number = random.randint(10000, 15000)
    port = str(random_number)
    os.environ["ROS_MASTER_URI"] = "http://localhost:" + port
    
    print(f"🌐 启动ROS环境，端口: {port}")

    # 启动roscore
    subprocess.Popen(["roscore", "-p", port])
    time.sleep(2)
    print("✅ Roscore启动成功!")

    # 启动Stage仿真器
    world_file = "d8888153.world"
    subprocess.Popen(["rosrun", "stage_ros1", "stageros", world_file])
    print("✅ Stage仿真环境启动成功!")
    time.sleep(2)
    
    main()


# ============= 核心变量说明 =============
"""
重要变量含义：

训练相关：
- episode: 当前episode编号
- T: 总训练步数计数器
- obs_dim: 观测维度(548 = 540激光雷达 + 8其他状态)
- act_dim: 动作维度(2 = 线速度 + 角速度)

课程学习相关：
- env_list[5]: 5个机器人尺寸组对应的环境难度等级(0-6)
- mean_rate[5]: 各尺寸组的平均成功率
- suc_record[5,50]: 成功记录，滑动窗口记录最近50次结果
- new_env: 当前解锁的最高环境等级

机器人参数：
- length1/length2: 机器人前后长度
- width: 机器人宽度
- length_index: 机器人尺寸组索引(0-4)

SAC算法参数：
- gamma: 折扣因子(0.99)
- polyak: 目标网络软更新系数(0.995)
- alpha: 熵正则化系数(0.01)
- lr1/lr2: 价值/策略网络学习率(1e-4)

测试相关：
- test_time: 测试轮次计数器
- test_result_plot: 测试结果存储数组[实验数, 尺寸组, 测试次数, 测试轮次, 指标数]

日志系统相关：
- logger: TrainingLogger实例，管理所有日志和可视化
- step_context: 当前step调用的上下文（initialization/training/testing）

PyTorch相关：
- device: 计算设备('cpu'或'cuda')
- main_model: 主训练网络
- target_model: 目标网络
- pi_optimizer: 策略网络优化器
- q_optimizer: Q网络优化器
- summary_writer: TensorBoard记录器
"""