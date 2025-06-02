# -*- coding: utf-8 -*-
"""
Soft Actor-Critic强化学习主训练文件 (集成日志系统版本)
=======================================================

本文件实现完整的SAC算法训练流程，包括：
1. SAC算法核心实现
2. 经验回放缓冲区管理
3. 课程学习和难度自适应
4. 多实验并行和性能评估
5. 完整的日志管理和可视化系统

调用关系：
- 使用core_config中的神经网络架构
- 与stage_obs环境进行交互
- 管理ROS+Stage仿真环境
- 使用TrainingLogger进行日志管理和可视化
"""

import numpy as np
import tensorflow as tf
tf.disable_v2_behavior()

import gym
import time
import core_config_GMM_comment as core
from core_config_GMM_comment import get_vars
from stage_obs_ada_shape_vel_comment import StageWorld
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
    actor_critic=core.mlp_actor_critic,
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
):
    """
    Soft Actor-Critic算法实现
    
    SAC核心思想：最大化期望奖励和策略熵的加权和
    目标函数：J(π) = E[r(s,a) + α·H(π(·|s))]
    
    参数说明：
        actor_critic - 神经网络架构函数
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
    """
    
    # ============= 基础参数设置 =============
    sac = 1                # 算法标识
    obs_dim = 540 + 8      # 观测维度：激光雷达540 + 其他状态8
    act_dim = 2            # 动作维度：线速度+角速度

    # ============= 构建计算图 =============
    # 创建占位符
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(
        obs_dim,    # 当前状态
        act_dim,    # 动作
        obs_dim,    # 下一状态
        None,       # 奖励
        None        # 终止标志
    )

    # 主网络：用于训练
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = actor_critic(x_ph, a_ph)

    # 目标网络：用于计算目标Q值，提高训练稳定性
    with tf.variable_scope('target'):
        _, _, _, _, _, q1_pi_targ, q2_pi_targ = actor_critic(x2_ph, a_ph)

    # 经验回放缓冲区
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # 网络参数统计
    var_counts = tuple(
        core.count_vars(scope)
        for scope in ['main/pi', 'main/q1', 'main/q2', 'main/values', 'main']
    )
    print(f'\n网络参数数量: pi: {var_counts[0]}, q1: {var_counts[1]}, '
          f'q2: {var_counts[2]}, values: {var_counts[3]}, total: {var_counts[4]}\n')

    # ============= SAC损失函数 =============
    # 双Q网络取最小值，减少过估计
    min_q_pi = tf.minimum(q1_pi_targ, q2_pi_targ)
    min_q = tf.minimum(q1_pi, q2_pi)

    # 目标值计算：r + γ(1-done)[Q_min - α*log_π]
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)
    backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * v_backup)

    # L2正则化
    regularizer = tf.keras.regularizers.l2(0.001)
    all_trainable_weights_pi = [var for var in tf.trainable_variables() if 'main/pi' in var.name]
    regularization_penalty_pi = tf.add_n([regularizer(var) for var in all_trainable_weights_pi])

    # 损失函数
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q) + regularization_penalty_pi  # 策略损失
    q1_loss = tf.reduce_mean((q1 - backup) ** 2)  # Q1损失
    q2_loss = tf.reduce_mean((q2 - backup) ** 2)  # Q2损失
    q_loss = q2_loss + q1_loss                    # 总Q损失

    # ============= 优化器 =============
    # 策略网络优化器
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr2)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # 价值网络优化器
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr1)
    value_params = get_vars('main/values')
    train_q_op = value_optimizer.minimize(q_loss, var_list=value_params)
    
    # 目标网络软更新：θ_target = ρ*θ_target + (1-ρ)*θ_main
    target_update = tf.group([
        tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
        for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
    ])

    # 训练操作组合
    step_ops1 = [q_loss, q1, q2, train_q_op]          # Q网络更新
    step_ops2 = [pi_loss, train_pi_op, target_update]  # 策略网络更新+目标网络软更新

    # 目标网络初始化
    target_init = tf.group([
        tf.assign(v_targ, v_main)
        for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
    ])

    # ============= TensorFlow会话 =============
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    
    # TensorBoard记录变量
    reward_var = tf.Variable(0.0, trainable=False)
    robot_size_var = tf.Variable(0.0, trainable=False)
    average_speed_var = tf.Variable(0.0, trainable=False)
    goal_reach_var = tf.Variable(0.0, trainable=False)
    
    reward_epi = tf.summary.scalar('reward', reward_var)
    robot_size_epi = tf.summary.scalar('robot_size', robot_size_var)
    average_speed_epi = tf.summary.scalar('average_speed', average_speed_var)
    goal_reach_epi = tf.summary.scalar('goal_reach', goal_reach_var)
    
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./logssac' + str(sac), sess.graph)

    # 初始化变量
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    
    # 模型保存器
    trainables = tf.trainable_variables()
    trainable_saver = tf.train.Saver(trainables, max_to_keep=None)
    sess.run(tf.global_variables_initializer())

    def get_action(o, deterministic=False):
        """
        获取动作
        
        参数：
            o - 状态观测
            deterministic - 是否使用确定性策略(测试时为True)
        
        返回：动作向量
        """
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]

    # ============= 训练初始化 =============
    episode = 0              # episode计数
    T = 0                   # 总训练步数
    epi_thr = 0

    # 实验数据存储
    suc_record_all = np.zeros((5, 150, 9))
    suc_record_all_new = np.zeros((5, 150, 9))
    test_result_plot = np.load('test_result_plot1.npy')  # 测试结果存储
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
        tf.set_random_seed(seed)
        np.random.seed(seed)
        
        # 重新初始化
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        sess.run(tf.global_variables_initializer())
        sess.run(target_init)
        trainables = tf.trainable_variables()
        trainable_saver = tf.train.Saver(trainables, max_to_keep=None)
        sess.run(tf.global_variables_initializer())
        
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
                    logger.log_episode_end(
                        episode=episode,
                        reward=return_epoch,
                        steps=ep_len,
                        success=bool(goal_reach),
                        crash=(not bool(goal_reach)),  # d=True但没成功，就是碰撞
                        timeout=False
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
                            timeout=True
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
                            feed_dict = {
                                x_ph: batch['obs1'],
                                x2_ph: batch['obs2'],
                                a_ph: batch['acts'],
                                r_ph: batch['rews'],
                                d_ph: batch['done'],
                            }
                            
                            # 更新Q网络
                            outs = sess.run(step_ops1, feed_dict)
                            # 更新策略网络和目标网络
                            outs = sess.run(step_ops2, feed_dict)

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
                                    if (mean_rate[shape_no] >= 0.90 and new_env < 8):
                                        env_list[shape_no] = env_list[shape_no] + 1
                                        succ_rate_test = 0
                                        mean_rate[shape_no] = 0.0
                                        suc_record[shape_no, :] = 0.0
                                        suc_pointer[shape_no] = 0.0
                                        logger.write_log(f"🎉尺寸组{shape_no}解锁环境{int(env_list[shape_no])}")
                                    
                                    np.save(f'test_result_plot{sac}.npy', test_result_plot)

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
                                        f'configback540GMMdense{sac}lambda{hyper_exp}'
                                    )
                                    trainable_saver.save(sess, save_path, global_step=test_time)
                                    logger.write_log(f"💾 模型已保存: {save_path}-{test_time}")
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


def main():
    """主函数：启动SAC训练"""
    print("🚀 开始SAC训练...")
    sac(actor_critic=core.mlp_actor_critic)
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
"""