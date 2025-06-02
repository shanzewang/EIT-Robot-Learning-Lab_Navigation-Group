# import torch
# import torch.optim as optim
# import numpy as np
# import random
# import os
import time
# import rospy
# import math
# import stage_obs_ada_shape_vel_test_dyna as StageWorld
# #import replay_buffer_multi_agent_dynamic as ReplayBuffer
# import sac_torch
# from torch.utils.tensorboard import SummaryWriter
# from copy import deepcopy
# import core_torch as core
import psutil
# import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gym
import time
import core_torch as core
from stage_obs_ada_shape_vel_test_dyna import StageWorld
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
from copy import deepcopy
import itertools
import datetime
import gc

# 设置设备（GPU如果可用，否则CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """
    用于SAC算法的简单FIFO经验回放缓冲区
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=128, start=0):
        idxs = np.random.randint(int(start), self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs]
        )
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

def sac(
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
    使用PyTorch实现的Soft Actor-Critic (SAC)算法
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 定义观察空间和动作空间
    obs_dim = 540 + 8
    act_dim = 2
    
    # 初始化Actor-Critic网络
    ac = core.MLPActorCritic(obs_dim, act_dim).to(device)
    ac_targ = deepcopy(ac).to(device)
    
    # 冻结目标网络参数（只通过polyak平均更新）
    for p in ac_targ.parameters():
        p.requires_grad = False
    
    # 优化器
    pi_optimizer = optim.Adam(ac.ac.policy.parameters(), lr=lr2, weight_decay=0.001)
    q_params = itertools.chain(ac.ac.q1_net.parameters(), ac.ac.q2_net.parameters())
    q_optimizer = optim.Adam(q_params, lr=lr1)
    
    # 经验回放缓冲区
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    
    # 日志记录 - 使用时间戳创建唯一的日志目录
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'./logs/sac_{current_time}'
    summary_writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志保存在: {log_dir}")
    
    # 计算Q函数损失
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        
        # Bellman备份
        with torch.no_grad():
            a2, logp_a2 = ac.pi(o2)
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
        
        # MSE损失
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        # 记录额外的Q网络统计信息
        with torch.no_grad():
            q_mean = q1.mean().item()
            q_std = q1.std().item()
            q_max = q1.max().item()
            q_min = q1.min().item()
        
        return loss_q, loss_q1.item(), loss_q2.item(), q_mean, q_std, q_max, q_min
    
    # 计算策略损失
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        
        # 熵正则化策略损失
        loss_pi = (alpha * logp_pi - q_pi).mean()
        
        # 记录策略统计信息
        with torch.no_grad():
            entropy = -logp_pi.mean().item()
            policy_mean = pi.mean().item()
            policy_std = pi.std().item()
        
        return loss_pi, entropy, policy_mean, policy_std
    
    # 更新函数
    def update(data, step_count):
        # 先运行Q函数的梯度下降
        q_optimizer.zero_grad()
        loss_q, loss_q1, loss_q2, q_mean, q_std, q_max, q_min = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        
        # 冻结Q网络以避免在策略学习步骤中浪费计算资源
        for p in q_params:
            p.requires_grad = False
        
        # 运行策略的梯度下降
        pi_optimizer.zero_grad()
        loss_pi, entropy, policy_mean, policy_std = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        
        # 解冻Q网络
        for p in q_params:
            p.requires_grad = True
        
        # 通过polyak平均更新目标网络
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        
        # 记录训练指标到TensorBoard
        if step_count % 100 == 0:
            summary_writer.add_scalar('Loss/Q', loss_q.item(), step_count)
            summary_writer.add_scalar('Loss/Q1', loss_q1, step_count)
            summary_writer.add_scalar('Loss/Q2', loss_q2, step_count)
            summary_writer.add_scalar('Loss/Policy', loss_pi.item(), step_count)
            summary_writer.add_scalar('Stats/Q_mean', q_mean, step_count)
            summary_writer.add_scalar('Stats/Q_std', q_std, step_count)
            summary_writer.add_scalar('Stats/Q_max', q_max, step_count)
            summary_writer.add_scalar('Stats/Q_min', q_min, step_count)
            summary_writer.add_scalar('Stats/Policy_entropy', entropy, step_count)
            summary_writer.add_scalar('Stats/Policy_mean', policy_mean, step_count)
            summary_writer.add_scalar('Stats/Policy_std', policy_std, step_count)
        
        # 每1000步记录网络权重分布
        if step_count % 10000 == 0:
            try:
                for name, param in ac.ac.policy.named_parameters():
                    # 确保参数正确转换为numpy数组
                    param_numpy = param.data.cpu().numpy()
                    summary_writer.add_histogram(f'Policy/{name}', param_numpy, step_count)
                for name, param in ac.ac.q1_net.named_parameters():
                    param_numpy = param.data.cpu().numpy()
                    summary_writer.add_histogram(f'Q1/{name}', param_numpy, step_count)
                for name, param in ac.ac.q2_net.named_parameters():
                    param_numpy = param.data.cpu().numpy()
                    summary_writer.add_histogram(f'Q2/{name}', param_numpy, step_count)
            except Exception as e:
                print(f"记录参数分布时出错: {e}")
                # 出错时不中断训练
        
        return loss_q.item(), loss_pi.item()
    
    # 获取动作函数
    def get_action(o, deterministic=False):
        o = torch.FloatTensor(o).to(device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                pi, _ = ac.pi(o, deterministic=True)
                return pi.squeeze(0).cpu().numpy()
            else:
                a = ac.act(o, deterministic=False)
                return a.squeeze(0).cpu().numpy()
    
    # 主循环
    episode = 0
    T = 0
    env = StageWorld(540)
    rate = rospy.Rate(10)
    
    # 初始化变量
    suc_record_all = np.zeros((5, 100, 9))
    suc_record_all_new = np.zeros((5, 100, 9))
    test_result_plot = np.zeros((5, 5, 50, 100, 5))
    env_record = np.zeros((5, 4))
    train_result = np.zeros((5, 12000))
    
    for hyper_exp in range(1, 5):
        goal_reach = 0
        suc_record = np.zeros((5, 50))
        suc_record1 = np.zeros((5, 50))
        suc_record2 = np.zeros((5, 50))
        suc_pointer = np.zeros(5)
        mean_rate = np.zeros(5)
        env_list = np.zeros(5)
        p = np.zeros(9)
        p[0] = 1.0
        mean_rate[0] = 0.0
        
        # 添加训练过程中的成功率追踪
        train_success_window = 50  # 滑动窗口大小
        train_success_history = []  # 存储最近的成功/失败记录
        current_train_success_rate = 0.0  # 当前训练成功率
        
        # 添加超时检测和监控
        last_progress_time = time.time()  # 记录最后进展时间
        step_timeout = 300  # 5分钟超时
        
        # 内存使用监控
        def check_memory_usage():
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            if memory_mb > 8000:  # 超过8GB时警告
                print(f"内存使用过高: {memory_mb:.1f}MB")
                gc.collect()  # 强制垃圾回收
                return memory_mb
            return memory_mb
        
        seed = hyper_exp
        torch.manual_seed(seed)
        np.random.seed(seed)
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        
        episode = 0
        T = 0
        test_time = 0
        b_test = True
        
        # 记录超参数
        summary_writer.add_text('Hyperparameters/seed', str(seed), 0)
        summary_writer.add_text('Hyperparameters/gamma', str(gamma), 0)
        summary_writer.add_text('Hyperparameters/polyak', str(polyak), 0)
        summary_writer.add_text('Hyperparameters/lr1', str(lr1), 0)
        summary_writer.add_text('Hyperparameters/lr2', str(lr2), 0)
        summary_writer.add_text('Hyperparameters/alpha', str(alpha), 0)
        summary_writer.add_text('Hyperparameters/batch_size', str(batch_size), 0)
        
        while test_time < 101:
            # 检查Stage仿真器健康状态
            if episode % 20 == 0:  # 每20个episode检查一次
                print(f"检查Stage仿真器健康状态...")
                try:
                    stage_healthy = env.check_stage_health()
                    if not stage_healthy:
                        print("Stage仿真器健康检查失败，等待恢复...")
                        rospy.sleep(1.0)
                        # 尝试重新发送一个简单命令来测试响应
                        env.Control([0.0, 0.0])  # 发送停止命令
                        rospy.sleep(1.0)
                except Exception as e:
                    print(f"Stage健康检查出错: {e}")
            
            print(f"开始新episode: episode={episode}, test_time={test_time}")
            
            # 环境设置
            length_index = np.random.choice(range(5))
            print(f"选择length_index: {length_index}")
            length1 = np.random.uniform(0.075, 0.6)
            length2 = np.random.uniform(0.075, 0.6)
            width = np.random.uniform(0.075, (length2 + length1) / 2.0)
            env_no = int(env_list[length_index])
            
            if length_index == 0:
                while length1 + length2 + width * 2 >= 0.8:
                    length1 = np.random.uniform(0.075, 0.6)
                    length2 = np.random.uniform(0.075, 0.6)
                    width = np.random.uniform(0.075, (length2 + length1) / 2.0)
            else:
                while (
                    length1 + length2 + width * 2 < (length_index + 1.0) * 0.4
                    or length1 + length2 + width * 2 >= (length_index + 2.0) * 0.4
                ):
                    length1 = np.random.uniform(0.075, 0.6)
                    length2 = np.random.uniform(0.075, 0.6)
                    width = np.random.uniform(0.075, (length2 + length1) / 2.0)
            
            if goal_reach == 1 and b_test:
                print("使用Reset方法重置环境")
                velocityMax = env.Reset(env_no)
                print(f"Reset完成，velocityMax={velocityMax}")
            else:
                print("使用ResetWorld方法重置环境")
                try:
                    velocityMax = env.ResetWorld(env_no, length1, length2, width)
                    print(f"ResetWorld完成，velocityMax={velocityMax}")
                    b_test = True
                except Exception as e:
                    print(f"环境重置失败: {e}")
                    # 尝试重新初始化环境
                    try:
                        rospy.sleep(5.0)
                        velocityMax = env.ResetWorld(env_no, length1, length2, width)
                        print("环境重新初始化成功")
                    except Exception as retry_e:
                        print(f"环境重新初始化也失败: {retry_e}")
                        # 如果仍然失败，跳过此次episode
                        continue
            
            print("开始生成目标点...")
            env.GenerateTargetPoint(mean_rate[length_index])
            print("目标点生成完成")
            
            print("执行第一步...")
            o, r, d, goal_reach, r2gd, robot_pose = env.step()
            print(f"第一步完成: r={r:.3f}, d={d}, goal_reach={goal_reach}, r2gd={r2gd:.3f}")
            rate.sleep()
            
            # 重置逻辑
            try_time = 0
            while r2gd < 0.3 and try_time < 100:
                try_time += 1
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()
            
            try_time = 0
            while d and try_time < 100:
                try_time += 1
                velocityMax = env.ResetWorld(env_no, length1, length2, width)
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()
            
            try_time = 0
            while r2gd < 0.3 and try_time < 1000:
                try_time += 1
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()
            
            max_ep_len = int(40 * 0.8**env_no / velocityMax * 5)
            
            # 添加尺寸分布的详细调试输出
            total_size = length1 + length2 + width * 2
            print(
                f"训练 length_index={length_index}, env_no={env_no}, 目标距离={round(r2gd, 1)}, "
                f"最近50个episode内的训练成功率={round(current_train_success_rate, 3)}"
            )
            print(
                f"  机器人尺寸: length1={round(length1, 3)}, length2={round(length2, 3)}, "
                f"width={round(width, 3)}, total_size={round(total_size, 3)}, 最大步数={max_ep_len}"
            )
            
            # # 记录环境参数
            # summary_writer.add_scalar(f'Env/length_index_{hyper_exp}', length_index, episode)
            # summary_writer.add_scalar(f'Env/env_no_{hyper_exp}', env_no, episode)
            # summary_writer.add_scalar(f'Env/length1_{hyper_exp}', length1, episode)
            # summary_writer.add_scalar(f'Env/length2_{hyper_exp}', length2, episode)
            # summary_writer.add_scalar(f'Env/width_{hyper_exp}', width, episode)
            # summary_writer.add_scalar(f'Env/max_ep_len_{hyper_exp}', max_ep_len, episode)
            # summary_writer.add_scalar(f'Env/target_distance_{hyper_exp}', r2gd, episode)
            
            # 单个回合循环
            reset = False
            return_epoch = 0
            total_vel = 0
            ep_len = 0
            d = False
            
            # 重置进度计时器
            last_progress_time = time.time()
            
            while not reset:
                # 检查是否超时
                current_time = time.time()
                if current_time - last_progress_time > step_timeout:
                    print(f"检测到步骤超时({step_timeout}秒)，强制结束当前episode")
                    print(f"可能的原因: Stage仿真器卡死或ROS通信阻塞")
                    
                    # 尝试发送停止命令
                    try:
                        print("尝试发送停止命令...")
                        env.Control([0.0, 0.0])
                        rospy.sleep(1.0)
                        print("停止命令发送成功")
                    except Exception as e:
                        print(f"发送停止命令失败: {e}")
                    
                    # 检查Stage健康状态
                    try:
                        stage_healthy = env.check_stage_health()
                        if not stage_healthy:
                            print("Stage仿真器健康检查失败")
                        else:
                            print("Stage仿真器健康检查通过")
                    except Exception as e:
                        print(f"无法进行Stage健康检查: {e}")
                    
                    d = True
                    reset = True
                    break
                
                # 每50步检查一次内存使用
                if ep_len % 50 == 0 and ep_len > 0:
                    memory_mb = check_memory_usage()
                    if memory_mb > 12000:  # 超过12GB强制结束
                        print(f"内存使用过高({memory_mb:.1f}MB)，强制结束episode")
                        d = True
                        reset = True
                        break
                
                if episode > start_epoch:
                    a = get_action(o, deterministic=False)
                    #print(f"训练action获取成功")
                else:
                    a = env.PIDController()
                    #print(f"PIDaction获取成功")
                
                env.Control(a)
                #print(f"训练action发布成功1")
                rate.sleep()
                #print(f"训练sleep1")
                env.Control(a)
                #print(f"训练action发布成功2")
                rate.sleep()
                #print(f"训练sleep2")
                
                try:
                    o2, r, d, goal_reach, r2gd, robot_pose2 = env.step()
                    #print(f"训练step成功")
                except Exception as e:
                    print(f"环境step操作失败，可能仿真器已崩溃: {e}")
                    # 检测ROS是否还在运行
                    if rospy.is_shutdown():
                        print("ROS已关闭，退出训练")
                        return
                    else:
                        print("尝试跳过此步骤并继续...")
                        # 设置默认值并继续
                        o2 = o  # 使用上一步的观测
                        r = -1.0  # 给一个小的负奖励
                        d = True  # 标记为done，结束当前episode
                        goal_reach = 0
                        r2gd = float('inf')
                        robot_pose2 = [0, 0, 0]
                
                return_epoch += r
                total_vel += a[0]
                
                #print(f"开始存储到回放缓冲区")
                try:
                    replay_buffer.store(o, a, r, o2, d)
                    #print(f"回放缓冲区存储成功")
                    last_progress_time = time.time()  # 更新进度时间
                except Exception as e:
                    print(f"回放缓冲区存储失败: {e}")
                    # 如果存储失败，跳过这一步但继续训练
                
                ep_len += 1
                o = o2
                #print(f"episode步数更新: ep_len={ep_len}, max_ep_len={max_ep_len}")
                
                if episode > start_epoch:
                    if goal_reach == 1 or env.crash_stop or (ep_len >= max_ep_len):
                        reset = True
                        average_vel = total_vel / ep_len if ep_len > 0 else 0
                        
                        print(f"Episode结束: goal_reach={goal_reach}, crash={env.crash_stop}, ep_len={ep_len}")
                        last_progress_time = time.time()  # 更新进度时间
                        
                        # 更新训练成功率历史记录
                        train_success_history.append(1 if goal_reach == 1 else 0)
                        if len(train_success_history) > train_success_window:
                            train_success_history.pop(0)  # 移除最旧的记录
                        current_train_success_rate = sum(train_success_history) / len(train_success_history)
                        
                        # # 记录回合结束的统计信息
                        # summary_writer.add_scalar(f'Episode/return_{hyper_exp}', return_epoch, episode)
                        # summary_writer.add_scalar(f'Episode/length_{hyper_exp}', ep_len, episode)
                        # summary_writer.add_scalar(f'Episode/average_vel_{hyper_exp}', average_vel, episode)
                        # summary_writer.add_scalar(f'Episode/goal_reach_{hyper_exp}', goal_reach, episode)
                        # summary_writer.add_scalar(f'Episode/crash_{hyper_exp}', int(env.crash_stop), episode)
                        # summary_writer.add_scalar(f'Train/success_rate_{hyper_exp}', current_train_success_rate, episode)
                        

                        for j in range(ep_len):
                            T += 1
                            #print(f"开始第{j+1}/{ep_len}次网络更新, T={T}")
                            
                            start = min(
                                replay_buffer.size * (1.0 - (0.996 ** (j * 1.0 / ep_len * 1000.0))),
                                max(replay_buffer.size - 10000, 0)
                            )
                            #print(f"采样起始位置: start={start}, buffer_size={replay_buffer.size}")
                            
                            try:
                                batch = replay_buffer.sample_batch(batch_size, start=start)
                                #print(f"采样批次成功")
                            except Exception as e:
                                #print(f"采样批次失败: {e}")
                                continue
                            
                            try:
                                q_loss, pi_loss = update(batch, T)
                                #print(f"网络更新成功: q_loss={q_loss:.4f}, pi_loss={pi_loss:.4f}")
                            except Exception as e:
                                #print(f"网络更新失败: {e}")
                                continue
                            
                            # 每5000步进行测试 (原为10000)
                            if T % 5000 == 0:
                                # 测试开始记录
                                # summary_writer.add_text(
                                #     'Testing/start',
                                #     f'Starting test at step {T}', T)
                                print(
                                    f"开始第 {T//5000} 轮测试，"
                                    f"训练步数: {T}, test_time: {test_time}"
                                )
                                
                                # 为测试结果创建一个临时存储
                                test_results = {
                                    'rewards': np.zeros((5, 50)),
                                    'goal_reach': np.zeros((5, 50)),
                                    'ep_len_ratio': np.zeros((5, 50)),
                                    'crash': np.zeros((5, 50))
                                }
                                
                                for shape_no in range(5):
                                    for k in range(50):
                                        total_vel_test = 0
                                        return_epoch_test = 0
                                        ep_len_test = 0
                                        rospy.sleep(2.0)
                                        #print(f"测试rospysleep成功")
                                        
                                        # velocityMax = env.ResetWorld_test(shape_no, int(env_list[shape_no]), k)
                                        # env.GenerateTargetPoint_test(k, int(env_list[shape_no]), shape_no)                                        # 【修改】使用与训练时完全相同的环境重置方法
                                        # 1. 生成与训练时相同的机器人尺寸参数
                                        length_index = shape_no  # 使用shape_no作为length_index
                                        length1 = np.random.uniform(0.075, 0.6)
                                        length2 = np.random.uniform(0.075, 0.6)
                                        width = np.random.uniform(0.075, (length2 + length1) / 2.0)
                                        
                                        # 使用与训练时相同的尺寸约束逻辑
                                        if length_index == 0:
                                            while length1 + length2 + width * 2 >= 0.8:
                                                length1 = np.random.uniform(0.075, 0.6)
                                                length2 = np.random.uniform(0.075, 0.6)
                                                width = np.random.uniform(0.075, (length2 + length1) / 2.0)
                                        else:
                                            while (
                                                length1 + length2 + width * 2 < (length_index + 1.0) * 0.4
                                                or length1 + length2 + width * 2 >= (length_index + 2.0) * 0.4
                                            ):
                                                length1 = np.random.uniform(0.075, 0.6)
                                                length2 = np.random.uniform(0.075, 0.6)
                                                width = np.random.uniform(0.075, (length2 + length1) / 2.0)
                                        
                                        # 2. 使用训练时的ResetWorld方法（包含随机位置和角度）
                                        velocityMax = env.ResetWorld(int(env_list[shape_no]), length1, length2, width)
                                        #print(f"测试重置成功")
                                        
                                        # 3. 使用训练时的目标点生成方法（动态范围，基于成功率）
                                        env.GenerateTargetPoint(mean_rate[shape_no])
                                        print(f"测试目标点生成成功")
                                        
                                        max_ep_len = int(40 * 0.8 ** int(env_list[shape_no]) / velocityMax * 5)
                                        o, r, d, goal_reach, r2gd, robot_pose = env.step()
                                        #print(f"测试step成功1")
                                        
                                        print(f"测试 shape_no={shape_no}, k={k}: 使用训练时相同的环境设置")
                                        print(f"  机器人尺寸: length1={length1:.3f}, length2={length2:.3f}, width={width:.3f}")
                                        print(f"  目标距离: {r2gd:.3f}, 最大步数: {max_ep_len}")
                                        
                                        # 【新增】打印详细的目标点信息，方便分析
                                        target_point = env.target_point  # 获取当前目标点
                                        robot_pos = robot_pose[:2]  # 机器人当前位置 [x, y]
                                        print(f"  目标点坐标: [{target_point[0]:.3f}, {target_point[1]:.3f}]")
                                        print(f"  机器人坐标: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}]")
                                        print(f"  成功率影响的生成范围: mean_rate[{shape_no}]={mean_rate[shape_no]:.3f}")
                                        
                                        # 计算生成窗口大小（与训练时GenerateTargetPoint中的逻辑一致）
                                        map_size = env.map_sizes[int(env_list[shape_no])]
                                        local_window = max(mean_rate[shape_no] * map_size[0], 0.5)
                                        print(f"  目标点生成窗口大小: {local_window:.3f} (地图尺寸: {map_size[0]:.3f})")
                                        
                                        for i in range(1000):
                                            # 【修改】使用与训练时相同的动作采样方式（非确定性）
                                            # a = get_action(o, deterministic=True)
                                            a = get_action(o, deterministic=False)  # 使用随机采样，与训练时一致
                                            #print(f"测试action获取成功")
                                            env.Control(a)
                                            #print(f"测试action发布成功1")
                                            rate.sleep()
                                            #print(f"测试sleep1成功")
                                            env.Control(a)
                                            #print(f"测试action发布成功2")
                                            rate.sleep()
                                            #print(f"测试sleep2成功")
                                            
                                            o2, r, d, goal_reach, r2gd, robot_pose2 = env.step()
                                            #print(f"测试step成功2")
                                            return_epoch_test += r
                                            total_vel_test += a[0]
                                            ep_len_test += 1
                                            o = o2
                                            
                                            if d or (ep_len_test >= max_ep_len):
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 0] = return_epoch_test
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 1] = goal_reach
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 2] = ep_len_test / max_ep_len
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 3] = 1.0 * goal_reach - ep_len_test * 2.0 / max_ep_len
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 4] = env.crash_stop
                                                
                                                # 【新增】打印每个测试回合的详细结果
                                                current_distance = r2gd  # 当前距离目标的距离
                                                final_robot_pos = robot_pose2[:2]  # 最终机器人位置
                                                result_type = "到达目标" if goal_reach == 1 else ("碰撞" if env.crash_stop else "超时")
                                                print(f"****回合{k}结束: {result_type}****, 步数{ep_len_test}/{max_ep_len}, "
                                                      f"奖励{return_epoch_test:.2f}, 最终距离{current_distance:.3f}")
                                                print(f"      最终位置: [{final_robot_pos[0]:.3f}, {final_robot_pos[1]:.3f}], "
                                                      f"目标位置: [{target_point[0]:.3f}, {target_point[1]:.3f}]")
                                                
                                                # 存储测试结果到临时数组
                                                test_results['rewards'][shape_no, k] = return_epoch_test
                                                test_results['goal_reach'][shape_no, k] = goal_reach
                                                test_results['ep_len_ratio'][shape_no, k] = ep_len_test / max_ep_len
                                                test_results['crash'][shape_no, k] = env.crash_stop
                                                
                                                # # 记录单个测试案例的结果 - 使用T作为x轴
                                                # summary_writer.add_scalar(
                                                #     f'Test/case_reward_{hyper_exp}_{shape_no}',
                                                #     return_epoch_test, T
                                                # )
                                                # summary_writer.add_scalar(
                                                #     f'Test/case_goal_reach_{hyper_exp}_{shape_no}',
                                                #     goal_reach, T
                                                # )
                                                
                                                if k == 49:
                                                    mean_rate[shape_no] = np.mean(test_result_plot[hyper_exp, shape_no, :, test_time, 1])
                                                    print(f"######尺寸组 {shape_no}: 50次测试的成功率: {mean_rate[shape_no]:.3f}###### (T={T}, test_time={test_time})")
                                                    
                                                    # 【新增】详细的测试统计信息
                                                    success_count = np.sum(test_result_plot[hyper_exp, shape_no, :, test_time, 1])
                                                    crash_count = np.sum(test_result_plot[hyper_exp, shape_no, :, test_time, 4])
                                                    timeout_count = 50 - success_count - crash_count
                                                    avg_steps = np.mean(test_result_plot[hyper_exp, shape_no, :, test_time, 2]) * max_ep_len
                                                    avg_reward = np.mean(test_result_plot[hyper_exp, shape_no, :, test_time, 0])
                                                    
                                                    print(f"  详细统计: 成功{int(success_count)}次, 碰撞{int(crash_count)}次, 超时{int(timeout_count)}次")
                                                    print(f"  平均步数: {avg_steps:.1f}/{max_ep_len}, 平均奖励: {avg_reward:.2f}")
                                                    print(f"  当前环境级别: {int(env_list[shape_no])}, 目标生成窗口: {local_window:.3f}")
                                                    
                                                    # # 记录环境的测试成功率 - 使用T作为x轴
                                                    # summary_writer.add_scalar(
                                                    #     f'Test/success_rate_{hyper_exp}_{shape_no}',
                                                    #     mean_rate[shape_no], T
                                                    # )
                                                    
                                                    if mean_rate[shape_no] >= 0.90 and shape_no < 8:
                                                        # 设置环境级别上限为4，共5个训练地图(0-4)
                                                        if env_list[shape_no] < 4:
                                                            env_list[shape_no] += 1
                                                            mean_rate[shape_no] = 0.0
                                                            suc_record[shape_no, :] = 0.0
                                                            suc_pointer[shape_no] = 0.0
                                                            
                                                            # # 记录环境提升事件 - 使用T作为x轴
                                                            # summary_writer.add_scalar(
                                                            #     f'Test/env_level_up_{hyper_exp}_{shape_no}',
                                                            #     env_list[shape_no], T
                                                            # )
                                                            print(f"环境 {shape_no} 升级到级别 {env_list[shape_no]}")
                                                        else:
                                                            # 当达到最高环境级别时，不再升级环境，但继续提升目标距离
                                                            print(f"环境 {shape_no} 已达到最高级别 {env_list[shape_no]}，"
                                                                  f"当前成功率 {mean_rate[shape_no]:.3f}，继续学习长距离导航")
                                                            # # 记录在最高级别的成功率提升
                                                            # summary_writer.add_scalar(
                                                            #     f'Test/max_level_success_rate_{hyper_exp}_{shape_no}',
                                                            #     mean_rate[shape_no], T
                                                            # )
                                                    
                                                    np.save('test_result_plot_torch.npy', test_result_plot)
                                                    
                                                    if shape_no == 4:
                                                        test_time += 1
                                                        
                                                        # 记录所有环境的平均成功率 - 使用T作为x轴
                                                        avg_success = np.mean([
                                                            np.mean(test_results['goal_reach'][i, :]) 
                                                            for i in range(5)
                                                        ])
                                                        # summary_writer.add_scalar(
                                                        #     f'Test/overall_success_rate_{hyper_exp}',
                                                        #     avg_success, T
                                                        # )
                                                        print(f"第 {test_time} 轮测试完成，总体成功率: {avg_success:.3f}")
                                                        
                                                        if test_time % 1 == 0:
                                                            model_path = f'ac_torch_lambda{hyper_exp}_{test_time}.pt'
                                                            torch.save(ac.state_dict(), model_path)
                                                            # summary_writer.add_text(
                                                            #     'Checkpoint', 
                                                            #     f'Saved model to {model_path} at test_time {test_time}',
                                                            #     T  # 使用T而不是test_time
                                                            # )
                                                            rospy.sleep(3.0)
                                                break
                
                elif d or ep_len >= max_ep_len:
                    reset = True
            
            if episode % 5 == 0:
                print(
                    f"回合 {hyper_exp}, episode{episode} / 奖励 {return_epoch} / update步数 {T}, "
                    f"各尺寸机器人训练进度 {env_list}, 测试中各尺寸机器人成功率 {mean_rate}"
                )
                
            print(f"完成episode {episode}，准备开始下一个episode")
            episode += 1
            print(f"episode计数器更新为: {episode}")
            
            # 检查内存使用
            if episode % 10 == 0:
                memory_mb = check_memory_usage()
                print(f"当前内存使用: {memory_mb:.1f}MB")
                
            # 检查是否需要进行测试
            print(f"检查测试条件: test_time={test_time}, test_time<101={test_time < 101}")
            
            # 强制垃圾回收，防止内存累积
            if episode % 50 == 0:
                print("执行垃圾回收...")
                gc.collect()
                print("垃圾回收完成")
    
    # 保存最终模型
    final_model_path = 'ac_torch_final.pt'
    torch.save(ac.state_dict(), final_model_path)
    summary_writer.add_text('Final', f'训练完成，最终模型保存为 {final_model_path}', 0)
    summary_writer.close()
    print(f"训练完成! 日志已保存到 {log_dir}")

if __name__ == '__main__':
    random_number = random.randint(10000, 15000)
    port = str(random_number)
    print({port})
    os.environ["ROS_MASTER_URI"] = "http://localhost:" + port
    subprocess.Popen(["roscore", "-p", port])
    time.sleep(2)
    print("Roscore 已启动!")
    subprocess.Popen(["rosrun", "stage_ros1", "stageros1", "dynamic.world"])
    print("环境已启动!")
    time.sleep(2)
    sac()
