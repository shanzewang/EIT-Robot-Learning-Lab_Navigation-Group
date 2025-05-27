import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gym
import time
import core_torch as core
from stage_obs_ada_shape_vel_test import StageWorld
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
    
    # 日志记录
    summary_writer = SummaryWriter(log_dir='./logssac')
    
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
        
        return loss_q
    
    # 计算策略损失
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        
        # 熵正则化策略损失
        loss_pi = (alpha * logp_pi - q_pi).mean()
        
        return loss_pi
    
    # 更新函数
    # def update(data):
    #     # 更新Critic（Q函数）
    #     critic1_optimizer.zero_grad()
    #     critic2_optimizer.zero_grad()
    #     loss_q = compute_loss_q(data)
    #     loss_q.backward()
    #     critic1_optimizer.step()
    #     critic2_optimizer.step()
        
    #     # 冻结Q网络以避免在策略学习步骤中浪费计算资源
    #     for p in ac.ac.q1.parameters():
    #         p.requires_grad = False
    #     for p in ac.ac.q2.parameters():
    #         p.requires_grad = False
        
    #     # 更新Actor（策略）
    #     actor_optimizer.zero_grad()
    #     loss_pi = compute_loss_pi(data)
    #     loss_pi.backward()
    #     actor_optimizer.step()
        
    #     # 解冻Q网络
    #     for p in ac.ac.q1.parameters():
    #         p.requires_grad = True
    #     for p in ac.ac.q2.parameters():
    #         p.requires_grad = True
        
    #     # 通过polyak平均更新目标网络
    #     with torch.no_grad():
    #         for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
    #             p_targ.data.mul_(polyak)
    #             p_targ.data.add_((1 - polyak) * p.data)
        
    #     return loss_q.item(), loss_pi.item()
    def update(data):
        # 先运行Q函数的梯度下降
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        
        # 冻结Q网络以避免在策略学习步骤中浪费计算资源
        for p in q_params:
            p.requires_grad = False
        
        # 运行策略的梯度下降
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
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
        
        return loss_q.item(), loss_pi.item()
    
    # 获取动作函数
    def get_action(o, deterministic=False):
        o = torch.FloatTensor(o).to(device).unsqueeze(0)
        with torch.no_grad():
            a = ac.act(o, deterministic=deterministic)
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
        seed = hyper_exp
        torch.manual_seed(seed)
        np.random.seed(seed)
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        
        episode = 0
        T = 0
        test_time = 0
        b_test = True
        
        while test_time < 101:
            # 环境设置
            length_index = np.random.choice(range(5))
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
                robot_size = env.Reset(env_no)
            else:
                robot_size = env.ResetWorld(env_no, length1, length2, width)
                b_test = True
            
            env.GenerateTargetPoint(mean_rate[length_index])
            o, r, d, goal_reach, r2gd, robot_pose = env.step()
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
                robot_size = env.ResetWorld(env_no, length1, length2, width)
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()
            
            try_time = 0
            while r2gd < 0.3 and try_time < 1000:
                try_time += 1
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()
            
            max_ep_len = int(40 * 0.8**env_no / robot_size * 5)
            print(
                f"训练环境编号 {env_no}, 目标距离 {round(r2gd, 1)}, "
                f"成功率 {mean_rate[length_index]}, 长度1 {round(length1, 2)}, "
                f"长度2 {round(length2, 2)}, 宽度 {round(width, 2)}, 最大步数 {max_ep_len}"
            )
            
            # 单个回合循环
            reset = False
            return_epoch = 0
            total_vel = 0
            ep_len = 0
            d = False
            
            while not reset:
                if episode > start_epoch:
                    a = get_action(o, deterministic=False)
                else:
                    a = env.PIDController()
                
                env.Control(a)
                rate.sleep()
                env.Control(a)
                rate.sleep()
                o2, r, d, goal_reach, r2gd, robot_pose2 = env.step()
                return_epoch += r
                total_vel += a[0]
                
                replay_buffer.store(o, a, r, o2, d)
                ep_len += 1
                o = o2
                
                if episode > start_epoch:
                    if goal_reach == 1 or env.crash_stop or (ep_len >= max_ep_len):
                        reset = True
                        average_vel = total_vel / ep_len if ep_len > 0 else 0
                        
                        for j in range(ep_len):
                            T += 1
                            start = min(
                                replay_buffer.size * (1.0 - (0.996 ** (j * 1.0 / ep_len * 1000.0))),
                                max(replay_buffer.size - 10000, 0)
                            )
                            batch = replay_buffer.sample_batch(batch_size, start=start)
                            q_loss, pi_loss = update(batch)
                            
                            # 每10000步进行测试
                            if T % 10000 == 0:
                                for shape_no in range(5):
                                    for k in range(50):
                                        total_vel_test = 0
                                        return_epoch_test = 0
                                        ep_len_test = 0
                                        rospy.sleep(2.0)
                                        
                                        velcity = env.set_robot_pose_test(k, int(env_list[shape_no]), shape_no)
                                        env.GenerateTargetPoint_test(k, int(env_list[shape_no]), shape_no)
                                        max_ep_len = int(40 * 0.8 ** int(env_list[shape_no]) / velcity * 5)
                                        o, r, d, goal_reach, r2gd, robot_pose = env.step()
                                        
                                        for i in range(1000):
                                            a = get_action(o, deterministic=True)
                                            env.Control(a)
                                            rate.sleep()
                                            env.Control(a)
                                            rate.sleep()
                                            
                                            o2, r, d, goal_reach, r2gd, robot_pose2 = env.step()
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
                                                
                                                if k == 49:
                                                    mean_rate[shape_no] = np.mean(test_result_plot[hyper_exp, shape_no, :, test_time, 1])
                                                    print(f"50次测试的成功率: {mean_rate[shape_no]}")
                                                    
                                                    if mean_rate[shape_no] >= 0.90 and shape_no < 8:
                                                        env_list[shape_no] += 1
                                                        mean_rate[shape_no] = 0.0
                                                        suc_record[shape_no, :] = 0.0
                                                        suc_pointer[shape_no] = 0.0
                                                        
                                                    np.save(f'test_result_plot_torch.npy', test_result_plot)
                                                    
                                                    if shape_no == 4:
                                                        test_time += 1
                                                        
                                                        if test_time % 1 == 0:
                                                            torch.save(ac.state_dict(), f'ac_torch_lambda{hyper_exp}_{test_time}.pt')
                                                            rospy.sleep(3.0)
                                                
                                                break
                
                elif d or ep_len >= max_ep_len:
                    reset = True
            
            if episode % 5 == 0:
                print(
                    f"回合 {hyper_exp}, {episode} / 奖励 {return_epoch} / 步数 {T}, "
                    f"环境列表 {env_list}, 各环境成功率 {mean_rate}"
                )
                
            episode += 1
    
    # 保存最终模型
    torch.save(ac.state_dict(), 'ac_torch_final.pt')

if __name__ == '__main__':
    random_number = random.randint(10000, 15000)
    port = str(random_number)
    print({port})
    os.environ["ROS_MASTER_URI"] = "http://localhost:" + port
    subprocess.Popen(["roscore", "-p", port])
    time.sleep(2)
    print("Roscore 已启动!")
    subprocess.Popen(["rosrun", "stage_ros1", "stageros1", "d8888153.world"])
    print("环境已启动!")
    time.sleep(2)
    sac()