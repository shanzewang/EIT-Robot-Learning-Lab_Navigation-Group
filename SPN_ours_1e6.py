import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gym
import time
import core_config_GMM_ada_shape_vel_acc as core
#from core_config_GMM_ada_shape_vel_acc import get_vars
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

# Set device for PyTorch (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=128, start=0):
        idxs = np.random.randint(int(start), self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

# Define Actor and Critic models using PyTorch
class Actor(nn.Module):
    """Policy network for SAC."""
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        self.fc_log_std = nn.Linear(256, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x).clamp(-20, 2)  # Stabilize log_std
        return mu, log_std

    def sample(self, x, deterministic=False):
        mu, log_std = self.forward(x)
        if deterministic:
            return mu
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std

class Critic(nn.Module):
    """Q-value network for SAC."""
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_q = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc_q(x)
        return q

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
    Soft Actor-Critic (SAC) implementation with PyTorch.
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = 540 + 8
    act_dim = 2

    # Initialize networks
    actor = Actor(obs_dim, act_dim).to(device)
    critic1 = Critic(obs_dim, act_dim).to(device)
    critic2 = Critic(obs_dim, act_dim).to(device)
    target_critic1 = Critic(obs_dim, act_dim).to(device)
    target_critic2 = Critic(obs_dim, act_dim).to(device)

    # Copy weights to target networks
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    for p in target_critic1.parameters():
        p.requires_grad = False
    for p in target_critic2.parameters():
        p.requires_grad = False

    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr2, weight_decay=0.001)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=lr1)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=lr1)

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Logging
    summary_writer = SummaryWriter(log_dir='./logssac')

    def get_action(o, deterministic=False):
        o = torch.FloatTensor(o).to(device).unsqueeze(0)
        with torch.no_grad():
            a = actor.sample(o, deterministic=deterministic)
        return a.squeeze(0).cpu().numpy()

    def update(batch, step):
        obs1 = torch.FloatTensor(batch['obs1']).to(device)
        obs2 = torch.FloatTensor(batch['obs2']).to(device)
        acts = torch.FloatTensor(batch['acts']).to(device)
        rews = torch.FloatTensor(batch['rews']).to(device)
        done = torch.FloatTensor(batch['done']).to(device)

        # Critic update
        with torch.no_grad():
            next_mu, next_log_std = actor(obs2)
            next_pi = next_mu + torch.randn_like(next_log_std) * torch.exp(next_log_std)
            q1_targ = target_critic1(obs2, next_pi)
            q2_targ = target_critic2(obs2, next_pi)
            min_q_targ = torch.min(q1_targ, q2_targ)
            v_backup = min_q_targ - alpha * (next_log_std + torch.log1p(-torch.exp(2 * next_log_std)))
            target = rews + gamma * (1 - done) * v_backup

        q1 = critic1(obs1, acts)
        q2 = critic2(obs1, acts)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        q_loss = q1_loss + q2_loss

        critic1_optimizer.zero_grad()
        critic2_optimizer.zero_grad()
        q_loss.backward()
        critic1_optimizer.step()
        critic2_optimizer.step()

        # Actor update
        mu, log_std = actor(obs1)
        pi = mu + torch.randn_like(log_std) * torch.exp(log_std)
        q1_pi = critic1(obs1, pi)
        q2_pi = critic2(obs1, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        logp_pi = -log_std.sum(dim=-1) - 0.5 * ((pi - mu) / (torch.exp(log_std) + 1e-8)).pow(2).sum(dim=-1)
        pi_loss = (alpha * logp_pi - min_q_pi).mean()

        actor_optimizer.zero_grad()
        pi_loss.backward()
        actor_optimizer.step()

        # Target network update
        with torch.no_grad():
            for p, p_targ in zip(critic1.parameters(), target_critic1.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(critic2.parameters(), target_critic2.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return q_loss.item(), pi_loss.item()

    # Main loop
    episode = 0
    T = 0
    env = StageWorld(540)
    rate = rospy.Rate(10)
    #xxxxxsf
    # Initialize variables (same as original)
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
            # Environment setup (unchanged logic)
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

            # Reset logic (unchanged)
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
                f"train env no. is {env_no}, target distance {round(r2gd, 1)}, "
                f"succ rate {mean_rate[length_index]}, length1 {round(length1, 2)}, "
                f"length2 {round(length2, 2)}, width {round(width, 2)}, max_ep_len {max_ep_len}"
            )

            # Episode loop
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
                            q_loss, pi_loss = update(batch, T)

                            # Testing every 10000 steps
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
                                                    print(f"success rate in 50 is: {mean_rate[shape_no]}")
                                                    if mean_rate[shape_no] >= 0.90 and shape_no < 8:
                                                        env_list[shape_no] += 1
                                                        mean_rate[shape_no] = 0.0
                                                        suc_record[shape_no, :] = 0.0
                                                        suc_pointer[shape_no] = 0.0
                                                    np.save(f'test_result_plot{sac}.npy', test_result_plot)
                                                    if shape_no == 4:
                                                        test_time += 1
                                                        if test_time % 1 == 0:
                                                            torch.save(actor.state_dict(), f'actor_{sac}_lambda{hyper_exp}_{test_time}.pth')
                                                            torch.save(critic1.state_dict(), f'critic1_{sac}_lambda{hyper_exp}_{test_time}.pth')
                                                            torch.save(critic2.state_dict(), f'critic2_{sac}_lambda{hyper_exp}_{test_time}.pth')
                                                            rospy.sleep(3.0)
                                                break
                elif d or ep_len >= max_ep_len:
                    reset = True

            if episode % 5 == 0:
                print(
                    f"EPISODE {sac}{hyper_exp}, {episode} / REWARD {return_epoch} / steps {T}, "
                    f"env list {env_list}, succ rate each env {mean_rate}"
                )
            episode += 1

    # Save final models
    torch.save(actor.state_dict(), 'actor_final.pth')
    torch.save(critic1.state_dict(), 'critic1_final.pth')
    torch.save(critic2.state_dict(), 'critic2_final.pth')

def main():
    sac(actor_critic=core.mlp_actor_critic)

if __name__ == '__main__':
    random_number = random.randint(10000, 15000)
    port = str(random_number)
    print({port})
    os.environ["ROS_MASTER_URI"] = "http://localhost:" + port
    subprocess.Popen(["roscore", "-p", port])
    time.sleep(2)
    print("Roscore launched!")
    subprocess.Popen(["rosrun", "stage_ros1", "stageros1", "d8888153.world"])
    print("environment launched!")
    time.sleep(2)
    main()
