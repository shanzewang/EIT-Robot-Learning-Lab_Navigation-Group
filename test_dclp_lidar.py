import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gym
import time
import core_config_GMM_ada_shape_vel_acc as core
from core_config_GMM_ada_shape_vel_acc import get_vars
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
    sac = 1
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

    # Logging
    summary_writer = SummaryWriter(log_dir='./logssac' + str(sac))

    # Load pre-trained model if available
    try:
        actor.load_state_dict(torch.load("network/configback540GMMdense101lambda0-100_actor.pth"))
        critic1.load_state_dict(torch.load("network/configback540GMMdense101lambda0-100_critic1.pth"))
        critic2.load_state_dict(torch.load("network/configback540GMMdense101lambda0-100_critic2.pth"))
    except FileNotFoundError:
        print("Pre-trained models not found, starting from scratch.")

    def get_action(o, deterministic=False):
        o = torch.FloatTensor(o).to(device).unsqueeze(0)
        with torch.no_grad():
            a = actor.sample(o, deterministic=deterministic)
        return a.squeeze(0).cpu().numpy()

    # Main loop: collect experience in env and update/log each epoch
    episode = 0
    T = 0
    env = StageWorld(540)
    rate = rospy.Rate(10)
    epi_thr = 0
    obs_metric_flage = 0

    suc_record_all = np.zeros((5, 100, 9))
    suc_record_all_new = np.zeros((5, 100, 9))
    test_result_plot = np.zeros((5, 5, 50, 100, 5))
    env_record = np.zeros((5, 4))
    train_result = np.zeros((5, 12000))
    test_time = 0

    for hyper_exp in range(4, 5):
        scores_window = deque(maxlen=50)
        for initial_zero in range(50):
            scores_window.append(0)

        episode = 0
        T = 0
        epi_thr = 0
        goal_reach = 0
        test_time = 0
        new_env = 0
        succ_rate_test = 0
        b_test = True
        length_index = 0
        train_result = np.zeros((5, 12000))
        train_result1 = np.zeros((5, 250000))

        while test_time < 20:
            env_no = 1  # obstacles3.world
            print("train env no. is Obstacles3")

            if episode <= start_epoch or goal_reach == 0:
                robot_size = env.ResetWorld(env_no)
            else:
                robot_size = env.Reset(env_no)

            env.GenerateTargetPoint(env_no)

            o, r, d, goal_reach, r2gd, robot_pose = env.step()

            while d:
                reset = False
                robot_size = env.ResetWorld(env_no)
                env.GenerateTargetPoint(env_no)
                o, r, d, goal_reach, r2gd, robot_pose = env.step()

            reset = False
            return_epoch = 0
            total_vel = 0
            ep_len = 0
            d = False
            last_d = 0
            while not reset:
                if episode > start_epoch:
                    a = get_action(o, deterministic=False)
                else:
                    a = env.PIDController()

                # Step the env
                env.Control(a)
                rate.sleep()
                env.Control(a)
                rate.sleep()
                past_a = a
                o2, r, d, goal_reach, r2gd, robot_pose2 = env.step()
                return_epoch = return_epoch + r
                total_vel = total_vel + a[0]

                if episode > start_epoch:
                    T = T + 1  # T记录有效训练步数
                ep_len += 1  # ep_len记录当前回合的步数
                o = o2
                if ep_len >= max_ep_len:
                    print("Time out")

                    # 回合结束时更新scores_window
                    if d or (ep_len >= max_ep_len):
                        if episode > start_epoch and new_env == env_no:
                            scores_window.append(goal_reach)
                        reset = True
                    if episode > start_epoch:
                        train_result1[hyper_exp, T] = np.mean(scores_window)
                        if T % 10000 == 0:
                            test_time = test_time + 1
                train_result[hyper_exp, episode] = np.mean(scores_window)
                print(
                    "EPISODE" + str(sac) + str(hyper_exp),
                    episode,
                    "/ REWARD",
                    return_epoch,
                    "/ class_num ",
                    obs_metric_flage,
                    "/ steps ",
                    T,
                    "success rate",
                    np.mean(scores_window),
                )
                episode = episode + 1
                epi_thr = epi_thr + 1
            np.save("train_result" + str(sac) + ".npy", train_result)
            np.save("train_result_1_" + str(sac) + ".npy", train_result1)
            break

def main():
    sac(actor_critic=core.mlp_actor_critic)

if __name__ == "__main__":
    random_number = random.randint(10000, 15000)
    port = str(random_number)  # os.environ["ROS_PORT_SIM"]
    os.environ["ROS_MASTER_URI"] = "http://localhost:" + port
    subprocess.Popen(["roscore", "-p", port])
    time.sleep(2)
    print("Roscore launched!")

    # Launch the simulation with the given launchfile name
    subprocess.Popen(["rosrun", "stage_ros1", "stageros1", "Obstacles3.world"])
    print("environment launched!")
    time.sleep(2)
    main()