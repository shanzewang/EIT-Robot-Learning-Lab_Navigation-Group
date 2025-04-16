import numpy as np

# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
# ... existing code ...
import tensorflow as tf

tf.disable_v2_behavior()
# ... existing code ...
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
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(
        obs_dim, act_dim, obs_dim, None, None
    )

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = actor_critic(x_ph, a_ph)

    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, q1_pi_targ, q2_pi_targ = actor_critic(x2_ph, a_ph)

    # Initializing targets to match main variables
    target_init = tf.group(
        [
            tf.assign(v_targ, v_main)
            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
        ]
    )

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    reward_var = tf.Variable(0.0, trainable=False)
    robot_size_var = tf.Variable(0.0, trainable=False)
    average_speed_var = tf.Variable(0.0, trainable=False)
    goal_reach_var = tf.Variable(0.0, trainable=False)
    reward_epi = tf.summary.scalar('reward', reward_var)
    robot_size_epi = tf.summary.scalar('robot_size', robot_size_var)
    average_speed_epi = tf.summary.scalar('average_speed', average_speed_var)
    goal_reach_epi = tf.summary.scalar('goal_reach', goal_reach_var)
    # define summary
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./logssac' + str(sac), sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    trainables = tf.trainable_variables()
    trainable_saver = tf.train.Saver(trainables, max_to_keep=None)
    sess.run(tf.global_variables_initializer())
    trainable_saver.restore(sess,"network/configback540GMMdense101lambda0-100")

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]
    
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
    # test_result_plot = np.load('test_result_plot1.npy')
    env_record = np.zeros((5, 4))
    train_result = np.zeros((5, 12000))
    #    np.save('suc_record_all'+str(sac)+'.npy',suc_record_all)
    #    np.save('suc_record_all_new'+str(sac)+'.npy',suc_record_all_new)
    #    np.save('test_result_plot'+str(sac)+'.npy',test_result_plot)
    test_time = 0

    for hyper_exp in range(4,5):

        # 初始化成功率窗口
        scores_window = deque(maxlen=50)  # 用于存储最近50个回合的成功率
        for initial_zero in range(50):    # 初始化成功率窗口为0
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
        train_result = np.zeros((5,12000))
        train_result1 = np.zeros((5,250000))

        while test_time < 20:
            env_no = 1  # obstacles3.world
            print("train env no. is Obstacles3")

            if episode <= start_epoch or goal_reach==0:
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
            return_epoch =0
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

                # T在每个训练步骤更新   
                if  episode > start_epoch:
                    T=T+1  # T记录有效训练步数
                ep_len += 1  # ep_len记录当前回合的步数
                o = o2
                if ep_len >= max_ep_len:
                    print("Time out")

                    # 回合结束时更新scores_window
                    if d or (ep_len >= max_ep_len):
                        if episode > start_epoch and new_env == env_no:
                            scores_window.append(goal_reach)
                        reset = True
                    if  episode > start_epoch:
                        train_result1[hyper_exp,T] = np.mean(scores_window)
                        if T%10000==0:
                            test_time = test_time+1
                train_result[hyper_exp,episode] = np.mean(scores_window)
                print("EPISODE"+str(sac)+str(hyper_exp), episode, "/ REWARD", return_epoch,"/ class_num ",obs_metric_flage, "/ steps ", T,"success rate",np.mean(scores_window))
                episode = episode + 1
                epi_thr = epi_thr+1
            np.save('train_result'+str(sac)+'.npy',train_result)
            np.save('train_result_1_'+str(sac)+'.npy',train_result1)
            break 


def main():
    sac(actor_critic=core.mlp_actor_critic)
if __name__ == '__main__':
    random_number = random.randint(10000, 15000)
    port = str(random_number) #os.environ["ROS_PORT_SIM"]
    os.environ["ROS_MASTER_URI"] = "http://localhost:"+port
    subprocess.Popen(["roscore", "-p", port])
    time.sleep(2)
    print ("Roscore launched!")

    # Launch the simulation with the given launchfile name
    subprocess.Popen(["rosrun","stage_ros1", "stageros", "Obstacles3.world"])
    print ("environment launched!")
    time.sleep(2)
    main()