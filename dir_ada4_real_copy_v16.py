'''
dot,
4 points are used
'''
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import gym
import time
import core_denseu_ada4_no_goal as core
from core_denseu_ada4_no_goal import get_vars
from stage_dir2_copy_1 import StageWorld
import random
import rospy
import os
import signal
import subprocess
import sys
from collections import deque
import scipy.stats as stats
import math

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
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=128):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sac( actor_critic=core.mlp_actor_critic, seed=5, 
        steps_per_epoch=5000, epochs=10000, replay_size=int(5e5), gamma=0.99, 
        polyak=0.995, lr1=1e-4, lr2=1e-4,alpha=0.01, batch_size=100, start_epoch=100, 
        max_ep_len=200,MAX_EPISODE=10000):

#    logger = EpochLogger(**logger_kwargs)
#    logger.save_config(locals())
    sac=7
    obs_dim = 34
    act_dim = 2

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
#    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
#    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
    train = tf.placeholder(dtype=tf.bool, shape=None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi,ALP = actor_critic(x_ph, a_ph,train)
    
    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, q1_pi_targ, q2_pi_targ,_  = actor_critic(x2_ph, a_ph,train)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi_targ, q2_pi_targ)
    min_q = tf.minimum(q1_pi, q2_pi)
#    min_q_pi = tf.maximum(min_q_pi, 10.0)
#    min_q_pi = tf.minimum(min_q_pi, (24-0.28+tf.log(4.0)))
#    min_q_pi = tf.minimum(min_q_pi, 24.0)

    # Targets for Q and V regression
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_backup)
    

#        policy_loss = (policy_kl_loss
#                       + policy_regularization_loss + regularization_penalty_pi - self.ent_coef * policy_entropy)

    # Soft actor-critic losses
    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q)
    q1_loss = tf.reduce_mean((q1-backup)**2)
    q2_loss = tf.reduce_mean((q2-backup)**2)
    q_loss =  q2_loss + q1_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr2)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr1)
    value_params = get_vars('main/q')
#    value_params1 = get_vars('main/q2')
#    with tf.control_dependencies([train_pi_op]):
    train_q_op = value_optimizer.minimize(q_loss, var_list=value_params)
#    with tf.control_dependencies([train_value_op1]):    
#        train_value_op2 = value_optimizer.minimize(q2_loss, var_list=value_params1)
    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
#    with tf.control_dependencies([train_value_op2]):
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops1 = [q_loss, q1, q2, train_q_op]
    step_ops2 = [pi_loss, train_pi_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    sess = tf.Session(config=config)
    reward_var = tf.Variable(0., trainable=False)
    robot_size_var = tf.Variable(0., trainable=False)
    average_speed_var = tf.Variable(0., trainable=False)
    goal_reach_var = tf.Variable(0., trainable=False)
    reward_epi = tf.summary.scalar('reward', reward_var)
    robot_size_epi = tf.summary.scalar('robot_size', robot_size_var)
    average_speed_epi = tf.summary.scalar('average_speed', average_speed_var)
    goal_reach_epi = tf.summary.scalar('goal_reach', goal_reach_var)
    # define summary
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./logssac'+str(sac), sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    trainables = tf.trainable_variables()
    trainable_saver = tf.train.Saver(trainables,max_to_keep=None)
    sess.run(tf.global_variables_initializer())
    # trainable_saver.restore(sess,"/home/eias/20240118/Monocular-Obstacle-Avoidance/D3QN/input_compare/real/network_new_5/sac7lambda4-25")
    # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
    # outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    def get_action(o, deterministic=False,istrain= False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1),train:istrain})[0]
    def print_alpha(o,istrain= False):
        return sess.run(ALP, feed_dict={x_ph: o.reshape(1,-1),train:istrain})

    # Main loop: collect experience in env and update/log each epoch
    episode=0
    T = 0
    env = StageWorld(540)
    rate = rospy.Rate(10)
    epi_thr = 0
    goal_reach=0
    robot_size_bound = 0.2
    test_scores_window = deque(maxlen=10)
    result_plot = np.zeros((MAX_EPISODE,3))
    test_result_plot = np.zeros((5,250,30,5))
    env_record = np.zeros((5,4))
    train_result = np.zeros((5,12000))
    sac = 107001535
    test_time = 0
    for hyper_exp in range(4,5):
        scores_window = deque(maxlen=50)
        low_bound = 1.0
        best_value = 0
        seed = hyper_exp
        tf.set_random_seed(seed)
        np.random.seed(seed)
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        sess.run(tf.global_variables_initializer())
        sess.run(target_init)
        trainables = tf.trainable_variables()
        trainable_saver = tf.train.Saver(trainables,max_to_keep=None)
        # trainable_saver.restore(sess,"/home/eias/20240118/Monocular-Obstacle-Avoidance/D3QN/input_compare/real/network_new_5/sac7lambda4-25")
        sess.run(tf.global_variables_initializer())
        episode=0
        T = 0
        epi_thr = 0
        goal_reach=0
        test_time = 0
        train_env =[0]
        new_env = 0
        T_last=0
        obs_metric_flage = 0
        while test_time<20:
            env_no = 0
            print("train env no. is",env_no,"total trained env is",new_env)
            if  episode <= start_epoch or goal_reach==0:
                robot_size = env.ResetWorld(env_no)
            else:
                robot_size = env.Reset(env_no)
            env.GenerateTargetPoint(env_no)
            o, r, d,goal_reach,obs_change =env.step()
            while d:
                reset = False
                robot_size = env.ResetWorld(env_no)
                env.GenerateTargetPoint(env_no)
                o, r, d,goal_reach,obs_change =env.step()
            reset = False
#            power= max(4.0*(0.995**episode),0.5)
            return_epoch =0
            total_vel = 0
            ep_len = 0
            obs_metric_1 = 1.0
            obs_metric_2 = 2.0
            obs_metric_3 = 2.5
            obs_metric_4 = 3.0
            obs_metric_5 = 3.5
    #        o = np.reshape(o,[1,56])
            d = False
            v_ratio = 0.5
            while not reset:
                if episode > start_epoch:
#                    if env_no<1:
#                        a = get_action(o, deterministic=False,istrain= True)
#                    else:
                    a= get_action(o, deterministic=False,istrain= False)
                else:
                    a = [0,0]
                    a[0] = np.random.uniform(-1,1)
                    a[1] = np.random.uniform(-1,1)
                # Step the env
                env.Control(a)
                rate.sleep()
                env.Control(a)
                rate.sleep()
                o2, r, d,goal_reach,obs_change= env.step()
################################# code start here ################################# 
                if np.mean(scores_window)>=0.90 and obs_metric_flage == 0:
                    obs_metric_flage = 1
                    scores_window = deque(maxlen=50)
                    for initial_zero in range(50):
                        scores_window.append(0)
                if np.mean(scores_window)>=0.90 and obs_metric_flage == 1:
                    obs_metric_flage = 2
                    scores_window = deque(maxlen=50)
                    for initial_zero in range(50):
                        scores_window.append(0)
                if np.mean(scores_window)>=0.90 and obs_metric_flage == 2:
                    obs_metric_flage = 3
                    scores_window = deque(maxlen=50)
                    for initial_zero in range(50):
                        scores_window.append(0)
                if np.mean(scores_window)>=0.90 and obs_metric_flage == 3:
                    obs_metric_flage = 4
                    scores_window = deque(maxlen=50)
                    for initial_zero in range(50):
                        scores_window.append(0)
################################# code start here #################################
                if obs_metric_flage == 0 :
                    if a[0] > 0.6:
                       r = r + v_ratio * a[0]
                    if a[0] < -0.6:
                       r = r + v_ratio * a[0]
                    obs_change = (obs_change -1.0)*10.0 + 1.0
                    if obs_change > 1.0:
                       obs_metric_r = (2.0 /obs_change) - 1.9
                    else :
                       obs_metric_r = (2.0 / (2.0 - obs_change)) - 1.9
                    if math.isnan(obs_metric_r):
                       obs_metric_r = 0
                    obs_metric_r = obs_metric_r * obs_metric_1
                    r = r + obs_metric_r
                if obs_metric_flage == 1 :
                    obs_change = (obs_change -1.0)*10.0 + 1.0
                    if obs_change > 1.0:
                       obs_metric_r = (2.0 /obs_change) - 1.9
                    else :
                       obs_metric_r = (2.0 / (2.0 - obs_change)) - 1.9
                    if math.isnan(obs_metric_r):
                       obs_metric_r = 0
                    obs_metric_r = obs_metric_r * obs_metric_2
                    r = r + obs_metric_r
                if obs_metric_flage == 2 :
                    obs_change = (obs_change -1.0)*10.0 + 1.0
                    if obs_change > 1.0:
                       obs_metric_r = (2.0 /obs_change) - 1.9
                    else :
                       obs_metric_r = (2.0 / (2.0 - obs_change)) - 1.9
                    if math.isnan(obs_metric_r):
                       obs_metric_r = 0
                    obs_metric_r = obs_metric_r * obs_metric_3
                    r = r + obs_metric_r
                if obs_metric_flage == 3 :
                    obs_change = (obs_change -1.0)*10.0 + 1.0
                    if obs_change > 1.0:
                       obs_metric_r = (2.0 /obs_change) - 1.9
                    else :
                       obs_metric_r = (2.0 / (2.0 - obs_change)) - 1.9
                    if math.isnan(obs_metric_r):
                       obs_metric_r = 0
                    obs_metric_r = obs_metric_r * obs_metric_4
                    r = r + obs_metric_r
                if obs_metric_flage == 4 :
                    obs_change = (obs_change -1.0)*10.0 + 1.0
                    if obs_change > 1.0:
                       obs_metric_r = (2.0 /obs_change) - 1.9
                    else :
                       obs_metric_r = (2.0 / (2.0 - obs_change)) - 1.9
                    if math.isnan(obs_metric_r):
                       obs_metric_r = 0
                    obs_metric_r = obs_metric_r * obs_metric_5
                    r = r + obs_metric_r
################################# code start here ################################# 
                return_epoch = return_epoch + r
                total_vel = total_vel + a[0]          
                replay_buffer.store(o, a, r, o2, d)
                if  episode > start_epoch:
                    T=T+1 
                    batch = replay_buffer.sample_batch(batch_size)
                    feed_dict = {x_ph: batch['obs1'],
                                 x2_ph: batch['obs2'],
                                 a_ph: batch['acts'],
                                 r_ph: batch['rews'],
                                 d_ph: batch['done'],
                                 train:True
                                }
                # print(batch['obs1'])
                    outs = sess.run(step_ops1, feed_dict) 
                    outs = sess.run(step_ops2, feed_dict)
                ep_len += 1
                # Super critical, easy to overlook step: make sure to update 
                # most recent observation!
                o = o2
                if ep_len >= max_ep_len:
                    print("Time out")
                if d or (ep_len >= max_ep_len):
                    if episode > start_epoch and new_env == env_no:
                        scores_window.append(goal_reach)
                    reset = True
                if  episode > start_epoch:
                    if T%3000==0:
                        test_time = test_time+1
                        print("trainable_saver: ")
                        if test_time % 1 == 0:
                            trainable_saver.save(sess, '/home/eias/20240118/Monocular-Obstacle-Avoidance/D3QN/input_compare/real/1125/network_minpool_new4_suqare_01535/sac'+str(sac)+'lambda'+str(hyper_exp), global_step=test_time)    
            train_result[hyper_exp,episode] = np.mean(scores_window)
            print("EPISODE"+str(sac)+str(hyper_exp), episode, "/ REWARD", return_epoch,"/ class_num ",obs_metric_flage, "/ steps ", T,"success rate",np.mean(scores_window))
            episode = episode + 1
            epi_thr = epi_thr+1
        np.save('train_result'+str(sac)+'.npy',train_result)
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
    subprocess.Popen(["rosrun","stage_ros1", "stageros", "/home/eias/20240118/Monocular-Obstacle-Avoidance/D3QN/input_compare/real/worlds/d6_1.world"])
    print ("environment launched!")
    time.sleep(2)
    main()
