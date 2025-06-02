# -*- coding: utf-8 -*-
"""
SAC强化学习神经网络架构配置文件
===============================

本文件定义了Soft Actor-Critic算法的神经网络架构，包括：
1. CNN特征提取网络 - 处理激光雷达数据
2. 高斯混合策略网络 - 生成多模态动作策略
3. 双Q价值网络 - 评估状态-动作价值
4. 各种辅助函数 - 支持网络训练和数值稳定

调用关系：被SPN_ours_1e6.py主训练文件调用
"""

import numpy as np
import tensorflow as tf
tf.disable_v2_behavior()

# ============= 全局常量 =============
EPS = 1e-8          # 防止除零的极小值
LOG_STD_MAX = 2     # 策略网络对数标准差上界
LOG_STD_MIN = -20   # 策略网络对数标准差下界


def clip_but_pass_gradient2(x, l=EPS):
    """
    带梯度传递的下界裁剪函数
    
    作用：对输入进行下界裁剪，但保持梯度传播
    参数：
        x - 输入张量
        l - 下界值
    返回：裁剪后的张量
    """
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((l - x) * clip_low)


def new_relu(x, alpha_actv):
    """
    自定义激活函数
    
    作用：实现倒数形式的自适应激活函数 1/(x + α + ε)
    参数：
        x - 输入特征
        alpha_actv - 可训练的激活参数
    返回：激活后的特征
    """
    r = tf.math.reciprocal(clip_but_pass_gradient2(x + alpha_actv, l=EPS))
    return r


def placeholder(dim=None):
    """
    创建TensorFlow占位符
    
    作用：统一创建浮点型占位符
    参数：
        dim - 特征维度，None表示标量
    返回：tf.placeholder对象
    """
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    """
    批量创建占位符
    
    作用：一次性创建多个占位符
    参数：
        args - 各占位符的维度列表
    返回：占位符列表
    """
    return [placeholder(dim) for dim in args]


def mlp(x, hidden_sizes=(32,), activation=tf.nn.leaky_relu, output_activation=None):
    """
    多层感知机网络
    
    作用：构建标准的全连接神经网络
    参数：
        x - 输入特征
        hidden_sizes - 隐藏层神经元数量元组
        activation - 隐藏层激活函数
        output_activation - 输出层激活函数
    返回：网络输出
    """
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def mlp_policy(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """
    策略网络专用MLP
    
    作用：为策略网络构建MLP，使用tanh激活保证训练稳定性
    参数：
        x - 输入特征
        hidden_sizes - 隐藏层大小
        activation - 激活函数，默认tanh
        output_activation - 输出激活函数
    返回：处理后的特征
    """
    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)
    return x


def CNN(x, y, activation=tf.nn.relu, output_activation=None):
    """
    基础CNN网络
    
    作用：使用1D卷积处理序列数据
    参数：
        x - 输入张量 [batch_size, sequence_length, features]
        y - 未使用参数（保持接口兼容性）
        activation - 激活函数
        output_activation - 输出激活函数
    返回：扁平化的特征向量
    """
    x1 = tf.layers.conv1d(x, filters=32, kernel_size=1, strides=1, padding='valid', activation=tf.nn.leaky_relu)
    x11 = tf.layers.conv1d(x1, filters=64, kernel_size=1, strides=1, padding='valid', activation=tf.nn.leaky_relu)
    x0 = tf.layers.conv1d(x11, filters=128, kernel_size=1, strides=1, padding='valid', activation=tf.nn.leaky_relu)
    x3 = tf.layers.max_pooling1d(x0, pool_size=90, strides=1, padding='valid')
    x3_flatten = tf.layers.flatten(x3)
    return x3_flatten


def CNN_dense(x, activation=tf.nn.leaky_relu, output_activation=None):
    """
    增强型CNN特征提取器
    
    作用：处理激光雷达数据的主要特征提取网络
    输入格式：
        x[:,0:540] - 激光雷达数据(90个方向×6个特征)
        x[:,540:] - 其他状态信息(目标位置、速度等)
    
    参数：
        x - 完整状态输入 [batch_size, 548]
        activation - 卷积层激活函数
        output_activation - 输出激活函数
    返回：CNN特征与其他状态信息的拼接 [batch_size, 136]
    """
    # 定义可训练激活参数
    alpha_actv2 = tf.Variable(initial_value=0.0, dtype='float32', trainable=True)
    
    # 提取和重塑激光雷达数据
    x_input = x[:, 0:6*90]  # 激光雷达数据段
    x_input = tf.reshape(x_input, [-1, 90, 6])  # [batch, 90方向, 6特征]
    
    # 对距离特征应用自定义激活
    x00 = new_relu(x_input[:, :, 2], alpha_actv2)  # 处理距离信息(第3个特征)
    
    # 重新组合特征
    x_input = tf.concat([
        x_input[:, :, 0:2],          # 方向信息(cos, sin)
        tf.reshape(x00, [-1, 90, 1]), # 处理后的距离
        x_input[:, :, 3:6]           # 机器人几何信息(length1, length2, width)
    ], axis=-1)
    
    # 三层1D卷积提取特征
    x1 = tf.layers.conv1d(x_input, filters=32, kernel_size=1, strides=1, padding='valid', activation=tf.nn.leaky_relu)
    x11 = tf.layers.conv1d(x1, filters=64, kernel_size=1, strides=1, padding='valid', activation=tf.nn.leaky_relu)
    x0 = tf.layers.conv1d(x11, filters=128, kernel_size=1, strides=1, padding='valid', activation=tf.nn.leaky_relu)
    
    # 全局最大池化
    x3 = tf.layers.max_pooling1d(x0, pool_size=90, strides=1, padding='valid')
    x3_flatten = tf.layers.flatten(x3)
    
    # 拼接CNN特征和其他状态信息
    return tf.concat([x3_flatten, x[:, 6*90:]], axis=-1)


def CNN2(x, activation=tf.nn.relu, output_activation=None):
    """
    备用CNN架构
    
    作用：使用大卷积核处理时序数据的替代CNN实现
    参数：
        x - 输入时序数据
        activation - 激活函数
        output_activation - 输出激活函数
    返回：时序特征表示
    """
    x1 = tf.layers.conv1d(x, filters=32, kernel_size=20, strides=10, padding='same', activation=tf.nn.relu)
    x2 = tf.layers.conv1d(x1, filters=16, kernel_size=10, strides=3, padding='same', activation=tf.nn.relu)
    x2_flatten = tf.layers.flatten(x2)
    x3 = tf.layers.dense(x2_flatten, units=128, activation=tf.nn.relu)
    return x3


def get_vars(scope):
    """
    获取指定作用域的变量
    
    作用：获取特定命名空间下的所有TensorFlow变量
    参数：
        scope - 变量作用域名称(如'main/pi', 'main/q1')
    返回：变量列表
    """
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    """
    计算作用域参数数量
    
    作用：统计指定作用域下的总参数数量
    参数：
        scope - 变量作用域名称
    返回：参数总数(整数)
    """
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    """
    高斯分布对数似然
    
    作用：计算多维高斯分布的对数概率密度
    参数：
        x - 观测值
        mu - 均值
        log_std - 对数标准差
    返回：对数似然值
    """
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def clip_but_pass_gradient(x, l=-1., u=1.):
    """
    双边梯度保持裁剪
    
    作用：对输入进行上下界裁剪，同时保持梯度传播
    参数：
        x - 输入张量
        l - 下界
        u - 上界
    返回：裁剪后的张量
    """
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


def create_log_gaussian(mu_t, log_sig_t, t):
    """
    高斯混合模型概率计算
    
    作用：计算高斯混合模型中每个组件的对数概率密度
    参数：
        mu_t - 均值张量 [..., K, D]
        log_sig_t - 对数标准差张量 [..., K, D]
        t - 样本点张量 [..., 1, D]
    返回：对数概率密度 [..., K]
    """
    # 计算标准化距离
    normalized_dist_t = (t - mu_t) * tf.exp(-log_sig_t)
    
    # 计算二次项
    quadratic = -0.5 * tf.reduce_sum(normalized_dist_t ** 2, axis=-1)

    # 计算归一化常数
    log_z = tf.reduce_sum(log_sig_t, axis=-1)
    D_t = tf.cast(tf.shape(mu_t)[-1], tf.float32)
    log_z += 0.5 * D_t * np.log(2 * np.pi)

    log_p = quadratic - log_z
    return log_p


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, alpha_actv1):
    """
    高斯混合策略网络
    
    作用：实现基于高斯混合模型的多模态策略网络
    网络流程：状态输入 → CNN特征提取 → MLP处理 → GMM参数生成 → 动作采样
    
    参数：
        x - 状态输入 [batch_size, 548]
        a - 动作张量(用于获取动作维度)
        hidden_sizes - MLP隐藏层配置
        activation - 激活函数
        output_activation - 输出激活函数
        alpha_actv1 - CNN自适应激活参数
    
    GMM参数：
        k=4 - 高斯组件数量
        每个组件包含：1个权重 + act_dim个均值 + act_dim个标准差
    
    返回：
        xz_mu_t - 选中组件的均值 [batch_size, act_dim]
        x_t - 采样的动作 [batch_size, act_dim]
        logp_pi - 动作的对数概率 [batch_size]
    """
    k = 4  # 高斯混合组件数量
    act_dim = a.shape.as_list()[-1]  # 动作维度(2维：线速度和角速度)
    
    # 处理激光雷达数据
    x_input = x[:, 0:6*90]
    x_input = tf.reshape(x_input, [-1, 90, 6])
    
    # 对距离特征应用自定义激活
    x0 = new_relu(x_input[:, :, 2], alpha_actv1)
    x_input = tf.concat([
        x_input[:, :, 0:2],
        tf.reshape(x0, [-1, 90, 1]),
        x_input[:, :, 3:6]
    ], axis=-1)
    
    # 提取其他状态信息
    w_input = x[:, 6*90:6*90+8]
    w_input = tf.reshape(w_input, [-1, 8])
    
    # CNN特征提取
    cnn_net = CNN(x_input, w_input)
    
    # 特征拼接
    y = tf.concat([cnn_net, x[:, 6*90:]], axis=-1)
    
    # MLP处理
    net = mlp_policy(y, list(hidden_sizes), activation, activation)
    
    # 生成GMM参数：(权重 + 均值 + 对数标准差) × k个组件
    w_and_mu_and_logsig_t = tf.layers.dense(net, (act_dim*2+1)*k, activation=output_activation)
    w_and_mu_and_logsig_t = tf.reshape(w_and_mu_and_logsig_t, shape=(-1, k, 2*act_dim+1))
    
    # 分离参数
    log_w_t = w_and_mu_and_logsig_t[..., 0]              # 混合权重 [N, K]
    mu_t = w_and_mu_and_logsig_t[..., 1:1+act_dim]       # 均值 [N, K, act_dim]
    log_sig_t = w_and_mu_and_logsig_t[..., 1+act_dim:]   # 对数标准差 [N, K, act_dim]

    # 约束对数标准差到安全范围
    log_sig_t = tf.tanh(log_sig_t)
    log_sig_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig_t + 1)
    xz_sigs_t = tf.exp(log_sig_t)

    # 采样高斯组件
    z_t = tf.multinomial(logits=log_w_t, num_samples=1)  # 选择组件
    mask_t = tf.one_hot(z_t[:, 0], depth=k, dtype=tf.bool, on_value=True, off_value=False)
    
    # 获取选中组件的参数
    xz_mu_t = tf.boolean_mask(mu_t, mask_t)      # 选中组件均值
    xz_sig_t = tf.boolean_mask(xz_sigs_t, mask_t) # 选中组件标准差

    # 重参数化采样：a = μ + σ * ε
    x_t = xz_mu_t + xz_sig_t * tf.random_normal((tf.shape(net)[0], act_dim))

    # 计算动作概率
    log_p_xz_t = create_log_gaussian(mu_t, log_sig_t, x_t[:, None, :])  # 各组件概率
    log_p_x_t = tf.reduce_logsumexp(log_p_xz_t + log_w_t, axis=1)      # 边际概率
    log_p_x_t -= tf.reduce_logsumexp(log_w_t, axis=1)                   # 归一化

    logp_pi = log_p_x_t
    return xz_mu_t, x_t, logp_pi


def apply_squashing_func(mu, pi, logp_pi):
    """
    Tanh压缩函数及雅可比校正
    
    作用：将无界策略输出映射到有界动作空间[-1,1]，并进行概率密度校正
    数学原理：a = tanh(u), log p(a) = log p(u) - log|det(∂a/∂u)|
    
    参数：
        mu - 策略均值(压缩前)
        pi - 采样动作(压缩前)
        logp_pi - 原始对数概率
    返回：
        压缩后的均值、动作和校正后的对数概率
    """
    # 应用tanh压缩
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    
    # 雅可比校正：log|det(∂tanh(u)/∂u)| = log(1 - tanh²(u))
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    
    return mu, pi, logp_pi


def mlp_actor_critic(x, a, hidden_sizes=(128,128,128,128), 
                     activation=tf.nn.leaky_relu, output_activation=None, 
                     policy=mlp_gaussian_policy, action_space=None):
    """
    Actor-Critic网络架构
    
    作用：构建完整的SAC算法神经网络，包括策略网络(Actor)和价值网络(Critic)
    
    网络结构：
        状态输入 → 共享CNN特征提取 → 分支为Actor和Critic
        Actor: 高斯混合策略网络
        Critic: 双Q网络(Q1和Q2)
    
    参数：
        x - 状态输入 [batch_size, 548]
        a - 动作输入 [batch_size, 2]
        hidden_sizes - MLP隐藏层配置，默认4层128神经元
        activation - 激活函数
        output_activation - 输出激活函数
        policy - 策略网络函数
        action_space - 动作空间(未使用)
    
    返回：
        mu - 策略均值 [batch_size, act_dim]
        pi - 采样动作 [batch_size, act_dim]
        logp_pi - 动作对数概率 [batch_size]
        q1, q2 - 双Q网络输出 [batch_size]
        q1_pi, q2_pi - 当前策略下的Q值 [batch_size]
    """
    
    # ============= Actor网络(策略网络) =============
    with tf.variable_scope('pi'):
        # CNN自适应激活参数
        alpha_actv1 = tf.Variable(initial_value=0.0, dtype='float32', trainable=True)
        
        # 构建高斯混合策略网络
        mu, pi, logp_pi = policy(x, a, [128,128,128,128], activation, output_activation, alpha_actv1)
        
        # 应用tanh压缩到有界动作空间
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # ============= Critic网络(价值网络) =============
    # Q网络MLP：输入特征+动作，输出Q值
    vf_mlp = lambda y: tf.squeeze(mlp(y, list(hidden_sizes)+[1], activation, None), axis=1)
    
    with tf.variable_scope('values'):   
        # 共享CNN特征提取器
        with tf.variable_scope('CNN'):
            y = CNN_dense(x, activation, None)  # 提取136维特征
        
        # 第一个Q网络
        with tf.variable_scope('q1'):
            q1 = vf_mlp(tf.concat([y, a], axis=-1))        # Q1(s,a)
        with tf.variable_scope('q1', reuse=True):
            q1_pi = vf_mlp(tf.concat([y, pi], axis=-1))    # Q1(s,π(s))
        
        # 第二个Q网络
        with tf.variable_scope('q2'):
            q2 = vf_mlp(tf.concat([y, a], axis=-1))        # Q2(s,a)
        with tf.variable_scope('q2', reuse=True):
            q2_pi = vf_mlp(tf.concat([y, pi], axis=-1))    # Q2(s,π(s))
    
    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi