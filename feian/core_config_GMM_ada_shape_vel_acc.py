import numpy as np
#import tensorflow._api.v2.compat.v1 as tf
#tf.disable_v2_behavior()
# ... existing code ...
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# ... existing code ...
EPS = 1e-8
def clip_but_pass_gradient2(x, l=EPS):
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((l - x)*clip_low)
def new_relu(x, alpha_actv):
    r = tf.math.reciprocal(clip_but_pass_gradient2(x+alpha_actv,l=EPS))
    return r #+ part_3*0
def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

# 标准MLP网络实现
def mlp(x, hidden_sizes=(32,), activation=tf.nn.leaky_relu, output_activation=None):
    """
    标准多层感知器网络
    参数:
        x: 输入张量
        hidden_sizes: 隐藏层神经元数量列表
        activation: 激活函数
        output_activation: 输出层激活函数
    返回:
        MLP网络输出
    """
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

# 策略MLP网络，与标准MLP略有不同
def mlp_policy(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """
    策略网络专用MLP，使用tanh激活
    参数:
        x: 输入张量
        hidden_sizes: 隐藏层神经元数量列表
        activation: 激活函数
        output_activation: 输出层激活函数
    返回:
        MLP网络输出
    """
    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)
    return x

def CNN(x, y,activation=tf.nn.relu, output_activation=None):
    x1 = tf.layers.conv1d(x, filters=32, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x11 = tf.layers.conv1d(x1, filters=64, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x0 = tf.layers.conv1d(x11, filters=128, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x3 = tf.layers.max_pooling1d(x0, pool_size=90, strides=1, padding='valid')
    x3_flatten = tf.layers.flatten(x3)
    return x3_flatten
def CNN_dense(x,activation=tf.nn.leaky_relu, output_activation=None):
    alpha_actv2= tf.Variable(initial_value=0.0, dtype='float32', trainable=True)
    x_input = x[:,0:6*90]
    x_input = tf.reshape(x_input,[-1,90,6])
    x00 = new_relu(x_input[:,:,2], alpha_actv2)
    x_input = tf.concat([x_input[:,:,0:2],tf.reshape(x00,[-1,90,1]),x_input[:,:,3:6]], axis=-1)
    x1 = tf.layers.conv1d(x_input, filters=32, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x11 = tf.layers.conv1d(x1, filters=64, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x0 = tf.layers.conv1d(x11, filters=128, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x3 = tf.layers.max_pooling1d(x0, pool_size=90, strides=1, padding='valid')
    x3_flatten = tf.layers.flatten(x3)
    return tf.concat([x3_flatten,x[:,6*90:]], axis=-1)
def CNN2(x, activation=tf.nn.relu, output_activation=None,):
    x1 = tf.layers.conv1d(x, filters=32, kernel_size=20, strides=10, padding='same',activation=tf.nn.relu)
    x2 = tf.layers.conv1d(x1, filters=16, kernel_size=10, strides=3, padding='same',activation=tf.nn.relu)
    x2_flatten = tf.layers.flatten(x2)
    x3 = tf.layers.dense(x2_flatten, units=128, activation=tf.nn.relu)
    return x3
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def create_log_gaussian(mu_t, log_sig_t, t):
    normalized_dist_t = (t - mu_t) * tf.exp(-log_sig_t)  # ... x D
    quadratic = - 0.5 * tf.reduce_sum(normalized_dist_t ** 2, axis=-1)
    # ... x (None)

    log_z = tf.reduce_sum(log_sig_t, axis=-1)  # ... x (None)
    D_t = tf.cast(tf.shape(mu_t)[-1], tf.float32)
    log_z += 0.5 * D_t * np.log(2 * np.pi)

    log_p = quadratic - log_z

    return log_p  # ... x (None)


def mlp_gaussian_policy(x, a,hidden_sizes, activation, output_activation,alpha_actv1):
    """
    基于高斯混合模型(GMM)的随机策略网络
    参数:
        x: 输入状态向量 [batch_size, obs_dim]
        a: 动作占位符 [batch_size, act_dim]
        hidden_sizes: 隐藏层神经元数量列表
        activation: 激活函数
        output_activation: 输出层激活函数
        alpha_actv1: 激光距离激活函数的可学习参数
    返回:
        mu: 均值动作 [batch_size, act_dim]
        pi: 采样动作 [batch_size, act_dim]
        logp_pi: 采样动作的对数概率 [batch_size]
    """
    k = 4  # 高斯混合模型的组件数
    act_dim = a.shape.as_list()[-1]  # 动作维度，通常为2(线速度和角速度)
    
    # 提取并重塑激光数据部分
    x_input = x[:,0:6*90]  # 取前540维激光数据
    x_input = tf.reshape(x_input,[-1,90,6])  # 重塑为[batch_size, 90, 6]
    
    # 对第3通道(距离)应用自定义激活函数
    x0 = new_relu(x_input[:,:,2], alpha_actv1)  # 距离的倒数激活 reciprocal
    # 重组特征通道
    x_input = tf.concat([x_input[:,:,0:2], tf.reshape(x0,[-1,90,1]), x_input[:,:,3:6]], axis=-1)
    
    # 提取目标位置和当前速度等信息(紧跟激光数据的8维)
    w_input = x[:,6*90:6*90+8]
    w_input = tf.reshape(w_input,[-1,8])
    
    # 通过CNN处理激光数据
    cnn_net = CNN(x_input, w_input)  # 输出维度[batch_size, 128]  只处理了x_input
    
    # 合并CNN处理后的540特征和后8维原始特征 
    y = tf.concat([cnn_net, x[:,6*90:]], axis=-1)
    
    # 通过策略MLP网络处理
    net = mlp_policy(y, list(hidden_sizes), activation, activation)
    
    # 输出层：为GMM生成所有参数
    # 每个动作维度有k个混合成分，每个成分有权重、均值、方差，总共(2*act_dim+1)*k
    w_and_mu_and_logsig_t = tf.layers.dense(net, (act_dim*2+1)*k, activation=output_activation)
    w_and_mu_and_logsig_t = tf.reshape(w_and_mu_and_logsig_t, shape=(-1, k, 2*act_dim+1))
    
    # 分离GMM参数：权重、均值、对数标准差
    log_w_t = w_and_mu_and_logsig_t[..., 0]  # 混合权重的对数 [batch_size, k]
    mu_t = w_and_mu_and_logsig_t[..., 1:1+act_dim]  # 均值 [batch_size, k, act_dim]
    log_sig_t = w_and_mu_and_logsig_t[..., 1+act_dim:]  # 对数标准差 [batch_size, k, act_dim]
    
    # 对对数标准差进行范围约束，增加数值稳定性
    log_sig_t = tf.tanh(log_sig_t)
    log_sig_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig_t + 1)
    xz_sigs_t = tf.exp(log_sig_t)  # 标准差 [batch_size, k, act_dim]
    
    # 采样潜在成分z
    z_t = tf.multinomial(logits=log_w_t, num_samples=1)  # [batch_size, 1]
    
    # 根据采样的z选择对应的混合成分
    mask_t = tf.one_hot(z_t[:, 0], depth=k, dtype=tf.bool, on_value=True, off_value=False)
    xz_mu_t = tf.boolean_mask(mu_t, mask_t)  # 选中的均值 [batch_size, act_dim]
    xz_sig_t = tf.boolean_mask(xz_sigs_t, mask_t)  # 选中的标准差 [batch_size, act_dim]
    
    # 基于选定的混合成分采样动作
    x_t = xz_mu_t + xz_sig_t * tf.random_normal((tf.shape(net)[0], act_dim))  # [batch_size, act_dim]
    
    # 计算采样动作的对数概率密度
    log_p_xz_t = create_log_gaussian(mu_t, log_sig_t, x_t[:, None, :])  # [batch_size, k]
    
    # 计算边缘概率密度(所有混合成分的加权和)
    log_p_x_t = tf.reduce_logsumexp(log_p_xz_t + log_w_t, axis=1)
    log_p_x_t -= tf.reduce_logsumexp(log_w_t, axis=1)  # 归一化项
    
    logp_pi = log_p_x_t  # 最终的对数概率 [batch_size]
    
    return xz_mu_t, x_t, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
#    pi_run = pi
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(x, a,hidden_sizes=(128,128,128,128), activation=tf.nn.leaky_relu, 
                     output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope('pi'):
        alpha_actv1 = tf.Variable(initial_value=0.0, dtype='float32', trainable=True)
        mu, pi, logp_pi = policy(x, a,[128,128,128,128], activation, output_activation,alpha_actv1)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
#    action_scale = action_space.high[0]
#    mu *= action_scale
#    pi *= action_scale

    # vfs
    # the dim of q function and v function is 1, and hence it use +[1]
#    vf_cnn = lambda x : CNN(x)
#    vf_mlp = lambda y : tf.squeeze(mlp(y, list(hidden_sizes)+[1], activation, None), axis=1)
#    with tf.variable_scope('q1'):
#        q1 = vf_mlp(tf.concat([x,a], axis=-1))
#    with tf.variable_scope('q1', reuse=True):
#        q1_pi = vf_mlp(tf.concat([x,pi], axis=-1))
#    with tf.variable_scope('q2'):
#        q2 = vf_mlp(tf.concat([x,a], axis=-1))
#    with tf.variable_scope('q2', reuse=True):
#        q2_pi = vf_mlp(tf.concat([x,pi], axis=-1))
#    with tf.variable_scope('v'):
#        v = vf_mlp(x)        
    vf_mlp = lambda y : tf.squeeze(mlp(y, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('values'):   
        with tf.variable_scope('CNN'):
            y = CNN_dense(x,activation, None)
        with tf.variable_scope('q1'):
            q1 = vf_mlp(tf.concat([y,a], axis=-1))
        with tf.variable_scope('q1', reuse=True):
            q1_pi = vf_mlp(tf.concat([y,pi], axis=-1))
        with tf.variable_scope('q2'):
            q2 = vf_mlp(tf.concat([y,a], axis=-1))
        with tf.variable_scope('q2', reuse=True):
            q2_pi = vf_mlp(tf.concat([y,pi], axis=-1))
    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi
