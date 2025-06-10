import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# 常量定义
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = (x > u).float()
    clip_low = (x < l).float()
    return x + (u - x) * clip_up.detach()

def clip_but_pass_gradient2(x, l=EPS):
    clip_low = (x < l).float()
    return x + (l - x) * clip_low.detach()

def new_relu(x, alpha_actv):
    r = torch.reciprocal(clip_but_pass_gradient2(x + alpha_actv, l=EPS))
    return r

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def create_log_gaussian(mu_t, log_sig_t, t):
    normalized_dist_t = (t - mu_t) * torch.exp(-log_sig_t)
    quadratic = -0.5 * (normalized_dist_t ** 2).sum(dim=-1)
    log_z = log_sig_t.sum(dim=-1)
    D_t = mu_t.size(-1)
    log_z += 0.5 * D_t * np.log(2 * np.pi)
    log_p = quadratic - log_z
    return log_p

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=nn.LeakyReLU(), output_activation=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes[:-1]:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, hidden_sizes[-1]))
        if output_activation:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPPolicy(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=nn.Tanh()):
        super(MLPPolicy, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1)
        self.pool = nn.MaxPool1d(90)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        return x

class CNNDense(nn.Module):
    def __init__(self, input_dim):
        super(CNNDense, self).__init__()
        self.cnn = CNN()
        self.input_dim = input_dim
        self.alpha_actv2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_input = x[:, :6*90].view(-1, 6, 90)
        x0 = new_relu(x_input[:, 2, :], self.alpha_actv2)
        x_input = torch.cat([x_input[:, :2, :], x0.unsqueeze(1), x_input[:, 3:, :]], dim=1)
        cnn_output = self.cnn(x_input)
        return torch.cat([cnn_output, x[:, 6*90:]], dim=1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh(), k=4):
        super(GaussianPolicy, self).__init__()
        self.k = k
        self.act_dim = act_dim
        self.cnn_dense = CNNDense(obs_dim)
        # 添加注意力模块和动态MLP
        self.attention = AttentionModule(feature_dim=10)  # 每个行人特征维度为10
        self.dynamic_mlp = dynamic_MLP(input_dim=8 * 10, output_dim=128)  # 8个行人展平
        # 输入维度：136（原始特征）+ 128（行人特征）
        self.net = MLPPolicy(136 + 128, hidden_sizes, activation)
        self.w_mu_logsig = nn.Linear(hidden_sizes[-1], (act_dim * 2 + 1) * k)

    def forward(self, x, time_dynamic, deterministic=False, with_logprob=True):
        #print(f"Debug: time_dynamic shape = {time_dynamic.shape}")
        # 处理原始观测，得到 136 维特征
        y = self.cnn_dense(x)

        # 处理行人数据，形状 [batch_size, 3, 8, 10]
        spatial_attended = self.attention.spatial_attention(time_dynamic)  # [batch_size, 3, 8, 10]
        temporal_attended = self.attention.temporal_attention(spatial_attended)  # [batch_size, 3, 8, 10]
        # 取最后一帧特征并展平，[batch_size, 8*10]
        pedestrian_features = temporal_attended[:, -1, :, :].reshape(-1, 8 * 10)
        pedestrian_features = self.dynamic_mlp(pedestrian_features)  # [batch_size, 128]

        # 拼接特征，[batch_size, 136 + 128]
        combined_features = torch.cat([y, pedestrian_features], dim=1)

        # 传入策略网络
        net_output = self.net(combined_features)
        w_mu_logsig = self.w_mu_logsig(net_output).view(-1, self.k, 2 * self.act_dim + 1)
        log_w_t = w_mu_logsig[:, :, 0]
        mu_t = w_mu_logsig[:, :, 1:1 + self.act_dim]
        log_sig_t = w_mu_logsig[:, :, 1 + self.act_dim:]

        log_sig_t = torch.tanh(log_sig_t)
        log_sig_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig_t + 1)
        sig_t = torch.exp(log_sig_t)

        if deterministic:
            weights = F.softmax(log_w_t, dim=1)
            weighted_mu = (weights.unsqueeze(-1) * mu_t).sum(dim=1)
            xz_mu_t = weighted_mu
            pi_action = weighted_mu
        else:
            z_t = torch.multinomial(F.softmax(log_w_t, dim=1), num_samples=1)
            mask_t = torch.zeros_like(log_w_t).scatter_(1, z_t, 1).unsqueeze(-1)
            xz_mu_t = (mu_t * mask_t).sum(dim=1)
            xz_sig_t = (sig_t * mask_t).sum(dim=1)
            pi_action = xz_mu_t + xz_sig_t * torch.randn_like(xz_mu_t)

        if with_logprob:
            log_p_xz_t = create_log_gaussian(mu_t, log_sig_t, pi_action.unsqueeze(1))
            logp_pi = torch.logsumexp(log_p_xz_t + log_w_t, dim=1) - torch.logsumexp(log_w_t, dim=1)
        else:
            logp_pi = None

        return xz_mu_t if deterministic else pi_action, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    if logp_pi is not None:
        logp_pi -= torch.log(clip_but_pass_gradient(1 - pi ** 2, l=0, u=1) + 1e-6).sum(dim=1)
    return mu, pi, logp_pi


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128, 128, 128), activation=nn.LeakyReLU(), policy=GaussianPolicy):
        super(ActorCritic, self).__init__()
        self.policy = policy(obs_dim, act_dim, hidden_sizes, activation)
        # Q 网络输入维度：136（原始特征）+ 128（行人特征）+ act_dim
        self.q1_net = MLP(136 + 128 + act_dim, list(hidden_sizes) + [1], activation)
        self.q2_net = MLP(136 + 128 + act_dim, list(hidden_sizes) + [1], activation)

    def pi(self, obs, time_dynamic, deterministic=False):
        mu, logp_pi = self.policy(obs, time_dynamic, deterministic, True)
        pi = mu
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        return pi, logp_pi

    def compute_q1(self, obs, act, time_dynamic):
        with torch.no_grad():
            obs_processed = self.policy.cnn_dense(obs)  # [batch_size, 136]
            # 处理行人数据
            spatial_attended = self.policy.attention.spatial_attention(time_dynamic)
            temporal_attended = self.policy.attention.temporal_attention(spatial_attended)
            pedestrian_features = temporal_attended[-1].view(-1, 8*10)
            pedestrian_features = self.policy.dynamic_mlp(pedestrian_features)  # [batch_size, 128]
            # 拼接特征
            combined_features = torch.cat([obs_processed, pedestrian_features, act], dim=1)
        q = self.q1_net(combined_features)
        return torch.squeeze(q, -1)

    def compute_q2(self, obs, act, time_dynamic):
        with torch.no_grad():
            obs_processed = self.policy.cnn_dense(obs)
            spatial_attended = self.policy.attention.spatial_attention(time_dynamic)
            temporal_attended = self.policy.attention.temporal_attention(spatial_attended)
            pedestrian_features = temporal_attended[-1].view(-1, 8*10)
            pedestrian_features = self.policy.dynamic_mlp(pedestrian_features)
            combined_features = torch.cat([obs_processed, pedestrian_features, act], dim=1)
        q = self.q2_net(combined_features)
        return torch.squeeze(q, -1)

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128, 128, 128), activation=nn.LeakyReLU()):
        super(MLPActorCritic, self).__init__()
        
        # Use original ActorCritic architecture
        self.ac = ActorCritic(obs_dim, act_dim, hidden_sizes, activation)
        
    def pi(self, obs, time_dynamic, deterministic=False):
        """
        生成动作策略
        参数：
            obs: 原始观测张量
            time_dynamic: 行人动态数据张量，形状为 [batch_size, 3, 8, 10]
            deterministic: 是否使用确定性策略
        返回：
            pi: 生成的动作
            logp_pi: 动作的对数概率
        """
        pi, logp_pi = self.ac.pi(obs, time_dynamic, deterministic)
        return pi, logp_pi
        
    def q1(self, obs, act, time_dynamic):
        """
        计算第一个Q值
        参数：
            obs: 原始观测张量
            act: 动作张量
            time_dynamic: 行人动态数据张量
        返回：
            Q值
        """
        return self.ac.compute_q1(obs, act, time_dynamic)
        
    def q2(self, obs, act, time_dynamic):
        """
        计算第二个Q值
        参数：
            obs: 原始观测张量
            act: 动作张量
            time_dynamic: 行人动态数据张量
        返回：
            Q值
        """
        return self.ac.compute_q2(obs, act, time_dynamic)
        
    def act(self, obs, time_dynamic, deterministic=False):
        """
        无梯度生成动作
        参数：
            obs: 原始观测张量
            time_dynamic: 行人动态数据张量
            deterministic: 是否使用确定性策略
        返回：
            动作
        """
        with torch.no_grad():
            pi, _ = self.pi(obs, time_dynamic, deterministic)
            return pi


class AttentionModule(nn.Module):
    def __init__(self, feature_dim, num_heads=2):
        """
        初始化注意力模块，包含空间和时间注意力机制

        参数说明：
            feature_dim (int): 每个行人特征的维度(例如10)
            num_heads (int): 注意力头的数量(默认: 2)
        """
        super(AttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # 确保特征维度能被注意力头数整除
        assert self.head_dim * num_heads == feature_dim, "特征维度必须能被注意力头数整除"

        # 多头注意力的线性变换层
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)

    def spatial_attention(self, x):
        """
        计算每个时间步上行人之间的空间注意力

        参数:
            x (torch.Tensor): 输入张量，形状为[batch_size, time_steps, num_pedestrians, feature_dim]

        返回:
            torch.Tensor: 输出张量，形状为[batch_size, time_steps, num_pedestrians, feature_dim]
        """
        batch_size, time_steps, num_pedestrians, _ = x.size()
        spatial_out = []

        # 对每个时间步进行处理
        for t in range(time_steps):
            x_t = x[:, t]  # 形状: [batch_size, num_pedestrians, feature_dim]

            # 计算多头注意力的查询、键和值
            Q = self.query(x_t).view(batch_size, num_pedestrians, self.num_heads, self.head_dim)
            K = self.key(x_t).view(batch_size, num_pedestrians, self.num_heads, self.head_dim)
            V = self.value(x_t).view(batch_size, num_pedestrians, self.num_heads, self.head_dim)

            # 计算注意力得分并归一化
            attention_scores = torch.einsum('bnhd,bmhd->bnm', Q, K) / (self.head_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)

            # 加权求和得到注意力输出
            attended = torch.einsum('bnm,bmhd->bnhd', attention_weights, V)
            attended = attended.view(batch_size, num_pedestrians, self.feature_dim)

            # 输出投影并添加残差连接
            out = self.out(attended) + x_t
            spatial_out.append(out)

        return torch.stack(spatial_out, dim=1)

    def temporal_attention(self, x):
        """
        计算每个行人在时间步之间的时间注意力

        参数:
            x (torch.Tensor): 输入张量，形状为[batch_size, time_steps, num_pedestrians, feature_dim]

        返回:
            torch.Tensor: 输出张量，形状为[batch_size, time_steps, num_pedestrians, feature_dim]
        """
        batch_size, time_steps, num_pedestrians, _ = x.size()
        temporal_out = []

        # 对每个行人进行处理
        for p in range(num_pedestrians):
            x_p = x[:, :, p, :]  # 形状: [batch_size, time_steps, feature_dim]

            # 计算多头注意力的查询、键和值
            Q = self.query(x_p).view(batch_size, time_steps, self.num_heads, self.head_dim)
            K = self.key(x_p).view(batch_size, time_steps, self.num_heads, self.head_dim)
            V = self.value(x_p).view(batch_size, time_steps, self.num_heads, self.head_dim)

            # 计算注意力得分并归一化
            attention_scores = torch.einsum('bthd,bmhd->btm', Q, K) / (self.head_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)

            # 加权求和得到注意力输出
            attended = torch.einsum('btm,bmhd->bthd', attention_weights, V)
            attended = attended.view(batch_size, time_steps, self.feature_dim)

            # 输出投影并添加残差连接
            out = self.out(attended) + x_p
            temporal_out.append(out)

        # 堆叠并转置以匹配输入形状
        return torch.stack(temporal_out, dim=2)


class dynamic_MLP(nn.Module):
    """
    处理注意力机制后的行人特征的MLP模块
    输入：展平后的行人特征（例如 [batch_size, 8*10]）
    输出：固定维度的特征向量（例如 128 维）
    """

    def __init__(self, input_dim, output_dim, hidden_sizes=(128, 128), activation=nn.LeakyReLU()):
        super(dynamic_MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)