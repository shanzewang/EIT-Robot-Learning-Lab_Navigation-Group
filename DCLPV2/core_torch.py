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
    return x + (u - x) * clip_up.detach() + (l - x) * clip_low.detach()

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
        self.net = MLPPolicy(obs_dim - 6*90 + 128, hidden_sizes, activation)
        self.w_mu_logsig = nn.Linear(hidden_sizes[-1], (act_dim * 2 + 1) * k)

    def forward(self, x, deterministic=False, with_logprob=True):
        y = self.cnn_dense(x)
        net_output = self.net(y)
        w_mu_logsig = self.w_mu_logsig(net_output).view(-1, self.k, 2 * self.act_dim + 1)
        log_w_t = w_mu_logsig[:, :, 0]
        mu_t = w_mu_logsig[:, :, 1:1 + self.act_dim]
        log_sig_t = w_mu_logsig[:, :, 1 + self.act_dim:]

        log_sig_t = torch.tanh(log_sig_t)
        log_sig_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig_t + 1)
        sig_t = torch.exp(log_sig_t)

        if deterministic:
            # 新方案：使用所有分量的加权平均作为确定性输出
            weights = F.softmax(log_w_t, dim=1)  # [batch_size, k]
            # 计算加权平均的均值：Σ(wᵢ * μᵢ)
            weighted_mu = (weights.unsqueeze(-1) * mu_t).sum(dim=1)  # [batch_size, act_dim]
            xz_mu_t = weighted_mu
            pi_action = weighted_mu  # 确定性输出：所有分量的加权平均
            
            # 原方案（注释掉）：使用最高权重的分量 - 这是有问题的方法
            # weights = F.softmax(log_w_t, dim=1)
            # max_weight_idx = torch.argmax(weights, dim=1, keepdim=True)
            # mask_t = torch.zeros_like(log_w_t).scatter_(1, max_weight_idx, 1).unsqueeze(-1)
            # xz_mu_t = (mu_t * mask_t).sum(dim=1)  # 确保在确定性模式下也计算xz_mu_t
            # pi_action = xz_mu_t  # 确定性模式下，pi_action就是xz_mu_t（无噪声）
        else:
            # 多项式采样选择分量
            z_t = torch.multinomial(F.softmax(log_w_t, dim=1), num_samples=1)
            mask_t = torch.zeros_like(log_w_t).scatter_(1, z_t, 1).unsqueeze(-1)
            xz_mu_t = (mu_t * mask_t).sum(dim=1)
            xz_sig_t = (sig_t * mask_t).sum(dim=1)
            
            # 采样动作
            pi_action = xz_mu_t + xz_sig_t * torch.randn_like(xz_mu_t)

        if with_logprob:
            # 计算对数概率
            log_p_xz_t = create_log_gaussian(mu_t, log_sig_t, pi_action.unsqueeze(1))
            logp_pi = torch.logsumexp(log_p_xz_t + log_w_t, dim=1) - torch.logsumexp(log_w_t, dim=1)
        else:
            logp_pi = None

        # 返回均值、动作和对数概率
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
        # Rename these to avoid conflicts with method names
        self.q1_net = MLP(obs_dim - 6*90 + 128 + act_dim, list(hidden_sizes) + [1], activation)
        self.q2_net = MLP(obs_dim - 6*90 + 128 + act_dim, list(hidden_sizes) + [1], activation)

    def pi(self, obs, deterministic=False):
        mu, logp_pi = self.policy(obs, deterministic, True)
        pi = mu  # 在policy的forward中，返回的第一个值就是正确的动作
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        return pi, logp_pi

    # Rename these methods to avoid conflict with network attributes
    def compute_q1(self, obs, act):
        with torch.no_grad():
            obs_processed = self.policy.cnn_dense(obs)
        q = self.q1_net(torch.cat([obs_processed, act], dim=-1))
        return torch.squeeze(q, -1)

    def compute_q2(self, obs, act):
        with torch.no_grad():
            obs_processed = self.policy.cnn_dense(obs)
        q = self.q2_net(torch.cat([obs_processed, act], dim=-1))
        return torch.squeeze(q, -1)

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128, 128, 128), activation=nn.LeakyReLU()):
        super(MLPActorCritic, self).__init__()
        
        # Use original ActorCritic architecture
        self.ac = ActorCritic(obs_dim, act_dim, hidden_sizes, activation)
        
    def pi(self, obs, deterministic=False):
        pi, logp_pi = self.ac.pi(obs, deterministic)
        return pi, logp_pi
        
    def q1(self, obs, act):
        return self.ac.compute_q1(obs, act)
        
    def q2(self, obs, act):
        return self.ac.compute_q2(obs, act)
        
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            pi, _ = self.pi(obs, deterministic)
            return pi