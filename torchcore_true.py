# -*- coding: utf-8 -*-
"""
SAC强化学习神经网络架构配置文件 (PyTorch版本)
===============================

本文件定义了Soft Actor-Critic算法的神经网络架构，包括：
1. CNN特征提取网络 - 处理激光雷达数据
2. 高斯混合策略网络 - 生成多模态动作策略
3. 双Q价值网络 - 评估状态-动作价值
4. 各种辅助函数 - 支持网络训练和数值稳定

调用关系：被SPN_ours_1e6.py主训练文件调用
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

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
    clip_low = (x < l).float()
    return x + (l - x) * clip_low


def new_relu(x, alpha_actv):
    """
    自定义激活函数
    
    作用：实现倒数形式的自适应激活函数 1/(x + α + ε)
    参数：
        x - 输入特征
        alpha_actv - 可训练的激活参数
    返回：激活后的特征
    """
    r = torch.reciprocal(clip_but_pass_gradient2(x + alpha_actv, l=EPS))
    return r


def mlp(x, hidden_sizes=(32,), activation=F.leaky_relu, output_activation=None):
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
    layers = []
    input_dim = x.shape[-1]
    
    for h in hidden_sizes[:-1]:
        layers.append(nn.Linear(input_dim, h))
        input_dim = h
    
    layers.append(nn.Linear(input_dim, hidden_sizes[-1]))
    
    net = nn.Sequential(*layers)
    
    # 应用激活函数
    for i, layer in enumerate(net):
        if i < len(net) - 1:  # 隐藏层
            x = activation(layer(x))
        else:  # 输出层
            x = layer(x)
            if output_activation is not None:
                x = output_activation(x)
    
    return x


class MLP(nn.Module):
    """MLP模块类"""
    def __init__(self, input_dim, hidden_sizes, activation=F.leaky_relu, output_activation=None):
        super(MLP, self).__init__()
        self.activation = activation
        self.output_activation = output_activation
        
        layers = []
        dims = [input_dim] + list(hidden_sizes)
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 隐藏层
                x = self.activation(x)
            else:  # 输出层
                if self.output_activation is not None:
                    x = self.output_activation(x)
        return x


class CNNNet(nn.Module):
    """
    基础CNN网络
    
    作用：使用1D卷积处理序列数据
    """
    def __init__(self, activation=F.relu, output_activation=None):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=90, stride=1, padding=0)
        self.activation = activation
    
    def forward(self, x, y=None):
        # x shape: [batch_size, sequence_length, features] -> [batch_size, features, sequence_length]
        x = x.transpose(1, 2)
        
        x1 = F.leaky_relu(self.conv1(x))
        x11 = F.leaky_relu(self.conv2(x1))
        x0 = F.leaky_relu(self.conv3(x11))
        x3 = self.pool(x0)
        x3_flatten = x3.view(x3.size(0), -1)
        return x3_flatten


class CNNDense(nn.Module):
    """
    增强型CNN特征提取器
    
    作用：处理激光雷达数据的主要特征提取网络
    输入格式：
        x[:,0:540] - 激光雷达数据(90个方向×6个特征)
        x[:,540:] - 其他状态信息(目标位置、速度等)
    """
    def __init__(self, activation=F.leaky_relu, output_activation=None):
        super(CNNDense, self).__init__()
        # 定义可训练激活参数
        self.alpha_actv2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        self.conv1 = nn.Conv1d(6, 32, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=90, stride=1, padding=0)
        self.activation = activation
    
    def forward(self, x):
        """
        参数：
            x - 完整状态输入 [batch_size, 548]
        返回：CNN特征与其他状态信息的拼接 [batch_size, 136]
        """
        # 提取和重塑激光雷达数据
        x_input = x[:, 0:6*90]  # 激光雷达数据段
        x_input = x_input.view(-1, 90, 6)  # [batch, 90方向, 6特征]
        
        # 对距离特征应用自定义激活
        x00 = new_relu(x_input[:, :, 2], self.alpha_actv2)  # 处理距离信息(第3个特征)
        
        # 重新组合特征
        x_input = torch.cat([
            x_input[:, :, 0:2],          # 方向信息(cos, sin)
            x00.unsqueeze(-1),           # 处理后的距离
            x_input[:, :, 3:6]           # 机器人几何信息(length1, length2, width)
        ], dim=-1)
        
        # 转换为Conv1d格式: [batch, features, sequence]
        x_input = x_input.transpose(1, 2)
        
        # 三层1D卷积提取特征
        x1 = F.leaky_relu(self.conv1(x_input))
        x11 = F.leaky_relu(self.conv2(x1))
        x0 = F.leaky_relu(self.conv3(x11))
        
        # 全局最大池化
        x3 = self.pool(x0)
        x3_flatten = x3.view(x3.size(0), -1)
        
        # 拼接CNN特征和其他状态信息
        return torch.cat([x3_flatten, x[:, 6*90:]], dim=-1)


def count_vars(model):
    """
    计算模型参数数量
    
    作用：统计模型的总参数数量
    参数：
        model - PyTorch模型
    返回：参数总数(整数)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    clip_up = (x > u).float()
    clip_low = (x < l).float()
    return x + (u - x) * clip_up + (l - x) * clip_low


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
    normalized_dist_t = (t - mu_t) * torch.exp(-log_sig_t)
    
    # 计算二次项
    quadratic = -0.5 * torch.sum(normalized_dist_t ** 2, dim=-1)

    # 计算归一化常数
    log_z = torch.sum(log_sig_t, dim=-1)
    D_t = float(mu_t.shape[-1])
    log_z += 0.5 * D_t * np.log(2 * np.pi)

    log_p = quadratic - log_z
    return log_p


class MLPGaussianPolicy(nn.Module):
    """
    高斯混合策略网络
    
    作用：实现基于高斯混合模型的多模态策略网络
    网络流程：状态输入 → CNN特征提取 → MLP处理 → GMM参数生成 → 动作采样
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=[128,128,128,128], 
                 activation=F.leaky_relu, output_activation=None):
        super(MLPGaussianPolicy, self).__init__()
        
        self.k = 4  # 高斯混合组件数量
        self.act_dim = action_dim
        self.activation = activation
        self.output_activation = output_activation
        
        # CNN自适应激活参数
        self.alpha_actv1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        # CNN网络
        self.cnn = CNNNet()
        
        # MLP网络
        # CNN输出128维 + 其他状态信息8维 = 136维
        self.mlp = MLP(136, hidden_sizes, activation, activation)
        
        # GMM参数输出层
        self.gmm_layer = nn.Linear(hidden_sizes[-1], (self.act_dim*2+1)*self.k)
    
    def forward(self, x):
        """
        参数：
            x - 状态输入 [batch_size, 548]
        
        返回：
            xz_mu_t - 选中组件的均值 [batch_size, act_dim]
            x_t - 采样的动作 [batch_size, act_dim]
            logp_pi - 动作的对数概率 [batch_size]
        """
        batch_size = x.shape[0]
        
        # 处理激光雷达数据
        x_input = x[:, 0:6*90]
        x_input = x_input.view(-1, 90, 6)
        
        # 对距离特征应用自定义激活
        x0 = new_relu(x_input[:, :, 2], self.alpha_actv1)
        x_input = torch.cat([
            x_input[:, :, 0:2],
            x0.unsqueeze(-1),
            x_input[:, :, 3:6]
        ], dim=-1)
        
        # 提取其他状态信息
        w_input = x[:, 6*90:6*90+8]
        w_input = w_input.view(-1, 8)
        
        # CNN特征提取
        cnn_net = self.cnn(x_input, w_input)
        
        # 特征拼接
        y = torch.cat([cnn_net, x[:, 6*90:]], dim=-1)
        
        # MLP处理
        net = self.mlp(y)
        
        # 生成GMM参数：(权重 + 均值 + 对数标准差) × k个组件
        w_and_mu_and_logsig_t = self.gmm_layer(net)
        w_and_mu_and_logsig_t = w_and_mu_and_logsig_t.view(-1, self.k, 2*self.act_dim+1)
        
        # 分离参数
        log_w_t = w_and_mu_and_logsig_t[..., 0]              # 混合权重 [N, K]
        mu_t = w_and_mu_and_logsig_t[..., 1:1+self.act_dim]       # 均值 [N, K, act_dim]
        log_sig_t = w_and_mu_and_logsig_t[..., 1+self.act_dim:]   # 对数标准差 [N, K, act_dim]

        # 约束对数标准差到安全范围
        log_sig_t = torch.tanh(log_sig_t)
        log_sig_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig_t + 1)
        xz_sigs_t = torch.exp(log_sig_t)

        # 采样高斯组件
        z_t = torch.multinomial(torch.softmax(log_w_t, dim=-1), num_samples=1)  # 选择组件
        
        # 获取选中组件的参数
        batch_indices = torch.arange(batch_size, device=x.device)
        xz_mu_t = mu_t[batch_indices, z_t.squeeze(-1)]      # 选中组件均值
        xz_sig_t = xz_sigs_t[batch_indices, z_t.squeeze(-1)] # 选中组件标准差

        # 重参数化采样：a = μ + σ * ε
        epsilon = torch.randn((batch_size, self.act_dim), device=x.device)
        x_t = xz_mu_t + xz_sig_t * epsilon

        # 计算动作概率
        log_p_xz_t = create_log_gaussian(mu_t, log_sig_t, x_t.unsqueeze(1))  # 各组件概率
        # log_p_x_t = torch.logsumexp(log_p_xz_t + log_w_t, dim=1)      # 边际概率
        # log_p_x_t -= torch.logsumexp(log_w_t, dim=1)                   # 归一化
        log_p_x_t_numerator = torch.logsumexp(log_p_xz_t + log_w_t, dim=1)    # 分子
        log_p_x_t_denominator = torch.logsumexp(log_w_t, dim=1)               # 分母  
        log_p_x_t = log_p_x_t_numerator - log_p_x_t_denominator              # 非原地操作

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
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    
    # 雅可比校正：log|det(∂tanh(u)/∂u)| = log(1 - tanh²(u))
    logp_pi -= torch.sum(torch.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), dim=1)
    
    return mu, pi, logp_pi


class MLPActorCritic(nn.Module):
    """
    Actor-Critic网络架构
    
    作用：构建完整的SAC算法神经网络，包括策略网络(Actor)和价值网络(Critic)
    
    网络结构：
        状态输入 → 共享CNN特征提取 → 分支为Actor和Critic
        Actor: 高斯混合策略网络
        Critic: 双Q网络(Q1和Q2)
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=(128,128,128,128), 
                 activation=F.leaky_relu, output_activation=None):
        super(MLPActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation
        
        # ============= Actor网络(策略网络) =============
        self.policy = MLPGaussianPolicy(state_dim, action_dim, list(hidden_sizes), 
                                      activation, output_activation)
        
        # ============= Critic网络(价值网络) =============
        # 共享CNN特征提取器
        self.cnn_dense = CNNDense(activation, None)
        
        # Q网络：输入特征+动作，输出Q值
        q_input_dim = 128 + 8 + action_dim  # CNN特征136维 + 动作2维 = 138维
        self.q1 = MLP(q_input_dim, list(hidden_sizes) + [1], activation, None)
        self.q2 = MLP(q_input_dim, list(hidden_sizes) + [1], activation, None)
    
    def forward(self, x, a=None):
        """
        参数：
            x - 状态输入 [batch_size, 548]
            a - 动作输入 [batch_size, 2] (可选)
        
        返回：
            mu - 策略均值 [batch_size, act_dim]
            pi - 采样动作 [batch_size, act_dim]
            logp_pi - 动作对数概率 [batch_size]
            q1, q2 - 双Q网络输出 [batch_size] (如果提供a)
            q1_pi, q2_pi - 当前策略下的Q值 [batch_size]
        """
        # ============= Actor网络(策略网络) =============
        mu, pi, logp_pi = self.policy(x)
        
        # 应用tanh压缩到有界动作空间
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

        # ============= Critic网络(价值网络) =============
        # 共享CNN特征提取
        y = self.cnn_dense(x)  # 提取136维特征
        
        # 计算当前策略下的Q值
        q1_pi = self.q1(torch.cat([y, pi], dim=-1)).squeeze(-1)    # Q1(s,π(s))
        q2_pi = self.q2(torch.cat([y, pi], dim=-1)).squeeze(-1)    # Q2(s,π(s))
        
        if a is not None:
            # 如果提供了动作，计算Q(s,a)
            q1 = self.q1(torch.cat([y, a], dim=-1)).squeeze(-1)        # Q1(s,a)
            q2 = self.q2(torch.cat([y, a], dim=-1)).squeeze(-1)        # Q2(s,a)
            return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi
        else:
            return mu, pi, logp_pi, q1_pi, q2_pi