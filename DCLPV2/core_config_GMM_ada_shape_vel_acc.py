import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def clip_but_pass_gradient2(x, l=EPS):
    clip_low = (x < l).float()
    return x + (l - x) * clip_low.detach()

def new_relu(x, alpha_actv):
    r = torch.reciprocal(clip_but_pass_gradient2(x + alpha_actv, l=EPS))
    return r

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
        x = x.view(x.size(0), -1)  # Flatten
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

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return pre_sum.sum(dim=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = (x > u).float()
    clip_low = (x < l).float()
    return x + (u - x) * clip_up.detach() + (l - x) * clip_low.detach()

def create_log_gaussian(mu_t, log_sig_t, t):
    normalized_dist_t = (t - mu_t) * torch.exp(-log_sig_t)
    quadratic = -0.5 * (normalized_dist_t ** 2).sum(dim=-1)
    log_z = log_sig_t.sum(dim=-1)
    D_t = mu_t.size(-1)
    log_z += 0.5 * D_t * np.log(2 * np.pi)
    log_p = quadratic - log_z
    return log_p

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh(), k=4):
        super(GaussianPolicy, self).__init__()
        self.k = k
        self.act_dim = act_dim
        self.cnn_dense = CNNDense(obs_dim)
        self.net = MLPPolicy(obs_dim - 6*90 + 128, hidden_sizes, activation)
        self.w_mu_logsig = nn.Linear(hidden_sizes[-1], (act_dim * 2 + 1) * k)

    def forward(self, x):
        y = self.cnn_dense(x)
        net_output = self.net(y)
        w_mu_logsig = self.w_mu_logsig(net_output).view(-1, self.k, 2 * self.act_dim + 1)
        log_w_t = w_mu_logsig[:, :, 0]
        mu_t = w_mu_logsig[:, :, 1:1 + self.act_dim]
        log_sig_t = w_mu_logsig[:, :, 1 + self.act_dim:]

        log_sig_t = torch.tanh(log_sig_t)
        log_sig_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig_t + 1)
        sig_t = torch.exp(log_sig_t)

        z_t = torch.multinomial(F.softmax(log_w_t, dim=1), num_samples=1)
        mask_t = torch.zeros_like(log_w_t).scatter_(1, z_t, 1).unsqueeze(-1)
        xz_mu_t = (mu_t * mask_t).sum(dim=1)
        xz_sig_t = (sig_t * mask_t).sum(dim=1)

        x_t = xz_mu_t + xz_sig_t * torch.randn_like(xz_mu_t)
        log_p_xz_t = create_log_gaussian(mu_t, log_sig_t, x_t.unsqueeze(1))
        log_p_x_t = torch.logsumexp(log_p_xz_t + log_w_t, dim=1) - torch.logsumexp(log_w_t, dim=1)
        return xz_mu_t, x_t, log_p_x_t

def apply_squashing_func(mu, pi, logp_pi):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    logp_pi -= torch.log(clip_but_pass_gradient(1 - pi ** 2, l=0, u=1) + 1e-6).sum(dim=1)
    return mu, pi, logp_pi

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128, 128, 128), activation=nn.LeakyReLU(), policy=GaussianPolicy):
        super(ActorCritic, self).__init__()
        self.policy = policy(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = MLP(obs_dim + act_dim, list(hidden_sizes) + [1], activation)
        self.q2 = MLP(obs_dim + act_dim, list(hidden_sizes) + [1], activation)

    def forward(self, x, a):
        mu, pi, logp_pi = self.policy(x)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        q1 = self.q1(torch.cat([x, a], dim=-1)).squeeze(-1)
        q1_pi = self.q1(torch.cat([x, pi], dim=-1)).squeeze(-1)
        q2 = self.q2(torch.cat([x, a], dim=-1)).squeeze(-1)
        q2_pi = self.q2(torch.cat([x, pi], dim=-1)).squeeze(-1)
        return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi

def get_vars(module, scope):
    return [param for name, param in module.named_parameters() if scope in name]

def count_vars(module, scope):
    vars = get_vars(module, scope)
    return sum([np.prod(var.shape) for var in vars])

def placeholder(dim=None):
    raise NotImplementedError("PyTorch does not use placeholders. Use torch tensors instead.")

def placeholders(*args):
    raise NotImplementedError("PyTorch does not use placeholders. Use torch tensors instead.")

def mlp(x, hidden_sizes=(32,), activation=F.leaky_relu, output_activation=None):
    net = MLP(x.shape[-1], hidden_sizes, activation, output_activation)
    return net(x)

def mlp_policy(x, hidden_sizes=(32,), activation=F.tanh, output_activation=None):
    net = MLPPolicy(x.shape[-1], hidden_sizes, activation)
    return net(x)

def CNN(x, y, activation=F.relu, output_activation=None):
    x1 = F.leaky_relu(nn.Conv1d(6, 32, kernel_size=1)(x))
    x11 = F.leaky_relu(nn.Conv1d(32, 64, kernel_size=1)(x1))
    x0 = F.leaky_relu(nn.Conv1d(64, 128, kernel_size=1)(x11))
    x3 = nn.MaxPool1d(90)(x0)
    x3_flatten = x3.view(x3.size(0), -1)
    return x3_flatten

def CNN2(x, activation=F.relu, output_activation=None):
    x1 = F.relu(nn.Conv1d(x.shape[1], 32, kernel_size=20, stride=10, padding='same')(x))
    x2 = F.relu(nn.Conv1d(32, 16, kernel_size=10, stride=3, padding='same')(x1))
    x2_flatten = x2.view(x2.size(0), -1)
    x3 = F.relu(nn.Linear(x2_flatten.shape[-1], 128)(x2_flatten))
    return x3

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, alpha_actv1):
    k = 4
    act_dim = a.shape[-1]
    x_input = x[:, :6*90].view(-1, 90, 6)
    x0 = new_relu(x_input[:, :, 2], alpha_actv1)
    x_input = torch.cat([x_input[:, :, :2], x0.unsqueeze(-1), x_input[:, :, 3:]], dim=-1)
    w_input = x[:, 6*90:6*90+8].view(-1, 8)
    cnn_net = CNN(x_input, w_input)
    y = torch.cat([cnn_net, x[:, 6*90:]], dim=-1)
    net = mlp_policy(y, hidden_sizes, activation, activation)
    w_and_mu_and_logsig_t = nn.Linear(hidden_sizes[-1], (act_dim * 2 + 1) * k)(net)
    w_and_mu_and_logsig_t = w_and_mu_and_logsig_t.view(-1, k, 2 * act_dim + 1)
    log_w_t = w_and_mu_and_logsig_t[:, :, 0]
    mu_t = w_and_mu_and_logsig_t[:, :, 1:1 + act_dim]
    log_sig_t = w_and_mu_and_logsig_t[:, :, 1 + act_dim:]

    log_sig_t = torch.tanh(log_sig_t)
    log_sig_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig_t + 1)
    xz_sigs_t = torch.exp(log_sig_t)

    z_t = torch.multinomial(F.softmax(log_w_t, dim=1), num_samples=1)
    mask_t = torch.zeros_like(log_w_t).scatter_(1, z_t, 1).unsqueeze(-1)
    xz_mu_t = (mu_t * mask_t).sum(dim=1)
    xz_sig_t = (xz_sigs_t * mask_t).sum(dim=1)

    x_t = xz_mu_t + xz_sig_t * torch.randn_like(xz_mu_t)
    log_p_xz_t = create_log_gaussian(mu_t, log_sig_t, x_t.unsqueeze(1))
    log_p_x_t = torch.logsumexp(log_p_xz_t + log_w_t, dim=1) - torch.logsumexp(log_w_t, dim=1)
    return xz_mu_t, x_t, log_p_x_t

def mlp_actor_critic(x, a, hidden_sizes=(128,128,128,128), activation=F.leaky_relu, 
                     output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    alpha_actv1 = torch.tensor(0.0, requires_grad=True)
    mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, alpha_actv1)
    mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
    vf_mlp = lambda y: MLP(y.shape[-1], list(hidden_sizes) + [1], activation)(y).squeeze(-1)
    with torch.no_grad():
        y = CNNDense(x.shape[-1])(x)
    q1 = vf_mlp(torch.cat([y, a], dim=-1))
    q1_pi = vf_mlp(torch.cat([y, pi], dim=-1))
    q2 = vf_mlp(torch.cat([y, a], dim=-1))
    q2_pi = vf_mlp(torch.cat([y, pi], dim=-1))
    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi
