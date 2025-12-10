import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 128]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden_sizes[1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim, requires_grad=True))

    def forward(self, obs):
        x = self.net(obs)
        mu = torch.tanh(self.mu_layer(x))  # [-1,1] actions
        std = torch.exp(self.log_std)
        return mu, std

    def distribution(self, obs):
        mu, std = self.forward(obs)
        return torch.distributions.Normal(mu, std)

    def log_prob(self, obs, act):
        dist = self.distribution(obs)
        return dist.log_prob(act).sum(axis=-1)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[256, 128]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
