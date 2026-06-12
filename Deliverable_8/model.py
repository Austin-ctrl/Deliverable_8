import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from config import HIDDEN_SIZE


class ActorCritic(nn.Module):


    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        # CartPole state is only 4 numbers so a simple MLP works fine

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

        self.actor  = nn.Linear(HIDDEN_SIZE, action_dim)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.actor(h)
        value  = self.critic(h).squeeze(-1)
        return logits, value

    def get_action(self, obs: np.ndarray):
        x = torch.FloatTensor(obs).unsqueeze(0)
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value