"""Actor-critic network.

The flat observation is unpacked back into (window, features), run through a
sequence encoder (GRU or dilated 1-D CNN), concatenated with the position/
drawdown/time scalars, then split into a categorical policy head and a scalar
value head. Orthogonal init with a small policy gain is the standard PPO recipe
and matters more than it looks -- a large final-layer gain makes the initial
policy near-deterministic and kills exploration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import ModelConfig


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias)
    return layer


class CNNEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv1d(n_features, hidden // 2, 5, padding=2, dilation=1)), nn.GELU(),
            layer_init(nn.Conv1d(hidden // 2, hidden // 2, 5, padding=4, dilation=2)), nn.GELU(),
            layer_init(nn.Conv1d(hidden // 2, hidden, 5, padding=8, dilation=4)), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, T, F)
        h = self.net(x.transpose(1, 2))                   # (B, H, T)
        return torch.cat([h[:, :, -1], h.mean(-1)], dim=-1)


class GRUEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.drop(out[:, -1])


class ActorCritic(nn.Module):
    def __init__(self, window: int, n_features: int, n_actions: int,
                 n_scalars: int, cfg: ModelConfig):
        super().__init__()
        self.window, self.n_features, self.n_scalars = window, n_features, n_scalars
        self.n_actions = n_actions

        if cfg.encoder == "gru":
            self.enc: nn.Module = GRUEncoder(n_features, cfg.hidden, cfg.layers, cfg.dropout)
        elif cfg.encoder == "cnn":
            self.enc = CNNEncoder(n_features, cfg.hidden, cfg.dropout)
        else:
            raise ValueError(cfg.encoder)

        d = self.enc.out_dim + n_scalars
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(d, cfg.mlp_hidden)), nn.GELU(),
            layer_init(nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden)), nn.GELU(),
        )
        self.pi = layer_init(nn.Linear(cfg.mlp_hidden, n_actions), std=0.01)
        self.v = layer_init(nn.Linear(cfg.mlp_hidden, 1), std=1.0)

    # ------------------------------------------------------------------ #
    def _split(self, obs: torch.Tensor):
        n = self.window * self.n_features
        seq = obs[:, :n].view(-1, self.window, self.n_features)
        return seq, obs[:, n:]

    def forward(self, obs: torch.Tensor):
        seq, scal = self._split(obs)
        h = self.trunk(torch.cat([self.enc(seq), scal], dim=-1))
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value
