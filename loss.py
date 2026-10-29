"""PPO objective, GAE advantage estimation, and financial evaluation metrics.

Kept in one place so the training loop stays readable and the loss terms can be
unit-tested independently of the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Advantage estimation
# --------------------------------------------------------------------------- #
def compute_gae(
    rewards: torch.Tensor,      # (T, N)
    values: torch.Tensor,       # (T, N)
    dones: torch.Tensor,        # (T, N) 1.0 if the step terminated the episode
    last_value: torch.Tensor,   # (N,)
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. Returns (advantages, returns)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        if t == T - 1:
            next_value, next_nonterm = last_value, 1.0 - dones[t]
        else:
            next_value, next_nonterm = values[t + 1], 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterm - values[t]
        last_gae = delta + gamma * lam * next_nonterm * last_gae
        adv[t] = last_gae
    return adv, adv + values


# --------------------------------------------------------------------------- #
# PPO loss
# --------------------------------------------------------------------------- #
@dataclass
class PPOLossOutput:
    total: torch.Tensor
    policy: float
    value: float
    entropy: float
    approx_kl: float
    clip_frac: float


def ppo_loss(
    new_logprob: torch.Tensor,
    entropy: torch.Tensor,
    new_value: torch.Tensor,
    old_logprob: torch.Tensor,
    old_value: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    clip_vloss: bool = True,
    norm_adv: bool = True,
) -> PPOLossOutput:
    logratio = new_logprob - old_logprob
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - logratio).mean().item()   # Schulman k3 estimator
        clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg1, pg2).mean()

    if clip_vloss:
        v_unclipped = (new_value - returns) ** 2
        v_clipped = (old_value + torch.clamp(new_value - old_value, -clip_coef, clip_coef)
                     - returns) ** 2
        v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
    else:
        v_loss = 0.5 * F.mse_loss(new_value, returns)

    ent = entropy.mean()
    total = pg_loss + vf_coef * v_loss - ent_coef * ent

    return PPOLossOutput(total, pg_loss.item(), v_loss.item(), ent.item(), approx_kl, clip_frac)


# --------------------------------------------------------------------------- #
# Financial metrics -- the numbers that actually decide if the policy is any good
# --------------------------------------------------------------------------- #
def performance_metrics(
    net_returns: np.ndarray,       # per-step net LOG returns of the strategy
    positions: np.ndarray | None = None,
    periods_per_year: int = 252,
    rf: float = 0.0,
) -> Dict[str, float]:
    r = np.asarray(net_returns, dtype=np.float64)
    if r.size == 0:
        return {}
    simple = np.expm1(r)
    equity = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0

    ann_ret = float(r.sum() / len(r) * periods_per_year)          # log CAGR
    ann_vol = float(simple.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0
    downside = simple[simple < 0]
    dvol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if downside.size > 1 else 0.0

    m = {
        "total_return": float(equity[-1] - 1.0),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float((ann_ret - rf) / ann_vol) if ann_vol > 1e-12 else 0.0,
        "sortino": float((ann_ret - rf) / dvol) if dvol > 1e-12 else 0.0,
        "max_drawdown": float(dd.min()),
        "calmar": float(ann_ret / abs(dd.min())) if dd.min() < -1e-12 else 0.0,
        "hit_rate": float((simple > 0).mean()),
        "n_periods": int(len(r)),
    }
    if positions is not None and len(positions):
        p = np.asarray(positions, dtype=np.float64)
        act = np.abs(p) > 1e-9
        # hit rate on days actually in the market -- otherwise flat days count as losses
        m["hit_rate"] = float((simple[act] > 0).mean()) if act.any() else 0.0
        m["avg_exposure"] = float(np.abs(p).mean())
        m["long_frac"] = float((p > 0).mean())
        m["short_frac"] = float((p < 0).mean())
        m["ann_turnover"] = float(np.abs(np.diff(p, prepend=0.0)).sum() / len(p) * periods_per_year)
    return m


def deflated_sharpe_note(sharpe: float, n_trials: int, n_periods: int) -> float:
    """Rough haircut on Sharpe for multiple-testing / selection bias.

    Subtracts the expected maximum Sharpe of `n_trials` random strategies. If the
    result is <= 0 the backtest is indistinguishable from luck. Use it whenever
    you have tuned hyperparameters more than once (i.e. always).
    """
    if n_trials <= 1 or n_periods <= 1:
        return sharpe
    e_max = np.sqrt(2.0 * np.log(max(n_trials, 2))) / np.sqrt(n_periods / 252.0)
    return float(sharpe - e_max)
