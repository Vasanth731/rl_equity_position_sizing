"""Vectorized single-asset trading environment.

Action space : Discrete(len(positions)) -> a *target* portfolio weight, e.g.
               [-1, -0.5, 0, 0.5, 1] for short / half-short / flat / half / long.
Observation  : window*F standardized features, flattened, plus 3 state scalars
               [current position, current drawdown, fraction of episode elapsed].
Reward       : r_t = (w_t * logret_t  -  costs_t) * reward_scale
               costs = (cost+slippage)/1e4 * |Δw|  +  holding/1e4 * |w_t|

Notes
-----
* Using log returns makes the reward additive in time, so the discounted return
  is (approximately) log terminal wealth -- a principled objective.
* Costs are charged on the *change* in position, which is what makes the agent
  learn to hold rather than flip every day. Set cost_bps=0 and watch it churn.
* Episodes start at random points in the split and each episode samples a random
  ticker. Training across many assets is the cheapest regularizer available:
  a policy that only works on one price path will not survive it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import EnvConfig
from dataloader import MarketData

N_STATE_SCALARS = 3


class VecTradingEnv:
    def __init__(
        self,
        md: MarketData,
        cfg: EnvConfig,
        split: str = "train",
        n_envs: int = 8,
        seed: int = 0,
        tickers: Optional[List[str]] = None,
    ):
        self.md = md
        self.cfg = cfg
        self.split = split
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.W = md.window
        self.F = md.n_features
        self.positions = np.asarray(cfg.positions, dtype=np.float32)

        self.tickers = tickers or md.tickers(split)
        if not self.tickers:
            raise ValueError(f"no tickers with data in split {split!r}")

        self.obs_dim = self.W * self.F + N_STATE_SCALARS
        self.n_actions = len(self.positions)

        self._ti = np.zeros(self.n, dtype=np.int64)     # ticker index
        self._t = np.zeros(self.n, dtype=np.int64)      # absolute time index
        self._t0 = np.zeros(self.n, dtype=np.int64)
        self._tend = np.zeros(self.n, dtype=np.int64)
        self._pos = np.zeros(self.n, dtype=np.float32)
        self._eq = np.ones(self.n, dtype=np.float32)
        self._peak = np.ones(self.n, dtype=np.float32)

        self._cost = (cfg.cost_bps + cfg.slippage_bps) / 1e4
        self._hold = cfg.holding_cost_bps / 1e4

    # ------------------------------------------------------------------ #
    def _reset_one(self, i: int) -> None:
        ti = int(self.rng.integers(len(self.tickers)))
        a = self.md.assets[self.tickers[ti]]
        lo, hi = a.ranges[self.split]
        span = min(self.cfg.episode_len, hi - lo)
        start = int(self.rng.integers(lo, max(lo + 1, hi - span + 1)))
        self._ti[i] = ti
        self._t[i] = self._t0[i] = start
        self._tend[i] = min(start + span, hi)
        self._pos[i] = 0.0
        self._eq[i] = self._peak[i] = 1.0

    def reset(self) -> np.ndarray:
        for i in range(self.n):
            self._reset_one(i)
        return self._obs()

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        out = np.empty((self.n, self.obs_dim), dtype=np.float32)
        for i in range(self.n):
            a = self.md.assets[self.tickers[self._ti[i]]]
            t = int(self._t[i])
            out[i, : self.W * self.F] = a.features[t - self.W + 1 : t + 1].ravel()
        dd = self._eq / self._peak - 1.0
        frac = (self._t - self._t0) / np.maximum(1, self._tend - self._t0)
        out[:, -3] = self._pos
        out[:, -2] = dd
        out[:, -1] = frac.astype(np.float32)
        return out

    def _vol_scale(self, i: int) -> float:
        if self.cfg.vol_target <= 0:
            return 1.0
        a = self.md.assets[self.tickers[self._ti[i]]]
        t = int(self._t[i])
        rv = float(np.std(a.ret[max(0, t - 20) : t]) * np.sqrt(252)) + 1e-6
        return float(np.clip(self.cfg.vol_target / rv, 0.0, 1.5))

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        target = self.positions[np.asarray(actions, dtype=np.int64)].copy()
        for i in range(self.n):
            target[i] *= self._vol_scale(i)

        rets = np.empty(self.n, dtype=np.float32)
        for i in range(self.n):
            a = self.md.assets[self.tickers[self._ti[i]]]
            rets[i] = a.ret[int(self._t[i])]

        turn = np.abs(target - self._pos)
        costs = self._cost * turn + self._hold * np.abs(target)
        pnl = target * rets
        net = pnl - costs

        self._eq *= np.exp(net)
        self._peak = np.maximum(self._peak, self._eq)
        dd = self._eq / self._peak - 1.0

        reward = net * self.cfg.reward_scale
        if self.cfg.dd_penalty > 0:
            reward = reward + self.cfg.dd_penalty * dd * self.cfg.reward_scale

        self._pos = target
        self._t += 1
        done = self._t >= self._tend

        infos: List[Dict] = []
        for i in range(self.n):
            info = {"turnover": float(turn[i]), "position": float(target[i]),
                    "net_ret": float(net[i])}
            if done[i]:
                info["episode"] = {
                    "ticker": self.tickers[self._ti[i]],
                    "equity": float(self._eq[i]),
                    "log_return": float(np.log(self._eq[i])),
                    "max_dd": float(self._eq[i] / self._peak[i] - 1.0),
                    "length": int(self._tend[i] - self._t0[i]),
                }
                self._reset_one(i)
            infos.append(info)

        return self._obs(), reward.astype(np.float32), done, infos


def make_obs_single(a, t: int, W: int, position: float, dd: float, frac: float) -> np.ndarray:
    """Build one observation outside the env (used by the sequential backtester)."""
    o = np.empty(W * a.features.shape[1] + N_STATE_SCALARS, dtype=np.float32)
    o[: W * a.features.shape[1]] = a.features[t - W + 1 : t + 1].ravel()
    o[-3], o[-2], o[-1] = position, dd, frac
    return o
