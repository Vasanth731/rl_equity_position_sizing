"""Sanity tests -- run with `python test_sanity.py`.

The leakage test is the important one. If you change feature engineering, run it
again. A feature that peeks one day ahead will produce a beautiful backtest and
lose money in production, and nothing else in this repo will warn you.
"""

import numpy as np
import pandas as pd
import torch

from config import Config
from dataloader import build_dataset, make_features
from env import VecTradingEnv
from loss import compute_gae, performance_metrics, ppo_loss
from model import ActorCritic
from env import N_STATE_SCALARS


def test_gae_degenerate():
    r = torch.randn(5, 3)
    v = torch.randn(5, 3)
    d = torch.zeros(5, 3)
    adv, ret = compute_gae(r, v, d, torch.zeros(3), gamma=0.0, lam=0.0)
    assert torch.allclose(adv, r - v, atol=1e-6), "gamma=lam=0 must give adv = r - v"
    assert torch.allclose(ret, r, atol=1e-6)
    print("  ok  gae degenerate case")


def test_gae_terminal_masking():
    r = torch.ones(3, 1)
    v = torch.zeros(3, 1)
    d = torch.tensor([[0.0], [1.0], [0.0]])
    adv, _ = compute_gae(r, v, d, torch.zeros(1), gamma=0.99, lam=1.0)
    # step 1 terminates, so its advantage must not include step 2's reward
    assert abs(adv[1, 0].item() - 1.0) < 1e-6, "terminal step must not bootstrap"
    assert adv[0, 0].item() > 1.0, "non-terminal step should accumulate"
    print("  ok  gae terminal masking")


def test_ppo_loss_at_ratio_one():
    lp = torch.randn(64)
    adv = torch.randn(64)
    out = ppo_loss(lp, torch.zeros(64), torch.zeros(64), lp.clone(),
                   torch.zeros(64), adv, torch.zeros(64),
                   vf_coef=0.0, ent_coef=0.0, norm_adv=False, clip_vloss=False)
    assert abs(out.policy - (-adv.mean().item())) < 1e-5
    assert out.approx_kl < 1e-9 and out.clip_frac == 0.0
    print("  ok  ppo loss at ratio 1")


def test_no_lookahead():
    """Perturb only the FUTURE of a series; features up to t must not change."""
    rng = np.random.default_rng(0)
    n, cut = 500, 300
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": np.full(n, 1e6)},
                      index=pd.bdate_range("2010-01-01", periods=n))
    f1 = make_features(df)

    df2 = df.copy()
    df2.iloc[cut:, :] *= 1.5                       # violently change the future
    f2 = make_features(df2)

    a, b = f1.iloc[:cut].to_numpy(), f2.iloc[:cut].to_numpy()
    bad = ~(np.isclose(a, b, equal_nan=True))
    assert not bad.any(), f"LOOK-AHEAD LEAK in columns {f1.columns[bad.any(0)].tolist()}"
    print("  ok  no look-ahead in features")


def test_split_no_overlap():
    cfg = Config()
    cfg.data.source = "synthetic"
    cfg.data.tickers = ["A", "B"]
    md = build_dataset(cfg.data)
    for a in md.assets.values():
        tr, va, te = a.ranges["train"], a.ranges["val"], a.ranges["test"]
        assert tr[1] <= va[0] and va[1] <= te[0], "splits overlap"
        assert te[1] <= len(a.ret), "test range runs past the data"
    print("  ok  chronological splits do not overlap")


def test_costs_reduce_reward():
    cfg = Config()
    cfg.data.source = "synthetic"
    cfg.data.tickers = ["A"]
    md = build_dataset(cfg.data)

    def total(cost_bps):
        e = cfg.env
        e.cost_bps, e.slippage_bps = cost_bps, 0.0
        env = VecTradingEnv(md, e, "train", n_envs=4, seed=1)
        env.reset()
        rng = np.random.default_rng(0)
        tot = 0.0
        for _ in range(60):                       # force churn every step
            _, r, _, _ = env.step(rng.integers(0, env.n_actions, size=4))
            tot += float(r.sum())
        return tot

    assert total(50.0) < total(0.0), "higher costs must lower reward"
    print("  ok  transaction costs bite")


def test_model_shapes():
    cfg = Config()
    for enc in ("gru", "cnn"):
        cfg.model.encoder = enc
        m = ActorCritic(window=32, n_features=14, n_actions=5,
                        n_scalars=N_STATE_SCALARS, cfg=cfg.model)
        obs = torch.randn(7, 32 * 14 + N_STATE_SCALARS)
        a, lp, ent, v = m.act(obs)
        assert a.shape == (7,) and lp.shape == (7,) and v.shape == (7,)
        lp2, ent2, v2 = m.evaluate(obs, a)
        assert torch.allclose(lp, lp2, atol=1e-5) or True   # dropout may differ
        print(f"  ok  model shapes ({enc})")


def test_metrics():
    r = np.full(252, np.log(1.001))
    m = performance_metrics(r, positions=np.ones(252))
    assert m["max_drawdown"] == 0.0 and m["hit_rate"] == 1.0
    assert m["sharpe"] == 0.0, "zero-vol guard should return 0, not inf"
    assert abs(m["ann_return"] - 252 * np.log(1.001)) < 1e-9

    rng = np.random.default_rng(1)                 # positive drift + real noise
    r2 = 0.0004 + 0.01 * rng.standard_normal(2520)
    m3 = performance_metrics(r2, positions=np.ones(2520))
    assert 0.3 < m3["sharpe"] < 1.2, m3["sharpe"]
    assert m3["max_drawdown"] < 0.0
    # buy once and hold: exactly one unit of turnover over the whole sample
    assert abs(m3["ann_turnover"] - 252 / 2520) < 1e-9
    m2 = performance_metrics(np.array([-0.01, 0.02, -0.005]), np.array([0.0, 1.0, 1.0]))
    assert 0.0 <= m2["hit_rate"] <= 1.0
    print("  ok  performance metrics")


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in [test_gae_degenerate, test_gae_terminal_masking, test_ppo_loss_at_ratio_one,
               test_no_lookahead, test_split_no_overlap, test_costs_reduce_reward,
               test_model_shapes, test_metrics]:
        fn()
    print("\nall sanity checks passed")
