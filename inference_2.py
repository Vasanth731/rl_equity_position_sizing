"""Walk-forward inference / backtest.

This is deliberately *not* the training env: it steps through the chosen split
once, in order, with no random episode starts and no resets, carrying the
position forward exactly as live trading would. All tickers are advanced in
lockstep so the policy forward pass is batched.

    python inference.py --ckpt runs/ppo/best.pt --split test --plot
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from config import Config, EnvConfig, resolve_device
from dataloader import MarketData, load_dataset
from env import N_STATE_SCALARS
from loss import performance_metrics
from model import ActorCritic


@torch.no_grad()
def run_backtest(
    model: ActorCritic,
    md: MarketData,
    split: str,
    ecfg: EnvConfig,
    device: str = "cpu",
    deterministic: bool = True,
    tickers: List[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    model.eval()
    tks = tickers or md.tickers(split)
    W, F = md.window, md.n_features
    positions = np.asarray(ecfg.positions, dtype=np.float32)
    cost = (ecfg.cost_bps + ecfg.slippage_bps) / 1e4
    hold = ecfg.holding_cost_bps / 1e4

    lo = np.array([md.assets[t].ranges[split][0] for t in tks])
    hi = np.array([md.assets[t].ranges[split][1] for t in tks])
    n = len(tks)
    pos = np.zeros(n, dtype=np.float32)
    eq = np.ones(n, dtype=np.float32)
    peak = np.ones(n, dtype=np.float32)
    rec: Dict[str, List[dict]] = {t: [] for t in tks}
    max_len = int((hi - lo).max())

    for k in range(max_len):
        t_idx = lo + k
        active = np.where(t_idx < hi)[0]
        if active.size == 0:
            break

        obs = np.empty((active.size, W * F + N_STATE_SCALARS), dtype=np.float32)
        for j, i in enumerate(active):
            a = md.assets[tks[i]]
            t = int(t_idx[i])
            obs[j, : W * F] = a.features[t - W + 1 : t + 1].ravel()
            obs[j, -3] = pos[i]
            obs[j, -2] = eq[i] / peak[i] - 1.0
            obs[j, -1] = k / max(1, max_len)

        act, _, _, _ = model.act(torch.as_tensor(obs, device=device),
                                 deterministic=deterministic)
        target = positions[act.cpu().numpy()]

        if ecfg.vol_target > 0:
            for j, i in enumerate(active):
                a = md.assets[tks[i]]
                t = int(t_idx[i])
                rv = float(np.std(a.ret[max(0, t - 20) : t]) * np.sqrt(252)) + 1e-6
                target[j] *= float(np.clip(ecfg.vol_target / rv, 0.0, 1.5))

        for j, i in enumerate(active):
            a = md.assets[tks[i]]
            t = int(t_idx[i])
            turn = abs(float(target[j]) - float(pos[i]))
            c = cost * turn + hold * abs(float(target[j]))
            gross = float(target[j]) * float(a.ret[t])
            net = gross - c
            eq[i] *= np.exp(net)
            peak[i] = max(peak[i], eq[i])
            pos[i] = target[j]
            rec[tks[i]].append({
                "date": a.dates[t],
                "close": float(a.close[t]),
                "asset_logret": float(a.ret[t]),
                "position": float(target[j]),
                "turnover": turn,
                "cost": c,
                "net_logret": net,
                "equity": float(eq[i]),
            })

    out = {}
    for t in tks:
        df = pd.DataFrame(rec[t]).set_index("date")
        df["buy_hold_equity"] = np.exp(df["asset_logret"].cumsum())
        out[t] = df
    return out


def summarize(bt: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tk, df in bt.items():
        m = performance_metrics(df["net_logret"].to_numpy(), df["position"].to_numpy())
        b = performance_metrics(df["asset_logret"].to_numpy())
        rows.append({
            "ticker": tk,
            "sharpe": m["sharpe"], "bh_sharpe": b["sharpe"],
            "ann_return": m["ann_return"], "bh_ann_return": b["ann_return"],
            "ann_vol": m["ann_vol"],
            "max_dd": m["max_drawdown"], "bh_max_dd": b["max_drawdown"],
            "hit_rate": m["hit_rate"], "exposure": m["avg_exposure"],
            "turnover": m["ann_turnover"], "total_cost": float(df["cost"].sum()),
        })
    out = pd.DataFrame(rows).set_index("ticker")
    if len(bt) == 1:
        return out          # a one-stock "portfolio" row would just repeat the row above

    # Equal-weight portfolio of all sleeves
    port = pd.concat([df["net_logret"].rename(tk) for tk, df in bt.items()], axis=1)
    pm = performance_metrics(port.mean(axis=1).to_numpy())
    bench = pd.concat([df["asset_logret"].rename(tk) for tk, df in bt.items()], axis=1)
    bm = performance_metrics(bench.mean(axis=1).to_numpy())
    out.loc["PORTFOLIO"] = {
        "sharpe": pm["sharpe"], "bh_sharpe": bm["sharpe"],
        "ann_return": pm["ann_return"], "bh_ann_return": bm["ann_return"],
        "ann_vol": pm["ann_vol"], "max_dd": pm["max_drawdown"],
        "bh_max_dd": bm["max_drawdown"], "hit_rate": pm["hit_rate"],
        "exposure": out["exposure"].mean(), "turnover": out["turnover"].mean(),
        "total_cost": out["total_cost"].sum(),
    }
    return out


def portfolio_curve(bt: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    strat = pd.concat([df["net_logret"].rename(tk) for tk, df in bt.items()], axis=1).mean(axis=1)
    bench = pd.concat([df["asset_logret"].rename(tk) for tk, df in bt.items()], axis=1).mean(axis=1)
    return pd.DataFrame({
        "strategy": np.exp(strat.cumsum()),
        "buy_hold": np.exp(bench.cumsum()),
    })


def ticker_curve(bt: Dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
    """Equity curve for ONE company: the policy on that stock vs holding it."""
    if ticker not in bt:
        raise KeyError(f"{ticker!r} not in backtest; have {sorted(bt)}")
    df = bt[ticker]
    return pd.DataFrame({"strategy": df["equity"], "buy_hold": df["buy_hold_equity"]})


def plot_curve(curve: pd.DataFrame, positions: pd.Series, path: str,
               title: str, equity_label: str = "equity") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(curve.index, curve["strategy"], label="RL policy", lw=1.6)
    ax[0].plot(curve.index, curve["buy_hold"], label="buy & hold", lw=1.2, alpha=0.75)
    ax[0].axhline(1.0, color="k", lw=0.6, alpha=0.4)
    ax[0].set_ylabel(equity_label)
    ax[0].set_title(title)
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(positions.index, positions, lw=0.9, color="tab:gray")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("position"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_model(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: Config = ck["config"]
    model = ActorCritic(window=ck["window"], n_features=ck["n_features"],
                        n_actions=ck["n_actions"], n_scalars=N_STATE_SCALARS,
                        cfg=cfg.model).to(device)
    model.load_state_dict(ck["model"])
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs2/ppo/best.pt")
    p.add_argument("--dataset", default="data2/nse.pkl")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--stochastic", action="store_true", help="sample instead of argmax")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--tickers", default=None,
                   help="comma separated; restrict the backtest to these companies")
    p.add_argument("--plot-each", action="store_true", dest="plot_each",
                   help="also write one equity-curve PNG per company")
    a = p.parse_args()

    device = resolve_device("auto")
    model, cfg = load_model(a.ckpt, device)
    md = load_dataset(a.dataset)

    available = md.tickers(a.split)
    want = None
    if a.tickers:
        want = [t.strip().upper() for t in a.tickers.split(",")]
        unknown = [t for t in want if t not in available]
        if unknown:
            raise SystemExit(f"not in the {a.split} split: {unknown}\n"
                             f"available: {sorted(available)}")
    out_dir = a.out_dir or os.path.join(os.path.dirname(a.ckpt) or ".", f"backtest_{a.split}")
    os.makedirs(out_dir, exist_ok=True)

    bt = run_backtest(model, md, a.split, cfg.env, device,
                      deterministic=not a.stochastic, tickers=want)
    summary = summarize(bt)
    curve = portfolio_curve(bt)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(f"\n=== {a.split.upper()} backtest "
          f"({curve.index[0].date()} .. {curve.index[-1].date()}) ===")
    print(summary.round(3).to_string())

    # Per-company artifacts: each ticker gets its own trade log and equity curve
    for tk, df in bt.items():
        df.to_csv(os.path.join(out_dir, f"{tk}_trades.csv"))
        ticker_curve(bt, tk).to_csv(os.path.join(out_dir, f"{tk}_equity.csv"))
    summary.to_csv(os.path.join(out_dir, "summary.csv"))
    curve.to_csv(os.path.join(out_dir, "portfolio_curve.csv"))

    if a.plot:
        if len(bt) == 1:
            tk = next(iter(bt))
            path = os.path.join(out_dir, f"{tk}_equity.png")
            plot_curve(ticker_curve(bt, tk), bt[tk]["position"], path,
                       f"{tk} -- {a.split} period, net of costs", f"{tk} equity")
        else:
            expo = pd.concat([d["position"].rename(t) for t, d in bt.items()],
                             axis=1).mean(axis=1)
            path = os.path.join(out_dir, "equity_curve.png")
            plot_curve(curve, expo, path,
                       f"equal-weight portfolio of {len(bt)} -- {a.split} period, "
                       f"net of costs", "equity (equal-weight)")
        print(f"plot -> {path}")

    if a.plot_each:
        for tk in bt:
            p_tk = os.path.join(out_dir, f"{tk}_equity.png")
            plot_curve(ticker_curve(bt, tk), bt[tk]["position"], p_tk,
                       f"{tk} -- {a.split} period, net of costs", f"{tk} equity")
        print(f"per-company plots -> {out_dir}/<TICKER>_equity.png")

    print(f"artifacts -> {out_dir}")


if __name__ == "__main__":
    main()