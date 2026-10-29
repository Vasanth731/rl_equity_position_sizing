# PPO trading agent (Yahoo Finance data)

A complete, runnable reinforcement-learning pipeline for daily equity position
sizing. PPO implemented from scratch in PyTorch — no `stable-baselines3`, no
`gym` dependency, so every piece is inspectable.

**Read this first:** RL does not predict prices. A supervised model predicts
`r_{t+1}`; an RL agent learns a *policy* — how much to hold, given the state,
after transaction costs. That is a different and usually more useful object, but
it means "accuracy" is not the metric here. Sharpe net of costs is.

---

## Files

| File | Role |
|---|---|
| `config.py` | Every hyperparameter, in one dataclass tree. Saved with each checkpoint. |
| `dataloader.py` | Download (yfinance/CSV/synthetic) → causal features → leak-free chronological splits. |
| `env.py` | Vectorized trading env: discrete target weights, costs, slippage, holding charge. |
| `model.py` | Actor-critic; GRU or dilated-CNN sequence encoder. |
| `loss.py` | GAE, clipped PPO objective, financial metrics (Sharpe, Sortino, Calmar, DD, turnover). |
| `train.py` | PPO loop, validation-based model selection, CSV logging, checkpoints. |
| `inference.py` | Walk-forward backtest, per-trade CSVs, summary table, equity-curve plot. |
| `test_sanity.py` | 9 checks, including an explicit look-ahead-leakage test. |

## Quickstart

```bash
pip install -r requirements.txt
python test_sanity.py                              # 10 seconds, do this first

# offline smoke test (no network needed)
python dataloader.py --source synthetic --out data/dataset.pkl
python train.py --dataset data/dataset.pkl --total_steps 50000 --out_dir runs/smoke

# real data
python dataloader.py --source yfinance --tickers AAPL,MSFT,JPM,XOM,JNJ,KO,CAT,PG \
    --start 2005-01-01 --end 2024-12-31 --out data/dataset.pkl
python train.py --dataset data/dataset.pkl --total_steps 400000 --out_dir runs/ppo
python inference.py --ckpt runs/ppo/best.pt --split test --plot
```

CSVs are cached in `data/`, so re-running the loader does not re-download.
Roughly 30–60 min for 400k steps on CPU, a few minutes on a GPU.

## How it's set up

**State** — 64 days × 14 features, standardized, plus 3 scalars (current
position, current drawdown, fraction of episode elapsed). Features: multi-horizon
log returns, realized and downside vol, RSI, MACD histogram, SMA gaps, ATR,
high-low range, volume z-score, drawdown from the 252-day high. All strictly
backward-looking.

**Action** — a *target* weight from `[-1, -0.5, 0, 0.5, 1]`. Discrete keeps the
policy a simple categorical, which is far more stable than a Gaussian head at
this signal-to-noise ratio.

**Reward** — `(w_t · logret_t − costs_t) × 100`, where
`costs = 7bps · |Δw| + 0.2bps · |w|`. Log returns make the reward additive, so
the discounted sum is (approximately) log terminal wealth. Charging cost on the
*change* in position is what teaches the agent to hold instead of churn — set
`--cost_bps 0` and watch turnover explode.

**Episodes** — random start date *and* random ticker each episode. Training
across eight assets is the cheapest regularizer available: a policy that only
works on one price path won't survive it.

## Three things in here that most tutorials get wrong

1. **Normalization is fit on the training slice only** (`dataloader.py`). Fitting
   a scaler on the full series leaks the future variance into the past and
   inflates every metric downstream.
2. **`gamma = 0.9`, not 0.99.** The action at *t* affects the reward at *t* and
   — through the turnover cost — at *t+1*, and essentially nothing after. At 0.99
   the critic must forecast ~100 days of market return, which is noise; it swamps
   the signal. In my runs the policy barely moved at 0.99 and began learning
   immediately at 0.9. This is the single highest-leverage setting in the file.
3. **Model selection happens on the validation period, by walk-forward Sharpe** —
   never on training reward. Training reward keeps improving long after
   generalization stops.

## Reading the results honestly

From an actual (deliberately short, 43k-step) run on synthetic data, best
validation Sharpe was **+0.97** — recorded at iteration 5 of 21, before the
policy had learned anything. Test Sharpe for that same checkpoint: **+0.14**.
That gap is not a bug; it is what selecting a maximum over ~5 noisy evaluations
buys you. Guard against it:

- `deflated_sharpe_note(sharpe, n_trials, n_periods)` in `loss.py` subtracts the
  expected maximum Sharpe of `n_trials` random strategies. Count *every*
  configuration you tried, not just the one you kept. If the deflated number is
  ≤ 0, you have found nothing.
- Run 5+ seeds and report the median and spread. A single seed is an anecdote.
- Compare against the right benchmark. `inference.py` prints `bh_sharpe`
  alongside; beating a bear market by holding cash is not skill.
- Check `turnover`. Annual turnover of 20× means costs dominate and the edge must
  be large and real to survive a 5bps error in your cost assumption.

## Known limitations

- **Single-asset policy**, applied independently per ticker and equal-weighted.
  No cross-sectional ranking, no portfolio-level risk budget, no correlation
  handling. A portfolio version would replace the categorical head with a
  simplex/weight output over all assets.
- **Costs are a flat linear model.** Real impact is nonlinear in size and
  regime-dependent; for anything above small size, 7bps is optimistic.
- **Daily close-to-close fills** assume you transact at the close you observed.
  In practice, decide on close *t* and fill at open *t+1*: shift `ret` by one
  more day to model that honestly.
- **No survivorship-bias correction.** A fixed list of tickers that exist today
  is a list of winners. For serious work, use a point-in-time index constituent
  file.
- **Delisting, splits, dividends** come from yfinance's adjusted close, which is
  restated over time and not point-in-time accurate.

## Extending it

- Swap the GRU for a small causal Transformer (`model.py`, keep the mask causal).
- Add regime features from FRED (yield-curve slope, VIX) — exogenous inputs help
  far more than a deeper network.
- Try continuous actions with a squashed Gaussian and compare turnover.
- Replace PPO with distributional or risk-sensitive RL if you care about the tail
  rather than the mean.

---

Research and educational code. Not investment advice; nothing here is validated
for trading real money.
