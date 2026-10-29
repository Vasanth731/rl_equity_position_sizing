"""dataloader2.py -- load a folder of per-ticker NSE CSVs.

Drop-in alternative to dataloader.py for a directory laid out like:

    nse_individual_stocks2/
        TATACAP.csv    TATACONSUM.csv   TATAPOWER.csv   TCS.csv    TTML.csv
        TATACHEM.csv   TATAELXSI.csv    TATASTEEL.csv   TMCV.csv
        TATACOMM.csv   TATAINVEST.csv   TATATECH.csv    TMPV.csv

Each file is the yfinance-style export:

    Date,Adj Close,Close,High,Low,Open,Volume,Symbol
    ,TATACAP.NS,TATACAP.NS,TATACAP.NS,TATACAP.NS,TATACAP.NS,TATACAP.NS,     <- dropped
    2025-10-13,329.949005,330.5,333.0,326.25,330.0,117611923,TATACAP

It emits the same MarketData/AssetSeries objects as dataloader.py, so env.py,
train.py and inference.py work against it unchanged:

    python dataloader2.py --dir ~/amb/rl_stock/nse_individual_stocks2 \
        --out data/nse.pkl --window 64
    python train.py --dataset data/nse.pkl --total_steps 400000

Three things this handles that the plain loader does not
-------------------------------------------------------
1. The second row of each file is a yfinance MultiIndex artifact (the ticker
   repeated across every column). Parsed as data it would poison the series, so
   any row whose Date will not parse is dropped.
2. `Adj Close` and `Close` genuinely differ in these files -- in the sample,
   196 of 219 rows, by up to ~0.17%. Returns are computed from the adjusted
   series, and open/high/low are rescaled by adj/close so intraday features
   (ATR, high-low range) stay consistent with it.
3. The files have wildly different lengths -- a recent listing may have ~200
   rows while TCS has years. Splitting each ticker at its own 70/85% marks
   would put every ticker's test period in a different calendar window, which
   makes a portfolio backtest meaningless. Default here is `--split-mode date`:
   one set of global cut dates for everyone.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config, DataConfig
from dataloader import (
    FEATURE_NAMES,
    AssetSeries,
    MarketData,
    make_features,
    save_dataset,
)

# Accepted spellings for each column we need, lowercased and stripped
COLUMN_ALIASES = {
    "date": {"date", "datetime", "timestamp", "index", ""},
    "open": {"open", "o"},
    "high": {"high", "h"},
    "low": {"low", "l"},
    "close": {"close", "c", "close_price"},
    "adj_close": {"adj close", "adj_close", "adjclose", "adjusted close",
                  "adjusted_close", "close_adj"},
    "volume": {"volume", "vol", "v", "totaltradedquantity"},
    "symbol": {"symbol", "ticker", "name"},
}


def _canonical_columns(cols: List[str]) -> Dict[str, str]:
    """Map the file's actual column names onto our canonical names."""
    out: Dict[str, str] = {}
    for raw in cols:
        key = str(raw).strip().lower().replace("*", "")
        for canon, aliases in COLUMN_ALIASES.items():
            if key in aliases and canon not in out:
                out[canon] = raw
                break
    return out


def read_stock_csv(
    path: str,
    use_adjusted: bool = True,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """Read one file into a clean OHLCV frame indexed by date.

    Returns None if the file cannot be used at all.
    """
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    except Exception as e:                                    # unreadable file
        print(f"[skip] {os.path.basename(path)}: unreadable ({e})")
        return None

    if raw.empty:
        print(f"[skip] {os.path.basename(path)}: empty")
        return None

    cmap = _canonical_columns(list(raw.columns))
    need = {"open", "high", "low", "close", "volume"}
    missing = need - set(cmap)
    if "date" not in cmap:
        # a nameless first column is often the date index
        first = raw.columns[0]
        if str(first).strip().lower() in {"", "unnamed: 0"}:
            cmap["date"] = first
        else:
            print(f"[skip] {os.path.basename(path)}: no date column")
            return None
    if missing:
        print(f"[skip] {os.path.basename(path)}: missing {sorted(missing)}")
        return None

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[cmap["date"]], errors="coerce", format="mixed"),
        "open": pd.to_numeric(raw[cmap["open"]], errors="coerce"),
        "high": pd.to_numeric(raw[cmap["high"]], errors="coerce"),
        "low": pd.to_numeric(raw[cmap["low"]], errors="coerce"),
        "close": pd.to_numeric(raw[cmap["close"]], errors="coerce"),
        "volume": pd.to_numeric(raw[cmap["volume"]], errors="coerce"),
    })
    if "adj_close" in cmap:
        df["adj_close"] = pd.to_numeric(raw[cmap["adj_close"]], errors="coerce")

    n0 = len(df)
    # The yfinance ticker sub-header row has an unparseable date -> dropped here
    df = df[df["date"].notna()].copy()
    dropped_header = n0 - len(df)

    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)

    # basic sanity: positive prices, high >= low, no duplicate dates
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"])
    n_bad = int(bad.sum())
    df = df[~bad]
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")

    if df.empty:
        print(f"[skip] {os.path.basename(path)}: no usable rows")
        return None

    # Apply the dividend/split adjustment to every price, not just the close,
    # so intraday-derived features stay on the same scale as the returns.
    if use_adjusted and "adj_close" in df.columns and df["adj_close"].notna().any():
        factor = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        for c in ("open", "high", "low"):
            df[c] = df[c] * factor
        df["close"] = df["adj_close"]
        adj_rows = int((factor.round(8) != 1.0).sum())
    else:
        adj_rows = 0

    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]

    if verbose:
        print(f"       rows={len(df):<6} {df.index[0].date()} -> {df.index[-1].date()}"
              f"  header_rows_dropped={dropped_header}  bad_rows={n_bad}"
              f"  adjusted_rows={adj_rows}")
    return df


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def _global_cut_dates(
    frames: Dict[str, pd.DataFrame], train_frac: float, val_frac: float
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Two calendar dates that split the union of all trading days."""
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
    n = len(all_dates)
    return all_dates[int(n * train_frac)], all_dates[int(n * (train_frac + val_frac))]


def _ranges_by_date(
    dates: np.ndarray, window: int, t_cut: pd.Timestamp, v_cut: pd.Timestamp
) -> Dict[str, Tuple[int, int]]:
    idx = pd.DatetimeIndex(dates)
    a = int(idx.searchsorted(t_cut))          # first val index
    b = int(idx.searchsorted(v_cut))          # first test index
    T = len(idx)
    W = window
    return {
        "train": (min(W - 1, max(0, a - 1)) if a > W - 1 else W - 1, max(W - 1, a - 1)),
        "val":   (max(W - 1, a), max(W - 1, b - 1)),
        "test":  (max(W - 1, b), T - 1),
    }


def _ranges_by_fraction(
    T: int, window: int, train_frac: float, val_frac: float
) -> Dict[str, Tuple[int, int]]:
    n_tr = int(T * train_frac)
    n_va = int(T * val_frac)
    W = window
    return {
        "train": (W - 1, max(W - 1, n_tr - 1)),
        "val":   (max(W - 1, n_tr), max(W - 1, n_tr + n_va - 1)),
        "test":  (max(W - 1, n_tr + n_va), T - 1),
    }


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #
def build_dataset_from_dir(
    directory: str,
    window: int = 64,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    clip_sigma: float = 5.0,
    split_mode: str = "date",
    use_adjusted: bool = True,
    min_rows: Optional[int] = None,
    min_train_rows: int = 40,
    tickers: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    pattern: str = "*.csv",
    verbose: bool = True,
) -> MarketData:
    directory = os.path.expanduser(directory)
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")

    want = {t.upper() for t in tickers} if tickers else None
    drop = {t.upper() for t in exclude} if exclude else set()
    if min_rows is None:
        min_rows = window + 60

    frames: Dict[str, pd.DataFrame] = {}
    for p in paths:
        tk = os.path.splitext(os.path.basename(p))[0].upper()
        if want and tk not in want:
            continue
        if tk in drop:
            continue
        if verbose:
            print(f"[read] {tk}")
        df = read_stock_csv(p, use_adjusted=use_adjusted, verbose=verbose)
        if df is None:
            continue
        if len(df) < min_rows:
            print(f"[skip] {tk}: {len(df)} rows < min_rows={min_rows}")
            continue
        frames[tk] = df

    if not frames:
        raise RuntimeError("no usable tickers -- try lowering --min-rows or --window")

    if split_mode == "date":
        t_cut, v_cut = _global_cut_dates(frames, train_frac, val_frac)
        if verbose:
            print(f"\nglobal cut dates: train < {t_cut.date()} <= val < {v_cut.date()} <= test")
    else:
        t_cut = v_cut = None

    assets: Dict[str, AssetSeries] = {}
    for tk, df in frames.items():
        feats = make_features(df)
        ret = np.log(df["close"]).diff().shift(-1)        # ret[t] = log(C[t+1]/C[t])

        keep = feats.notna().all(axis=1) & ret.notna()
        arr = keep.to_numpy()
        if not arr.any():
            print(f"[skip] {tk}: no rows survive feature warm-up")
            continue
        first, last = int(np.argmax(arr)), len(arr) - int(np.argmax(arr[::-1]))
        feats = feats.iloc[first:last].ffill().bfill()
        close = df["close"].iloc[first:last]
        ret = ret.iloc[first:last]
        T = len(feats)
        if T < window + 10:
            print(f"[skip] {tk}: only {T} rows after warm-up")
            continue

        dates = feats.index
        if split_mode == "date":
            ranges = _ranges_by_date(dates.to_numpy(), window, t_cut, v_cut)
            n_train_rows = int(pd.DatetimeIndex(dates).searchsorted(t_cut))
        else:
            ranges = _ranges_by_fraction(T, window, train_frac, val_frac)
            n_train_rows = int(T * train_frac)

        # Standardize on TRAIN ROWS ONLY. A ticker with no training history
        # cannot be scaled without leaking the future, so it is dropped.
        if n_train_rows < min_train_rows:
            print(f"[skip] {tk}: only {n_train_rows} training rows "
                  f"(needs {min_train_rows}) -- listed too recently for this split")
            continue

        X = feats.to_numpy(dtype=np.float64)
        mu = X[:n_train_rows].mean(0)
        sd = X[:n_train_rows].std(0) + 1e-8
        Z = np.clip((X - mu) / sd, -clip_sigma, clip_sigma).astype(np.float32)

        assets[tk] = AssetSeries(
            ticker=tk,
            features=Z,
            ret=ret.to_numpy(dtype=np.float32),
            close=close.to_numpy(dtype=np.float32),
            dates=dates.to_numpy(),
            ranges=ranges,
        )

    if not assets:
        raise RuntimeError("every ticker was dropped -- see the reasons above")

    md = MarketData(assets=assets, feature_names=list(FEATURE_NAMES), window=window)
    if verbose:
        print()
        print(report(md))
    return md


def report(md: MarketData) -> str:
    rows = []
    for tk, a in sorted(md.assets.items()):
        r = a.ranges
        rows.append({
            "ticker": tk,
            "rows": len(a.close),
            "from": str(a.dates[0])[:10],
            "to": str(a.dates[-1])[:10],
            "train": max(0, r["train"][1] - r["train"][0]),
            "val": max(0, r["val"][1] - r["val"][0]),
            "test": max(0, r["test"][1] - r["test"][0]),
        })
    df = pd.DataFrame(rows).set_index("ticker")
    usable = [c for c in ("train", "val", "test")]
    tail = (f"\n{len(md.assets)} tickers | {md.n_features} features | window={md.window}"
            f" | usable decision steps: "
            + ", ".join(f"{c}={int(df[c].sum())}" for c in usable))
    warn = ""
    for c in usable:
        empty = df.index[df[c] <= 0].tolist()
        if empty:
            warn += f"\n[warn] no {c} steps for: {', '.join(empty)}"
    return df.to_string() + tail + warn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True, help="folder of per-ticker CSVs")
    p.add_argument("--out", default="data2/nse.pkl")
    p.add_argument("--pattern", default="*.csv")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--split-mode", default="date", choices=["date", "fraction"],
                   help="'date' = one global calendar split for all tickers")
    p.add_argument("--min-rows", type=int, default=None,
                   help="skip files shorter than this (default: window + 60)")
    p.add_argument("--min-train-rows", type=int, default=40)
    p.add_argument("--no-adjust", action="store_true",
                   help="use raw Close instead of Adj Close")
    p.add_argument("--tickers", default=None, help="comma separated whitelist")
    p.add_argument("--exclude", default=None, help="comma separated blacklist")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    md = build_dataset_from_dir(
        directory=a.dir,
        window=a.window,
        train_frac=a.train_frac,
        val_frac=a.val_frac,
        split_mode=a.split_mode,
        use_adjusted=not a.no_adjust,
        min_rows=a.min_rows,
        min_train_rows=a.min_train_rows,
        tickers=[t.strip() for t in a.tickers.split(",")] if a.tickers else None,
        exclude=[t.strip() for t in a.exclude.split(",")] if a.exclude else None,
        pattern=a.pattern,
        verbose=not a.quiet,
    )
    save_dataset(md, a.out)
    print(f"\nsaved -> {a.out}")
    print(f"next:  python train.py --dataset {a.out} --total_steps 400000")


if __name__ == "__main__":
    main()