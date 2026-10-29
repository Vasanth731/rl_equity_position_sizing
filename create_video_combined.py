# """Create a single video showing 5 companies trading simultaneously.

# Shows all 5 stocks in one video with their equity curves, positions, and stats.

# Usage:
#     python3 create_video_combined.py --ckpt runs2/ppo/best.pt --n_tickers 5 --split test --out combined_trading.webm
# """

# from __future__ import annotations

# import argparse
# import os
# from typing import Dict, List

# import cv2
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import numpy as np
# import pandas as pd
# import torch

# from config import Config, EnvConfig, resolve_device
# from dataloader import MarketData, load_dataset
# from env import N_STATE_SCALARS
# from loss import performance_metrics
# from model import ActorCritic


# def load_model(ckpt_path: str, device: str):
#     """Load trained model from checkpoint."""
#     ck = torch.load(ckpt_path, map_location=device, weights_only=False)
#     cfg: Config = ck["config"]
#     model = ActorCritic(
#         window=ck["window"],
#         n_features=ck["n_features"],
#         n_actions=ck["n_actions"],
#         n_scalars=N_STATE_SCALARS,
#         cfg=cfg.model
#     ).to(device)
#     model.load_state_dict(ck["model"])
#     return model, cfg


# @torch.no_grad()
# def collect_trading_data(
#     model: ActorCritic,
#     md: MarketData,
#     ticker: str,
#     split: str,
#     ecfg: EnvConfig,
#     device: str = "cpu",
# ) -> pd.DataFrame:
#     """Run backtest and collect all trading decisions."""
#     model.eval()
#     asset = md.assets[ticker]
#     W, F = md.window, md.n_features
#     positions = np.asarray(ecfg.positions, dtype=np.float32)
#     cost = (ecfg.cost_bps + ecfg.slippage_bps) / 1e4
#     hold = ecfg.holding_cost_bps / 1e4

#     lo, hi = asset.ranges[split]
#     pos = 0.0
#     eq = 1.0
#     peak = 1.0
#     records = []

#     for k in range(hi - lo):
#         t = lo + k

#         # Build observation
#         obs = np.empty(W * F + N_STATE_SCALARS, dtype=np.float32)
#         obs[:W * F] = asset.features[t - W + 1 : t + 1].ravel()
#         obs[-3] = pos
#         obs[-2] = eq / peak - 1.0
#         obs[-1] = k / max(1, hi - lo)

#         # Get action from model
#         act, _, _, value = model.act(
#             torch.as_tensor(obs, device=device).unsqueeze(0),
#             deterministic=True
#         )
#         target = float(positions[act.item()])

#         # Apply volatility targeting if enabled
#         if ecfg.vol_target > 0:
#             rv = float(np.std(asset.ret[max(0, t - 20):t]) * np.sqrt(252)) + 1e-6
#             target *= float(np.clip(ecfg.vol_target / rv, 0.0, 1.5))

#         # Calculate costs and returns
#         turn = abs(target - pos)
#         c = cost * turn + hold * abs(target)
#         gross = target * float(asset.ret[t])
#         net = gross - c
#         eq *= np.exp(net)
#         peak = max(peak, eq)

#         # Record state
#         records.append({
#             "date": asset.dates[t],
#             "close": float(asset.close[t]),
#             "position": target,
#             "action": int(act.item()),
#             "value_estimate": float(value.item()),
#             "asset_ret": float(asset.ret[t]),
#             "net_ret": net,
#             "equity": eq,
#             "turnover": turn,
#             "cost": c,
#             "drawdown": eq / peak - 1.0,
#         })

#         pos = target

#     df = pd.DataFrame(records).set_index("date")
#     df["buy_hold_equity"] = np.exp(df["asset_ret"].cumsum())
#     return df


# def create_frame(
#     axes_dict: Dict,
#     data_dict: Dict[str, pd.DataFrame],
#     tickers: List[str],
#     current_idx: int,
#     positions_list: List[float],
# ) -> None:
#     """Draw a single frame showing all companies."""
#     # Clear all axes
#     for ax in axes_dict.values():
#         ax.clear()

#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

#     # --- Top: Combined Equity Curves ---
#     ax_equity = axes_dict['equity']
#     for i, ticker in enumerate(tickers):
#         df = data_dict[ticker]
#         if current_idx < len(df):
#             df_window = df.iloc[:current_idx+1]
#             ax_equity.plot(range(len(df_window)), df_window["equity"],
#                           label=f"{ticker} (Agent)", linewidth=1.8,
#                           color=colors[i], alpha=0.9)
#             ax_equity.plot(range(len(df_window)), df_window["buy_hold_equity"],
#                           linewidth=1.2, linestyle='--',
#                           color=colors[i], alpha=0.5)

#     ax_equity.axhline(1.0, color='black', linewidth=0.6, alpha=0.3)
#     ax_equity.set_ylabel("Equity", fontsize=11, fontweight='bold')
#     ax_equity.set_title("Multi-Stock Trading Agent - Portfolio View",
#                        fontsize=13, fontweight='bold')
#     ax_equity.legend(loc='upper left', fontsize=7, ncol=2)
#     ax_equity.grid(alpha=0.3)

#     # --- Middle: Position Indicators for Each Company ---
#     ax_positions = axes_dict['positions']
#     y_positions = np.arange(len(tickers))

#     for i, ticker in enumerate(tickers):
#         df = data_dict[ticker]
#         if current_idx < len(df):
#             pos = df.iloc[current_idx]["position"]
#             color = colors[i]

#             # Draw position bar
#             if pos >= 0:
#                 ax_positions.barh(y_positions[i], pos, height=0.6,
#                                  left=0, color=color, alpha=0.7)
#             else:
#                 ax_positions.barh(y_positions[i], -pos, height=0.6,
#                                  left=pos, color=color, alpha=0.7)

#             # Position text
#             pos_text = "LONG" if pos > 0.1 else "SHORT" if pos < -0.1 else "FLAT"
#             ax_positions.text(-1.05, y_positions[i], ticker,
#                             ha='right', va='center', fontsize=9, fontweight='bold')
#             ax_positions.text(0, y_positions[i], f"{pos_text}\n{pos:.2f}",
#                             ha='center', va='center', fontsize=8,
#                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

#     ax_positions.axvline(0, color='black', linewidth=1)
#     ax_positions.set_xlim(-1.3, 1.1)
#     ax_positions.set_ylim(-0.5, len(tickers) - 0.5)
#     ax_positions.set_yticks([])
#     ax_positions.set_xlabel("Position", fontsize=10, fontweight='bold')
#     ax_positions.grid(axis='x', alpha=0.3)

#     # --- Bottom: Stats Table ---
#     ax_stats = axes_dict['stats']
#     ax_stats.axis('off')

#     # Build stats table
#     table_data = []
#     headers = ["Ticker", "Equity", "B&H", "Sharpe", "Return%", "MaxDD%", "Action"]

#     action_names = ["FULL SHORT", "HALF SHORT", "FLAT", "HALF LONG", "FULL LONG"]

#     for ticker in tickers:
#         df = data_dict[ticker]
#         if current_idx < len(df):
#             df_so_far = df.iloc[:current_idx+1]
#             current = df.iloc[current_idx]

#             metrics = performance_metrics(
#                 df_so_far["net_ret"].to_numpy(),
#                 df_so_far["position"].to_numpy()
#             )

#             action_idx = int(current["action"])
#             action_label = action_names[action_idx] if action_idx < len(action_names) else str(action_idx)

#             table_data.append([
#                 ticker,
#                 f"{current['equity']:.3f}",
#                 f"{current['buy_hold_equity']:.3f}",
#                 f"{metrics['sharpe']:.2f}",
#                 f"{metrics['ann_return']*100:.1f}",
#                 f"{metrics['max_drawdown']*100:.1f}",
#                 action_label,
#             ])

#     # Create table
#     table = ax_stats.table(cellText=table_data, colLabels=headers,
#                           cellLoc='center', loc='center',
#                           bbox=[0, 0, 1, 1])
#     table.auto_set_font_size(False)
#     table.set_fontsize(9)
#     table.scale(1, 2)

#     # Style header
#     for i in range(len(headers)):
#         cell = table[(0, i)]
#         cell.set_facecolor('#4CAF50')
#         cell.set_text_props(weight='bold', color='white')

#     # Color rows by performance
#     for i, ticker in enumerate(tickers):
#         df = data_dict[ticker]
#         if current_idx < len(df):
#             current = df.iloc[current_idx]
#             outperform = current['equity'] > current['buy_hold_equity']
#             color = '#E8F5E9' if outperform else '#FFEBEE'

#             for j in range(len(headers)):
#                 cell = table[(i+1, j)]
#                 cell.set_facecolor(color)

#     # --- Bottom Right: Progress Info ---
#     ax_info = axes_dict['info']
#     ax_info.axis('off')

#     # Get date from first ticker
#     first_ticker = tickers[0]
#     df_first = data_dict[first_ticker]
#     if current_idx < len(df_first):
#         current_date = df_first.index[current_idx].date()
#         start_date = df_first.index[0].date()
#         end_date = df_first.index[-1].date()
#         progress = (current_idx + 1) / len(df_first) * 100

#         # Calculate portfolio stats (equal weight)
#         all_rets = []
#         for ticker in tickers:
#             df = data_dict[ticker]
#             if current_idx < len(df):
#                 all_rets.append(df.iloc[:current_idx+1]["net_ret"].to_numpy())

#         if all_rets:
#             # Align lengths
#             min_len = min(len(r) for r in all_rets)
#             all_rets = [r[:min_len] for r in all_rets]
#             portfolio_rets = np.mean(all_rets, axis=0)
#             portfolio_metrics = performance_metrics(portfolio_rets)

#             info_text = f"""PORTFOLIO (Equal Weight)
# Sharpe: {portfolio_metrics['sharpe']:.2f}
# Ann. Return: {portfolio_metrics['ann_return']*100:.1f}%
# Max DD: {portfolio_metrics['max_drawdown']*100:.1f}%

# PROGRESS
# Date: {current_date}
# Period: {start_date} - {end_date}
# Day {current_idx + 1} / {len(df_first)}
# {progress:.1f}% complete"""

#             ax_info.text(0.1, 0.9, info_text.strip(),
#                         transform=ax_info.transAxes,
#                         fontsize=9, verticalalignment='top',
#                         family='monospace',
#                         bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.5))


# def create_video(
#     data_dict: Dict[str, pd.DataFrame],
#     tickers: List[str],
#     output_path: str,
#     positions_list: List[float],
#     fps: int = 15,
#     speed: int = 2,
# ) -> None:
#     """Create combined video from trading data."""
#     # Find max length
#     max_len = max(len(df) for df in data_dict.values())
#     print(f"Creating combined video with {max_len} frames at {fps} fps...")

#     # Setup figure with layout
#     fig = plt.figure(figsize=(16, 12))
#     gs = fig.add_gridspec(4, 2, height_ratios=[3, 1.2, 1.8, 1],
#                          hspace=0.35, wspace=0.3)

#     axes_dict = {
#         'equity': fig.add_subplot(gs[0, :]),
#         'positions': fig.add_subplot(gs[1, :]),
#         'stats': fig.add_subplot(gs[2, :]),
#         'info': fig.add_subplot(gs[3, :]),
#     }

#     # Video writer setup
#     if output_path.endswith('.webm'):
#         fourcc = cv2.VideoWriter_fourcc(*'VP09')
#     elif output_path.endswith('.mp4'):
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     else:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')

#     height, width = 1200, 1600
#     fig.set_size_inches(width/100, height/100)

#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     try:
#         for i in range(0, max_len, speed):
#             if i % 50 == 0:
#                 print(f"  Processing frame {i}/{max_len}...")

#             create_frame(axes_dict, data_dict, tickers, i, positions_list)

#             # Convert to video frame
#             fig.canvas.draw()
#             img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#             img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
#             img = img[:, :, :3]  # Drop alpha
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             img = cv2.resize(img, (width, height))

#             out.write(img)

#         # Hold last frame
#         for _ in range(fps * 2):
#             out.write(img)

#     finally:
#         out.release()
#         plt.close(fig)

#     print(f"Video saved to: {output_path}")


# def main():
#     parser = argparse.ArgumentParser(description="Create combined trading video")
#     parser.add_argument("--ckpt", default="runs2/ppo/best.pt")
#     parser.add_argument("--dataset", default="data2/nse.pkl")
#     parser.add_argument("--tickers", default=None,
#                        help="Comma-separated tickers")
#     parser.add_argument("--n_tickers", type=int, default=5)
#     parser.add_argument("--split", default="test", choices=["train", "val", "test"])
#     parser.add_argument("--out", default="combined_trading.webm")
#     parser.add_argument("--fps", type=int, default=15)
#     parser.add_argument("--speed", type=int, default=2)
#     args = parser.parse_args()

#     # Load model and data
#     device = resolve_device("auto")
#     print(f"Loading model from {args.ckpt}...")
#     model, cfg = load_model(args.ckpt, device)

#     print(f"Loading dataset from {args.dataset}...")
#     md = load_dataset(args.dataset)

#     # Select tickers
#     available = md.tickers(args.split)
#     if not available:
#         raise ValueError(f"No tickers in {args.split} split")

#     if args.tickers:
#         tickers = [t.strip().upper() for t in args.tickers.split(",")]
#         tickers = [t for t in tickers if t in available]
#     else:
#         tickers = available[:args.n_tickers]

#     if not tickers:
#         raise ValueError("No valid tickers")

#     print(f"\nProcessing {len(tickers)} tickers: {', '.join(tickers)}")

#     # Collect data for all tickers
#     data_dict = {}
#     for ticker in tickers:
#         print(f"  Backtesting {ticker}...", end=' ')
#         df = collect_trading_data(model, md, ticker, args.split, cfg.env, device)
#         data_dict[ticker] = df
#         print(f"✓ ({len(df)} days, equity: {df['equity'].iloc[-1]:.3f})")

#     # Create video
#     print()
#     create_video(data_dict, tickers, args.out, cfg.env.positions, args.fps, args.speed)

#     # Summary
#     print("\nSUMMARY:")
#     for ticker in tickers:
#         df = data_dict[ticker]
#         final_eq = df['equity'].iloc[-1]
#         final_bh = df['buy_hold_equity'].iloc[-1]
#         symbol = "✓" if final_eq > final_bh else "✗"
#         print(f"  {symbol} {ticker}: {final_eq:.3f} vs B&H {final_bh:.3f}")

#     duration = max(len(df) for df in data_dict.values()) / args.speed / args.fps
#     print(f"\nVideo duration: ~{duration:.1f} seconds")


# if __name__ == "__main__":
#     main()


##########################################################################################




















######################################################################################################



# """Create a single video showing 5 companies trading simultaneously.

# Shows all 5 stocks in one video with their equity curves, positions, and stats.

# Usage:
#     python3 create_video_combined.py --ckpt runs2/ppo/best.pt --n_tickers 5 --split test --out combined_trading.webm
# """

# from __future__ import annotations

# import argparse
# import os
# from typing import Dict, List

# import cv2
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import numpy as np
# import pandas as pd
# import torch

# from config import Config, EnvConfig, resolve_device
# from dataloader import MarketData, load_dataset
# from env import N_STATE_SCALARS
# from loss import performance_metrics
# from model import ActorCritic


# def load_model(ckpt_path: str, device: str):
#     """Load trained model from checkpoint."""
#     ck = torch.load(ckpt_path, map_location=device, weights_only=False)
#     cfg: Config = ck["config"]
#     model = ActorCritic(
#         window=ck["window"],
#         n_features=ck["n_features"],
#         n_actions=ck["n_actions"],
#         n_scalars=N_STATE_SCALARS,
#         cfg=cfg.model
#     ).to(device)
#     model.load_state_dict(ck["model"])
#     return model, cfg


# @torch.no_grad()
# def collect_trading_data(
#     model: ActorCritic,
#     md: MarketData,
#     ticker: str,
#     split: str,
#     ecfg: EnvConfig,
#     device: str = "cpu",
# ) -> pd.DataFrame:
#     """Run backtest and collect all trading decisions."""
#     model.eval()
#     asset = md.assets[ticker]
#     W, F = md.window, md.n_features
#     positions = np.asarray(ecfg.positions, dtype=np.float32)
#     cost = (ecfg.cost_bps + ecfg.slippage_bps) / 1e4
#     hold = ecfg.holding_cost_bps / 1e4

#     lo, hi = asset.ranges[split]
#     pos = 0.0
#     eq = 1.0
#     peak = 1.0
#     records = []

#     for k in range(hi - lo):
#         t = lo + k

#         # Build observation
#         obs = np.empty(W * F + N_STATE_SCALARS, dtype=np.float32)
#         obs[:W * F] = asset.features[t - W + 1 : t + 1].ravel()
#         obs[-3] = pos
#         obs[-2] = eq / peak - 1.0
#         obs[-1] = k / max(1, hi - lo)

#         # Get action from model
#         act, _, _, value = model.act(
#             torch.as_tensor(obs, device=device).unsqueeze(0),
#             deterministic=True
#         )
#         target = float(positions[act.item()])

#         # Apply volatility targeting if enabled
#         if ecfg.vol_target > 0:
#             rv = float(np.std(asset.ret[max(0, t - 20):t]) * np.sqrt(252)) + 1e-6
#             target *= float(np.clip(ecfg.vol_target / rv, 0.0, 1.5))

#         # Calculate costs and returns
#         turn = abs(target - pos)
#         c = cost * turn + hold * abs(target)
#         gross = target * float(asset.ret[t])
#         net = gross - c
#         eq *= np.exp(net)
#         peak = max(peak, eq)

#         # Record state
#         records.append({
#             "date": asset.dates[t],
#             "close": float(asset.close[t]),
#             "position": target,
#             "action": int(act.item()),
#             "value_estimate": float(value.item()),
#             "asset_ret": float(asset.ret[t]),
#             "net_ret": net,
#             "equity": eq,
#             "turnover": turn,
#             "cost": c,
#             "drawdown": eq / peak - 1.0,
#         })

#         pos = target

#     df = pd.DataFrame(records).set_index("date")
#     df["buy_hold_equity"] = np.exp(df["asset_ret"].cumsum())
#     return df


# def create_frame(
#     axes_dict: Dict,
#     data_dict: Dict[str, pd.DataFrame],
#     tickers: List[str],
#     current_idx: int,
#     positions_list: List[float],
# ) -> None:
#     """Draw a single frame showing all companies."""
#     # Clear all axes
#     for ax in axes_dict.values():
#         ax.clear()

#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

#     # --- Top: Combined Equity Curves ---
#     ax_equity = axes_dict['equity']
#     for i, ticker in enumerate(tickers):
#         df = data_dict[ticker]
#         if current_idx < len(df):
#             df_window = df.iloc[:current_idx+1]
#             ax_equity.plot(range(len(df_window)), df_window["equity"],
#                           label=f"{ticker} (Agent)", linewidth=1.8,
#                           color=colors[i], alpha=0.9)
#             ax_equity.plot(range(len(df_window)), df_window["buy_hold_equity"],
#                           linewidth=1.2, linestyle='--',
#                           color=colors[i], alpha=0.5)

#     ax_equity.axhline(1.0, color='black', linewidth=0.6, alpha=0.3)
#     ax_equity.set_ylabel("Equity", fontsize=11, fontweight='bold')
#     ax_equity.set_title("Multi-Stock Trading Agent - Portfolio View",
#                        fontsize=13, fontweight='bold')
#     ax_equity.legend(loc='upper left', fontsize=7, ncol=2)
#     ax_equity.grid(alpha=0.3)

#     # --- Middle: Position Indicators for Each Company ---
#     ax_positions = axes_dict['positions']
#     y_positions = np.arange(len(tickers))

#     for i, ticker in enumerate(tickers):
#         df = data_dict[ticker]
#         if current_idx < len(df):
#             pos = df.iloc[current_idx]["position"]
#             color = colors[i]

#             # Draw position bar
#             if pos >= 0:
#                 ax_positions.barh(y_positions[i], pos, height=0.6,
#                                  left=0, color=color, alpha=0.7)
#             else:
#                 ax_positions.barh(y_positions[i], -pos, height=0.6,
#                                  left=pos, color=color, alpha=0.7)

#             # Position text
#             pos_text = "LONG" if pos > 0.1 else "SHORT" if pos < -0.1 else "FLAT"
#             ax_positions.text(-1.05, y_positions[i], ticker,
#                             ha='right', va='center', fontsize=9, fontweight='bold')
#             ax_positions.text(0, y_positions[i], f"{pos_text}\n{pos:.2f}",
#                             ha='center', va='center', fontsize=8,
#                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

#     ax_positions.axvline(0, color='black', linewidth=1)
#     ax_positions.set_xlim(-1.3, 1.1)
#     ax_positions.set_ylim(-0.5, len(tickers) - 0.5)
#     ax_positions.set_yticks([])
#     ax_positions.set_xlabel("Position", fontsize=10, fontweight='bold')
#     ax_positions.grid(axis='x', alpha=0.3)

#     # --- Bottom: Stats Table ---
#     ax_stats = axes_dict['stats']
#     ax_stats.axis('off')

#     # Build stats table
#     table_data = []
#     headers = ["Ticker", "Equity", "B&H", "Sharpe", "Return%", "MaxDD%", "Action"]

#     action_names = ["FULL SHORT", "HALF SHORT", "FLAT", "HALF LONG", "FULL LONG"]

#     for ticker in tickers:
#         df = data_dict[ticker]
#         if current_idx < len(df):
#             df_so_far = df.iloc[:current_idx+1]
#             current = df.iloc[current_idx]

#             metrics = performance_metrics(
#                 df_so_far["net_ret"].to_numpy(),
#                 df_so_far["position"].to_numpy()
#             )

#             action_idx = int(current["action"])
#             action_label = action_names[action_idx] if action_idx < len(action_names) else str(action_idx)

#             table_data.append([
#                 ticker,
#                 f"{current['equity']:.3f}",
#                 f"{current['buy_hold_equity']:.3f}",
#                 f"{metrics['sharpe']:.2f}",
#                 f"{metrics['ann_return']*100:.1f}",
#                 f"{metrics['max_drawdown']*100:.1f}",
#                 action_label,
#             ])

#     # Create table
#     table = ax_stats.table(cellText=table_data, colLabels=headers,
#                           cellLoc='center', loc='center',
#                           bbox=[0, 0, 1, 1])
#     table.auto_set_font_size(False)
#     table.set_fontsize(9)
#     table.scale(1, 2)

#     # Style header
#     for i in range(len(headers)):
#         cell = table[(0, i)]
#         cell.set_facecolor('#4CAF50')
#         cell.set_text_props(weight='bold', color='white')

#     # Color rows by performance
#     for i, ticker in enumerate(tickers):
#         df = data_dict[ticker]
#         if current_idx < len(df):
#             current = df.iloc[current_idx]
#             outperform = current['equity'] > current['buy_hold_equity']
#             color = '#E8F5E9' if outperform else '#FFEBEE'

#             for j in range(len(headers)):
#                 cell = table[(i+1, j)]
#                 cell.set_facecolor(color)

#     # --- Beside Stats Table: Portfolio and Progress Info ---
#     ax_info = axes_dict['info']
#     ax_info.axis('off')

#     # Get date from first ticker
#     first_ticker = tickers[0]
#     df_first = data_dict[first_ticker]
#     if current_idx < len(df_first):
#         current_date = df_first.index[current_idx].date()
#         start_date = df_first.index[0].date()
#         end_date = df_first.index[-1].date()
#         progress = (current_idx + 1) / len(df_first) * 100

#         # Calculate portfolio stats (equal weight)
#         all_rets = []
#         for ticker in tickers:
#             df = data_dict[ticker]
#             if current_idx < len(df):
#                 all_rets.append(df.iloc[:current_idx+1]["net_ret"].to_numpy())

#         if all_rets:
#             # Align lengths
#             min_len = min(len(r) for r in all_rets)
#             all_rets = [r[:min_len] for r in all_rets]
#             portfolio_rets = np.mean(all_rets, axis=0)
#             portfolio_metrics = performance_metrics(portfolio_rets)

#             info_text = f"""PORTFOLIO (Equal Weight)
# Sharpe: {portfolio_metrics['sharpe']:.2f}
# Ann. Return: {portfolio_metrics['ann_return']*100:.1f}%
# Max DD: {portfolio_metrics['max_drawdown']*100:.1f}%

# PROGRESS
# Date: {current_date}
# Period: {start_date} - {end_date}
# Day {current_idx + 1} / {len(df_first)}
# {progress:.1f}% complete"""

#             ax_info.text(0.02, 0.98, info_text.strip(),
#                         transform=ax_info.transAxes,
#                         fontsize=9, verticalalignment='top',
#                         family='monospace',
#                         bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.5))


# def create_video(
#     data_dict: Dict[str, pd.DataFrame],
#     tickers: List[str],
#     output_path: str,
#     positions_list: List[float],
#     fps: int = 15,
#     speed: int = 2,
# ) -> None:
#     """Create combined video from trading data."""
#     # Find max length
#     max_len = max(len(df) for df in data_dict.values())
#     print(f"Creating combined video with {max_len} frames at {fps} fps...")

#     # Setup figure with layout
#     fig = plt.figure(figsize=(16, 12))
#     gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 1.8],
#                          width_ratios=[3.5, 1.4], hspace=0.35, wspace=0.15)

#     axes_dict = {
#         'equity': fig.add_subplot(gs[0, :]),
#         'positions': fig.add_subplot(gs[1, :]),
#         'stats': fig.add_subplot(gs[2, 0]),
#         'info': fig.add_subplot(gs[2, 1]),
#     }

#     # Video writer setup
#     if output_path.endswith('.webm'):
#         fourcc = cv2.VideoWriter_fourcc(*'VP09')
#     elif output_path.endswith('.mp4'):
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     else:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')

#     height, width = 1200, 1600
#     fig.set_size_inches(width/100, height/100)

#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     try:
#         for i in range(0, max_len, speed):
#             if i % 50 == 0:
#                 print(f"  Processing frame {i}/{max_len}...")

#             create_frame(axes_dict, data_dict, tickers, i, positions_list)

#             # Convert to video frame
#             fig.canvas.draw()
#             img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#             img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
#             img = img[:, :, :3]  # Drop alpha
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             img = cv2.resize(img, (width, height))

#             out.write(img)

#         # Hold last frame
#         for _ in range(fps * 2):
#             out.write(img)

#     finally:
#         out.release()
#         plt.close(fig)

#     print(f"Video saved to: {output_path}")


# def main():
#     parser = argparse.ArgumentParser(description="Create combined trading video")
#     parser.add_argument("--ckpt", default="runs2/ppo/best.pt")
#     parser.add_argument("--dataset", default="data2/nse.pkl")
#     parser.add_argument("--tickers", default=None,
#                        help="Comma-separated tickers")
#     parser.add_argument("--n_tickers", type=int, default=5)
#     parser.add_argument("--split", default="test", choices=["train", "val", "test"])
#     parser.add_argument("--out", default="combined_trading_2.webm")
#     parser.add_argument("--fps", type=int, default=15)
#     parser.add_argument("--speed", type=int, default=2)
#     args = parser.parse_args()

#     # Load model and data
#     device = resolve_device("auto")
#     print(f"Loading model from {args.ckpt}...")
#     model, cfg = load_model(args.ckpt, device)

#     print(f"Loading dataset from {args.dataset}...")
#     md = load_dataset(args.dataset)

#     # Select tickers from the dataset split; default is the first N, not random.
#     available = md.tickers(args.split)
#     if not available:
#         raise ValueError(f"No tickers in {args.split} split")

#     if args.tickers:
#         tickers = [t.strip().upper() for t in args.tickers.split(",")]
#         tickers = [t for t in tickers if t in available]
#     else:
#         tickers = available[:args.n_tickers]

#     if not tickers:
#         raise ValueError("No valid tickers")

#     print(f"\nProcessing {len(tickers)} tickers: {', '.join(tickers)}")

#     # Collect data for all tickers
#     data_dict = {}
#     for ticker in tickers:
#         print(f"  Backtesting {ticker}...", end=' ')
#         df = collect_trading_data(model, md, ticker, args.split, cfg.env, device)
#         data_dict[ticker] = df
#         print(f"✓ ({len(df)} days, equity: {df['equity'].iloc[-1]:.3f})")

#     # Create video
#     print()
#     create_video(data_dict, tickers, args.out, cfg.env.positions, args.fps, args.speed)

#     # Summary
#     print("\nSUMMARY:")
#     for ticker in tickers:
#         df = data_dict[ticker]
#         final_eq = df['equity'].iloc[-1]
#         final_bh = df['buy_hold_equity'].iloc[-1]
#         symbol = "✓" if final_eq > final_bh else "✗"
#         print(f"  {symbol} {ticker}: {final_eq:.3f} vs B&H {final_bh:.3f}")

#     duration = max(len(df) for df in data_dict.values()) / args.speed / args.fps
#     print(f"\nVideo duration: ~{duration:.1f} seconds")


# if __name__ == "__main__":
#     main()


##################################################################################################################


















###########################################################################################################################


"""Create a single video showing 5 companies trading simultaneously.

Shows all 5 stocks in one video with their equity curves, positions, and stats.

Usage:
    python3 create_video_combined.py --ckpt runs2/ppo/best.pt --n_tickers 5 --split test --out combined_trading.webm
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch

from config import Config, EnvConfig, resolve_device
from dataloader import MarketData, load_dataset
from env import N_STATE_SCALARS
from loss import performance_metrics
from model import ActorCritic


def load_model(ckpt_path: str, device: str):
    """Load trained model from checkpoint."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: Config = ck["config"]
    model = ActorCritic(
        window=ck["window"],
        n_features=ck["n_features"],
        n_actions=ck["n_actions"],
        n_scalars=N_STATE_SCALARS,
        cfg=cfg.model
    ).to(device)
    model.load_state_dict(ck["model"])
    return model, cfg


@torch.no_grad()
def collect_trading_data(
    model: ActorCritic,
    md: MarketData,
    ticker: str,
    split: str,
    ecfg: EnvConfig,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run backtest and collect all trading decisions."""
    model.eval()
    asset = md.assets[ticker]
    W, F = md.window, md.n_features
    positions = np.asarray(ecfg.positions, dtype=np.float32)
    cost = (ecfg.cost_bps + ecfg.slippage_bps) / 1e4
    hold = ecfg.holding_cost_bps / 1e4

    lo, hi = asset.ranges[split]
    pos = 0.0
    eq = 1.0
    peak = 1.0
    records = []

    for k in range(hi - lo):
        t = lo + k

        # Build observation
        obs = np.empty(W * F + N_STATE_SCALARS, dtype=np.float32)
        obs[:W * F] = asset.features[t - W + 1 : t + 1].ravel()
        obs[-3] = pos
        obs[-2] = eq / peak - 1.0
        obs[-1] = k / max(1, hi - lo)

        # Get action from model
        act, _, _, value = model.act(
            torch.as_tensor(obs, device=device).unsqueeze(0),
            deterministic=True
        )
        target = float(positions[act.item()])

        # Apply volatility targeting if enabled
        if ecfg.vol_target > 0:
            rv = float(np.std(asset.ret[max(0, t - 20):t]) * np.sqrt(252)) + 1e-6
            target *= float(np.clip(ecfg.vol_target / rv, 0.0, 1.5))

        # Calculate costs and returns
        turn = abs(target - pos)
        c = cost * turn + hold * abs(target)
        gross = target * float(asset.ret[t])
        net = gross - c
        eq *= np.exp(net)
        peak = max(peak, eq)

        # Record state
        records.append({
            "date": asset.dates[t],
            "close": float(asset.close[t]),
            "position": target,
            "action": int(act.item()),
            "value_estimate": float(value.item()),
            "asset_ret": float(asset.ret[t]),
            "net_ret": net,
            "equity": eq,
            "turnover": turn,
            "cost": c,
            "drawdown": eq / peak - 1.0,
        })

        pos = target

    df = pd.DataFrame(records).set_index("date")
    df["buy_hold_equity"] = np.exp(df["asset_ret"].cumsum())
    return df


def create_frame(
    axes_dict: Dict,
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    current_idx: int,
    positions_list: List[float],
) -> None:
    """Draw a single frame showing all companies."""
    # Clear all axes
    for ax in axes_dict.values():
        ax.clear()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # --- Top: Combined Equity Curves ---
    ax_equity = axes_dict['equity']
    for i, ticker in enumerate(tickers):
        df = data_dict[ticker]
        if current_idx < len(df):
            df_window = df.iloc[:current_idx+1]
            ax_equity.plot(range(len(df_window)), df_window["equity"],
                          label=f"{ticker} (Agent)", linewidth=1.8,
                          color=colors[i], alpha=0.9)
            ax_equity.plot(range(len(df_window)), df_window["buy_hold_equity"],
                          linewidth=1.2, linestyle='--',
                          color=colors[i], alpha=0.5)

    ax_equity.axhline(1.0, color='black', linewidth=0.6, alpha=0.3)
    ax_equity.set_ylabel("Equity", fontsize=11, fontweight='bold')
    ax_equity.set_title("Multi-Stock Trading Agent - Portfolio View",
                       fontsize=13, fontweight='bold')
    ax_equity.legend(loc='upper left', fontsize=7, ncol=2)
    ax_equity.grid(alpha=0.3)

    # --- Middle: Position Indicators for Each Company ---
    ax_positions = axes_dict['positions']
    y_positions = np.arange(len(tickers))

    for i, ticker in enumerate(tickers):
        df = data_dict[ticker]
        if current_idx < len(df):
            pos = df.iloc[current_idx]["position"]
            color = colors[i]

            # Draw position bar
            if pos >= 0:
                ax_positions.barh(y_positions[i], pos, height=0.6,
                                 left=0, color=color, alpha=0.7)
            else:
                ax_positions.barh(y_positions[i], -pos, height=0.6,
                                 left=pos, color=color, alpha=0.7)

            # Position text
            pos_text = "LONG" if pos > 0.1 else "SHORT" if pos < -0.1 else "FLAT"
            ax_positions.text(-1.05, y_positions[i], ticker,
                            ha='right', va='center', fontsize=9, fontweight='bold')
            ax_positions.text(0, y_positions[i], f"{pos_text}\n{pos:.2f}",
                            ha='center', va='center', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax_positions.axvline(0, color='black', linewidth=1)
    ax_positions.set_xlim(-1.3, 1.1)
    ax_positions.set_ylim(-0.5, len(tickers) - 0.5)
    ax_positions.set_yticks([])
    ax_positions.set_xlabel("Position", fontsize=10, fontweight='bold')
    ax_positions.grid(axis='x', alpha=0.3)

    # --- Bottom: Stats Table ---
    ax_stats = axes_dict['stats']
    ax_stats.axis('off')

    # Build stats table
    table_data = []
    table_tickers = []
    headers = ["Ticker", "Equity", "B&H", "Sharpe", "Return%", "MaxDD%", "Action"]

    action_names = ["FULL SHORT", "HALF SHORT", "FLAT", "HALF LONG", "FULL LONG"]

    for ticker in tickers:
        df = data_dict[ticker]
        if current_idx < len(df):
            df_so_far = df.iloc[:current_idx+1]
            current = df.iloc[current_idx]

            metrics = performance_metrics(
                df_so_far["net_ret"].to_numpy(),
                df_so_far["position"].to_numpy()
            )

            action_idx = int(current["action"])
            action_label = action_names[action_idx] if action_idx < len(action_names) else str(action_idx)

            table_tickers.append(ticker)
            table_data.append([
                ticker,
                f"{current['equity']:.3f}",
                f"{current['buy_hold_equity']:.3f}",
                f"{metrics['sharpe']:.2f}",
                f"{metrics['ann_return']*100:.1f}",
                f"{metrics['max_drawdown']*100:.1f}",
                action_label,
            ])

    # Create table
    table = ax_stats.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Color rows by performance
    for i, ticker in enumerate(table_tickers):
        df = data_dict[ticker]
        if current_idx < len(df):
            current = df.iloc[current_idx]
            outperform = current['equity'] > current['buy_hold_equity']
            color = '#E8F5E9' if outperform else '#FFEBEE'

            for j in range(len(headers)):
                cell = table[(i+1, j)]
                cell.set_facecolor(color)

    # --- Beside Stats Table: Portfolio and Progress Info ---
    ax_info = axes_dict['info']
    ax_info.axis('off')

    # Get date from first ticker
    first_ticker = tickers[0]
    df_first = data_dict[first_ticker]
    if current_idx < len(df_first):
        current_date = df_first.index[current_idx].date()
        start_date = df_first.index[0].date()
        end_date = df_first.index[-1].date()
        progress = (current_idx + 1) / len(df_first) * 100

        # Calculate portfolio stats (equal weight)
        all_rets = []
        for ticker in tickers:
            df = data_dict[ticker]
            if current_idx < len(df):
                all_rets.append(df.iloc[:current_idx+1]["net_ret"].to_numpy())

        if all_rets:
            # Align lengths
            min_len = min(len(r) for r in all_rets)
            all_rets = [r[:min_len] for r in all_rets]
            portfolio_rets = np.mean(all_rets, axis=0)
            portfolio_metrics = performance_metrics(portfolio_rets)

            info_text = f"""PORTFOLIO (Equal Weight)
Sharpe: {portfolio_metrics['sharpe']:.2f}
Ann. Return: {portfolio_metrics['ann_return']*100:.1f}%
Max DD: {portfolio_metrics['max_drawdown']*100:.1f}%

PROGRESS
Date: {current_date}
Period: {start_date} - {end_date}
Day {current_idx + 1} / {len(df_first)}
{progress:.1f}% complete"""

            ax_info.text(0.02, 0.98, info_text.strip(),
                        transform=ax_info.transAxes,
                        fontsize=9, verticalalignment='top',
                        family='monospace',
                        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.5))


def create_video(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    output_path: str,
    positions_list: List[float],
    fps: int = 15,
    speed: int = 2,
) -> None:
    """Create combined video from trading data."""
    if fps <= 0 or speed <= 0:
        raise ValueError("fps and speed must be positive")
    if not data_dict or any(df.empty for df in data_dict.values()):
        raise ValueError("Trading data must contain non-empty dataframes")

    # Find max length
    max_len = max(len(df) for df in data_dict.values())
    print(f"Creating combined video with {max_len} frames at {fps} fps...")

    # Setup figure with layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 1.8],
                         width_ratios=[3.5, 1.4], hspace=0.35, wspace=0.15)

    axes_dict = {
        'equity': fig.add_subplot(gs[0, :]),
        'positions': fig.add_subplot(gs[1, :]),
        'stats': fig.add_subplot(gs[2, 0]),
        'info': fig.add_subplot(gs[2, 1]),
    }

    # Video writer setup
    if output_path.endswith('.webm'):
        fourcc = cv2.VideoWriter_fourcc(*'VP09')
    elif output_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

    height, width = 1200, 1600
    fig.set_size_inches(width/100, height/100)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        frame_indices = list(range(0, max_len, speed))
        if frame_indices[-1] != max_len - 1:
            frame_indices.append(max_len - 1)
        for i in frame_indices:
            if i % 50 == 0:
                print(f"  Processing frame {i}/{max_len}...")

            create_frame(axes_dict, data_dict, tickers, i, positions_list)

            # Convert to video frame
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, :3]  # Drop alpha
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (width, height))

            out.write(img)

        # Save the final rendered frame as a PNG thumbnail.
        thumbnail_path = os.path.splitext(output_path)[0] + "_thumbnail.png"
        if not cv2.imwrite(thumbnail_path, img):
            raise OSError(f"Could not save thumbnail: {thumbnail_path}")
        print(f"Thumbnail saved to: {thumbnail_path}")

        # Hold last frame
        for _ in range(fps * 2):
            out.write(img)

    finally:
        out.release()
        plt.close(fig)

    print(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create combined trading video")
    parser.add_argument("--ckpt", default="runs2/ppo/best.pt")
    parser.add_argument("--dataset", default="data2/nse.pkl")
    parser.add_argument("--tickers", default=None,
                       help="Comma-separated tickers")
    parser.add_argument("--n_tickers", type=int, default=5)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="combined_trading_3.webm")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--speed", type=int, default=2)
    args = parser.parse_args()

    # Load model and data
    device = resolve_device("auto")
    print(f"Loading model from {args.ckpt}...")
    model, cfg = load_model(args.ckpt, device)

    print(f"Loading dataset from {args.dataset}...")
    md = load_dataset(args.dataset)

    # Select tickers from the dataset split; default is the first N, not random.
    available = md.tickers(args.split)
    if not available:
        raise ValueError(f"No tickers in {args.split} split")

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        tickers = [t for t in tickers if t in available]
    else:
        tickers = available[:args.n_tickers]

    if not tickers:
        raise ValueError("No valid tickers")

    print(f"\nProcessing {len(tickers)} tickers: {', '.join(tickers)}")

    # Collect data for all tickers
    data_dict = {}
    for ticker in tickers:
        print(f"  Backtesting {ticker}...", end=' ')
        df = collect_trading_data(model, md, ticker, args.split, cfg.env, device)
        data_dict[ticker] = df
        print(f"✓ ({len(df)} days, equity: {df['equity'].iloc[-1]:.3f})")

    # Create video
    print()
    create_video(data_dict, tickers, args.out, cfg.env.positions, args.fps, args.speed)

    # Summary
    print("\nSUMMARY:")
    for ticker in tickers:
        df = data_dict[ticker]
        final_eq = df['equity'].iloc[-1]
        final_bh = df['buy_hold_equity'].iloc[-1]
        symbol = "✓" if final_eq > final_bh else "✗"
        print(f"  {symbol} {ticker}: {final_eq:.3f} vs B&H {final_bh:.3f}")

    duration = max(len(df) for df in data_dict.values()) / args.speed / args.fps
    print(f"\nVideo duration: ~{duration:.1f} seconds")


if __name__ == "__main__":
    main()


# # python3 create_video_combined.py --tickers "BIRLACORPN,BOSCHLTD,IOB,LEXUS,PAYTM"