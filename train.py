"""PPO training loop.

    python train.py --dataset data/dataset.pkl --total_steps 400000

Model selection is done on the *validation* period by walk-forward Sharpe, never
on training reward. Training reward will keep improving long after the policy has
stopped generalizing -- that gap is the whole story in this problem.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from collections import deque
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from config import Config, resolve_device
from dataloader import build_dataset, load_dataset, save_dataset
from env import N_STATE_SCALARS, VecTradingEnv
from inference import portfolio_curve, run_backtest, summarize
from loss import compute_gae, performance_metrics, ppo_loss
from model import ActorCritic


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(model: ActorCritic, md, cfg: Config, device: str, split: str = "val") -> Dict[str, float]:
    bt = run_backtest(model, md, split, cfg.env, device, deterministic=True)
    curve = portfolio_curve(bt)
    r = np.log(curve["strategy"]).diff().dropna().to_numpy()
    m = performance_metrics(r)
    bh = performance_metrics(np.log(curve["buy_hold"]).diff().dropna().to_numpy())
    expo = float(np.mean([df["position"].abs().mean() for df in bt.values()]))
    model.train()
    return {"sharpe": m["sharpe"], "ann_return": m["ann_return"],
            "max_dd": m["max_drawdown"], "bh_sharpe": bh["sharpe"],
            "exposure": expo}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data2/nse.pkl")
    p.add_argument("--source", default=None, help="build dataset on the fly instead")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--n_envs", type=int, default=None)
    p.add_argument("--encoder", default=None, choices=["gru", "cnn"])
    p.add_argument("--cost_bps", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ent_coef", type=float, default=None)
    p.add_argument("--rollout_len", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=None)
    a = p.parse_args()

    cfg = Config()
    for k, attr in [("total_steps", ("ppo", "total_steps")), ("n_envs", ("ppo", "n_envs")),
                    ("encoder", ("model", "encoder")), ("cost_bps", ("env", "cost_bps")),
                    ("gamma", ("ppo", "gamma")), ("lr", ("ppo", "lr")),
                    ("ent_coef", ("ppo", "ent_coef")),
                    ("rollout_len", ("ppo", "rollout_len")),
                    ("seed", (None, "seed")), ("eval_every", (None, "eval_every")),
                    ("out_dir", (None, "out_dir"))]:
        v = getattr(a, k)
        if v is not None:
            grp, name = attr
            setattr(getattr(cfg, grp) if grp else cfg, name, v)
    if a.source:
        cfg.data.source = a.source

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))

    # ----------------------------------------------------------------- data
    if a.source or not os.path.exists(a.dataset):
        md = build_dataset(cfg.data, seed=cfg.seed)
        save_dataset(md, a.dataset)
    else:
        md = load_dataset(a.dataset)
        cfg.data.window = md.window
    print(md.describe())

    envs = VecTradingEnv(md, cfg.env, split="train", n_envs=cfg.ppo.n_envs, seed=cfg.seed)
    print(f"device={device}  obs_dim={envs.obs_dim}  n_actions={envs.n_actions}")

    model = ActorCritic(window=md.window, n_features=md.n_features,
                        n_actions=envs.n_actions, n_scalars=N_STATE_SCALARS,
                        cfg=cfg.model).to(device)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"params={n_par:,}")
    opt = optim.Adam(model.parameters(), lr=cfg.ppo.lr, eps=1e-5)

    # ------------------------------------------------------------- buffers
    P = cfg.ppo
    T, N = P.rollout_len, P.n_envs
    obs_buf = torch.zeros((T, N, envs.obs_dim), device=device)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)

    batch_size = T * N
    mb_size = batch_size // P.minibatches
    n_iters = max(1, P.total_steps // batch_size)

    obs = torch.as_tensor(envs.reset(), device=device)
    ep_returns: deque = deque(maxlen=200)
    ep_dd: deque = deque(maxlen=200)
    best_val = -np.inf
    global_step = 0
    t_start = time.time()

    log_path = os.path.join(cfg.out_dir, "train_log.csv")
    log_f = open(log_path, "w", newline="")
    logger = csv.DictWriter(log_f, fieldnames=[
        "iter", "step", "ep_logret", "ep_maxdd", "policy_loss", "value_loss",
        "entropy", "approx_kl", "clip_frac", "lr", "val_sharpe", "val_ann_return",
        "val_max_dd", "val_exposure", "sps"])
    logger.writeheader()

    # ------------------------------------------------------------ PPO loop
    for it in range(1, n_iters + 1):
        frac = 1.0 - (it - 1) / n_iters
        if P.anneal_lr:
            for g in opt.param_groups:
                g["lr"] = P.lr * frac
        ent_coef = P.ent_coef_final + (P.ent_coef - P.ent_coef_final) * frac

        # --- rollout
        for t in range(T):
            obs_buf[t] = obs
            with torch.no_grad():
                action, logp, _, value = model.act(obs)
            act_buf[t], logp_buf[t], val_buf[t] = action, logp, value

            nobs, rew, done, infos = envs.step(action.cpu().numpy())
            rew_buf[t] = torch.as_tensor(rew, device=device)
            done_buf[t] = torch.as_tensor(done.astype(np.float32), device=device)
            obs = torch.as_tensor(nobs, device=device)
            global_step += N
            for info in infos:
                if "episode" in info:
                    ep_returns.append(info["episode"]["log_return"])
                    ep_dd.append(info["episode"]["max_dd"])

        with torch.no_grad():
            _, last_value = model(obs)
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_value,
                                   P.gamma, P.gae_lambda)

        # --- update
        b_obs = obs_buf.reshape(-1, envs.obs_dim)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)

        idx = np.arange(batch_size)
        stats = {}
        for _ in range(P.epochs):
            np.random.shuffle(idx)
            for s in range(0, batch_size, mb_size):
                mb = idx[s : s + mb_size]
                new_logp, ent, new_val = model.evaluate(b_obs[mb], b_act[mb])
                out = ppo_loss(new_logp, ent, new_val, b_logp[mb], b_val[mb],
                               b_adv[mb], b_ret[mb], clip_coef=P.clip_coef,
                               vf_coef=P.vf_coef, ent_coef=ent_coef,
                               clip_vloss=P.clip_vloss, norm_adv=P.norm_adv)
                opt.zero_grad(set_to_none=True)
                out.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), P.max_grad_norm)
                opt.step()
                stats = {"policy_loss": out.policy, "value_loss": out.value,
                         "entropy": out.entropy, "approx_kl": out.approx_kl,
                         "clip_frac": out.clip_frac}
            if P.target_kl and stats.get("approx_kl", 0.0) > P.target_kl:
                break

        # --- log / validate
        row = {
            "iter": it, "step": global_step,
            "ep_logret": float(np.mean(ep_returns)) if ep_returns else float("nan"),
            "ep_maxdd": float(np.mean(ep_dd)) if ep_dd else float("nan"),
            "lr": opt.param_groups[0]["lr"],
            "sps": int(global_step / (time.time() - t_start)),
            **stats,
        }
        if it % cfg.eval_every == 0 or it == n_iters:
            v = validate(model, md, cfg, device, "val")
            row.update({f"val_{k}" if not k.startswith("val") else k: val
                        for k, val in {"val_sharpe": v["sharpe"],
                                       "val_ann_return": v["ann_return"],
                                       "val_max_dd": v["max_dd"],
                                       "val_exposure": v["exposure"]}.items()})
            print(f"it {it:4d}/{n_iters}  step {global_step:>8,}  "
                  f"ep_logret {row['ep_logret']:+.3f}  ent {stats['entropy']:.3f}  "
                  f"kl {stats['approx_kl']:.4f}  |  VAL sharpe {v['sharpe']:+.2f} "
                  f"(b&h {v['bh_sharpe']:+.2f})  ret {v['ann_return']:+.1%}  "
                  f"dd {v['max_dd']:.1%}  expo {v['exposure']:.2f}")
            if v["sharpe"] > best_val:
                best_val = v["sharpe"]
                torch.save({"model": model.state_dict(), "config": cfg,
                            "window": md.window, "n_features": md.n_features,
                            "n_actions": envs.n_actions, "iter": it,
                            "val_sharpe": best_val},
                           os.path.join(cfg.out_dir, "best.pt"))
        elif it % max(1, cfg.eval_every // 2) == 0:
            print(f"it {it:4d}/{n_iters}  step {global_step:>8,}  "
                  f"ep_logret {row['ep_logret']:+.3f}  ent {stats['entropy']:.3f}  "
                  f"kl {stats['approx_kl']:.4f}  sps {row['sps']}")
        logger.writerow(row)
        log_f.flush()

    torch.save({"model": model.state_dict(), "config": cfg, "window": md.window,
                "n_features": md.n_features, "n_actions": envs.n_actions,
                "iter": n_iters, "val_sharpe": best_val},
               os.path.join(cfg.out_dir, "last.pt"))
    log_f.close()
    print(f"\nbest val sharpe = {best_val:+.3f}   ckpt -> {cfg.out_dir}/best.pt")

    # final held-out evaluation, printed once
    ck = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    bt = run_backtest(model, md, "test", cfg.env, device, deterministic=True)
    print("\n=== TEST (best-on-val checkpoint) ===")
    print(summarize(bt).round(3).to_string())


if __name__ == "__main__":
    main()
