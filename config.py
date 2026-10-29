"""Central configuration for the RL trading pipeline. 

Every module reads from this so that data preprocessing, training and inference
can never silently disagree about window sizes, feature sets or splits.
"""

from dataclasses import dataclass, field, asdict
from typing import List
import json


@dataclass
class DataConfig:
    # "yfinance" pulls live data; "synthetic" generates GBM+regime data offline
    # (useful for smoke tests / CI); "csv" reads <cache_dir>/<TICKER>.csv
    source: str = "yfinance"
    tickers: List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "KO", "CAT", "PG"]
    )
    start: str = "2005-01-01"
    end: str = "2024-12-31"
    cache_dir: str = "data"

    window: int = 64          # observation lookback in trading days
    train_frac: float = 0.70  # chronological split -- never random
    val_frac: float = 0.15
    # remaining 0.15 is test

    clip_sigma: float = 5.0   # winsorize standardized features


@dataclass
class EnvConfig:
    positions: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    episode_len: int = 252            # ~1 trading year per episode
    cost_bps: float = 5.0            # round-trip cost, basis points of notional traded
    slippage_bps: float = 2.0        # extra cost on every position change
    holding_cost_bps: float = 0.2    # daily financing/borrow charge on |position|
    reward_scale: float = 100.0      # keeps rewards O(1) for stable value learning
    vol_target: float = 0.0          # >0 scales position by vol_target/realized_vol
    dd_penalty: float = 0.0          # penalty coefficient on new drawdown


@dataclass
class ModelConfig:
    encoder: str = "gru"      # "gru" or "cnn"
    hidden: int = 128
    layers: int = 1
    mlp_hidden: int = 128
    dropout: float = 0.1


@dataclass
class PPOConfig:
    n_envs: int = 16
    rollout_len: int = 128
    total_steps: int = 400_000
    epochs: int = 4
    minibatches: int = 4
    lr: float = 5e-4
    anneal_lr: bool = True
    # NOTE: gamma is deliberately LOW. The action at time t affects the reward at
    # t and (via turnover cost) at t+1 -- almost nothing further. With gamma=0.99
    # the critic has to forecast ~100 days of market return, which is essentially
    # noise, and it drowns the learnable signal. Empirically the policy barely
    # moves at 0.99 and starts learning immediately at 0.9.
    gamma: float = 0.9
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float = 0.001    # linearly annealed
    max_grad_norm: float = 0.5
    target_kl: float = 0.03          # early-stop the epoch loop if exceeded
    norm_adv: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    seed: int = 0
    device: str = "auto"             # "auto" | "cpu" | "cuda"
    out_dir: str = "runs2/ppo"
    eval_every: int = 10             # validate every N PPO iterations

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            raw = json.load(f)
        return Config(
            data=DataConfig(**raw["data"]),
            env=EnvConfig(**raw["env"]),
            model=ModelConfig(**raw["model"]),
            ppo=PPOConfig(**raw["ppo"]),
            seed=raw["seed"],
            device=raw["device"],
            out_dir=raw["out_dir"],
            eval_every=raw["eval_every"],
        )


def resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
