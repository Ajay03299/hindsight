"""
Randomized agent — the null-hypothesis baseline.

A 'monkey' that ignores all analysis and picks a target position uniformly at
random each rebalance date. Running it across many random seeds builds a
DISTRIBUTION of outcomes achievable by pure luck. We then ask whether the real
LLM agent's performance sits out in the tail of that distribution (genuine
signal) or in the middle of it (indistinguishable from chance).

This makes no LLM calls, so hundreds of runs take only seconds.

It is matched to the LLM agent's setup: same window, same weekly rebalance,
same cost model, and positions drawn from the SAME range the LLM agent uses
(roughly 0.0-1.0). This fairness matters — we want the monkey to play the same
game, just without thinking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.feed import PointInTimeFeed
from src.backtest.costs import DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS
from src.evaluation import metrics


def run_random_backtest(
    symbol: str,
    start: str,
    end: str,
    seed: int,
    rebalance_days: int = 7,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.Series:
    """One random-agent run. Returns the net equity curve."""
    feed = PointInTimeFeed(symbol)
    all_dates = feed.dates
    window = all_dates[(all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))]
    rebalance_dates = set(window[::rebalance_days])

    rng = np.random.default_rng(seed)
    closes = feed._df.loc[window, "close"]
    daily_returns = closes.pct_change().fillna(0.0)

    position_by_day = pd.Series(0.0, index=window)
    current = 0.0
    for date in window:
        if date in rebalance_dates:
            current = float(rng.uniform(0.0, 1.0))  # random position each rebalance
        position_by_day.loc[date] = current

    acted = position_by_day.shift(1).fillna(0.0)
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    trades = acted.diff().abs().fillna(acted.abs())
    costs = trades * cost_rate
    net_returns = acted * daily_returns - costs
    return (1 + net_returns).cumprod()


def random_distribution(
    symbol: str, start: str, end: str, n_runs: int = 200, rebalance_days: int = 7
) -> pd.DataFrame:
    """Run the random agent n_runs times; return a DataFrame of metrics per run."""
    rows = []
    for seed in range(n_runs):
        eq = run_random_backtest(symbol, start, end, seed, rebalance_days)
        rows.append(metrics.summary(metrics.to_returns(eq)))
    return pd.DataFrame(rows)


def percentile_of(value: float, distribution: pd.Series) -> float:
    """What percentile does `value` fall at within `distribution`? (0-100)

    95+ means the value beats 95% of random runs — strong evidence of signal.
    ~50 means it's average for a monkey — no evidence of skill.
    """
    return float((distribution < value).mean() * 100)


if __name__ == "__main__":
    # Quick demo on BTC
    SYM, START, END = "BTC/USDT", "2023-01-01", "2025-06-01"
    dist = random_distribution(SYM, START, END, n_runs=200)
    print(f"\nRandom agent distribution over 200 runs — {SYM} {START}..{END}")
    print(f"  Sharpe: mean {dist['sharpe_ratio'].mean():.3f}, "
          f"median {dist['sharpe_ratio'].median():.3f}, "
          f"95th pct {dist['sharpe_ratio'].quantile(0.95):.3f}, "
          f"max {dist['sharpe_ratio'].max():.3f}")
