"""
Null-hypothesis test: is the LLM agent distinguishable from random chance?

For each symbol, builds the random-agent Sharpe distribution and reports the
percentile at which the LLM agent's Sharpe falls. Saves a histogram per symbol.

Usage:  python -m src.backtest.run_null_test
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.backtest.agent_backtest import run_agent_backtest
from src.backtest.random_agent import random_distribution, percentile_of
from src.evaluation import metrics

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
START, END = "2023-01-01", "2025-06-01"
N_RUNS = 200


def main() -> None:
    Path("notebooks").mkdir(exist_ok=True)

    for sym in SYMBOLS:
        # LLM agent Sharpe (cached from earlier runs -> fast)
        agent_eq = run_agent_backtest(sym, START, END, verbose=False).equity_curve
        agent_sharpe = metrics.sharpe_ratio(metrics.to_returns(agent_eq))

        # Random distribution
        dist = random_distribution(sym, START, END, n_runs=N_RUNS)
        rand_sharpe = dist["sharpe_ratio"]
        pct = percentile_of(agent_sharpe, rand_sharpe)

        print(f"\n{'='*70}\n{sym}\n{'='*70}")
        print(f"LLM agent Sharpe:        {agent_sharpe:.3f}")
        print(f"Random Sharpe (median):  {rand_sharpe.median():.3f}")
        print(f"Random Sharpe (95th):    {rand_sharpe.quantile(0.95):.3f}")
        print(f"Random Sharpe (max):     {rand_sharpe.max():.3f}")
        print(f"--> LLM agent beats {pct:.1f}% of random agents")
        if pct >= 95:
            print("    Interpretation: strong evidence of signal (top 5%).")
        elif pct >= 80:
            print("    Interpretation: suggestive, but not conclusive.")
        else:
            print("    Interpretation: NOT clearly distinguishable from chance.")

        # Histogram
        plt.figure(figsize=(10, 5))
        plt.hist(rand_sharpe, bins=30, alpha=0.7, label=f"Random agents (n={N_RUNS})")
        plt.axvline(agent_sharpe, color="red", linestyle="--", linewidth=2,
                    label=f"LLM agent ({agent_sharpe:.2f})")
        plt.axvline(rand_sharpe.median(), color="gray", linestyle=":",
                    label=f"Random median ({rand_sharpe.median():.2f})")
        plt.title(f"LLM Agent vs Random-Agent Sharpe Distribution — {sym}")
        plt.xlabel("Sharpe ratio")
        plt.ylabel("Number of random runs")
        plt.legend()
        plt.tight_layout()
        safe = sym.replace("/", "_")
        out = Path("notebooks") / f"null_test_{safe}.png"
        plt.savefig(out, dpi=120)
        print(f"    Histogram saved -> {out}")


if __name__ == "__main__":
    main()
