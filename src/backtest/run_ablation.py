"""
Ablation: full multi-agent system vs. minimal single-analyst agent.

Does the debate machinery add measurable value? Runs both on the same window
and compares risk-adjusted performance, plus each against the random null.

Usage:  python -m src.backtest.run_ablation
"""

from __future__ import annotations

import pandas as pd

from src.backtest.agent_backtest import run_agent_backtest
from src.backtest.minimal_agent import run_minimal_backtest
from src.backtest.random_agent import random_distribution, percentile_of
from src.evaluation import metrics

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
START, END = "2023-01-01", "2025-06-01"


def main():
    rows = []
    for sym in SYMBOLS:
        print(f"\n=== {sym} ===")
        full_eq = run_agent_backtest(sym, START, END, verbose=False).equity_curve
        print("  running minimal agent...")
        min_eq = run_minimal_backtest(sym, START, END, verbose=False).equity_curve

        rand = random_distribution(sym, START, END, n_runs=200)["sharpe_ratio"]

        full_s = metrics.summary(metrics.to_returns(full_eq))
        min_s = metrics.summary(metrics.to_returns(min_eq))
        full_s["beats_random_%"] = percentile_of(full_s["sharpe_ratio"], rand)
        min_s["beats_random_%"] = percentile_of(min_s["sharpe_ratio"], rand)

        rows.append((f"Full (debate) {sym}", full_s))
        rows.append((f"Minimal (1 analyst) {sym}", min_s))

    df = pd.DataFrame({label: s for label, s in rows}).T
    df = df[["cumulative_return", "sharpe_ratio", "max_drawdown", "beats_random_%"]]
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\n" + "=" * 80)
    print(f"ABLATION: does the debate add value?  ({START} to {END}, net of costs)")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    df.to_csv("notebooks/ablation.csv")
    print("Saved -> notebooks/ablation.csv")


if __name__ == "__main__":
    main()
