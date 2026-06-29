"""
Full backtest: agent vs. baselines, net of costs, with a comparison table.

Runs the multi-agent system across a multi-year window for each symbol, then
scores it against buy-and-hold and SMA on the SAME window, all net of costs.
Produces a results table and an equity-curve chart.

The first run is slow (many uncached LLM calls); subsequent runs hit the cache
and are near-instant. Progress prints as it goes.

Usage:  python -m src.backtest.run_agent_comparison
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.feed import PointInTimeFeed
from src.backtest import baselines
from src.backtest.agent_backtest import run_agent_backtest
from src.evaluation import metrics

# Two coins for the first full run to keep it tractable; add more later.
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
START = "2023-01-01"
END = "2025-06-01"
REBALANCE_DAYS = 7


def _window_prices(symbol: str) -> pd.Series:
    feed = PointInTimeFeed(symbol)
    closes = feed._df["close"]
    mask = (closes.index >= pd.Timestamp(START)) & (closes.index <= pd.Timestamp(END))
    return closes[mask]


def main() -> None:
    rows = []
    curves = {}

    for sym in SYMBOLS:
        print(f"\n=== {sym} ===")
        price = _window_prices(sym)

        # Buy-and-hold baseline on the same window
        bh = baselines.buy_and_hold(price)
        rows.append((f"BuyHold {sym}", metrics.summary(metrics.to_returns(bh))))
        curves[f"BuyHold {sym}"] = bh

        # SMA (net) baseline on the same window
        sma = baselines.sma_crossover(price)
        rows.append((f"SMA-net {sym}", metrics.summary(metrics.to_returns(sma["net"]))))

        # The agent (this is the slow part on first run)
        print(f"Running agent backtest {START}..{END} (weekly)...")
        result = run_agent_backtest(sym, START, END, rebalance_days=REBALANCE_DAYS, verbose=False)
        agent_eq = result.equity_curve
        rows.append((f"Agent {sym}", metrics.summary(metrics.to_returns(agent_eq))))
        curves[f"Agent {sym}"] = agent_eq
        print(f"  agent decisions: {len(result.decisions)}")

    # ---- Table ----
    df = pd.DataFrame({label: stats for label, stats in rows}).T
    df = df[["cumulative_return", "annualized_return", "annualized_volatility",
             "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio"]]
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\n" + "=" * 90)
    print(f"AGENT vs BASELINES  ({START} to {END}, weekly rebalance, net of costs)")
    print("=" * 90)
    print(df.to_string())
    print("=" * 90)

    # ---- Chart ----
    Path("notebooks").mkdir(exist_ok=True)
    plt.figure(figsize=(12, 6))
    for label, curve in curves.items():
        style = "--" if label.startswith("Agent") else "-"
        plt.plot(curve.index, curve.values, style, label=label)
    plt.yscale("log")
    plt.title(f"Agent vs Buy-and-Hold ({START} to {END}, net of costs)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (log)")
    plt.legend()
    plt.tight_layout()
    out = Path("notebooks") / "agent_vs_baselines.png"
    plt.savefig(out, dpi=120)
    print(f"\nChart saved -> {out}")

    # Save the table as CSV for the writeup
    df.to_csv(Path("notebooks") / "agent_vs_baselines.csv")
    print(f"Table saved -> notebooks/agent_vs_baselines.csv")


if __name__ == "__main__":
    main()
