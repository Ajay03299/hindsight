"""
Run all baseline strategies on the downloaded data and print a results table.

Usage:  python src/backtest/run_baselines.py
Saves an equity-curve chart to notebooks/baselines_equity.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.feed import PointInTimeFeed, DATA_DIR
from src.backtest import baselines
from src.evaluation import metrics

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]


def load_prices() -> dict[str, pd.Series]:
    """Load close-price series for every symbol that has data on disk."""
    prices = {}
    for sym in SYMBOLS:
        try:
            feed = PointInTimeFeed(sym)
            prices[sym] = feed._df["close"]
        except FileNotFoundError:
            print(f"  (skipping {sym}: no data — run fetch.py)")
    return prices


def main() -> None:
    prices = load_prices()
    if not prices:
        print("No data found. Run `python src/data/fetch.py` first.")
        return

    rows = []
    curves = {}  # label -> equity curve, for plotting

    # Per-coin buy-and-hold and SMA crossover
    for sym, price in prices.items():
        bh = baselines.buy_and_hold(price)
        rows.append(("BuyHold " + sym, metrics.summary(metrics.to_returns(bh))))
        curves["BuyHold " + sym] = bh

        sma = baselines.sma_crossover(price)
        rows.append(("SMA " + sym, metrics.summary(metrics.to_returns(sma))))
        # (curves for SMA omitted from the plot to keep it readable)

    # Equal-weight basket across all coins
    ew = baselines.equal_weight(prices)
    rows.append(("EqualWeight basket", metrics.summary(metrics.to_returns(ew))))
    curves["EqualWeight basket"] = ew

    # ---- Print results table ----
    df = pd.DataFrame(
        {label: stats for label, stats in rows}
    ).T
    # Order and format columns nicely
    df = df[
        ["cumulative_return", "annualized_return", "annualized_volatility",
         "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio"]
    ]
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    print("\nReminder: these are GROSS of transaction costs (we add costs next).")

    # ---- Save equity-curve chart ----
    Path("notebooks").mkdir(exist_ok=True)
    plt.figure(figsize=(12, 6))
    for label, curve in curves.items():
        plt.plot(curve.index, curve.values, label=label)
    plt.yscale("log")  # log scale: equal % moves look equal regardless of level
    plt.title("Baseline Equity Curves (start = 1.0, log scale)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (log)")
    plt.legend()
    plt.tight_layout()
    out = Path("notebooks") / "baselines_equity.png"
    plt.savefig(out, dpi=120)
    print(f"\nChart saved -> {out}")


if __name__ == "__main__":
    main()
