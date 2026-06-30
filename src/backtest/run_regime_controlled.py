"""
Regime-controlled leakage test — the clean version of the alpha-decay test.

Compares the agent's edge over random in TWO BEAR markets:
  - 2022 bear  (likely seen during model training)
  - recent bear (2025-09 onward, likely unseen / post-cutoff)

Holding the regime constant (both down markets) removes the bull-vs-bear
confound from the original alpha-decay test, so a difference in edge-over-random
points more cleanly at a memorization (look-ahead) effect.

Usage:  python -m src.backtest.run_regime_controlled
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.backtest.agent_backtest import run_agent_backtest
from src.backtest.random_agent import random_distribution, percentile_of
from src.evaluation import metrics

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
N_RUNS = 200

WINDOWS = {
    "2022 bear (likely seen)": ("2022-01-01", "2022-12-31"),
    "recent bear (likely unseen)": ("2025-09-01", "2026-06-27"),
}


def evaluate(symbol, start, end):
    agent_eq = run_agent_backtest(symbol, start, end, verbose=False).equity_curve
    agent_sharpe = metrics.sharpe_ratio(metrics.to_returns(agent_eq))
    dist = random_distribution(symbol, start, end, n_runs=N_RUNS)["sharpe_ratio"]
    return {
        "agent_sharpe": agent_sharpe,
        "random_median": float(dist.median()),
        "percentile": percentile_of(agent_sharpe, dist),
    }


def main():
    results = {}
    for sym in SYMBOLS:
        print(f"\n{'='*70}\n{sym}\n{'='*70}")
        results[sym] = {}
        for label, (start, end) in WINDOWS.items():
            print(f"  {label}: {start}..{end}")
            r = evaluate(sym, start, end)
            results[sym][label] = r
            print(f"    agent Sharpe {r['agent_sharpe']:+.3f}, "
                  f"random median {r['random_median']:+.3f} "
                  f"-> beats {r['percentile']:.1f}% of random")

    print(f"\n{'='*70}\nREGIME-CONTROLLED VERDICT (both windows are bear markets)\n{'='*70}")
    for sym in SYMBOLS:
        seen = results[sym]["2022 bear (likely seen)"]["percentile"]
        unseen = results[sym]["recent bear (likely unseen)"]["percentile"]
        print(f"{sym}: 2022 bear {seen:.1f}%  ->  recent bear {unseen:.1f}%   "
              f"(change: {seen - unseen:+.1f} pts)")
        if seen - unseen > 20:
            print("    Edge dropped on unseen bear data — now WITHOUT the regime")
            print("    confound, this is cleaner evidence consistent with memorization.")
        elif abs(seen - unseen) <= 20:
            print("    Edge similar across both bears — no clear memorization signal;")
            print("    the agent simply shows little edge in either.")
        else:
            print("    Edge higher on recent data — inconsistent with memorization.")

    Path("notebooks").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(WINDOWS.keys())
    x = range(len(labels))
    width = 0.35
    for i, sym in enumerate(SYMBOLS):
        vals = [results[sym][lab]["percentile"] for lab in labels]
        ax.bar([xi + i * width for xi in x], vals, width, label=sym)
    ax.axhline(50, color="gray", linestyle=":", label="random chance (50%)")
    ax.axhline(95, color="red", linestyle="--", alpha=0.6, label="signal threshold (95%)")
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Percentile of random agents beaten")
    ax.set_title("Regime-Controlled Leakage Test (both windows = bear markets)")
    ax.legend()
    plt.tight_layout()
    out = Path("notebooks") / "regime_controlled.png"
    plt.savefig(out, dpi=120)
    print(f"\nChart saved -> {out}")


if __name__ == "__main__":
    main()
