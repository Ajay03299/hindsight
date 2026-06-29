"""
Alpha-decay / leakage test — the project's closing experiment.

Compares the LLM agent's edge over random in an EARLY window (likely seen
during model training) vs. a RECENT window (likely unseen, post-cutoff). If
the edge decays from early to recent, that is consistent with the agent's
apparent performance coming from MEMORIZATION rather than reasoning.

We measure 'edge' as the percentile of the random-agent Sharpe distribution
that the LLM agent beats, computed separately IN EACH WINDOW. Measuring
relative to random within each window controls for the two periods being
different market regimes.

NOTE: model training cutoffs are not published precisely, so 'seen' vs
'unseen' is a reasoned hypothesis, not a certainty. We report accordingly.

Usage:  python -m src.backtest.run_alpha_decay
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
    "early (likely seen)": ("2023-01-01", "2024-06-30"),
    "recent (likely unseen)": ("2025-09-01", "2026-06-27"),
}


def evaluate_window(symbol: str, start: str, end: str) -> dict:
    """Agent Sharpe, random distribution, and the agent's percentile within it."""
    agent_eq = run_agent_backtest(symbol, start, end, verbose=False).equity_curve
    agent_sharpe = metrics.sharpe_ratio(metrics.to_returns(agent_eq))
    dist = random_distribution(symbol, start, end, n_runs=N_RUNS)["sharpe_ratio"]
    pct = percentile_of(agent_sharpe, dist)
    return {
        "agent_sharpe": agent_sharpe,
        "random_median": float(dist.median()),
        "percentile": pct,
    }


def main() -> None:
    results = {}
    for sym in SYMBOLS:
        print(f"\n{'='*70}\n{sym}\n{'='*70}")
        results[sym] = {}
        for label, (start, end) in WINDOWS.items():
            print(f"  evaluating {label}: {start}..{end}")
            r = evaluate_window(sym, start, end)
            results[sym][label] = r
            print(f"    agent Sharpe {r['agent_sharpe']:.3f}, "
                  f"random median {r['random_median']:.3f} "
                  f"-> beats {r['percentile']:.1f}% of random")

    # ---- Summary verdict ----
    print(f"\n{'='*70}\nALPHA-DECAY SUMMARY (edge over random, by window)\n{'='*70}")
    for sym in SYMBOLS:
        early = results[sym]["early (likely seen)"]["percentile"]
        recent = results[sym]["recent (likely unseen)"]["percentile"]
        decay = early - recent
        print(f"{sym}: early {early:.1f}%  ->  recent {recent:.1f}%   "
              f"(change: {decay:+.1f} pts)")
        if decay > 20:
            print("    Edge DECAYED on likely-unseen data — consistent with memorization.")
        elif abs(decay) <= 20:
            print("    Edge roughly STABLE across windows — no clear decay signal.")
        else:
            print("    Edge INCREASED on recent data — inconsistent with memorization.")

    # ---- Chart: percentile by window, per symbol ----
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
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentile of random agents beaten")
    ax.set_title("Alpha-Decay Test: does the agent's edge survive on unseen data?")
    ax.legend()
    plt.tight_layout()
    out = Path("notebooks") / "alpha_decay.png"
    plt.savefig(out, dpi=120)
    print(f"\nChart saved -> {out}")


if __name__ == "__main__":
    main()
