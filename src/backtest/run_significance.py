"""
Apply formal significance tests to the LLM agent's performance.

For each symbol: computes the empirical p-value of the agent's Sharpe against
the random-agent null, and the Deflated Sharpe Ratio treating the random runs
as the pool of 'trials' whose Sharpe variance we deflate by.

Usage:  python -m src.backtest.run_significance
"""

from __future__ import annotations

from src.backtest.agent_backtest import run_agent_backtest
from src.backtest.random_agent import random_distribution
from src.evaluation import metrics
from src.evaluation.significance import (
    empirical_p_value, probabilistic_sharpe_ratio, deflated_sharpe_ratio,
)

from src.config import SYMBOLS, START, END, N_RUNS


def main():
    for sym in SYMBOLS:
        print(f"\n{'='*70}\n{sym}  ({START} to {END})\n{'='*70}")

        agent_eq = run_agent_backtest(sym, START, END, verbose=False).equity_curve
        agent_returns = metrics.to_returns(agent_eq)
        agent_sharpe = metrics.sharpe_ratio(agent_returns)

        rand = random_distribution(sym, START, END, n_runs=N_RUNS)["sharpe_ratio"]

        # 1. Empirical p-value vs random null
        p = empirical_p_value(agent_sharpe, rand)

        # 2. Probabilistic Sharpe (prob true Sharpe > 0, non-normality corrected)
        psr = probabilistic_sharpe_ratio(agent_returns, benchmark_sr=0.0)

        # 3. Deflated Sharpe: treat the spread of random Sharpes as the trial variance
        dsr = deflated_sharpe_ratio(
            agent_returns, n_trials=N_RUNS, variance_of_trial_sharpes=float(rand.var())
        )

        print(f"Agent annualized Sharpe:            {agent_sharpe:.3f}")
        print(f"Empirical p-value (vs random null): {p:.3f}"
              f"   ({'significant' if p < 0.05 else 'NOT significant'} at 0.05)")
        print(f"Probabilistic Sharpe (P[SR>0]):     {psr:.3f}")
        print(f"Expected max Sharpe under null:     "
              f"{dsr['expected_max_sharpe_under_null']:.3f}")
        print(f"Deflated Sharpe Ratio (P[SR>max]):  {dsr['deflated_sharpe_ratio']:.3f}")
        print(f"  Note: DSR benchmarks vs zero/luck-max; the empirical p-value above")
        print(f"  is the stricter test because random agents ALSO capture market beta.")
        if p >= 0.05:
            print(f"  Verdict: no edge over a beta-matched random agent (p={p:.3f}).")

if __name__ == "__main__":
    main()
