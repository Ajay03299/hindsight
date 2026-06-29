"""
Event-driven backtester for the multi-agent system.

Walks forward through time, one rebalance date at a time. On each rebalance
date the agent system produces a target position (0.0-1.0) using ONLY data
before that date. The position is held until the next rebalance. Daily
portfolio returns are computed from the held position and the asset's actual
returns, net of transaction costs on every position change.

This loop is the no-look-ahead guarantee in action: the agent is only ever
called with `as_of = rebalance_date`, and the feed slices off everything from
that date onward.

Rebalancing weekly (not daily) keeps the LLM call count tractable while
remaining a legitimate systematic-strategy design choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.feed import PointInTimeFeed
from src.agents import technical_analyst, momentum_analyst
from src.agents import debate
from src.backtest.costs import DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS


@dataclass
class BacktestResult:
    symbol: str
    equity_curve: pd.Series          # net portfolio value over time
    positions: pd.Series             # target position held each day
    decisions: list = field(default_factory=list)  # the TradeDecision log


def run_agent_backtest(
    symbol: str,
    start: str,
    end: str,
    rebalance_days: int = 7,
    n_rounds: int = 2,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    verbose: bool = True,
) -> BacktestResult:
    feed = PointInTimeFeed(symbol)
    all_dates = feed.dates
    # Restrict to the requested window
    window = all_dates[(all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))]
    if len(window) == 0:
        raise ValueError(f"No data in window {start}..{end} for {symbol}")

    # Rebalance dates: every `rebalance_days`-th date in the window
    rebalance_dates = window[::rebalance_days]

    # Daily price returns over the window (for settling held positions)
    closes = feed._df.loc[window, "close"]
    daily_returns = closes.pct_change().fillna(0.0)

    # Build a daily position series by deciding on each rebalance date and
    # holding until the next one.
    position_by_day = pd.Series(0.0, index=window)
    decisions = []
    current_position = 0.0

    for i, date in enumerate(window):
        if date in rebalance_dates:
            as_of = date.strftime("%Y-%m-%d")
            reports = [
                technical_analyst.analyze(symbol, as_of, feed),
                momentum_analyst.analyze(symbol, as_of, feed),
            ]
            decision = debate.decide(symbol, as_of, reports, n_rounds=n_rounds)
            current_position = decision.target_position
            decisions.append(decision)
            if verbose:
                print(f"  {as_of}: {decision.action} -> position {current_position:.2f}")
        position_by_day.loc[date] = current_position

    # The position we ACT on each day must be lagged by one day: we decide
    # using data before `date`, so we can only hold the new position starting
    # the NEXT day. This prevents same-day look-ahead.
    acted_position = position_by_day.shift(1).fillna(0.0)

    # Net returns: position * asset return, minus costs on position changes
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    trades = acted_position.diff().abs().fillna(acted_position.abs())
    costs = trades * cost_rate
    net_returns = acted_position * daily_returns - costs
    equity = (1 + net_returns).cumprod()

    return BacktestResult(symbol, equity, acted_position, decisions)


if __name__ == "__main__":
    # Small smoke test: 3 months of weekly rebalancing on BTC
    result = run_agent_backtest("BTC/USDT", "2023-06-01", "2023-09-01", rebalance_days=7)
    final = result.equity_curve.iloc[-1]
    print(f"\nFinal equity (start 1.0): {final:.3f}  ({(final-1)*100:+.1f}%)")
    print(f"Decisions made: {len(result.decisions)}")
