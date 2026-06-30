"""
Minimal agent — the ablation baseline.

Strips the architecture down to a SINGLE analyst whose stance is mapped
directly to a position. No researchers, no bull/bear debate, no trader
synthesis. One LLM call per decision instead of ~six.

Comparing this to the full multi-agent system answers: does the debate
machinery actually add value, or is it expensive theatre?

stance -> position mapping:
  bullish -> 0.5 + 0.5*confidence   (more confident bull = more invested)
  bearish -> 0.5 - 0.5*confidence
  neutral -> 0.5
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.feed import PointInTimeFeed
from src.agents import technical_analyst
from src.backtest.costs import DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS


@dataclass
class MinimalResult:
    symbol: str
    equity_curve: pd.Series
    positions: pd.Series
    decisions: list = field(default_factory=list)


def _stance_to_position(stance: str, confidence: float) -> float:
    if stance == "bullish":
        return min(1.0, 0.5 + 0.5 * confidence)
    if stance == "bearish":
        return max(0.0, 0.5 - 0.5 * confidence)
    return 0.5


def run_minimal_backtest(symbol, start, end, rebalance_days=7,
                         fee_bps=DEFAULT_FEE_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS,
                         verbose=False):
    feed = PointInTimeFeed(symbol)
    all_dates = feed.dates
    window = all_dates[(all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))]
    rebalance_dates = set(window[::rebalance_days])

    closes = feed._df.loc[window, "close"]
    daily_returns = closes.pct_change().fillna(0.0)

    position_by_day = pd.Series(0.0, index=window)
    decisions = []
    current = 0.0
    for date in window:
        if date in rebalance_dates:
            as_of = date.strftime("%Y-%m-%d")
            report = technical_analyst.analyze(symbol, as_of, feed)
            current = _stance_to_position(report.stance, report.confidence)
            decisions.append((as_of, report.stance, current))
            if verbose:
                print(f"  {as_of}: {report.stance} -> {current:.2f}")
        position_by_day.loc[date] = current

    acted = position_by_day.shift(1).fillna(0.0)
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    trades = acted.diff().abs().fillna(acted.abs())
    net_returns = acted * daily_returns - trades * cost_rate
    return MinimalResult(symbol, (1 + net_returns).cumprod(), acted, decisions)
