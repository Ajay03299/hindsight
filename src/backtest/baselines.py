"""
Baseline strategies — the honest competition every fancy model must beat.

Each function takes a price DataFrame (or dict of them) and returns an
equity curve: a pandas Series of portfolio value over time, starting at 1.0.
Feed that curve through evaluation.metrics to score it.

A note on look-ahead: the moving-average strategy decides today's position
using indicators computed from data up to YESTERDAY (via .shift(1)). You
cannot act on an average that includes a close you don't yet know.
"""

from __future__ import annotations

import pandas as pd


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """Buy on day one, hold. Equity curve = price rescaled to start at 1.0."""
    returns = prices.pct_change().fillna(0.0)
    return (1 + returns).cumprod()


def equal_weight(price_dict: dict[str, pd.Series]) -> pd.Series:
    """Split capital evenly across all assets on day one, hold.

    The basket return each day is the average of the assets' daily returns.
    """
    # Align all price series on common dates, then take daily returns
    aligned = pd.DataFrame(price_dict).dropna()
    asset_returns = aligned.pct_change().fillna(0.0)
    basket_returns = asset_returns.mean(axis=1)  # equal weight = simple average
    return (1 + basket_returns).cumprod()


def sma_crossover(
    prices: pd.Series, short_window: int = 20, long_window: int = 50
) -> pd.Series:
    """Moving-average crossover trend-following strategy.

    Position is 1 (invested) when the short SMA is above the long SMA,
    else 0 (in cash). The signal is LAGGED by one day so we only act on
    information available before the day we trade.
    """
    short_sma = prices.rolling(short_window).mean()
    long_sma = prices.rolling(long_window).mean()

    # Signal: 1 when short above long, else 0
    signal = (short_sma > long_sma).astype(int)
    # Act on YESTERDAY's signal (no look-ahead): shift forward by one day
    position = signal.shift(1).fillna(0)

    asset_returns = prices.pct_change().fillna(0.0)
    # We earn the asset's return only on days we held a position
    strategy_returns = position * asset_returns
    return (1 + strategy_returns).cumprod()
