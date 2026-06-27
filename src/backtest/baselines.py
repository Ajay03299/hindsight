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
    prices: pd.Series,
    short_window: int = 20,
    long_window: int = 50,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> dict[str, pd.Series]:
    """Moving-average crossover, returning BOTH gross and net equity curves.

    Position is 1 (invested) when short SMA > long SMA, else 0 (cash),
    lagged one day to avoid look-ahead. Net curve charges transaction costs
    on every change in position.

    Returns a dict: {"gross": curve, "net": curve}.
    """
    from src.backtest.costs import apply_costs

    short_sma = prices.rolling(short_window).mean()
    long_sma = prices.rolling(long_window).mean()
    signal = (short_sma > long_sma).astype(int)
    position = signal.shift(1).fillna(0)

    asset_returns = prices.pct_change().fillna(0.0)

    gross = (1 + position * asset_returns).cumprod()
    net = apply_costs(position, asset_returns, fee_bps, slippage_bps)
    return {"gross": gross, "net": net}
