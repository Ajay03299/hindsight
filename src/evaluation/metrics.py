"""
Performance metrics for evaluating trading strategies.

Everything is computed from a *returns series*: the daily fractional change
in portfolio value (0.02 == +2% that day). Each function takes that series
and answers one question about risk or reward.

Conventions:
- Daily data, so we annualize with 365 (crypto trades every day, unlike
  stocks which use ~252 trading days).
- `risk_free_rate` is an ANNUAL rate (e.g. 0.04 for 4%); we convert it to
  daily internally where needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Crypto markets trade 365 days a year (no weekends/holidays)
PERIODS_PER_YEAR = 365


def to_returns(equity: pd.Series) -> pd.Series:
    """Convert an equity curve (portfolio value over time) to daily returns.

    Example: [100, 102, 101] -> [NaN, 0.02, -0.0098]
    The first value is NaN (no prior day to compare) and is dropped.
    """
    return equity.pct_change().dropna()


def cumulative_return(returns: pd.Series) -> float:
    """Total return over the whole period, as a fraction.

    We compound the daily returns: (1+r1)(1+r2)...(1+rn) - 1.
    Example: two +10% days -> 1.1 * 1.1 - 1 = 0.21 (21%, not 20%).
    """
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series) -> float:
    """Cumulative return scaled to a per-year rate (geometric)."""
    total = (1 + returns).prod()
    years = len(returns) / PERIODS_PER_YEAR
    if years == 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """Standard deviation of returns, scaled to a yearly figure.

    Volatility scales with the square root of time, hence sqrt(365).
    """
    return float(returns.std() * np.sqrt(PERIODS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Risk-adjusted return: excess return per unit of total volatility.

    The headline metric. High = good return without a wild ride.
    Reality check: ~1 good, ~2 very good, ~3 excellent and rare.
    """
    daily_rf = risk_free_rate / PERIODS_PER_YEAR
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(PERIODS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe, but only penalizes DOWNSIDE volatility.

    Upswings aren't 'risk' to an investor, so the denominator is the
    'downside deviation': the root-mean-square of returns that fall below
    the target (here, the risk-free rate), measured FROM that target.

    Note this is measured from the target, NOT from the downside mean, and
    the sum is divided by the count of ALL periods, not just the negative
    ones. This is the standard definition and (unlike using std of the
    losses) it doesn't collapse to zero when the losses happen to be equal.
    """
    daily_rf = risk_free_rate / PERIODS_PER_YEAR
    excess = returns - daily_rf
    # Only the shortfalls below target contribute; everything else is 0
    downside = excess.clip(upper=0.0)
    downside_deviation = np.sqrt((downside ** 2).mean())
    if downside_deviation == 0:
        return 0.0
    return float(excess.mean() / downside_deviation * np.sqrt(PERIODS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline, as a NEGATIVE fraction.

    We rebuild the equity curve, track the running peak, and find the
    largest drop below that peak. Example: -0.33 means a 33% fall from a
    high before recovering.
    """
    equity = (1 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Annualized return divided by the worst drawdown.

    'How much do I earn relative to the most pain I'd endure?'
    """
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns) / mdd)


def summary(returns: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Compute all metrics at once, returned as a labeled dict.

    This is what the backtester will call to score any strategy.
    """
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
    }
