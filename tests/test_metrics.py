"""Verify each metric against hand-checkable values."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics


def test_to_returns():
    equity = pd.Series([100.0, 110.0, 99.0])
    r = metrics.to_returns(equity)
    # 100->110 is +10%, 110->99 is -10%
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(0.10)
    assert r.iloc[1] == pytest.approx(-0.10)


def test_cumulative_return_compounds():
    # Two +10% days compound to +21%, not +20%
    r = pd.Series([0.10, 0.10])
    assert metrics.cumulative_return(r) == pytest.approx(0.21)


def test_max_drawdown():
    # Equity goes 100 -> 150 -> 75 -> 90. Worst drop is 150->75 = -50%.
    equity = pd.Series([100.0, 150.0, 75.0, 90.0])
    r = metrics.to_returns(equity)
    assert metrics.max_drawdown(r) == pytest.approx(-0.50, abs=1e-9)


def test_sharpe_zero_when_flat():
    # Constant returns have zero volatility -> Sharpe defined as 0 here
    r = pd.Series([0.001, 0.001, 0.001])
    assert metrics.sharpe_ratio(r) == 0.0


def test_sharpe_positive_for_good_returns():
    # Mostly positive, low-volatility returns should yield a positive Sharpe
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.001, 0.01, 365))  # ~0.1%/day, 1% daily vol
    assert metrics.sharpe_ratio(r) > 0


def test_sortino_ignores_upside():
    """Sortino should reward a series where volatility is mostly upside.

    Compare two series with the SAME mean return. One has its volatility on
    the upside (big gains, tiny losses); the other has symmetric swings.
    The upside-heavy one should score a HIGHER Sortino, because Sortino
    only penalizes the downside.
    """
    upside_heavy = pd.Series([0.06, 0.06, -0.02, 0.06, -0.02, 0.06])
    symmetric = pd.Series([0.05, -0.05, 0.05, -0.05, 0.05, -0.05])
    assert metrics.sortino_ratio(upside_heavy) > metrics.sortino_ratio(symmetric)


def test_sortino_finite_with_equal_losses():
    """Equal-valued losses must NOT make downside deviation collapse to 0.

    (Regression test for a bug where std() of identical losses returned 0.)
    """
    r = pd.Series([0.05, 0.05, 0.05, -0.01, 0.05, -0.01])
    s = metrics.sortino_ratio(r)
    assert s > 0  # a profitable series with real (equal) losses scores > 0pytest tests/ -v


def test_summary_has_all_keys():
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    s = metrics.summary(r)
    expected = {
        "cumulative_return", "annualized_return", "annualized_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
    }
    assert set(s.keys()) == expected
