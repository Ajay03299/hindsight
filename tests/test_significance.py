"""Sanity checks for significance tools."""

import numpy as np
import pandas as pd

from src.evaluation import significance


def test_empirical_p_value_bounds():
    null = pd.Series(np.arange(100))  # 0..99
    # A value at the top should have a small p; at the bottom, large p
    assert significance.empirical_p_value(200, null) < 0.05
    assert significance.empirical_p_value(-1, null) > 0.95


def test_psr_high_for_clearly_good_returns():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.005, 0.01, 365))  # strong, steady positive
    assert significance.probabilistic_sharpe_ratio(good) > 0.95


def test_dsr_returns_expected_keys():
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.02, 365))
    out = significance.deflated_sharpe_ratio(r, n_trials=100, variance_of_trial_sharpes=0.5)
    assert set(out) == {"n_trials", "expected_max_sharpe_under_null", "deflated_sharpe_ratio"}
    assert 0.0 <= out["deflated_sharpe_ratio"] <= 1.0
