"""
Statistical significance tools: empirical p-values and the Deflated Sharpe Ratio.

These convert "beats X% of random" into the formal language quants expect, and
correct the raw Sharpe for the two things that most inflate it: non-normal
(fat-tailed, skewed) returns, and multiple-testing selection bias.

References:
  Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", J. Portfolio Mgmt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PERIODS_PER_YEAR = 365


def empirical_p_value(observed: float, null_distribution: pd.Series) -> float:
    """One-sided p-value: fraction of null samples >= the observed value.

    p small (e.g. < 0.05) => observed is unlikely under the null (real signal).
    p large (e.g. 0.235)  => observed is common under the null (no evidence).
    """
    n = len(null_distribution)
    # +1 in numerator and denominator is the standard finite-sample correction
    at_or_above = (null_distribution >= observed).sum()
    return float((at_or_above + 1) / (n + 1))


def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sr: float = 0.0
) -> float:
    """Probability that the TRUE (annualized) Sharpe exceeds `benchmark_sr`,
    correcting for sample length, skewness, and kurtosis of the returns.

    Returns a probability in [0, 1]. This is the building block of the DSR.
    """
    r = returns.dropna()
    n = len(r)
    if n < 3 or r.std() == 0:
        return 0.0

    # Non-annualized (per-period) Sharpe, and the benchmark on the same scale
    sr = r.mean() / r.std()
    sr_bench = benchmark_sr / np.sqrt(PERIODS_PER_YEAR)

    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # non-excess (normal = 3)

    # Standard error of the Sharpe estimate under non-normality
    se = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    if se == 0:
        return 0.0
    return float(stats.norm.cdf((sr - sr_bench) / se))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, variance_of_trial_sharpes: float
) -> dict:
    """Deflated Sharpe Ratio: probability the true Sharpe > 0 AFTER accounting
    for having selected the best of `n_trials` strategies.

    The more strategies you tried, the higher the Sharpe you'd expect from luck
    alone — DSR deflates by that expected maximum.

    Returns a dict with the expected-max benchmark and the resulting DSR.
    """
    if n_trials < 1:
        n_trials = 1
    # Expected maximum Sharpe from n_trials independent random strategies
    # (Bailey & Lopez de Prado's approximation using the Gaussian expected max)
    euler_mascheroni = 0.5772156649
    e_max_z = (
        (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / n_trials)
        + euler_mascheroni * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    # Expected max annualized Sharpe under the null
    sr_star_annual = np.sqrt(variance_of_trial_sharpes) * e_max_z

    dsr = probabilistic_sharpe_ratio(returns, benchmark_sr=sr_star_annual)
    return {
        "n_trials": n_trials,
        "expected_max_sharpe_under_null": float(sr_star_annual),
        "deflated_sharpe_ratio": dsr,
    }
