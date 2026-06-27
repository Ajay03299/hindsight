"""
Transaction cost model — the reality check that deflates fantasy backtests.

Every time you change position you pay:
  1. A fee  — the exchange's cut (a fixed fraction of the traded amount).
  2. Slippage — the price moving against you between deciding and filling.

We model both as a combined cost charged on the SIZE of each position
change. Going from 0% invested to 100% invested trades 100% of capital and
pays the full cost; nudging from 90% to 100% trades only 10% and pays a
tenth. This is intentionally simple and conservative — defensible, not fancy.

Defaults reflect a realistic retail crypto taker on a major exchange:
  fee  = 10 bps (0.10%)   slippage = 5 bps (0.05%)
A 'bp' (basis point) is 1/100th of a percent. 10 bps = 0.10% = 0.0010.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_FEE_BPS = 10.0       # 0.10% exchange fee per trade
DEFAULT_SLIPPAGE_BPS = 5.0   # 0.05% slippage per trade


def apply_costs(
    position: pd.Series,
    asset_returns: pd.Series,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.Series:
    """Turn a position series + asset returns into a NET equity curve.

    Parameters
    ----------
    position : fraction invested each day (0.0 to 1.0), already lagged so it
               represents what you actually held.
    asset_returns : the asset's daily returns.

    Returns
    -------
    Net equity curve (starts at 1.0) after deducting costs on every trade.
    """
    cost_rate = (fee_bps + slippage_bps) / 10_000.0  # bps -> fraction

    # How much the position changed each day = how much we traded
    trades = position.diff().abs().fillna(position.abs())
    costs = trades * cost_rate

    # Gross return from holding, minus the cost incurred that day
    net_returns = position * asset_returns - costs
    return (1 + net_returns).cumprod()
