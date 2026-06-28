"""
Technical indicators, computed by hand from price history.

Pure functions: price data in, numbers out. No LLM, no look-ahead (each
indicator at time t uses only data up to t). These feed the analyst agent
a factual snapshot it then interprets in words.
"""

from __future__ import annotations

import pandas as pd


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average: the average close over the last `window` days.
    A smoothed view of price; rising SMA = uptrend."""
    return prices.rolling(window).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (0-100): momentum oscillator.

    Compares the size of recent gains to recent losses. Above ~70 is often
    called 'overbought' (risen fast, may pull back); below ~30 'oversold'.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, pd.Series]:
    """Moving Average Convergence Divergence: trend + momentum.

    MACD line = fast EMA minus slow EMA. Signal line = EMA of the MACD line.
    MACD above its signal line suggests strengthening upward momentum.
    (EMA = Exponential Moving Average, which weights recent prices more.)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}


def volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Annualized rolling volatility of daily returns — the 'bumpiness'."""
    returns = prices.pct_change()
    return returns.rolling(window).std() * (365 ** 0.5)


def snapshot(prices: pd.Series) -> dict[str, float]:
    """Compute a one-moment summary of all indicators at the LATEST available
    date in `prices`. This is the factual packet handed to the analyst agent.

    Returns plain floats (the most recent value of each indicator), plus a
    little context like where the latest price sits vs its moving averages.
    """
    latest_close = float(prices.iloc[-1])
    sma20 = sma(prices, 20).iloc[-1]
    sma50 = sma(prices, 50).iloc[-1]
    rsi14 = rsi(prices, 14).iloc[-1]
    macd_vals = macd(prices)
    vol20 = volatility(prices, 20).iloc[-1]

    # 30-day price change for trend context
    pct_change_30d = (
        float(prices.iloc[-1] / prices.iloc[-31] - 1) if len(prices) > 31 else float("nan")
    )

    return {
        "latest_close": latest_close,
        "sma_20": float(sma20),
        "sma_50": float(sma50),
        "price_vs_sma20_pct": float((latest_close / sma20 - 1) * 100),
        "price_vs_sma50_pct": float((latest_close / sma50 - 1) * 100),
        "rsi_14": float(rsi14),
        "macd": float(macd_vals["macd"].iloc[-1]),
        "macd_signal": float(macd_vals["signal"].iloc[-1]),
        "macd_histogram": float(macd_vals["histogram"].iloc[-1]),
        "annualized_volatility": float(vol20),
        "pct_change_30d": pct_change_30d * 100 if pct_change_30d == pct_change_30d else float("nan"),
    }
