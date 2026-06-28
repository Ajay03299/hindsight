"""
Momentum Analyst Agent — focuses on TREND STRUCTURE and recent price behavior
across multiple horizons, rather than oscillators. A deliberately different
lens from the technical analyst, so the two can legitimately disagree.
Sees only data before `as_of`.
"""

from __future__ import annotations

from src.data.feed import PointInTimeFeed
from src.agents.base import AnalystReport, run_analyst

ROLE = "momentum_analyst"

SYSTEM_PROMPT = """You are a momentum-focused analyst at a trading firm.
You judge an asset by the strength and consistency of its price trend across
multiple time horizons (short, medium, long). Strong, consistent positive
momentum is bullish; deteriorating or negative momentum is bearish. You care
about the direction and persistence of the trend, not oscillator levels."""


def _momentum_snapshot(prices) -> dict:
    """Multi-horizon returns and a simple trend-consistency check."""
    def pct(n):
        return float(prices.iloc[-1] / prices.iloc[-1 - n] - 1) * 100 if len(prices) > n else float("nan")
    ret_7 = pct(7)
    ret_30 = pct(30)
    ret_90 = pct(90)
    # Trend consistency: is each longer horizon's direction aligned?
    aligned = (ret_7 > 0) == (ret_30 > 0) == (ret_90 > 0)
    return {
        "return_7d": ret_7,
        "return_30d": ret_30,
        "return_90d": ret_90,
        "trend_aligned": aligned,
        "latest_close": float(prices.iloc[-1]),
    }


def _user_prompt(symbol: str, as_of: str, snap: dict) -> str:
    return (
        f"Asset: {symbol}\nAs of: {as_of}\n\n"
        f"Momentum snapshot:\n"
        f"- 7-day return:  {snap['return_7d']:+.2f}%\n"
        f"- 30-day return: {snap['return_30d']:+.2f}%\n"
        f"- 90-day return: {snap['return_90d']:+.2f}%\n"
        f"- Trend aligned across horizons: {snap['trend_aligned']}\n\n"
        f"Judge the strength and persistence of the trend. Give JSON."
    )


def analyze(symbol: str, as_of: str, feed: PointInTimeFeed | None = None) -> AnalystReport:
    if feed is None:
        feed = PointInTimeFeed(symbol)
    history = feed.history_before(as_of)
    if len(history) < 95:
        return AnalystReport(ROLE, symbol, as_of, "neutral", 0.0,
                             "Insufficient history.", {})
    snap = _momentum_snapshot(history["close"])
    return run_analyst(ROLE, symbol, as_of, SYSTEM_PROMPT,
                       _user_prompt(symbol, as_of, snap), snap)


if __name__ == "__main__":
    r = analyze("BTC/USDT", "2023-06-01")
    print(f"[{r.role}] {r.symbol} @ {r.as_of}: {r.stance} ({r.confidence:.2f})")
    print(f"  {r.reasoning}")
