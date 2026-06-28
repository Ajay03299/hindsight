"""
Technical Analyst Agent — reads momentum/oscillator indicators (RSI, MACD,
SMA relationships) and forms a stance. Sees only data before `as_of`.
"""

from __future__ import annotations

from src.data.feed import PointInTimeFeed
from src.agents import indicators
from src.agents.base import AnalystReport, run_analyst

ROLE = "technical_analyst"

SYSTEM_PROMPT = """You are a professional technical analyst at a trading firm.
You are given a factual snapshot of momentum and trend indicators for a crypto
asset. Interpret them soberly, weighing conflicting signals honestly."""


def _user_prompt(symbol: str, as_of: str, snap: dict) -> str:
    return (
        f"Asset: {symbol}\nAs of: {as_of}\n\n"
        f"Technical indicator snapshot:\n"
        f"- Latest close: {snap['latest_close']:.2f}\n"
        f"- Price vs 20-day SMA: {snap['price_vs_sma20_pct']:+.2f}%\n"
        f"- Price vs 50-day SMA: {snap['price_vs_sma50_pct']:+.2f}%\n"
        f"- RSI (14): {snap['rsi_14']:.1f}  (>70 overbought, <30 oversold)\n"
        f"- MACD: {snap['macd']:.2f}, signal: {snap['macd_signal']:.2f}, "
        f"histogram: {snap['macd_histogram']:+.2f}\n"
        f"- 30-day price change: {snap['pct_change_30d']:+.2f}%\n"
        f"- Annualized volatility: {snap['annualized_volatility']:.2f}\n\n"
        f"Give your technical read as JSON."
    )


def analyze(symbol: str, as_of: str, feed: PointInTimeFeed | None = None) -> AnalystReport:
    if feed is None:
        feed = PointInTimeFeed(symbol)
    history = feed.history_before(as_of)
    if len(history) < 60:
        return AnalystReport(ROLE, symbol, as_of, "neutral", 0.0,
                             "Insufficient history.", {})
    snap = indicators.snapshot(history["close"])
    return run_analyst(ROLE, symbol, as_of, SYSTEM_PROMPT,
                       _user_prompt(symbol, as_of, snap), snap)


if __name__ == "__main__":
    r = analyze("BTC/USDT", "2023-06-01")
    print(f"[{r.role}] {r.symbol} @ {r.as_of}: {r.stance} ({r.confidence:.2f})")
    print(f"  {r.reasoning}")
