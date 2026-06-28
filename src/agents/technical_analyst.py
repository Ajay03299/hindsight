"""
Technical Analyst Agent.

Given a symbol and an 'as of' date, it:
  1. Pulls price history STRICTLY BEFORE that date (no look-ahead).
  2. Computes an indicator snapshot.
  3. Asks the local LLM to interpret it as a short professional read.
  4. Returns a structured result: stance + reasoning + the raw indicators.

The agent only ever touches `feed.history_before(as_of)`, so it cannot see
the present or future. That guarantee is the whole point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import ollama

from src.data.feed import PointInTimeFeed
from src.agents import indicators

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """You are a professional technical analyst at a trading firm.
You are given a factual snapshot of technical indicators for a crypto asset.
Interpret them soberly. Do not invent data you were not given. Do not reference
any events, news, or prices beyond the indicators provided.

Respond ONLY with a JSON object of this exact shape:
{
  "stance": "bullish" | "bearish" | "neutral",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<2-4 sentences citing the specific indicators>"
}
Do not include any text outside the JSON."""


@dataclass
class AnalystReport:
    symbol: str
    as_of: str
    stance: str
    confidence: float
    reasoning: str
    indicators: dict = field(default_factory=dict)


def _build_user_prompt(symbol: str, as_of: str, snap: dict) -> str:
    return (
        f"Asset: {symbol}\nAs of: {as_of}\n\n"
        f"Technical indicator snapshot (most recent available values):\n"
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
    """Run the technical analyst for one symbol as of one date."""
    if feed is None:
        feed = PointInTimeFeed(symbol)

    history = feed.history_before(as_of)
    if len(history) < 60:
        # Not enough history to compute 50-day indicators meaningfully
        return AnalystReport(symbol, as_of, "neutral", 0.0,
                             "Insufficient history for technical analysis.", {})

    snap = indicators.snapshot(history["close"])
    user_prompt = _build_user_prompt(symbol, as_of, snap)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",       # ask Ollama to constrain output to valid JSON
        options={"temperature": 0.0},  # deterministic -> reproducible
    )
    raw = response["message"]["content"]

    try:
        parsed = json.loads(raw)
        stance = str(parsed.get("stance", "neutral")).lower()
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError):
        stance, confidence, reasoning = "neutral", 0.0, f"Unparseable response: {raw[:200]}"

    return AnalystReport(symbol, as_of, stance, confidence, reasoning, snap)


if __name__ == "__main__":
    # Quick manual test: analyze BTC as of a fixed past date
    report = analyze("BTC/USDT", "2023-06-01")
    print(f"\n{'='*70}")
    print(f"TECHNICAL ANALYST — {report.symbol} as of {report.as_of}")
    print(f"{'='*70}")
    print(f"Stance:     {report.stance}  (confidence {report.confidence:.2f})")
    print(f"Reasoning:  {report.reasoning}")
    print(f"{'='*70}")
