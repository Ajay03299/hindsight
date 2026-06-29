"""
Debate orchestrator + Trader agent.

The debate runs the bull and bear for N rounds, each responding to the other's
most recent argument so positions can sharpen across rounds. The trader then
reads the full debate plus the analyst reports and commits to a concrete,
actionable decision: BUY, SELL, or HOLD, with a target position size.

This is the agent system's final output — the thing the backtester consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.agents.base import AnalystReport
from src.agents import researchers
from src.agents.researchers import ResearchArgument
from src.agents.llm_cache import cached_chat

MODEL = "qwen2.5:7b"


@dataclass
class DebateTranscript:
    rounds: list[dict] = field(default_factory=list)  # [{"bull":..,"bear":..}, ...]


@dataclass
class TradeDecision:
    symbol: str
    as_of: str
    action: str            # "BUY" | "SELL" | "HOLD"
    target_position: float # 0.0 to 1.0 (fraction of capital to hold long)
    rationale: str
    transcript: DebateTranscript = field(default_factory=DebateTranscript)


def run_debate(symbol: str, as_of: str, reports: list[AnalystReport],
               n_rounds: int = 2) -> DebateTranscript:
    """Run N rounds of bull/bear, each responding to the other's last argument."""
    transcript = DebateTranscript()
    last_bull: str | None = None
    last_bear: str | None = None

    for _ in range(n_rounds):
        bull_arg = researchers.bull(symbol, as_of, reports, opposing_argument=last_bear)
        bear_arg = researchers.bear(symbol, as_of, reports, opposing_argument=last_bull)
        transcript.rounds.append({
            "bull": bull_arg.argument,
            "bull_risks": bull_arg.risks_acknowledged,
            "bear": bear_arg.argument,
            "bear_risks": bear_arg.risks_acknowledged,
        })
        last_bull, last_bear = bull_arg.argument, bear_arg.argument

    return transcript


TRADER_SYSTEM = """You are the trader at a trading firm. You have read your
analysts' reports and a debate between a bullish and a bearish researcher. Your
job is to set ONE number: your target position in the asset.

target_position is a float from 0.0 to 1.0:
- 0.0  = fully in cash (you expect the price to fall or want no exposure)
- 0.5  = neutral / moderate conviction
- 1.0  = fully invested long (you strongly expect the price to rise)

Choose the number that reflects your honest conviction after weighing both
sides. Higher means more bullish, lower means more bearish.

Respond ONLY with a JSON object of this exact shape:
{
  "target_position": <float 0.0 to 1.0>,
  "rationale": "<2-4 sentences explaining the number, weighing both sides>"
}
Base everything ONLY on the provided reports and debate. Invent nothing."""

def _format_debate(transcript: DebateTranscript) -> str:
    out = []
    for i, rnd in enumerate(transcript.rounds, 1):
        out.append(
            f"Round {i}:\n"
            f"  BULL: {rnd['bull']}\n"
            f"  BEAR: {rnd['bear']}"
        )
    return "\n".join(out)


def _format_reports(reports: list[AnalystReport]) -> str:
    return "\n".join(
        f"[{r.role}] {r.stance} ({r.confidence:.2f}): {r.reasoning}" for r in reports
    )


def decide(symbol: str, as_of: str, reports: list[AnalystReport],
           n_rounds: int = 2) -> TradeDecision:
    """Run the debate and have the trader commit to a decision."""
    transcript = run_debate(symbol, as_of, reports, n_rounds)

    user = (
        f"Asset: {symbol}\nAs of: {as_of}\n\n"
        f"Analyst reports:\n{_format_reports(reports)}\n\n"
        f"Researcher debate:\n{_format_debate(transcript)}\n\n"
        f"Commit to your decision as JSON."
    )
    raw = cached_chat(model=MODEL, system=TRADER_SYSTEM, user=user,
                      options={"temperature": 0.0}, use_json=True)
    try:
        parsed = json.loads(raw)
        target = float(parsed.get("target_position", 0.5))
        rationale = str(parsed.get("rationale", ""))
        target = max(0.0, min(1.0, target))  # clamp to [0,1]
    except (json.JSONDecodeError, ValueError):
        target, rationale = 0.5, f"Unparseable: {raw[:200]}"

    # Derive a human-readable label FROM the position, so they can never
    # contradict each other. The position is the single source of truth.
    if target >= 0.6:
        action = "BUY"
    elif target <= 0.4:
        action = "SELL"
    else:
        action = "HOLD"

    return TradeDecision(symbol, as_of, action, target, rationale, transcript)

if __name__ == "__main__":
    from src.agents import technical_analyst, momentum_analyst

    SYM, DATE = "BTC/USDT", "2023-06-01"
    reports = [
        technical_analyst.analyze(SYM, DATE),
        momentum_analyst.analyze(SYM, DATE),
    ]
    decision = decide(SYM, DATE, reports, n_rounds=2)
    print(f"\n{'='*70}\nTRADE DECISION — {SYM} @ {DATE}\n{'='*70}")
    print(f"Action:          {decision.action}")
    print(f"Target position: {decision.target_position:.2f}")
    print(f"Rationale:       {decision.rationale}")
    print(f"{'='*70}")
    print(f"(Debate ran {len(decision.transcript.rounds)} rounds)")
