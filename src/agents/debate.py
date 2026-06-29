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
job is to commit to ONE concrete decision for the asset. You must decide, not
hedge endlessly — but HOLD is a legitimate choice when conviction is genuinely
low.

Decision meanings:
- BUY: open or increase a long position (you expect the price to rise).
- SELL: close/avoid the position (you expect the price to fall or stall).
- HOLD: maintain current stance with low conviction either way.

Set target_position between 0.0 (fully in cash) and 1.0 (fully invested long),
reflecting your conviction. BUY implies a higher target_position, SELL implies
near 0.0.

Respond ONLY with a JSON object of this exact shape:
{
  "action": "BUY" | "SELL" | "HOLD",
  "target_position": <float 0.0 to 1.0>,
  "rationale": "<2-4 sentences explaining the decision, weighing both sides>"
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
        action = str(parsed.get("action", "HOLD")).upper()
        target = float(parsed.get("target_position", 0.0))
        rationale = str(parsed.get("rationale", ""))
        target = max(0.0, min(1.0, target))  # clamp to [0,1]
    except (json.JSONDecodeError, ValueError):
        action, target, rationale = "HOLD", 0.0, f"Unparseable: {raw[:200]}"

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
