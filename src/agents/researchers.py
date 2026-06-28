"""
Bull and Bear Researcher Agents.

Unlike analysts (neutral observers of data), researchers are ADVOCATES. Each
receives the analyst reports and builds the strongest case for one side:
  - Bull: argues for buying / holding a long position.
  - Bear: argues for selling / staying out.

The partisanship is intentional. A single optimistic pass misses risks; a
single pessimistic pass misses opportunity. Pitting two advocates against each
other surfaces both. Each researcher is also asked to engage with the opposing
view when one is provided, so this can run as a multi-round debate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.agents.base import AnalystReport
from src.agents.llm_cache import cached_chat

MODEL = "qwen2.5:7b"

RESEARCHER_OUTPUT_SPEC = """Respond ONLY with a JSON object of this exact shape:
{
  "argument": "<your strongest 3-5 sentence case>",
  "key_risks_acknowledged": "<1-2 sentences honestly noting the strongest point against you>"
}
Do not include any text outside the JSON. Base your argument ONLY on the
analyst reports provided. Do not invent data, news, or prices."""

BULL_SYSTEM = """You are a bullish researcher at a trading firm. Your job is to
build the strongest honest case for BUYING or HOLDING a long position in the
asset, based on the analyst reports. Emphasize opportunity and positive signals,
but acknowledge the most serious risk to your view."""

BEAR_SYSTEM = """You are a bearish researcher at a trading firm. Your job is to
build the strongest honest case for SELLING or STAYING OUT of the asset, based
on the analyst reports. Emphasize risks and negative signals, but acknowledge
the most serious point against your view."""


@dataclass
class ResearchArgument:
    side: str          # "bull" or "bear"
    argument: str
    risks_acknowledged: str


def _format_analyst_reports(reports: list[AnalystReport]) -> str:
    lines = []
    for r in reports:
        lines.append(
            f"[{r.role}] stance={r.stance} confidence={r.confidence:.2f}\n"
            f"  reasoning: {r.reasoning}"
        )
    return "\n".join(lines)


def _run_researcher(
    side: str,
    system: str,
    symbol: str,
    as_of: str,
    reports: list[AnalystReport],
    opposing_argument: str | None,
) -> ResearchArgument:
    user = (
        f"Asset: {symbol}\nAs of: {as_of}\n\n"
        f"Analyst reports:\n{_format_analyst_reports(reports)}\n"
    )
    if opposing_argument:
        user += (
            f"\nThe opposing researcher argued:\n\"{opposing_argument}\"\n"
            f"Engage with their points directly while making your case.\n"
        )
    user += "\nMake your case as JSON."

    full_system = system.strip() + "\n\n" + RESEARCHER_OUTPUT_SPEC
    raw = cached_chat(model=MODEL, system=full_system, user=user,
                      options={"temperature": 0.0}, use_json=True)
    try:
        parsed = json.loads(raw)
        argument = str(parsed.get("argument", ""))
        risks = str(parsed.get("key_risks_acknowledged", ""))
    except (json.JSONDecodeError, ValueError):
        argument, risks = f"Unparseable: {raw[:200]}", ""
    return ResearchArgument(side, argument, risks)


def bull(symbol, as_of, reports, opposing_argument=None) -> ResearchArgument:
    return _run_researcher("bull", BULL_SYSTEM, symbol, as_of, reports, opposing_argument)


def bear(symbol, as_of, reports, opposing_argument=None) -> ResearchArgument:
    return _run_researcher("bear", BEAR_SYSTEM, symbol, as_of, reports, opposing_argument)


if __name__ == "__main__":
    from src.agents import technical_analyst, momentum_analyst

    SYM, DATE = "BTC/USDT", "2023-06-01"
    reports = [
        technical_analyst.analyze(SYM, DATE),
        momentum_analyst.analyze(SYM, DATE),
    ]
    b = bull(SYM, DATE, reports)
    r = bear(SYM, DATE, reports)
    print(f"\n{'='*70}\nBULL CASE — {SYM} @ {DATE}\n{'='*70}")
    print(b.argument)
    print(f"  (acknowledges: {b.risks_acknowledged})")
    print(f"\n{'='*70}\nBEAR CASE — {SYM} @ {DATE}\n{'='*70}")
    print(r.argument)
    print(f"  (acknowledges: {r.risks_acknowledged})")
