"""
Shared scaffolding for analyst-style agents.

An analyst agent is essentially: a role (system prompt) + a way to turn data
into a user prompt + JSON parsing of the response. This base captures that
common shape so each concrete agent is small and focused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.agents.llm_cache import cached_chat

MODEL = "qwen2.5:7b"

# Every analyst returns this same structured shape, so downstream agents
# (researchers, trader) can consume any analyst uniformly.
ANALYST_OUTPUT_SPEC = """Respond ONLY with a JSON object of this exact shape:
{
  "stance": "bullish" | "bearish" | "neutral",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<2-4 sentences citing the specific data provided>"
}
Do not include any text outside the JSON. Do not invent data you were not given.
Do not reference events, news, or prices beyond what is provided."""


@dataclass
class AnalystReport:
    role: str
    symbol: str
    as_of: str
    stance: str
    confidence: float
    reasoning: str
    data: dict = field(default_factory=dict)


def run_analyst(
    role: str,
    symbol: str,
    as_of: str,
    role_system_prompt: str,
    user_prompt: str,
    data: dict,
) -> AnalystReport:
    """Execute one analyst: call the cached LLM, parse the JSON, return a report."""
    system = role_system_prompt.strip() + "\n\n" + ANALYST_OUTPUT_SPEC
    raw = cached_chat(model=MODEL, system=system, user=user_prompt,
                      options={"temperature": 0.0}, use_json=True)
    try:
        parsed = json.loads(raw)
        stance = str(parsed.get("stance", "neutral")).lower()
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError):
        stance, confidence, reasoning = "neutral", 0.0, f"Unparseable: {raw[:200]}"
    return AnalystReport(role, symbol, as_of, stance, confidence, reasoning, data)
