# Key Results

## Agent vs. baselines (2023-01-01 to 2025-06-01, weekly, net of costs)

| Strategy | Cum. Return | Sharpe | Max Drawdown |
|---|---|---|---|
| BuyHold BTC | 535.8% | 1.811 | -28.1% |
| Agent BTC | 202.1% | 1.711 | -21.6% |
| BuyHold ETH | 111.5% | 0.805 | -63.8% |
| Agent ETH | 83.1% | 0.877 | -36.2% |

The agent **reduces risk** (lower volatility and drawdown) but **underperforms
buy-and-hold on returns**. Best Sharpe ~1.7 — versus the original TradingAgents
paper's claimed **8.2**.

## Null-hypothesis test: is the agent better than random?

Compared the LLM agent's Sharpe against a distribution of 200 randomized agents
(random weekly positions, same window/costs):

| Symbol | Agent Sharpe | Random median | Random 95th | Agent beats |
|---|---|---|---|---|
| BTC | 1.711 | 1.520 | 1.949 | **76.5%** of random |
| ETH | 0.877 | 0.695 | 1.108 | **73.5%** of random |

**Finding:** the LLM agent is *not* statistically distinguishable from chance
(would need to beat ~95% of random agents to claim signal). Its apparent
performance is consistent with luck plus crypto's bull-market beta — not skill
from the multi-agent architecture.

This is the central result: a faithfully-built multi-agent LLM trader, evaluated
honestly, shows **no significant edge over random** — the question the original
paper never asked.

## Alpha-decay / leakage test (early vs. recent window)

Compared the agent's edge over random in an early window (2023-01 to 2024-06,
likely seen in training, bull market) vs. a recent window (2025-09 to 2026-06,
likely unseen, bear market):

| Symbol | Early (seen) | Recent (unseen) |
|---|---|---|
| BTC | beats 68.0% of random | beats 50.5% (no edge) |
| ETH | beats 58.0% of random | beats 15.5% (no edge) |

**What this supports:** on recent, likely-unseen data the agent shows **no edge
over random** (both at or below chance).

**Honest caveat:** the recent window is a *down* market while the early window
is a *bull* market. Because the percentile-vs-random measure is not symmetric
across regimes, this does **not** cleanly isolate a memorization effect — it is
*suggestive*, not conclusive. A cleaner future test would compare two windows
of the *same* regime (e.g. the 2022 bear vs. a recent bear). Model training
cutoffs are also not published precisely, limiting how cleanly "seen" vs
"unseen" can be defined.

**Combined with the null test**, the overall picture is consistent and modest:
a faithfully-built multi-agent LLM trader, evaluated honestly, shows no
statistically significant edge over chance — and certainly none that survives
on recent data.

## Regime-controlled leakage test (bear vs. bear) — the clean version

To remove the bull/bear confound, compared the agent's edge over random in two
*bear* markets: 2022 (likely seen in training) and recent 2025-26 (likely unseen).

| Symbol | 2022 bear (seen) | recent bear (unseen) |
|---|---|---|
| BTC | beats 33.5% of random | beats 50.5% of random |
| ETH | beats 31.0% of random | beats 15.5% of random |

**Finding:** the agent beats fewer than ~50% of random agents in *every* bear
window — i.e. it performs at or below chance regardless of regime. Crucially,
it is NOT better on the likely-seen 2022 data than on unseen data, so there is
**no memorization signature.**

## Unified conclusion

Across all three experiments the story is consistent: the multi-agent LLM
trader shows **no statistically significant edge over random**, from skill or
from memorization. Its strong bull-market Sharpe (~1.7) reflects crypto
*beta* (being long while the market rose) — a randomized agent captured the
same. Stripped of beta in bear markets, no edge remains. This stands in sharp
contrast to the original paper's claimed Sharpe of 8.2 over a 3-month,
bull-only window with no null baseline and no cost accounting.

## Ablation: does the multi-agent debate add value?

Compared the full system (2 analysts + 2-round bull/bear debate + trader
synthesis, ~6 LLM calls/decision) against a minimal agent (1 analyst's stance
mapped directly to a position, 1 LLM call/decision):

| Strategy | Cum. Return | Sharpe | Max DD | Beats random |
|---|---|---|---|---|
| Full debate BTC | 202.1% | 1.711 | -21.6% | 76.5% |
| Minimal BTC | 199.7% | 1.648 | -22.1% | 66.0% |
| Full debate ETH | 83.1% | 0.877 | -36.2% | 73.5% |
| Minimal ETH | 81.8% | 0.867 | -37.9% | 73.5% |

**Finding:** the full debate architecture matches a single analyst within noise
(~0.06 Sharpe on BTC, ~0.01 on ETH) for ~6x the compute. The multi-agent
debate — the original paper's central contribution — adds **no measurable
performance** in this setup.

## Formal significance (empirical p-value + Deflated Sharpe)

| Symbol | Agent Sharpe | Empirical p (vs random) | Deflated Sharpe |
|---|---|---|---|
| BTC | 1.711 | 0.239 (not significant) | 0.925 |
| ETH | 0.877 | 0.269 (not significant) | 0.513 |

**Reading these together:** the agent's returns are reliably positive (high DSR /
probabilistic Sharpe) — but that is **crypto beta**, not skill. The Deflated
Sharpe only benchmarks against zero, which a bull market clears trivially. The
empirical p-value is the stricter test: it compares the agent to random agents
that *also* captured that beta, and there the edge is **not significant**
(p ≈ 0.24). The Sharpe measures market exposure, not alpha.

## Four-coin robustness (BTC, ETH, SOL, BNB)

Extending all tests to four coins confirms the pattern holds — it was not an
artifact of two hand-picked assets.

**Significance (empirical p-value vs random null):**

| Symbol | Agent Sharpe | p-value | Significant? |
|---|---|---|---|
| BTC | 1.711 | 0.239 | No |
| ETH | 0.877 | 0.269 | No |
| SOL | 1.497 | 0.443 | No |
| BNB | 1.090 | 0.264 | No |

No coin shows a statistically significant edge over a beta-matched random agent.
(BNB's apparent edge over buy-and-hold in the returns table is not significant.)

**Ablation (full debate vs single analyst), Sharpe:**

| Symbol | Full (debate) | Minimal (1 analyst) |
|---|---|---|
| BTC | 1.711 | 1.648 |
| ETH | 0.877 | 0.867 |
| SOL | 1.497 | **1.602** |
| BNB | **1.090** | 0.774 |

Across four coins the debate helps on BNB, *hurts* on SOL, and is a wash on
BTC/ETH — **no consistent direction.** The multi-agent debate adds variance,
not value, for ~6x the compute. (Note: the cleaner two-coin result suggested
mere redundancy; the fuller four-coin picture shows it is actively noisy.)
