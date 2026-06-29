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
