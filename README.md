# Hindsight

**A rigorous reimplementation of multi-agent LLM trading - does the edge survive once you control for what the model already knew?**

## Motivation

Recent work (e.g. *TradingAgents*, arXiv:2412.20138) reports spectacular results
from multi-agent LLM trading systems - Sharpe ratios above 8 over a three-month
backtest. Numbers like that should arouse suspicion, not excitement: sustained
Sharpe ratios above 3 are rare even at elite quant funds.

This project reimplements the multi-agent idea on crypto markets and subjects it
to the scrutiny the original version lacked. The central question:

> When an LLM "analyzes" a past period, is it reasoning or just recalling an outcome it already saw during training?

## What this project does differently

- **Leak-aware evaluation.** A pre-/post-training-cutoff experiment measures
  *alpha decay* to separate genuine signal from memorization.
- **Honest accounting.** Realistic transaction costs and slippage; results
  reported gross *and* net.
- **Real baselines.** Buy-and-hold, equal-weight, rule-based strategies, and a
  randomized-decision agent as a null hypothesis.
- **Statistical rigor.** Walk-forward analysis, Deflated Sharpe Ratio, and
  Probability of Backtest Overfitting.
- **Regime awareness.** Results split across bull and bear markets.
- **Fully reproducible & free.** Open-weight LLMs run locally; all data from
  free sources; every LLM call cached.

## Status

 Early development. Building in stages: data layer → baselines → agent system
→ leak-audit experiment → writeup.

## Stack

Python · CCXT · pandas/numpy · vectorbt · LangGraph · Ollama (local LLMs)

## License

MIT
