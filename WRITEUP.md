# Does Multi-Agent LLM Trading Actually Work? A Rigorous Reimplementation

I rebuilt a paper claiming a Sharpe ratio of 8, and found the edge was beta in disguise.

## The claim that started it

In late 2024, a paper called *TradingAgents* (arXiv:2412.20138) made the rounds: a team of LLM "agents" - analysts, a bull and bear who debate, a trader, a risk manager - collaborating to trade stocks, reporting a **Sharpe ratio above 8** over a 3 month backtest.

For context, a Sharpe ratio above 3 is considered amazing and is rare even at elite quant funds. Renaissance Technologies, the most successful quant fund in history, is estimated to run around 2 on its public-facing funds. A Sharpe of 8 from a few LLM prompts should trigger skepticism.

The paper had three tells that its result might be an artifact rather than a discovery:

1. **A three-month backtest** - far too short to distinguish skill from luck.
2. **A bull-only window** (early 2024) - when everything went up, so *anything* long looked brilliant.
3. **No null hypothesis** - they never checked whether the elaborate system beat random chance, and reported no transaction costs.

So I rebuilt it - on crypto, with the rigor the original skipped - to answer one question: **when you strip away the short window, the bull-market tailwind, and the missing cost accounting, is there any edge left?**

## What I built

A reimplementation of the multi-agent architecture, on daily crypto data (BTC, ETH, SOL, BNB), 2023–2025:

- **Analysts** read a point-in-time snapshot of technical and momentum indicators.
- **A bull and a bear researcher** build opposing cases and debate for two rounds.
- **A trader** synthesizes the debate into a target position.
- **An event-driven backtester** walks forward through time — structurally incapable of look-ahead, because on each date it only ever sees data *before* that date.

Everything runs on a **local open-weight LLM** (Qwen2.5-7B via Ollama) - free, no API bills, no GPU beyond a MacBook - with every LLM call cached to disk so the entire multi-year backtest reproduces in seconds.

Then I ran the four tests the original paper didn't.

## Test 1: Does it beat buy-and-hold?

No. Across all four coins, the agent **underperformed lazy buy-and-hold on returns** - sometimes dramatically (on SOL, buy-and-hold made 1,478% vs the agent's 403%). Its *only* consistent benefit was lower risk: smaller drawdowns and lower volatility on every coin.

So it's not a money machine. At best it's a risk-reducer that costs you upside.

## Test 2: Is it better than a monkey?

This is the test almost no LLM-trading paper runs. I built a **randomized agent** - one that ignores all analysis and picks a position at random each week - and ran it 200 times to build a distribution of what pure luck achieves. Then I asked where the LLM agent falls on that distribution.

| Coin | LLM agent Sharpe | Beats % of random | Empirical p-value |
|--|--|--|--|
| BTC | 1.711 | 76.5% | 0.239 |
| ETH | 0.877 | 73.5% | 0.269 |
| SOL | 1.497 | 56.0% | 0.443 |
| BNB | 1.090 | 74.0% | 0.264 |

The agent's Sharpe of 1.7 on BTC *sounds* good until you see that random agents averaged 1.5, and one in four random runs beat the LLM. **On no coin was the edge statistically significant** (all p-values far above 0.05). The Deflated Sharpe Ratio told the same story: the returns are reliably positive, but only because *being long crypto in a bull market* is reliably positive. That's beta, not alpha. A monkey captured it too.

## Test 3: Did it just memorize the past?

The LLM was trained on data that may include the backtest period - so maybe its faint apparent edge was hindsight, not reasoning. To test this, I compared the agent's edge in an **early bear market (2022, likely seen in training)** against a **recent bear market (2025–26, likely unseen)**, holding the market regime constant to isolate the memorization question.

If the agent had memorized 2022, it should have done *well* there. Instead it beat only **33.5%** (BTC) and **31.0%** (ETH) of random agents in 2022 - *worse* than chance on the period it supposedly memorized. There was no memorization edge to find, because there was no edge at all.

## Test 4: Does the multi-agent debate even help?

The paper's central contribution is the elaborate debate. So I ablated it: I stripped the system down to a **single analyst** making one LLM call, versus the full six-call debate, and compared.

| Coin | Full debate (Sharpe) | Single analyst (Sharpe) |
|--|--|--|
| BTC | 1.711 | 1.648 |
| ETH | 0.877 | 0.867 |
| SOL | 1.497 | **1.602** |
| BNB | **1.090** | 0.774 |

The debate helps on BNB, *hurts* on SOL, and is a wash on BTC/ETH - **no consistent direction.** The multi-agent architecture, running six times the compute of a single analyst, adds *variance, not value*. (Interestingly, with only two coins the debate looked like harmless redundancy; it took all four to reveal it was actively noisy - a small lesson in why sample size matters.)

## The verdict

Four independent tests, one consistent story: **a faithfully-built multi-agent LLM trader, evaluated honestly, shows no statistically significant edge** - not from skill, not from memorization, and not from the multi-agent structure itself. The impressive-looking Sharpe was crypto beta wearing a lab coat.

This doesn't prove LLM trading is impossible, or that a frontier model wouldn't do better. It proves something narrower and more useful: **the specific result that made this genre exciting evaporates under honest evaluation.** The three things the original omitted — a long enough window, transaction costs, and a null hypothesis - weren't oversights. They were load-bearing.

## What I actually learned

The technical skills here - multi-agent orchestration, event-driven backtesting, local LLM deployment, the statistics — matter. But the real lesson is about *evaluation discipline*:

- **A backtest that looks amazing is a bug report, not a result.** The right first reaction to a Sharpe of 8 is "what leaked?"
- **Always build the null.** "My system got X" means nothing without "random got Y."
- **Beta masquerades as alpha** in any rising market. Controlling for it is the whole game.
- **Let the data correct your hypothesis.** I set out to find memorization bias; the data told me something cleaner and I followed it.

## Reproduce it

Everything is open and free. See the [README](README.md) for one-command reproduction. All results, caches, and the code for every test above are in this repo.


*Built with Python, CCXT, Ollama (Qwen2.5-7B, local), and a healthy distrust of my own backtests.*
