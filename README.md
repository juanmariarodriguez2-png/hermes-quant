# Hermes Quant

[![Hermes daily run](https://github.com/juanmariarodriguez2-png/hermes-quant/actions/workflows/daily.yml/badge.svg)](https://github.com/juanmariarodriguez2-png/hermes-quant/actions/workflows/daily.yml)
[![tests](https://github.com/juanmariarodriguez2-png/hermes-quant/actions/workflows/tests.yml/badge.svg)](https://github.com/juanmariarodriguez2-png/hermes-quant/actions/workflows/tests.yml)

A systematic, multi-asset trading model that runs by itself every night: it sizes positions in US equities, gold, silver and crypto from fixed public rules, trades them on one Kraken account, and publishes everything it sees and does.

**Live dashboard:** https://juanmariarodriguez2-png.github.io/hermes-quant/ · **Live terminal:** https://juanmariarodriguez2-png.github.io/hermes-quant/terminal.html · **Guía en español:** [docs/GUIA.md](docs/GUIA.md) · **Research log:** [docs/RESEARCH.md](docs/RESEARCH.md)

It replaced an LLM agent that decided trades by reading the market. That agent made two trades, was stopped out of both, and cost more to run than it could plausibly earn. This is what was built instead, and how it was checked.

## Result

Every design decision used data up to 31 Dec 2021. The years after were locked away and evaluated once, with the model already frozen.

| | Annual return | Volatility | Sharpe | Worst drawdown |
|---|---:|---:|---:|---:|
| **Hermes Quant, blind exam 2022 → Nov 2026** | **11.9%** | 12.4% | **0.96** | **−11.3%** |
| S&P 500 buy & hold (in EUR) | 12.0% | 19.0% | 0.69 | −23.0% |
| 60% S&P 500 / 40% gold | 14.4% | 16.0% | 0.92 | −14.9% |
| Bitcoin buy & hold | 8.9% | 50.8% | 0.42 | −64.6% |
| Hermes Quant, design sample 2007 → 2021 | 11.5% | 12.3% | 0.95 | −17.2% |

In euros, after Kraken's real fees for a small account, a one-day execution delay and a currency-conversion charge. Benchmarks pay no costs.

How to read it honestly:

- **The Sharpe ratio did not decay out of sample** (0.95 designed, 0.96 examined). That is the main thing a hold-out can tell you.
- **It matched the S&P 500's return with two thirds of its volatility and half its drawdown.** It did not beat it on return.
- **A plain 60/40 stocks-gold mix returned more** (14.4% vs 11.9%). Gold had exceptional years; the model's edge over that mix was a smaller drawdown, nothing else.
- **There is no alpha.** Regressed on stocks, gold and bitcoin, the intercept is +0.9% a year with a t-statistic of 0.4. The model does not outguess markets. It budgets and controls risk.
- **The error bar is wide.** The 95% bootstrap interval for the blind-exam Sharpe is 0.19 to 1.74. Five years is not long.

## What the research found

Stated plainly, including what did not work:

1. **Costs decide everything at this size.** Spot crypto costs 0.80% per trade on Kraken's entry tier; tokenized stocks cost 0.08%. The first version of the model traded seven times its capital a year and lost most of its return to fees.
2. **Trend-following did not pay for itself on the funds.** Applied to equities and metals it lowered the Sharpe ratio from 0.95 to 0.81 after costs on the design sample, and from 0.96 to 0.79 out of sample. It is not used there.
3. **Trend-following did pay on crypto.** It cut the crypto sleeve's worst drawdown from −56% to −30% at a higher Sharpe. Crypto is held only when its trend is up.
4. **The sophisticated parts add robustness, not return.** Removing the Kalman filter, the random-matrix cleaning or the hierarchical allocation changes the Sharpe ratio by 0.01–0.02. They lower turnover and drawdown slightly. Diversifying by risk, targeting volatility and trading lazily do the real work.
5. **A regime model has to be judged in real time.** Fitted with hindsight, the "turbulent" state of the S&P 500 looks terrible. Judged on returns that could still be captured after the signal, it is still clearly bad for equities (Sharpe −0.06 vs 0.75). For crypto it looked promising in the same test but lowered the Sharpe ratio of every portfolio it was added to. It is used for equities only.
6. **Picking the best backtest would have been a coin toss.** Six structures finished within 0.05 Sharpe of each other; the in-sample winner landed in the worse half out of sample 65% of the time. The model was therefore assembled one tested component at a time, not chosen by rank.
7. **Listed is not the same as tradable.** Kraken lists 177 tokenized stocks and funds, but among the broad funds only four trade in real size. The Europe, small-cap and emerging-market tokens go days without a single trade; their stale prices looked calm and the risk model wanted to load up on them. A liquidity gate now checks every market every day.
8. **Derivatives do not help today.** Perpetual futures are 16× cheaper in fees but cost about 4.8% a year in funding for long positions, more than spot. Trend-following shorts lost money. The delta-neutral carry trade that paid 10–25% in 2020–21 pays roughly nothing in 2026. The engine monitors funding and will say when that changes.
9. **Picking single stocks did not beat holding the index.** Twelve rules (momentum, the model's own trend filter, both; 10 or 20 names; with and without a market gate) were tried on the companies that were in the S&P 500 on each date, not today's list. The best returned 5.7% a year against the index fund's 10.6% in 2007–2021, and 0.9% against 11.5% since 2022. Prices exist for only 59–91% of each year's members in the design years, which is the study's weak point. The model keeps holding companies through the index funds; the dashboard shows what that is in euros per company.

## How it works

Six steps, every one causal (day *t* uses data up to *t* only), and the same code runs the backtest and the live account.

| Step | Method | Why |
|---|---|---|
| Volatility | Garman-Klass range estimator, bias-corrected, wick-filtered, two time-scales | 9% better forecasts for crypto than a plain EWMA; chosen by forecast loss, not profit |
| Trend | Bank of Kalman filters (30/60/120/240-day) on volatility-normalised returns → P(latent drift > 0) | A probability with an error bar, instead of a moving-average crossover |
| Regime | Two-state Gaussian hidden Markov model, refitted walk-forward | Equity returns in the turbulent state do not compensate for the risk |
| Correlations | Marchenko-Pastur eigenvalue cleaning (random matrix theory) | Keeps real structure, flattens sampling noise before it reaches the allocator |
| Allocation | A third of risk per sleeve; hierarchical risk parity inside; 15% volatility target | No covariance inversion, so no noise amplification |
| Execution | No-trade bands, partial rebalancing, minimum tickets | Trade only when the deviation is worth the commission |

## What it trades

One Kraken account, three sleeves:

- **Equities:** S&P 500 and Nasdaq 100, as tokenized ETFs (SPYx, QQQx). Always held, scaled by volatility and by the regime model.
- **Real assets:** gold and silver (GLDx, SLVx). Always held, scaled by volatility.
- **Crypto:** the four most liquid coins at the time, spot in EUR. Held only while their trend probability is positive.

Single stocks and perpetual futures are scored every day and shown on the dashboard, but the model does not trade them: neither has passed a test that would justify it.

There is one door for single stocks, and it is outside the model: the owner's **favourites basket** (`config/favourites.json`, off unless `HERMES_FAVOURITES=yes`). At most four companies at 2.5% of the account each, held only while the model's trend filter says their trend is up, bought only with money the model leaves idle. It is a preference with a cap, not a finding, and the code keeps it apart: the model's weights are computed first from a panel the favourites are not in, and with the basket off the nightly run is byte-for-byte what it was.

## How it was checked

- **Design / hold-out split** fixed in advance, hold-out opened once (`research/05_holdout.py` refuses to run unless the model's defaults equal the frozen configuration).
- **Survivorship-free crypto universe:** all 688 USDT markets Binance has ever listed, delisted ones included (Terra, FTX...). Membership is decided day by day from trailing volume.
- **Trial registry:** 77 attempts are counted. The Deflated Sharpe Ratio on the design sample is 95% treating all of them as independent.
- **Block bootstrap** for confidence intervals; Newey-West errors for regressions; ECB deposit rate for excess returns.
- **227 automated tests**, including look-ahead tests (rewrite the future, nothing decided earlier may change) and the full live path against a simulated exchange.
- **Independent adversarial review** of the code by OpenAI Codex before the hold-out was opened: 16 findings, all addressed ([docs/RESEARCH.md](docs/RESEARCH.md#independent-review)).

## Run it

```bash
pip install -r requirements.txt
python -m hermes_quant.live.run_daily        # paper trading, the default
python -m pytest tests -q
```

The GitHub Actions workflow in `.github/workflows/daily.yml` does this every night, commits the new state and publishes the dashboard. No server, no laptop, no API bill.

Real trading is opt-in and has three locks: `HERMES_MODE=live`, Kraken API keys without withdrawal permission, and `HERMES_LIVE_ARMED=yes`. Until the third is set, every order is sent with Kraken's validate-only flag. A manual workflow, *Kraken connection check*, rehearses the day's orders against the real account with the validate flag and places nothing; run it before arming. No real order has been executed yet: what is verified is that Kraken accepts them. Details in [docs/GUIA.md](docs/GUIA.md).

**A second venue, for accounts Kraken will not serve by API.** `HERMES_VENUE=trading212` sends every order through Trading 212's API instead: real shares, and for each of the model's funds the European fund that tracks the same index or metal (`config/trading212.json`; a European retail account cannot buy the US-listed ones). Orders go out only while the instrument's exchange is open, are sized by quantity from a reference price and read back from the broker's own history; a request that times out is looked up, never repeated. It keeps its own record under `data/live/trading212/` and starts on the practice account (`T212_ENV=demo`). Written with OpenAI Codex from the broker's API document and tested against a simulated API; not yet run against a funded account.

One limit comes from the venue, not the model: Kraken takes no tokenized-stock orders by API from accounts in the European Economic Area, only by hand on kraken.com. On such an account (the default) the model trades crypto by itself and lists the equity, gold and silver orders for the owner to place; it never converts currency for an order it cannot send. Set `HERMES_TOKENIZED_API=yes` where the API does accept them. Buying by hand has a price: Kraken's Convert cost 1.07% all in when measured against the market, not the 0.38% the backtest assumes for the order book. Re-costed with that, the hold-out result is 10.9% a year (Sharpe 0.89), not 11.9%; the dashboard shows both.

To reproduce the research:

```bash
pip install -r requirements-research.txt
python research/01_download_data.py          # about 40 minutes, public data, no keys
python research/02_volatility_forecast.py
python research/03_regime_test.py
python research/04_design_sample.py
python research/05_holdout.py
```

## Layout

```
hermes_quant/
  filters/      Kalman trend, range volatility, random-matrix cleaning
  regime/       hidden Markov model, turbulence and absorption gauges
  portfolio/    hierarchical risk parity, order planning shared by backtest and live
  backtest/     causal simulator, metrics, statistical tests
  live/         daily runner, price store, paper broker, Kraken and Trading 212 brokers, monitors, favourites basket
  model.py      the model: features and target weights
research/       numbered scripts, one per research step, and their results
tests/          227 tests
config/         the owner's list of single stocks, and the map of markets to Trading 212 instruments
site/           the dashboard (static; reads live.json and research.json)
tradingview/    Pine Script indicator showing the trend probability on any chart
```

## Limits

Past performance does not guarantee future results. The backtest uses Binance prices and the original ETFs as a proxy for Kraken's markets (the two agree to within about 0.2% where they overlap). Funds were restricted to those that trade on Kraken today, and what is liquid today is partly what did well yesterday. Nothing here is financial advice.

MIT licence.
