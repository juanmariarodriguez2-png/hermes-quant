# Research log

How the model was built, in the order it happened, with the dead ends left in.

## Protocol

| | |
|---|---|
| Design sample | 2007-01-01 → 2021-12-31. Every decision was taken here. |
| Hold-out | 2022-01-01 → 2026-11-13. Opened once, with the model frozen. |
| Account | 236 EUR, the size of the real account, so minimum tickets bind. |
| Costs | Kraken's published entry-tier fees: spot crypto 0.80% taker, tokenized stocks 0.08% taker, EUR↔USD conversion 0.20%; plus 0.05–0.10% slippage. |
| Timing | Decision at the close of day *d*, execution at the close of day *d+1*. |
| Currency | Everything in EUR, FX included. Cash earns nothing. |
| Crypto universe | All 688 USDT markets Binance ever listed, delisted included. Top 4 by trailing 90-day volume among those with 400 days of history, decided day by day. |
| Fund universe | The broad funds whose tokenized version really trades on Kraken: SPY, QQQ, GLD, SLV. |

The scripts in `research/` are numbered in the order they must run. `05_holdout.py` refuses to run unless the model's defaults equal the configuration recorded by `04_design_sample.py`.

## Step 0. What the old bot did

The previous system was an LLM agent making discretionary trades on Kraken with real money. Its own e-mail log gives the full record: thirteen days, two trades, two stop-outs, −1.06%.

| Trade | Size | Stop distance | Daily volatility | P(stop hit in 24 h by noise alone) | Held |
|---|---:|---:|---:|---:|---|
| ETH long | 35.01 € | 2.9% | 2.4% | 23% | 23 hours |
| BTC long | 29.99 € | 1.6% | 2.0% | 44% | 19 minutes |

The probability is 2·Φ(−d/σ) for a driftless random walk (reflection principle). Each round trip cost 1.6% in commissions against a target of 3–5%. The language-model calls cost 0.19 € a day: about 29% of the capital per year. The arithmetic could not work, whatever the quality of the market read.

It also stopped running: it lived in Docker on a laptop, and the last message is dated 10 November.

## Step 1. Data

Binance's public archive (daily candles for 688 markets, funding rates from 2020), Coinbase for BTC and ETH before 2017-08, Yahoo Finance for the funds and EUR/USD, the ECB for the risk-free rate. No API keys.

Two data problems had to be fixed before anything else was believable:

- **Flash-crash wicks.** Coinbase printed BTC at $0.06 on 2017-04-15 and ETH at $0.10 on 2017-06-21. A range-based volatility estimator saw 700% days and kept the model out of crypto for the whole of 2017.
- **A reused ticker.** `LUNAUSDT` is two different coins, before and after the Terra collapse. Any gap longer than five days now starts a new instrument.

## Step 2. Volatility: chosen by forecast accuracy

Target: realised variance over the next seven days. Loss: QLIKE. Design sample only.

| Estimator | Crypto QLIKE | Fund QLIKE |
|---|---:|---:|
| Close-to-close EWMA, half-life 21 (RiskMetrics) | 0.634 | 0.445 |
| Range blend, half-life 21 | 0.614 | 0.442 |
| Range blend, half-life 10 | 0.618 | 0.433 |
| **Range blend, 0.7 × hl 10 + 0.3 × hl 126 (adopted)** | **0.577** | **0.431** |

"Range blend" is the squared close-to-close return averaged with a Garman-Klass range term that is winsorised against wicks and rescaled, causally, so that its trailing mean matches the close-to-close variance. Lower loss by 9% for crypto and 3% for funds. This is the one component selected on a metric other than backtest profit.

## Step 3. Regime: tested in real time

A two-state Gaussian HMM fitted with hindsight always looks clever. The test that matters uses the walk-forward probability as it stood each evening, smoothed as in production, and the first return the strategy could still capture (from session *s+1* to *s+2*).

| P(turbulent) | S&P 500: Sharpe of capturable return | Bitcoin |
|---|---:|---:|
| below 0.2 | 0.75 | 2.65 |
| 0.2 – 0.5 | 0.96 | 0.65 |
| 0.5 – 0.8 | 1.24 | 1.31 |
| above 0.8 | **−0.06** | 0.40 |

For equities the signal is real: when the model is confident the market is turbulent, returns do not pay for the risk. For bitcoin the buckets are not even monotone, and in step 4 the crypto overlay lowered the Sharpe ratio of every structure it was added to. The overlay is applied to equities only.

## Step 4. Structure

Twelve structures on the design sample: which sleeves use the trend filter, and which get the regime overlay.

| Structure | Regime overlay | Return | Sharpe | Worst drawdown | Turnover |
|---|---|---:|---:|---:|---:|
| A. Everything always held | off | 13.4% | 0.99 | −20.7% | 0.7× |
| A | equities | 12.4% | 0.98 | −19.3% | 1.3× |
| A | equities + crypto | 10.8% | 0.90 | −17.2% | 1.3× |
| B. Trend filter on everything | off | 7.6% | 0.81 | −15.7% | 4.4× |
| B | equities | 7.3% | 0.81 | −15.3% | 4.4× |
| C. Half core, half trend; crypto trend-only | off | 11.3% | 0.97 | −17.8% | 1.4× |
| C | equities | 10.4% | 0.95 | −15.6% | 1.9× |
| **D. Funds always held; crypto trend-only** | **equities** | **11.5%** | **0.95** | **−17.3%** | **1.3×** |
| D | off | 12.5% | 0.94 | −20.7% | 0.6× |

(The three remaining rows, with the crypto overlay, are in `research/results/design_sample.json`.)

**The decision, and the rule that was not followed.** A rule had been written first: highest Sharpe among structures with a drawdown no worse than −20%. It picks A with the equity overlay. It was not adopted. Six structures sit within 0.05 Sharpe of each other, the standard error of a fifteen-year Sharpe ratio is about 0.26, and the rule was choosing on noise. The structure was assembled from one test per component instead:

- trend filter on the funds: B loses to A and D net of costs → off
- trend filter on crypto: halves the sleeve's worst drawdown at a higher Sharpe (below), and an asset with no cash flows and −80% bear markets gets no permanent seat → on
- regime overlay on equities: passes step 3, cuts drawdown in every structure at no Sharpe cost → on
- regime overlay on crypto: lowers Sharpe everywhere → off

That is D with the equity overlay. The rule's own pick was carried into the hold-out as context: it scored 0.88 against D's 0.96. That could easily have gone the other way and is not offered as vindication.

**The crypto sleeve on its own** (2016-09 → 2021-12, 40% volatility target):

| | Return | Sharpe | Worst drawdown |
|---|---:|---:|---:|
| Bitcoin buy & hold | 127% | 1.43 | −82.6% |
| Top 4, volatility-targeted, no trend filter | 64.7% | 1.48 | −55.8% |
| **Top 4, trend filter (Kalman bank)** | **55.9%** | **1.61** | **−30.3%** |
| Same, classic time-series momentum | 53.8% | 1.60 | −29.0% |

### Ablations

One component changed at a time, design sample.

| Variant | Return | Sharpe | Worst drawdown | Turnover |
|---|---:|---:|---:|---:|
| **Full model** | 11.5% | 0.95 | −17.2% | 1.3× |
| No random-matrix cleaning | 11.3% | 0.94 | −17.1% | 1.3× |
| Inverse-volatility instead of HRP | 11.4% | 0.93 | −18.1% | 1.3× |
| Classic momentum instead of Kalman | 11.3% | 0.94 | −17.0% | 1.3× |
| Close-to-close volatility | 11.7% | 0.96 | −16.8% | 1.4× |
| No regime overlay | 12.5% | 0.94 | −20.7% | 0.6× |
| Crypto: top 2 / top 8 | 11.8% / 10.5% | 0.96 / 0.89 | −18.4% / −15.5% | 1.3× |
| Execution delay 0 / 2 days | 11.1% / 11.7% | 0.92 / 0.96 | −16.9% / −17.1% | 1.3× |
| Decide daily / monthly | 11.5% / 11.9% | 0.96 / 0.95 | −16.1% / −25.6% | 1.4× / 0.9× |
| No partial rebalancing | 11.5% | 0.93 | −16.8% | 1.8× |
| All costs doubled | 10.9% | 0.90 | −17.5% | 1.3× |
| Zero costs | 12.1% | 0.99 | −17.2% | 1.3× |

The plateau is the finding. Nothing here is a knife-edge optimum, and nothing sophisticated moves the Sharpe ratio by more than 0.02. The result does not depend on trading at the close: a two-day delay changes nothing.

## Step 5. The hold-out

| | Return | Volatility | Sharpe | Sharpe over € cash | Worst drawdown |
|---|---:|---:|---:|---:|---:|
| **Hermes Quant** | **11.9%** | 12.4% | **0.96** | 0.78 | **−11.3%** |
| Growth profile | 12.1% | 14.2% | 0.87 | 0.71 | −13.2% |
| S&P 500 (EUR) | 12.0% | 19.0% | 0.69 | 0.57 | −23.0% |
| 60/40 stocks-gold | 14.4% | 16.0% | 0.92 | 0.78 | −14.9% |
| Bitcoin | 8.9% | 50.8% | 0.42 | 0.38 | −64.6% |
| *Context: crypto always held (the rule's pick)* | 12.2% | 14.2% | 0.88 | 0.72 | −12.8% |
| *Context: trend filter on everything* | 7.8% | 10.2% | 0.79 | 0.57 | −10.7% |
| *Context: no regime overlay* | 13.1% | 14.1% | 0.94 | 0.78 | −13.9% |

Year by year: 2022 −5.5%, 2023 +11.2%, 2024 +34.4%, 2025 +17.9%, 2026 to mid-November +3.6%.

### Inference

- **Bootstrap** (stationary, mean block 20 days, 5,000 resamples): Sharpe 95% interval 0.19 to 1.74; 0.7% of resamples at or below zero. The interval barely moves with block lengths from 5 to 80 days.
- **Deflated Sharpe Ratio**, design sample, 77 attempts: 95% treating every attempt as an independent no-edge strategy; 99.9% using the dispersion actually observed across them. The first is the headline because the attempts are close cousins and their observed dispersion flatters.
- **Probability of backtest overfitting** (CSCV, 24 model variants): 65%. Read correctly, this says the *ranking* among variants is unstable, which is why no variant was picked by rank. It does not say the variants are unprofitable: the average out-of-sample Sharpe of the in-sample winner was 90% of its in-sample value.
- **Factor regression**, weekly, Newey-West: β(S&P 500) 0.30, β(gold) 0.36, β(bitcoin) 0.05, R² 0.78, **alpha +0.9% a year, t = 0.4**. Leave gold out and "alpha" becomes 6.5% with t = 1.9: that would have been the metals' return wearing a costume.

### Stress episodes

| | Hermes | S&P 500 (EUR) | Bitcoin |
|---|---:|---:|---:|
| Global financial crisis, Sep 2008 – Mar 2009 | +1.0% | −38.1% | n/a |
| Crypto winter, 2018 | −9.5% | +2.5% | −78.9% |
| COVID crash, Feb – Mar 2020 | −14.8% | −33.1% | −32.0% |
| 2022 bear market | −6.1% | −3.0% | −72.7% |
| Tariff shock, Feb – Apr 2025 | −11.0% | −22.2% | −24.4% |
| Crypto slide, Oct 2025 – Nov 2026 | +7.5% | +20.3% | −41.9% |

### What the wider fund list would have done

With Europe, small caps, emerging markets and energy included (tokens that exist on Kraken but do not trade), the design-sample Sharpe falls to 0.72 and the hold-out is 0.99. The four funds that are liquid today are the ones that did well in 2007–2021. That is a selection effect the design-sample figure benefits from and the hold-out figure does not.

## Step 6. What the venue really charges

The backtest prices tokenized funds as Kraken's order book would: 0.08% fee, 0.10% slippage and 0.20% to turn euros into dollars, 0.38% a side. That turned out not to be available to the account this runs on. Kraken does not let clients in the European Economic Area trade tokenized stocks through the order book or the API; they buy by hand with "Convert", which shows a fee of zero and carries its margin in the price.

Measured on 16 November 2026 with a real quote: 57 EUR for 0.1754081 GLDx, 324.96 EUR a unit, against a market mid of 321.51 EUR at the same moment. That is **1.07% all in, per side**: nearly three times the assumption.

`research/08_execution_costs.py` re-costs the frozen model with that figure. Nothing is re-fitted or re-selected; the order-book row reproduces the published hold-out to the last digit.

| Funds cost per side | Hold-out return | Sharpe (over cash) | Worst drawdown | Costs per year |
|---|---:|---:|---:|---:|
| 0.38% (order book, as designed) | 11.9% | 0.96 (0.78) | −11.3% | 0.72% |
| 1.07% (Convert, as measured) | 10.9% | 0.89 (0.70) | −11.5% | 1.65% |
| 1.50% (Convert, pessimistic) | 10.2% | 0.84 (0.65) | −11.5% | 2.23% |

One measurement of one asset at one moment is a thin basis; the silver fund's own order book was 1% wide that evening, so its Convert price is probably worse. The honest figure for a European account is the second row, and the dashboard shows it beside the headline.

The lesson is the one the old bot taught, again: the result is decided by what a trade costs, and the cost that matters is the one actually charged, not the published schedule.

## Step 7. Derivatives (exploratory)

Funding history starts in 2020, too short to split. Parameters fixed a priori, one sample, indicative only.

| Crypto sleeve, Mar 2020 → Nov 2026 | Return | Sharpe | Fees / yr | Funding / yr |
|---|---:|---:|---:|---:|
| Spot, long only | 13.9% | 0.67 | 2.95% | 0 |
| Perpetuals, long only | 11.8% | 0.58 | 0.43% | −4.82% |
| Perpetuals, long and short | 5.8% | 0.34 | 0.64% | −4.86% |

Cash-and-carry (own the coin, short the perpetual, enter when 30-day funding exceeds 6% a year): BTC net +10.5% in 2020, +25.3% in 2021, +2.2%, +5.6%, +7.9%, +4.2%, and +0.1% in 2026. The trade is crowded out. The live engine reads Kraken's funding every day and flags it when it is worth doing again.

## Independent review

Before the hold-out was opened, the code was given to OpenAI Codex (GPT-6 Astra, high reasoning effort, read-only) with instructions to find anything that would make the results untrustworthy, and a hard rule not to touch data after 2021. Sixteen findings.

| # | Severity | Finding | What was done |
|---|---|---|---|
| 1 | High | Forecast and regime labels near the cutoff reached into January 2022 | Data is cut at the boundary before any label is built |
| 2 | High | Fund orders are sized with a NAV that includes the later crypto close | Documented; the live system trades everything at one instant, and a 0–2 day delay does not change the result |
| 3 | High | A delisted coin was sold at its last print with no loss | 50% write-down by default, all costs recorded |
| 4 | High | The crypto candidate list was compiled with hindsight | Replaced by Binance's complete listing history, 688 markets |
| 5 | High | Being priced on Binance does not make a coin tradable on Kraken | Labelled as a proxy simulation; the live engine has a venue-liquidity gate; prices agree within 0.2% |
| 6 | High | The trial registry mixed samples and could absorb hold-out results | Registry frozen by step 4 from design data only |
| 7 | Medium | Returns across market closures dropped FX moves on holidays | Returns are computed session to session |
| 8 | Medium | The HMM was fitted per session but filtered per calendar day | One clock: the chain advances once per observation |
| 9 | Medium | The regime test used a return the strategy could not capture | Test redone with production timing and smoothing |
| 10 | Medium | Sharpe inference assumed independent returns | Autocorrelation correction; block bootstrap is the primary test |
| 11 | Medium | PBO does not replay the actual selection rule | Reported as a diagnostic of rank stability, and explained as such |
| 12 | Medium | A zero-rate Sharpe was presented as excess return | Both reported; excess uses the ECB deposit rate; regression gained a gold factor |
| 13 | Low | The FX fee reserve could leave cash at −0.007 € | Reserve on gross dollar flow; property test added |
| 14 | Low | Bootstrap drawdown ignored a loss on day one | Fixed |
| 15 | Low | A benchmark could own an asset before it existed | Raises an error |
| 16 | Low | Degenerate inputs could silently disable the regime model | Rejected explicitly; unavailable is reported as unavailable |

A second pass, on the live trading path, was cut short when the API credit ran out. It had already flagged two real problems, both fixed: orders that timed out were being recorded as filled, and currency conversions escaped the per-order cap. The rest of that review was finished by hand against a simulated exchange, which found a third: buys were sized to the full balance although a limit order can fill above its reference price, so the account could try to spend 0.5% more than it had.

## What is still weak

- **Five years of hold-out.** The interval on the Sharpe ratio runs from 0.19 to 1.74.
- **Four funds.** The portfolio is US large caps, gold and silver, plus crypto. It is diversified across kinds of risk, not across regions.
- **A euro investor in dollar assets.** Part of the return since 2008 is the dollar's rise against the euro. The benchmarks carry the same exposure, so the comparison is fair, but the level is not guaranteed to repeat.
- **The fund list was filtered on today's liquidity,** which favours past winners.
- **Proxy prices.** Binance and the original ETFs stand in for Kraken's markets before those existed.
- **The live adapter has never touched a funded account.**
