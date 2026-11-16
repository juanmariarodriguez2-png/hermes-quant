"""What the favourites basket would have done. Read the warning first.

The basket (hermes_quant/live/favourites.py) holds the owner's own list of companies:
Nvidia, Tesla, Apple, Alphabet, Microsoft, Amazon, Meta. Every one of them is on the
list because it is famous today, and it is famous today because its share price went up
enormously. A backtest of such a list answers "what if I had known in 2016 which
companies would win", which nobody could. The returns below are therefore NOT evidence
that the basket earns anything; research/10_stock_sleeve.py is the honest test of stock
picking, and it found nothing.

What this script can say honestly is how the *rule* behaves on real share prices: how
often it trades, what that costs at each venue, how much of the time it sits in cash and
how deep its falls were next to simply holding the same names. Those are properties of
the rule, not of the hindsight.

The rule is not re-implemented here: the script runs the live code (the same scoring and
the same ``select``), decides weekly, executes a day late, and sends the orders through
the same order planner as the nightly run. Nothing was tuned on this output: the two
thresholds, the seat count and the six-month wait were fixed before it was first run.

    python research/12_favourites_basket.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import ROOT

from hermes_quant.backtest.metrics import summary
from hermes_quant.live import favourites
from hermes_quant.live.market import PriceStore
from hermes_quant.model import PROFILES
from hermes_quant.portfolio.execution import CostModel, TradingRules, plan_orders, trading_costs

START = "2016-01-01"                      # the store begins in 2014; the filters get a year to warm up
CAPITAL = 10_000.0                        # large enough that the minimum ticket never binds
ORDER_BOOK = CostModel()                  # 0.18% a side + 0.20% currency conversion: Kraken's order book
CONVERT = CostModel(fee={"stock": 0.0107}, slippage={"stock": 0.0}, fx_fee=0.0)     # 1.07% a side, all in: Kraken's Convert
SP500 = "SPY"


def simulate(panel, decide, costs: CostModel, rules: TradingRules, start: str) -> dict:
    """An account that holds nothing but what ``decide(day)`` asks for. Cash earns nothing."""
    index, symbols = panel.close.index, list(panel.close.columns)
    price = panel.close.to_numpy(dtype=float)
    can_trade = panel.tradable.to_numpy(dtype=bool)
    classes = [panel.classes[s] for s in symbols]
    first = int(index.searchsorted(pd.Timestamp(start)))
    units, cash, target = np.zeros(len(symbols)), CAPITAL, np.zeros(len(symbols))
    equity, invested, paid, traded = [], [], 0.0, 0.0
    entries = {s: 0 for s in symbols}
    held_days = {s: 0 for s in symbols}
    for t in range(first, len(index)):
        px = price[t]
        valid = np.isfinite(px) & (px > 0)
        value = np.where(valid, units * px, 0.0)
        decision = t - rules.lag
        if decision >= first and (decision - first) % rules.rebalance_every == 0:
            before = target > 0
            wanted = decide(decision)
            target = np.array([wanted.get(s, 0.0) for s in symbols])
            for s, was, now in zip(symbols, before, target > 0):
                entries[s] += int(now and not was)
        order = plan_orders(target, value, cash, can_trade[t] & valid, classes, costs, rules)
        if np.any(order != 0):
            fees, fx_cost = trading_costs(order, classes, costs)
            units = units + np.where(order != 0, order / np.where(valid, px, 1.0), 0.0)
            cash -= float(order.sum()) + float(fees.sum()) + fx_cost
            paid += float(fees.sum()) + fx_cost
            traded += float(np.abs(order).sum())
        value = np.where(valid, units * px, 0.0)
        nav = cash + float(value.sum())
        equity.append(nav)
        invested.append(float(value.sum()) / nav)
        for s, v in zip(symbols, value):
            held_days[s] += int(v > 0.01 * nav * 0.1)
    eq = pd.Series(equity, index=index[first:])
    years = len(eq) / 365.0
    return {"equity": eq, "invested": float(np.mean(invested)), "cost_per_year": paid / years / CAPITAL,
            "traded_per_year": traded / years / CAPITAL, "entries_per_year": {s: n / years for s, n in entries.items()},
            "held_share": {s: d / len(eq) for s, d in held_days.items()}}


def on_budget(eq: pd.Series, budget: float) -> pd.Series:
    """Growth of the money allotted to the basket: the account's daily gain or loss over
    the budget's share of it. Seats left empty count as cash, as they are."""
    r = eq.pct_change().fillna(0.0) / budget
    return (1.0 + r).cumprod()


def describe(label: str, run: dict, budget: float) -> dict:
    pot = on_budget(run["equity"], budget)
    s = summary(pot)
    out = {"cagr": s["cagr"], "vol": s["vol"], "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"],
           "share_of_budget_invested": run["invested"] / budget, "cost_per_year_on_budget": run["cost_per_year"] / budget,
           "traded_per_year_on_budget": run["traded_per_year"] / budget,
           "account_cagr": summary(run["equity"])["cagr"], "account_max_drawdown": summary(run["equity"])["max_drawdown"]}
    print(f"  {label:44s} {out['cagr']:+7.1%} a year  vol {out['vol']:5.1%}  Sharpe {out['sharpe']:5.2f}  worst fall {out['max_drawdown']:+6.1%}  "
          f"invested {out['share_of_budget_invested']:4.0%}  costs {out['cost_per_year_on_budget']:.2%}/yr  "
          f"| on the whole account: {out['account_cagr']:+.2%} a year, worst fall {out['account_max_drawdown']:+.1%}")
    return out


def main() -> None:
    store = PriceStore(ROOT / "data" / "live" / "prices")
    cfg = PROFILES["balanced"]
    basket = favourites.load()
    last = store.load("BTC").index[-1]
    seasoned = {s: n for s, n in basket.stocks.items() if len(store.load(s)[:START]) > 250}     # SpaceX has no history yet
    basket = favourites.Basket(**{**basket.__dict__, "stocks": seasoned})
    panel, features = favourites.score(store, basket, cfg, last)
    symbols = list(panel.close.columns)
    rules = TradingRules()
    print(f"{', '.join(seasoned.values())}: {START} -> {last:%Y-%m-%d}. Names chosen with hindsight; see the warning in this file.\n")

    held: set[str] = set()

    def rule(t: int) -> dict:
        nonlocal held
        prob = {s: float(features.prob_up[s].iloc[t]) for s in symbols}
        eligible = {s: bool(features.eligible[s].iloc[t]) for s in symbols}
        weights = favourites.select(prob, eligible, held, basket)
        held = set(weights)
        return weights

    everything = {s: basket.budget / len(symbols) for s in symbols}          # the same budget, no rule: all seven, always
    out = {"window": {"start": START, "end": f"{last:%Y-%m-%d}"}, "names": seasoned,
           "basket": {"per_stock": basket.per_stock, "max_stocks": basket.max_stocks, "enter": basket.enter, "leave": basket.leave},
           "warning": "names chosen with hindsight: the returns are not evidence", "cases": {}}
    for venue, costs in (("order book (0.38% a side)", ORDER_BOOK), ("Convert (1.07% a side)", CONVERT)):
        print(venue)
        held = set()
        run = simulate(panel, rule, costs, rules, START)
        hold = simulate(panel, lambda t: everything, costs, rules, START)
        out["cases"][venue] = {"rule": describe("the basket rule (4 seats, trend up only)", run, basket.budget),
                               "hold_all": describe("the same seven, always held", hold, basket.budget),
                               "entries_per_stock_per_year": float(np.mean(list(run["entries_per_year"].values()))),
                               "held_share": run["held_share"]}
        print(f"    a company comes into the basket {out['cases'][venue]['entries_per_stock_per_year']:.1f} times a year on average; held share of the time: "
              + ", ".join(f"{s} {v:.0%}" for s, v in run["held_share"].items()) + "\n")

    spy = store.load(SP500)["close"] / store.load("EURUSD")["close"].reindex(store.load(SP500).index).ffill()
    spy = spy.loc[START:last]
    s = summary((spy / spy.iloc[0]).reindex(pd.date_range(spy.index[0], last, freq="D")).ffill())
    out["sp500"] = {"cagr": s["cagr"], "vol": s["vol"], "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"]}
    print(f"  for scale, the S&P 500 fund in euros over the same years: {s['cagr']:+.1%} a year, vol {s['vol']:.1%}, worst fall {s['max_drawdown']:+.1%}")
    (ROOT / "research" / "results" / "favourites_basket.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
