"""Event-driven daily backtester with venue-realistic frictions.

What is modelled
----------------
* Decision at the close of day d from data up to d; execution at the close of day
  d + ``lag`` (lag = 1 by default: a full day late, deliberately pessimistic).
* Fees per asset class, taken from Kraken's published schedule for a small account,
  plus slippage, plus a currency-conversion charge on euro <-> dollar flows.
* A no-trade band and a minimum ticket size: with a small account most "optimal"
  rebalances are not worth their commission, and the simulator knows it.
* Assets that cannot trade that day (a fund on a weekend) are simply retried later.
* Holdings whose price history stops (delisting) are liquidated at the last print
  once the price has been missing for several days.
* Optional daily carry (perpetual-futures funding) for the derivatives variant.

Cash earns nothing, as on the exchange.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..model import Features, ModelConfig, target_weights
from ..portfolio.execution import CostModel, TradingRules, plan_orders, trading_costs

__all__ = ["BacktestResult", "CostModel", "TradingRules", "run_backtest", "static_mix"]


@dataclass
class BacktestResult:
    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series                    # traded notional / NAV, per day
    costs: pd.Series                       # euros paid per day (fees + slippage + FX)
    carry: pd.Series                       # euros of funding paid(-)/received(+) per day
    trades: pd.DataFrame
    targets: pd.DataFrame

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)


def run_backtest(panel: Panel, features: Features, cfg: ModelConfig,
                 costs: CostModel | None = None, rules: TradingRules | None = None,
                 capital: float = 236.0, start: str | None = None, end: str | None = None,
                 carry: pd.DataFrame | None = None) -> BacktestResult:
    """Simulate the model. ``carry`` (optional) holds daily rates paid by a long
    position (positive = longs pay, shorts receive), aligned to the panel."""
    costs = costs or CostModel()
    rules = rules or TradingRules()
    index = panel.close.index
    symbols = list(panel.close.columns)
    n = len(symbols)
    price = panel.close.to_numpy(dtype=float)
    can_trade = panel.tradable.to_numpy(dtype=bool)
    cls = [panel.classes[s] for s in symbols]
    rate = np.array([costs.rate(c) for c in cls])
    carry_rate = None
    if carry is not None:
        carry_rate = carry.reindex(index=index, columns=symbols).fillna(0.0).to_numpy(dtype=float)

    first = int(index.searchsorted(pd.Timestamp(start))) if start else 0
    last = int(index.searchsorted(pd.Timestamp(end), side="right")) if end else len(index)

    units = np.zeros(n)
    cash = float(capital)
    last_price = np.full(n, np.nan)
    missing = np.zeros(n, dtype=int)
    target = np.zeros(n)
    equity = np.full(len(index), np.nan)
    weights_hist = np.zeros((len(index), n))
    targets_hist = np.full((len(index), n), np.nan)
    turnover = np.zeros(len(index))
    cost_paid = np.zeros(len(index))
    carry_paid = np.zeros(len(index))
    trade_log: list[tuple] = []

    for t in range(first, last):
        px = price[t]
        valid = np.isfinite(px) & (px > 0)
        last_price[valid] = px[valid]
        missing[valid] = 0
        missing[~valid] += 1

        # price history has stopped while we hold the asset -> forced liquidation
        gone = (missing >= rules.stale_days) & (units != 0) & np.isfinite(last_price)
        for i in np.flatnonzero(gone):
            quoted = units[i] * last_price[i]
            proceeds = quoted * (1.0 - rules.delist_haircut)
            fee = abs(proceeds) * (rate[i] + (costs.fx_fee if cls[i] in costs.usd_classes else 0.0))
            cash += proceeds - fee
            cost_paid[t] += fee + (quoted - proceeds)        # the write-down is a cost too
            trade_log.append((index[t], symbols[i], -units[i], last_price[i], -proceeds, fee, "delisted"))
            units[i] = 0.0

        mark = np.where(valid, px, np.nan_to_num(last_price))
        position_value = units * mark

        # funding on open positions accrues before today's trades
        if carry_rate is not None:
            flow = -float((position_value * carry_rate[t]).sum())
            cash += flow
            carry_paid[t] = flow

        nav = cash + position_value.sum()

        # refresh the target from a decision taken `lag` days ago
        decision = t - rules.lag
        if decision >= first and (decision - first) % rules.rebalance_every == 0:
            target = target_weights(features, cfg, decision).weights.to_numpy(dtype=float)
        targets_hist[t] = target

        # orders: only where the deviation is worth the commission
        order = plan_orders(target, position_value, cash, can_trade[t] & valid, cls, costs, rules)
        if np.any(order != 0):
            fees, fx_cost = trading_costs(order, cls, costs)
            for i in np.flatnonzero(order):
                units[i] += order[i] / px[i]
                if abs(units[i] * px[i]) < 0.01:
                    units[i] = 0.0
                trade_log.append((index[t], symbols[i], order[i] / px[i], px[i], order[i], fees[i], "rebalance"))
            cash -= float(order.sum()) + float(fees.sum()) + fx_cost
            cost_paid[t] += float(fees.sum()) + fx_cost
            turnover[t] = float(np.abs(order).sum()) / nav if nav > 0 else 0.0

        position_value = units * mark
        nav = cash + position_value.sum()
        equity[t] = nav
        weights_hist[t] = position_value / nav if nav > 0 else 0.0

    sl = slice(first, last)
    trades = pd.DataFrame(trade_log, columns=["date", "symbol", "units", "price", "notional", "cost", "reason"])
    return BacktestResult(
        equity=pd.Series(equity[sl], index=index[sl], name="equity"),
        weights=pd.DataFrame(weights_hist[sl], index=index[sl], columns=symbols),
        turnover=pd.Series(turnover[sl], index=index[sl]),
        costs=pd.Series(cost_paid[sl], index=index[sl]),
        carry=pd.Series(carry_paid[sl], index=index[sl]),
        trades=trades,
        targets=pd.DataFrame(targets_hist[sl], index=index[sl], columns=symbols),
    )


def static_mix(panel: Panel, weights: dict[str, float], start: str, end: str | None = None,
               rebalance_days: int | None = None) -> pd.Series:
    """Frictionless static-mix benchmark (weights sum to <= 1; the rest sits in cash).

    Every asset must already be trading on the first day: a benchmark may not own
    something before it exists."""
    close = panel.close.loc[start:end, list(weights)].ffill()
    if close.iloc[0].isna().any():
        missing = list(close.columns[close.iloc[0].isna()])
        raise ValueError(f"static_mix: no price on {close.index[0].date()} for {missing}")
    r = close.pct_change().fillna(0.0).to_numpy()
    w0 = np.array(list(weights.values()), dtype=float)
    w = w0.copy()
    cash_w = 1.0 - w.sum()
    nav = np.empty(len(close))
    value = 1.0
    for t in range(len(close)):
        grown = w * (1.0 + r[t])
        total = grown.sum() + cash_w
        value *= total
        w, cash_w = grown / total, cash_w / total
        if rebalance_days and t % rebalance_days == 0:
            w, cash_w = w0.copy(), 1.0 - w0.sum()
        nav[t] = value
    return pd.Series(nav, index=close.index)
