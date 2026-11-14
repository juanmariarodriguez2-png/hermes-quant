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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..model import Features, ModelConfig, target_weights


@dataclass(frozen=True)
class CostModel:
    """Per-side trading frictions, as fractions of traded notional."""

    fee: dict = field(default_factory=lambda: {"crypto": 0.0080, "equity": 0.0008, "real": 0.0008})
    slippage: dict = field(default_factory=lambda: {"crypto": 0.0005, "equity": 0.0010, "real": 0.0010})
    fx_fee: float = 0.0020                 # EUR <-> USD conversion for dollar-quoted assets
    usd_classes: tuple = ("equity", "real")

    def rate(self, cls: str) -> float:
        return self.fee.get(cls, 0.0) + self.slippage.get(cls, 0.0)


@dataclass(frozen=True)
class TradingRules:
    lag: int = 1                           # days between decision and execution
    rebalance_every: int = 7               # days between target updates
    band_abs: dict = field(default_factory=lambda: {"crypto": 0.02, "equity": 0.02, "real": 0.02})
    band_rel: dict = field(default_factory=lambda: {"crypto": 0.25, "equity": 0.25, "real": 0.25})
    min_ticket: float = 4.0                # euros; Kraken's minimum order is ~3.5-4 EUR
    min_exit_ticket: float = 1.0           # full exits may be smaller
    edge_fraction: float = 0.5             # after trading, stay this far inside the band (0 = at target)
    stale_days: int = 5                    # price missing this long -> treat as delisted
    delist_haircut: float = 0.0
    allow_leverage: bool = False           # True for margin/perpetual books


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
    is_usd = np.array([c in costs.usd_classes for c in cls])
    band_abs = np.array([rules.band_abs.get(c, 0.02) for c in cls])
    band_rel = np.array([rules.band_rel.get(c, 0.20) for c in cls])
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
            proceeds = units[i] * last_price[i] * (1.0 - rules.delist_haircut)
            fee = abs(proceeds) * rate[i]
            cash += proceeds - fee
            cost_paid[t] += fee
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
        current_w = position_value / nav if nav > 0 else np.zeros(n)
        delta_w = target - current_w
        threshold = np.maximum(band_abs, band_rel * np.abs(target))
        exiting = (target == 0) & (units != 0)
        wants = (np.abs(delta_w) > threshold) | exiting
        # with proportional costs it is optimal to trade only part of the way back
        partial = delta_w - np.sign(delta_w) * rules.edge_fraction * threshold
        desired = np.where(exiting, delta_w, partial)
        order = np.where(wants & can_trade[t] & valid, desired * nav, 0.0)
        floor = np.where(exiting, rules.min_exit_ticket, rules.min_ticket)
        order[np.abs(order) < floor] = 0.0

        if np.any(order != 0):
            if not rules.allow_leverage:
                sells = np.minimum(order, 0.0)
                buys = np.maximum(order, 0.0)
                fx_reserve = abs(float((order * is_usd).sum())) * costs.fx_fee
                budget = cash - float(sells.sum()) - float((-sells * rate).sum()) - fx_reserve
                buy_total = float((buys * (1.0 + rate)).sum())
                if budget <= 0:
                    buys[:] = 0.0
                elif buy_total > budget:
                    buys *= budget / buy_total               # never spend cash we lack
                order = sells + buys
                order[np.abs(order) < floor] = 0.0
            fees = np.abs(order) * rate
            fx_cost = abs(float((order * is_usd).sum())) * costs.fx_fee
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
    """Frictionless static-mix benchmark (weights sum to <= 1; the rest sits in cash)."""
    close = panel.close.loc[start:end, list(weights)].ffill()
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
