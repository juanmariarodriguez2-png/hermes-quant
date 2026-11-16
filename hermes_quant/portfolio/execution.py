"""Order planning shared by the backtester and the live runner.

Turning "target weights" into orders is where small accounts quietly lose money:
every tiny adjustment pays a commission. The rules here are deliberately lazy.

* No-trade band. Nothing happens until a position is further from its target than
  max(absolute band, relative band x target).
* Partial rebalancing. When a trade is triggered it only goes part of the way back,
  which is the optimal policy when costs are proportional to size.
* Minimum ticket. Orders below the exchange minimum (or not worth the fee) are dropped.
* Cash discipline. Sells are assumed to fill first; buys are scaled so that cash can
  never go negative once fees and currency conversion are paid.

Because both simulation and production call :func:`plan_orders`, a backtested rule
and a live rule cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Per-side trading frictions, as fractions of traded notional.

    Defaults are Kraken's published schedule for a small account (Nov 2026):
    crypto spot 0.80% taker, tokenized stocks 0.08% taker, FX conversion 0.20%.
    "stock" is the owner's optional basket of single stocks: the same tokenized venue
    as the funds, so the same charges. The model itself never holds that class.
    """

    fee: dict = field(default_factory=lambda: {"crypto": 0.0080, "equity": 0.0008, "real": 0.0008, "stock": 0.0008})
    slippage: dict = field(default_factory=lambda: {"crypto": 0.0005, "equity": 0.0010, "real": 0.0010, "stock": 0.0010})
    fx_fee: float = 0.0020                 # EUR <-> USD conversion for dollar-quoted assets
    usd_classes: tuple = ("equity", "real", "stock")

    def rate(self, cls: str) -> float:
        return self.fee.get(cls, 0.0) + self.slippage.get(cls, 0.0)


@dataclass(frozen=True)
class TradingRules:
    lag: int = 1                           # backtest only: days between decision and execution
    rebalance_every: int = 7               # days between target updates
    # single stocks are held in small slices (2-3% of the account), so their band is narrower
    band_abs: dict = field(default_factory=lambda: {"crypto": 0.02, "equity": 0.02, "real": 0.02, "stock": 0.01})
    band_rel: dict = field(default_factory=lambda: {"crypto": 0.25, "equity": 0.25, "real": 0.25, "stock": 0.25})
    min_ticket: float = 4.0                # euros; Kraken's minimum order is ~3.5-4 EUR
    min_exit_ticket: float = 1.0           # full exits may be smaller
    edge_fraction: float = 0.5             # after trading, stay this far inside the band (0 = at target)
    stale_days: int = 5                    # price missing this long -> treat as delisted
    delist_haircut: float = 0.50           # assume half the last quoted value is lost on a forced exit
    allow_leverage: bool = False           # True for margin/perpetual books


def plan_orders(target: np.ndarray, position_value: np.ndarray, cash: float, tradable: np.ndarray,
                classes: list[str], costs: CostModel, rules: TradingRules, outside: float = 0.0) -> np.ndarray:
    """Signed euro notional to trade per asset (positive = buy).

    Parameters
    ----------
    target : target weights (fractions of net asset value).
    position_value : current euro value of each holding.
    cash : free euros.
    tradable : whether each asset can be traded right now (market open, price valid).
    outside : euros the account holds in things that are not on this list (never traded
        here). They are part of what the account is worth, so targets are fractions of
        a total that includes them. Zero in the backtest and on Kraken.
    """
    target = np.asarray(target, dtype=float)
    position_value = np.asarray(position_value, dtype=float)
    nav = cash + float(position_value.sum()) + float(outside)
    n = len(target)
    if nav <= 0:
        return np.zeros(n)

    rate = np.array([costs.rate(c) for c in classes])
    is_usd = np.array([c in costs.usd_classes for c in classes])
    band_abs = np.array([rules.band_abs.get(c, 0.02) for c in classes])
    band_rel = np.array([rules.band_rel.get(c, 0.25) for c in classes])

    delta_w = target - position_value / nav
    threshold = np.maximum(band_abs, band_rel * np.abs(target))
    exiting = (target == 0) & (position_value != 0)
    wants = (np.abs(delta_w) > threshold) | exiting
    partial = delta_w - np.sign(delta_w) * rules.edge_fraction * threshold
    desired = np.where(exiting, delta_w, partial)
    order = np.where(wants & np.asarray(tradable, dtype=bool), desired * nav, 0.0)
    floor = np.where(exiting, rules.min_exit_ticket, rules.min_ticket)
    order[np.abs(order) < floor] = 0.0

    if np.any(order != 0) and not rules.allow_leverage:
        sells = np.minimum(order, 0.0)
        buys = np.maximum(order, 0.0)
        # reserve for the worst case: scaling the buys below can change the net dollar flow
        fx_reserve = float((np.abs(order) * is_usd).sum()) * costs.fx_fee
        budget = cash - float(sells.sum()) - float((-sells * rate).sum()) - fx_reserve
        buy_total = float((buys * (1.0 + rate)).sum())
        if budget <= 0:
            buys[:] = 0.0
        elif buy_total > budget:
            buys *= budget / buy_total               # never spend cash we lack
        order = sells + buys
        order[np.abs(order) < floor] = 0.0
    return order


def trading_costs(order: np.ndarray, classes: list[str], costs: CostModel) -> tuple[np.ndarray, float]:
    """Per-asset fees+slippage and the currency-conversion charge for a set of orders."""
    rate = np.array([costs.rate(c) for c in classes])
    is_usd = np.array([c in costs.usd_classes for c in classes])
    fees = np.abs(order) * rate
    fx_cost = abs(float((order * is_usd).sum())) * costs.fx_fee
    return fees, fx_cost
