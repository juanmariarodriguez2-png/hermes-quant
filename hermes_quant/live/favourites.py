"""The owner's basket of single stocks: optional, capped, and outside the model.

Why it is separate
------------------
The model holds companies through index funds because no rule for picking single
stocks beat the index when it was tested on the companies that were really in the
S&P 500 on each date (research/10_stock_sleeve.py). A list of favourites is therefore
a preference, not a finding, and it is treated as one:

* It is off until the owner switches it on (``HERMES_FAVOURITES=yes``).
* It never changes what the model does. The model's weights are computed first, from
  a panel the favourites are not part of; the basket only uses money the model
  leaves idle, and is scaled down when there is not enough of it.
* It is small by construction: at most ``max_stocks`` companies at a time, each
  ``per_stock`` of the account (2.5% x 4 = 10% as configured), and the file is refused
  if the two multiply to more than ``MAX_BUDGET``.
* A company is held only while the model's own trend filter says its trend is up,
  and only once it has traded long enough for that filter to mean something.

The rule (:func:`select`)
-------------------------
Decided once a week, with the model's target. A company comes in when its trend
probability is at least ``enter`` and goes when it falls below ``leave``; the gap
between the two stops a stock that hovers around 50% from being bought and sold
every week, which matters at 1% a side. A company already held keeps its seat while
it qualifies; free seats go to the strongest trends among the others. Every seat is
the same size.

The list and the two limits live in ``config/favourites.json``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..model import ModelConfig, compute_features
from ..universe import LIVE_BY_SYMBOL, Instrument, stock_instrument
from .market import PriceStore, build_live_panel

CONFIG = Path(__file__).resolve().parents[2] / "config" / "favourites.json"
MAX_BUDGET = 0.30            # the whole basket may never be configured above this share of the account
MAX_PER_STOCK = 0.10
SYMBOL = re.compile(r"^[A-Z]{1,6}$")


@dataclass(frozen=True)
class Basket:
    stocks: dict = field(default_factory=dict)     # symbol -> company name
    per_stock: float = 0.025                       # share of the account one company takes
    max_stocks: int = 4                            # companies held at the same time
    enter: float = 0.55                            # trend probability to come in...
    leave: float = 0.45                            # ...and below which a holding goes
    min_history_days: int = 180                    # calendar days of prices before the trend filter is trusted
    waiting: tuple = ()                            # names the owner wants that are not listed yet (shown, never traded)

    @property
    def budget(self) -> float:
        return self.per_stock * self.max_stocks


def load(path: Path | None = None) -> Basket:
    """Read and check the owner's list. Raises ``ValueError`` on anything doubtful: a
    typo in a limit must never become a position."""
    raw = json.loads(Path(path or CONFIG).read_text())
    stocks = {str(k).strip().upper(): str(v) for k, v in (raw.get("stocks") or {}).items()}
    known = set(Basket.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"favourites: unknown setting(s) {sorted(unknown)}")
    for symbol in stocks:
        if not SYMBOL.match(symbol):
            raise ValueError(f"favourites: {symbol!r} is not a ticker")
        if symbol in LIVE_BY_SYMBOL:
            raise ValueError(f"favourites: {symbol} is already one of the model's own markets")
    basket = Basket(stocks=stocks, **{k: raw[k] for k in ("per_stock", "max_stocks", "enter", "leave", "min_history_days") if k in raw},
                    waiting=tuple(str(w) for w in raw.get("waiting") or ()))
    if not (isinstance(basket.max_stocks, int) and 1 <= basket.max_stocks <= 10):
        raise ValueError("favourites: max_stocks must be a whole number from 1 to 10")
    if not (0.0 < float(basket.per_stock) <= MAX_PER_STOCK):
        raise ValueError(f"favourites: per_stock must be above 0 and at most {MAX_PER_STOCK:.0%} (0.025 means 2.5%)")
    if basket.budget > MAX_BUDGET + 1e-12:
        raise ValueError(f"favourites: per_stock x max_stocks is {basket.budget:.0%}; the limit is {MAX_BUDGET:.0%}")
    if not (0.0 < basket.leave <= basket.enter < 1.0):
        raise ValueError("favourites: need 0 < leave <= enter < 1")
    if basket.min_history_days < 90:
        raise ValueError("favourites: min_history_days below 90 leaves the trend filter with nothing to go on")
    return basket


def enabled() -> bool:
    return os.environ.get("HERMES_FAVOURITES", "no") == "yes"


def instruments(basket: Basket) -> list[Instrument]:
    return [stock_instrument(symbol) for symbol in basket.stocks]


def select(prob_up: dict[str, float], eligible: dict[str, bool], held: set[str], basket: Basket) -> dict[str, float]:
    """The rule: which companies the basket holds, and at what weight.

    ``prob_up`` is the model's trend probability per company, ``eligible`` whether it
    may be held at all (enough history, a market that really trades), ``held`` the
    companies in the basket's target so far."""
    ok = {s for s in basket.stocks if eligible.get(s) and np.isfinite(prob_up.get(s, np.nan))}
    keep = sorted((s for s in ok if s in held and prob_up[s] >= basket.leave), key=lambda s: -prob_up[s])[:basket.max_stocks]
    new = sorted((s for s in ok if s not in held and prob_up[s] >= basket.enter), key=lambda s: -prob_up[s])
    chosen = keep + new[:basket.max_stocks - len(keep)]
    return {s: float(basket.per_stock) for s in chosen}


def evaluate(store: PriceStore, basket: Basket, cfg: ModelConfig, bar_date: pd.Timestamp, held: set[str]) -> dict:
    """Score every favourite on ``bar_date`` and apply the rule. Uses the model's own
    volatility and trend filters on a panel of its own, so the model's panel (and with
    it every number the model computes) is untouched by this list."""
    listed = [i for i in instruments(basket) if len(store.load(i.symbol)[:bar_date])]
    rows = {s: {"symbol": s, "name": name, "pair": stock_instrument(s).kraken_pair, "status": "no_data"}
            for s, name in basket.stocks.items()}
    if not listed:
        return {"rows": list(rows.values()), "weights": {}, "last_close": {}}

    panel = build_live_panel(store, listed, end=bar_date)
    cfg = cfg.with_(min_history_days=basket.min_history_days, regime=False)
    features = compute_features(panel, cfg, p_turbulent=pd.DataFrame(index=panel.close.index))
    t = len(panel.close) - 1
    close = panel.close.ffill()
    prob = {s: float(features.prob_up[s].iloc[t]) for s in panel.close.columns}
    eligible = {s: bool(features.eligible[s].iloc[t]) for s in panel.close.columns}
    weights = select(prob, eligible, held, basket)

    seats_full = len(weights) >= basket.max_stocks
    for s in panel.close.columns:
        history = int(panel.close[s].notna().sum())
        liquid = bool(panel.investable[s].iloc[t])
        last = float(close[s].iloc[t])
        if s in weights:
            status = "picked"
        elif history < basket.min_history_days:
            status = "too_new"
        elif not liquid:
            status = "illiquid"
        elif not eligible[s] or not np.isfinite(prob[s]):
            status = "no_signal"
        elif prob[s] >= basket.enter and seats_full:
            status = "no_seat"
        else:
            status = "trend_down" if prob[s] < 0.5 else "trend_weak"
        back = lambda days: (round(last / float(close[s].iloc[t - days]) - 1.0, 4)            # noqa: E731
                             if t >= days and np.isfinite(close[s].iloc[t - days]) else None)
        rows[s].update(status=status, prob_up=round(prob[s], 4) if np.isfinite(prob[s]) else None,
                       vol=round(float(features.vol[s].iloc[t]), 4) if np.isfinite(features.vol[s].iloc[t]) else None,
                       history_days=history, days_to_wait=max(0, basket.min_history_days - history), liquid=liquid,
                       price=round(last, 6) if np.isfinite(last) else None, ret_30d=back(30), ret_180d=back(180),
                       weight=weights.get(s, 0.0))
    return {"rows": list(rows.values()), "weights": weights,
            "last_close": {s: float(close[s].iloc[t]) for s in panel.close.columns if np.isfinite(close[s].iloc[t])}}
