"""Synthetic market fixtures: tests never touch the network or downloaded data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_quant.data.panel import Panel  # noqa: E402


def make_panel(n_days: int = 1400, seed: int = 0, crypto: int = 5) -> Panel:
    """A small fake market: 3 funds that only trade on weekdays, a few 24/7 coins."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01", periods=n_days, freq="D")
    funds = ["SPY", "QQQ", "GLD"]
    coins = ["BTC", "ETH", "XRP", "SOL", "ADA"][:crypto]
    weekday = index.dayofweek < 5
    market = rng.normal(0.0004, 0.008, n_days)
    cols, tradable = {}, {}
    for name in funds:
        beta = {"SPY": 1.0, "QQQ": 1.2, "GLD": 0.0}[name]
        r = beta * market + rng.normal(0.0002, 0.006, n_days)
        r[~weekday] = 0.0
        cols[name] = 100.0 * np.exp(np.cumsum(r))
        tradable[name] = weekday
    crypto_factor = rng.normal(0.0008, 0.03, n_days)
    for name in coins:
        r = crypto_factor + rng.normal(0.0, 0.02, n_days)
        cols[name] = 1000.0 * np.exp(np.cumsum(r))
        tradable[name] = np.ones(n_days, dtype=bool)
    close = pd.DataFrame(cols, index=index)
    noise = pd.DataFrame(np.abs(rng.normal(0.0, 0.006, close.shape)), index=index, columns=close.columns)
    open_ = close.shift(1).fillna(close)
    high = np.maximum(open_, close) * (1.0 + noise)
    low = np.minimum(open_, close) * (1.0 - noise)
    trad = pd.DataFrame(tradable, index=index)
    for name in funds:  # flat candles on days the market is shut
        shut = ~trad[name]
        open_.loc[shut, name] = close.loc[shut, name]
        high.loc[shut, name] = close.loc[shut, name]
        low.loc[shut, name] = close.loc[shut, name]
    volume = pd.DataFrame(np.nan, index=index, columns=close.columns)
    for k, name in enumerate(coins):
        volume[name] = 1e9 / (k + 1) * (1.0 + 0.1 * rng.random(n_days))
    classes = {**{f: "equity" for f in ("SPY", "QQQ")}, "GLD": "real", **{c: "crypto" for c in coins}}
    return Panel(open_, high, low, close, close.copy(), volume, trad,
                 pd.Series(1.0, index=index), classes, pd.DataFrame(index=index))


@pytest.fixture(scope="session")
def panel() -> Panel:
    return make_panel()
