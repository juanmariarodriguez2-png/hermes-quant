"""Builds one aligned, euro-denominated research panel from the raw downloads.

Conventions
-----------
* Index: every calendar day (UTC). Crypto trades 24/7; exchange-traded funds are
  forward-filled on days their market is shut, so their return is 0 on those days.
* A row dated D contains only information known by the end of D (UTC): crypto
  closes at 24:00 UTC, US-listed funds at 20:00/21:00 UTC the same day.
* Prices are converted to EUR, the account's currency, so that signals, risk and
  P&L are all measured in what the investor actually experiences (FX included).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..universe import CRYPTO_EXCLUDE, ETF_EXTENDED, ETF_RESEARCH, asset_class

FIELDS = ("open", "high", "low", "close")


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame            # EUR
    close_usd: pd.DataFrame
    dollar_volume: pd.DataFrame    # USD traded per day (crypto only; NaN elsewhere)
    tradable: pd.DataFrame         # bool: could an order be executed on that day?
    eurusd: pd.Series              # USD per EUR
    classes: dict[str, str]
    funding: pd.DataFrame          # daily perp funding rate paid by longs (crypto, 2020+)
    investable: pd.DataFrame | None = None   # optional venue gate (live): False = do not hold

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def log_returns(self) -> pd.DataFrame:
        return np.log(self.close / self.close.shift(1))

    def simple_returns(self) -> pd.DataFrame:
        return self.close / self.close.shift(1) - 1.0


def _split_on_gaps(df: pd.DataFrame, symbol: str, max_gap_days: int = 5) -> dict[str, pd.DataFrame]:
    """A long hole in a symbol's history means it is a *different* listing afterwards
    (Terra's LUNA ticker was reused for a new coin; FTT was delisted and relisted)."""
    gaps = np.flatnonzero(np.diff(df.index.values).astype("timedelta64[D]").astype(int) > max_gap_days)
    if len(gaps) == 0:
        return {symbol: df}
    parts, start = {}, 0
    for n, g in enumerate(list(gaps) + [len(df) - 1]):
        name = symbol if n == 0 else f"{symbol}.{n + 1}"
        parts[name] = df.iloc[start:g + 1]
        start = g + 1
    return parts


def _wide(frames: dict[str, pd.DataFrame], field: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({s: f[field] for s, f in frames.items()}).reindex(index)


def _ever_liquid(crypto: dict[str, pd.DataFrame], index: pd.DatetimeIndex, keep_top: int,
                 min_history: int, window: int) -> list[str]:
    """Coins that ranked in the top ``keep_top`` by trailing traded volume on at least
    one day, judged day by day with the information available that day. Everything
    else can never be selected by the model, so dropping it only saves time."""
    volume = pd.DataFrame({s: f["quote_volume"] for s, f in crypto.items()}).reindex(index)
    close = pd.DataFrame({s: f["close"] for s, f in crypto.items()}).reindex(index)
    seasoned = (close.notna().cumsum() >= min_history) & close.notna()
    liquidity = volume.rolling(window, min_periods=window // 2).median().where(seasoned)
    rank = liquidity.rank(axis=1, ascending=False, method="first")
    return list(rank.columns[(rank <= keep_top).any()])


def build_panel(research_dir: str | Path, start: str = "2015-07-20", end: str | None = None,
                crypto_keep_top: int = 12, min_history: int = 400, liquidity_window: int = 90,
                extended_funds: bool = False) -> Panel:
    """Assemble the research panel.

    Crypto comes from Binance's complete USDT listing history when that file is
    present (``binance_klines_full.parquet``: every market ever listed, including the
    delisted ones), otherwise from the smaller hand-picked download.
    """
    research_dir = Path(research_dir)
    full = research_dir / "binance_klines_full.parquet"
    binance = pd.read_parquet(full if full.exists() else research_dir / "binance_klines.parquet")
    coinbase = pd.read_parquet(research_dir / "coinbase_klines.parquet")
    yahoo = pd.read_parquet(research_dir / "yahoo_daily.parquet")
    funding_path = research_dir / "binance_funding.parquet"
    funding = pd.read_parquet(funding_path) if funding_path.exists() else pd.DataFrame()

    # ---- crypto (USD) -------------------------------------------------------
    crypto: dict[str, pd.DataFrame] = {}
    for symbol, df in binance.groupby("symbol"):
        if symbol in CRYPTO_EXCLUDE:
            continue
        df = df.drop(columns="symbol").sort_index()
        if symbol in ("BTC", "ETH"):  # extend back with Coinbase before Binance listed it
            early = coinbase[coinbase["symbol"] == symbol].drop(columns="symbol").sort_index()
            early = early[early.index < df.index[0]]
            df = pd.concat([early[df.columns.intersection(early.columns)], df]).sort_index()
        crypto.update(_split_on_gaps(df, symbol))

    last_day = max(f.index[-1] for f in crypto.values())
    index = pd.date_range(start, end or last_day, freq="D")
    keep = set(_ever_liquid(crypto, index, crypto_keep_top, min_history, liquidity_window))
    crypto = {s: f for s, f in crypto.items() if s in keep}

    # ---- funds (USD) --------------------------------------------------------
    funds: dict[str, pd.DataFrame] = {}
    session: dict[str, pd.Series] = {}
    fund_list = list(ETF_RESEARCH) + (list(ETF_EXTENDED) if extended_funds else [])
    for symbol in fund_list:
        df = yahoo[yahoo["symbol"] == symbol].drop(columns="symbol").sort_index()
        df = df[(df["close"] > 0) & df["close"].notna()]
        session[symbol] = pd.Series(True, index=df.index).reindex(index, fill_value=False)
        full = df.reindex(index)
        filled_close = full["close"].ffill()
        for field in ("open", "high", "low"):  # flat candle when the market is shut
            full[field] = full[field].where(session[symbol], filled_close)
        full["close"] = filled_close
        funds[symbol] = full

    fx = yahoo[yahoo["symbol"] == "EURUSD=X"]["close"].sort_index()
    fx = fx[fx > 0].reindex(index).ffill()        # forward only: never borrow a later rate

    frames = {**{s: f.reindex(index) for s, f in crypto.items()}, **funds}
    usd = {field: _wide(frames, field, index) for field in FIELDS}
    eur = {field: usd[field].div(fx, axis=0) for field in FIELDS}

    dollar_volume = pd.DataFrame({s: f["quote_volume"] for s, f in crypto.items()}).reindex(index)
    dollar_volume = dollar_volume.reindex(columns=eur["close"].columns)

    tradable = eur["close"].notna()
    for symbol, open_days in session.items():
        tradable[symbol] = tradable[symbol] & open_days

    classes = {s: asset_class(s.split(".")[0]) for s in eur["close"].columns}
    funding = funding.reindex(index) if len(funding) else pd.DataFrame(index=index)
    return Panel(eur["open"], eur["high"], eur["low"], eur["close"], usd["close"],
                 dollar_volume, tradable, fx, classes, funding)
