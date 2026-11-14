"""Live market data: a small on-disk price store fed by Kraken, and the panel built from it.

The store is one CSV per instrument under ``data/live/prices`` in the instrument's
own quote currency. Kraken only serves ~720 daily candles, so the files are seeded
once with longer public history and then extended every day; candles re-served by
Kraken overwrite the stored ones, so late corrections are picked up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import kraken
from ..data.panel import Panel
from ..regime import hmm
from ..universe import Instrument

COLUMNS = ["open", "high", "low", "close", "volume"]
FX_SYMBOL = "EURUSD"


class PriceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, symbol: str) -> pd.DataFrame:
        path = self.root / f"{symbol}.csv"
        if not path.exists():
            return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"))
        return pd.read_csv(path, index_col="date", parse_dates=True)[COLUMNS]

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        df = df[COLUMNS].sort_index()
        df.index.name = "date"
        df.to_csv(self.root / f"{symbol}.csv", float_format="%.10g", date_format="%Y-%m-%d")

    def merge(self, symbol: str, fresh: pd.DataFrame) -> pd.DataFrame:
        """Fresh candles win over stored ones on the same date."""
        old = self.load(symbol)
        fresh = fresh[COLUMNS].dropna(subset=["close"])
        combined = pd.concat([old[~old.index.isin(fresh.index)], fresh]).sort_index()
        self.save(symbol, combined)
        return combined


def refresh_prices(store: PriceStore, instruments: list[Instrument]) -> dict[str, str]:
    """Pull the latest completed daily candles for every instrument. Returns any
    per-symbol errors instead of raising: one broken market must not stop the run."""
    errors: dict[str, str] = {}
    for symbol, pair, tokenized in [(FX_SYMBOL, "EURUSD", False)] + [
        (i.symbol, i.kraken_pair, i.tokenized) for i in instruments
    ]:
        try:
            store.merge(symbol, kraken.ohlc_daily(pair, tokenized))
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            errors[symbol] = repr(exc)[:200]
    return errors


MIN_DAILY_USD = 2000.0        # median traded value a tokenized market must show to be held
MAX_DEAD_SESSIONS = 0.20      # ...and at most this share of recent sessions without a trade
GATE_SESSIONS = 30


def venue_liquidity(df: pd.DataFrame, kraken_from: pd.Timestamp | None) -> pd.Series:
    """Is this tokenized market really trading? Judged on its last 30 weekday candles.
    Before Kraken listed it (seeded history) the answer is irrelevant and set to True."""
    weekdays = df[df.index.dayofweek < 5]
    traded = (weekdays["volume"] * weekdays["close"]).fillna(0.0)
    median = traded.rolling(GATE_SESSIONS, min_periods=10).median()
    dead = (weekdays["volume"].fillna(0.0) <= 0).rolling(GATE_SESSIONS, min_periods=10).mean()
    ok = (median >= MIN_DAILY_USD) & (dead <= MAX_DEAD_SESSIONS)
    if kraken_from is not None:
        ok[ok.index < kraken_from + pd.Timedelta(days=45)] = True
    return ok


def build_live_panel(store: PriceStore, instruments: list[Instrument],
                     start: str = "2015-01-01", listed_from: dict[str, pd.Timestamp] | None = None) -> Panel:
    """Same conventions as the research panel: calendar days, euro prices, flat
    candles (and no trading) for tokenized stocks while their market is shut.

    Tokenized markets additionally pass through a liquidity gate: several of them are
    listed but barely trade, and a stale price looks deceptively calm to a risk model."""
    fx_raw = store.load(FX_SYMBOL)["close"]
    frames = {i.symbol: store.load(i.symbol) for i in instruments}
    last = max(f.index[-1] for f in frames.values() if len(f))
    index = pd.date_range(start, last, freq="D")
    fx = fx_raw.reindex(index).ffill()

    fields: dict[str, dict[str, pd.Series]] = {f: {} for f in ("open", "high", "low", "close", "usd", "vol")}
    tradable, classes, investable = {}, {}, {}
    for inst in instruments:
        gate = pd.Series(True, index=index)
        if inst.tokenized:
            since = (listed_from or {}).get(inst.symbol)
            gate = venue_liquidity(frames[inst.symbol], since).reindex(index).ffill().fillna(True).astype(bool)
        investable[inst.symbol] = gate
        df = frames[inst.symbol].reindex(index)
        if inst.tokenized:
            session = (index.dayofweek < 5) & df["close"].notna().to_numpy()   # 24/5 market
        else:
            session = df["close"].notna().to_numpy()
        close = df["close"].where(session).ffill()
        ohl = {f: df[f].where(session, close) for f in ("open", "high", "low")}
        to_eur = (1.0 / fx) if inst.quote == "USD" else 1.0
        to_usd = 1.0 if inst.quote == "USD" else fx
        fields["close"][inst.symbol] = close * to_eur
        for f in ("open", "high", "low"):
            fields[f][inst.symbol] = ohl[f] * to_eur
        fields["usd"][inst.symbol] = close * to_usd
        volume = (df["volume"] * close * to_usd) if inst.asset_class == "crypto" else df["volume"] * np.nan
        fields["vol"][inst.symbol] = volume
        tradable[inst.symbol] = pd.Series(session, index=index)
        classes[inst.symbol] = inst.asset_class

    frame = lambda key: pd.DataFrame(fields[key], index=index)  # noqa: E731
    return Panel(frame("open"), frame("high"), frame("low"), frame("close"), frame("usd"),
                 frame("vol"), pd.DataFrame(tradable, index=index), fx, classes,
                 pd.DataFrame(index=index), pd.DataFrame(investable, index=index))


def live_regime(panel: Panel, regime_proxy: dict[str, str], max_window: int = 3650,
                tail: int = 150) -> pd.DataFrame:
    """P(turbulent) for the most recent days. The HMM is fitted on everything up to
    the day before the last candle and then filtered forward, mirroring the
    research procedure (fit on the past, infer today)."""
    out = {}
    for sleeve, proxy in regime_proxy.items():
        if proxy not in panel.close_usd.columns:
            continue
        usd = panel.close_usd[proxy]
        r = np.log(usd / usd.shift(1)).where(panel.tradable[proxy]).to_numpy()
        n = len(r)
        lo = max(0, n - max_window)
        train = r[lo:n - 1]
        train = train[np.isfinite(train)]
        series = np.full(n, np.nan)
        if len(train) >= 500:
            params = hmm.fit(train, 2)
            probs = hmm.filtered_probabilities(r[lo:], params)[:, -1]
            series[max(lo, n - tail):] = probs[-min(tail, n - lo):]
        out[sleeve] = pd.Series(series, index=panel.close.index)
    return pd.DataFrame(out)
