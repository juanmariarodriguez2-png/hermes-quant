"""One-off: seed the live price store with long history.

Kraken serves only ~720 daily candles (and tokenized stocks only since their
listing in August 2025), but the model wants years of data to warm up its filters
and its regime model. This script splices public history in front of Kraken's own
candles, rescaled so there is no jump at the join:

  crypto          Binance USDT candles converted to EUR
  tokenized ETFs  the underlying fund's dividend-adjusted candles (Yahoo)
  EUR/USD         Yahoo before Kraken's first candle

After this, ``hermes_quant.live.run_daily`` keeps the store current by itself.

    python scripts/seed_live_history.py
    python scripts/seed_live_history.py --favourites     # only the owner's single stocks

The single stocks of the optional favourites basket (config/favourites.json) get the
same treatment from the company's own share price. Seeding them is optional: a stock
added later simply starts from Kraken's own candles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes_quant.data import kraken  # noqa: E402
from hermes_quant.live import favourites  # noqa: E402
from hermes_quant.live.market import COLUMNS, FX_SYMBOL, PriceStore  # noqa: E402
from hermes_quant.universe import LIVE_INSTRUMENTS  # noqa: E402

RESEARCH = ROOT / "data" / "research"
SEED_START = "2014-01-01"


def splice(history: pd.DataFrame, venue: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Put ``history`` in front of ``venue``, rescaling prices to the venue's level
    over the first weeks they overlap."""
    overlap = history.index.intersection(venue.index)[:30]
    ratio = float((venue.loc[overlap, "close"] / history.loc[overlap, "close"]).median()) if len(overlap) else 1.0
    older = history[history.index < venue.index[0]].copy()
    older[["open", "high", "low", "close"]] *= ratio
    return pd.concat([older[COLUMNS], venue[COLUMNS]]).sort_index(), ratio


def seed_favourites(store: PriceStore) -> None:
    """Company share prices (Yahoo, dividend-adjusted) in front of Kraken's token candles."""
    import yfinance as yf

    for inst in favourites.instruments(favourites.load()):
        try:
            venue = kraken.ohlc_daily(inst.kraken_pair, inst.tokenized)
        except Exception as exc:  # noqa: BLE001
            print(f"{inst.symbol:6s} NOT AVAILABLE on Kraken ({exc})")
            continue
        raw = yf.download(inst.symbol, start=SEED_START, interval="1d", auto_adjust=True, progress=False, multi_level_index=False)
        hist = raw.rename(columns=str.lower)
        if len(hist):
            hist.index = pd.DatetimeIndex(hist.index).tz_localize(None).normalize()
            hist = hist[np.isfinite(hist["close"]) & (hist["close"] > 0)]
        if len(hist):
            merged, ratio = splice(hist, venue)
        else:
            merged, ratio = venue[COLUMNS], float("nan")
        store.save(inst.symbol, merged)
        print(f"{inst.symbol:6s} {merged.index[0].date()} -> {merged.index[-1].date()}  {len(merged):5d} rows  "
              f"Kraken from {venue.index[0].date()} ({len(venue)} candles)  join ratio {ratio:.4f}")


def main() -> None:
    store = PriceStore(ROOT / "data" / "live" / "prices")
    if "--favourites" in sys.argv:
        seed_favourites(store)
        return
    yahoo = pd.read_parquet(RESEARCH / "yahoo_daily.parquet")
    full = RESEARCH / "binance_klines_full.parquet"
    binance = pd.read_parquet(full if full.exists() else RESEARCH / "binance_klines.parquet")
    coinbase = pd.read_parquet(RESEARCH / "coinbase_klines.parquet")

    fx_hist = yahoo[yahoo.symbol == "EURUSD=X"].drop(columns="symbol").sort_index().loc[SEED_START:]
    fx_hist = fx_hist[fx_hist.close > 0].assign(volume=0.0)
    fx_venue = kraken.ohlc_daily("EURUSD")
    fx, ratio = splice(fx_hist, fx_venue)
    store.save(FX_SYMBOL, fx)
    print(f"{FX_SYMBOL:6s} {fx.index[0].date()} -> {fx.index[-1].date()}  {len(fx):5d} rows  join ratio {ratio:.4f}")
    usd_per_eur = fx["close"]

    for inst in LIVE_INSTRUMENTS:
        try:
            venue = kraken.ohlc_daily(inst.kraken_pair, inst.tokenized)
        except Exception as exc:  # noqa: BLE001
            print(f"{inst.symbol:6s} NOT AVAILABLE on Kraken ({exc}); remove it from LIVE_INSTRUMENTS")
            continue
        if inst.asset_class == "crypto":
            hist = binance[binance.symbol == inst.symbol].drop(columns="symbol").sort_index()
            if inst.symbol in ("BTC", "ETH"):
                early = coinbase[coinbase.symbol == inst.symbol].drop(columns="symbol").sort_index()
                hist = pd.concat([early[early.index < hist.index[0]], hist])
            hist = hist.loc[SEED_START:]
            rate = usd_per_eur.reindex(hist.index).ffill().bfill()
            hist = hist.assign(**{c: hist[c] / rate for c in ("open", "high", "low", "close")})
        else:
            hist = yahoo[yahoo.symbol == inst.symbol].drop(columns="symbol").sort_index().loc[SEED_START:]
        hist = hist[np.isfinite(hist["close"]) & (hist["close"] > 0)]
        merged, ratio = splice(hist, venue)
        store.save(inst.symbol, merged)
        print(f"{inst.symbol:6s} {merged.index[0].date()} -> {merged.index[-1].date()}  {len(merged):5d} rows  "
              f"Kraken from {venue.index[0].date()} ({len(venue)} candles)  join ratio {ratio:.4f}")


if __name__ == "__main__":
    main()
