"""Step 1 — download every research dataset into data/research/ (raw files cached in data/raw/).

    python research/01_download_data.py

Sources (all public, no API keys):
  * Binance public archive  -> daily OHLCV for ~48 coins incl. delisted ones, perp funding
  * Coinbase Exchange       -> BTC / ETH history before 2017-08
  * Yahoo Finance           -> ETFs (total-return adjusted), EUR/USD, reference series
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes_quant.data import binance_vision as bv  # noqa: E402
from hermes_quant.data import coinbase  # noqa: E402
from hermes_quant.universe import CRYPTO_CANDIDATES, ETF_RESEARCH, REFERENCE  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "research"
OUT.mkdir(parents=True, exist_ok=True)

FUNDING_SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "LINK", "ADA", "DOGE", "LTC", "AVAX", "BNB"]


def crypto() -> None:
    def one(coin: str) -> tuple[str, pd.DataFrame]:
        return coin, bv.download_klines(f"{coin}USDT", RAW / "binance", workers=6)

    with ThreadPoolExecutor(6) as pool:
        frames = dict(pool.map(one, CRYPTO_CANDIDATES))
    rows = []
    for coin, df in frames.items():
        if df.empty:
            print(f"  {coin:7s} no data")
            continue
        df = df.assign(symbol=coin)
        rows.append(df)
        print(f"  {coin:7s} {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  {len(df):5d} bars")
    pd.concat(rows).to_parquet(OUT / "binance_klines.parquet")


def early_majors() -> None:
    rows = []
    for product in ("BTC-USD", "ETH-USD"):
        df = coinbase.download_daily(product, start="2015-01-01").assign(symbol=product.split("-")[0])
        print(f"  {product}  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  {len(df)} bars")
        rows.append(df)
    pd.concat(rows).to_parquet(OUT / "coinbase_klines.parquet")


def funding() -> None:
    series = {}
    for coin in FUNDING_SYMBOLS:
        s = bv.download_funding(f"{coin}USDT", RAW / "binance")
        if len(s):
            series[coin] = s
            print(f"  {coin:5s} {s.index[0]:%Y-%m-%d} -> {s.index[-1]:%Y-%m-%d}  mean {s.mean() * 365:6.2%}/yr")
    pd.DataFrame(series).to_parquet(OUT / "binance_funding.parquet")


def yahoo() -> None:
    import yfinance as yf

    tickers = list(ETF_RESEARCH) + REFERENCE
    raw = yf.download(tickers, start="2005-01-01", interval="1d", auto_adjust=True,
                      progress=False, threads=False, group_by="ticker")
    rows = []
    for t in tickers:
        df = raw[t].dropna(how="all").rename(columns=str.lower)
        if df.empty:
            print(f"  {t:9s} no data")
            continue
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        rows.append(df[["open", "high", "low", "close", "volume"]].assign(symbol=t))
        print(f"  {t:9s} {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  {len(df)} bars")
    pd.concat(rows).to_parquet(OUT / "yahoo_daily.parquet")


if __name__ == "__main__":
    steps = {"crypto": crypto, "early": early_majors, "funding": funding, "yahoo": yahoo}
    wanted = sys.argv[1:] or list(steps)
    for name in wanted:
        print(f"[{name}]")
        steps[name]()
    print("done ->", OUT)
