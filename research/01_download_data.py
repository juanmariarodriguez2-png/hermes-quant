"""Step 1 — download every research dataset into data/research/ (raw files cached in data/raw/).

    python research/01_download_data.py

Sources (all public, no API keys):
  * Binance public archive  -> daily OHLCV for every USDT market ever listed, perp funding
  * Coinbase Exchange       -> BTC / ETH history before 2017-08
  * Yahoo Finance           -> ETFs (total-return adjusted), EUR/USD, reference series
"""

from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes_quant.data import binance_vision as bv  # noqa: E402
from hermes_quant.data import coinbase  # noqa: E402
from hermes_quant.universe import CRYPTO_CANDIDATES, ETF_EXTENDED, ETF_RESEARCH, REFERENCE  # noqa: E402

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


STABLE_OR_WRAPPED = {
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "UST", "USTC", "SUSD", "EUR", "GBP", "AUD",
    "EURI", "AEUR", "USD1", "XUSD", "BFUSD", "USDS", "USDSB", "PYUSD", "EURC", "USDE", "BKRW", "IDRT",
    "BIDR", "NGN", "RUB", "TRY", "UAH", "BRL", "ZAR", "VAI", "WBTC", "BETH", "WBETH", "BNSOL", "PAXG",
}
LEVERAGED_BASES = {"BTC", "ETH", "BNB", "ADA", "XRP", "LINK", "DOT", "TRX", "EOS", "LTC", "UNI", "SXP",
                   "XTZ", "FIL", "YFI", "BCH", "AAVE", "SUSHI", "XLM", "1INCH"}


def _is_leveraged(base: str) -> bool:
    for suffix in ("UP", "DOWN"):
        if base.endswith(suffix) and base[: -len(suffix)] in LEVERAGED_BASES:
            return True
    return base in {"BULL", "BEAR"} or (base.endswith(("BULL", "BEAR")) and base[:-4] in LEVERAGED_BASES)


def crypto_full() -> None:
    """EVERY USDT spot market Binance has ever listed, delisted ones included.

    The hand-written candidate list in ``universe.py`` was compiled with hindsight,
    which is a subtle form of survivorship bias. This step removes the judgement:
    the universe is whatever the exchange actually listed, and the model picks from
    it using only the trading volume observable at the time."""
    listing = bv.list_keys("data/spot/monthly/klines/")
    if not listing:                      # prefixes come back as CommonPrefixes, not Keys
        import requests

        symbols, marker = [], ""
        while True:
            url = f"{bv.LISTING}?delimiter=/&prefix=data/spot/monthly/klines/" + (f"&marker={marker}" if marker else "")
            text = requests.get(url, headers=bv.UA, timeout=60).text
            page = re.findall(r"<Prefix>data/spot/monthly/klines/([^/<]+)/</Prefix>", text)
            symbols += page
            if "<IsTruncated>true</IsTruncated>" not in text or not page:
                break
            marker = f"data/spot/monthly/klines/{page[-1]}/"
    bases = sorted({s[:-4] for s in symbols if s.endswith("USDT")})
    bases = [b for b in bases if b not in STABLE_OR_WRAPPED and not _is_leveraged(b)]
    print(f"  {len(bases)} USDT markets after removing stablecoins, wrapped and leveraged tokens")

    def one(base: str):
        try:
            return base, bv.download_klines(f"{base}USDT", RAW / "binance", workers=4)
        except Exception as exc:  # noqa: BLE001
            return base, repr(exc)

    rows, failed = [], []
    with ThreadPoolExecutor(10) as pool:
        for n, (base, df) in enumerate(pool.map(one, bases), 1):
            if isinstance(df, str):
                failed.append((base, df))
            elif len(df) >= 30:
                rows.append(df.assign(symbol=base))
            if n % 50 == 0:
                print(f"  {n}/{len(bases)} markets", flush=True)
    pd.concat(rows).to_parquet(OUT / "binance_klines_full.parquet")
    print(f"  saved {len(rows)} markets; {len(failed)} failed: {failed[:8]}")


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

    tickers = list(ETF_RESEARCH) + list(ETF_EXTENDED) + REFERENCE
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
    steps = {"crypto": crypto, "crypto_full": crypto_full, "early": early_majors, "funding": funding,
             "yahoo": yahoo}
    wanted = sys.argv[1:] or ["crypto_full", "early", "funding", "yahoo"]
    for name in wanted:
        print(f"[{name}]")
        steps[name]()
    print("done ->", OUT)
