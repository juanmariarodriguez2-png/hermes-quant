"""Daily candles from Coinbase Exchange's public API (no key).

Used to extend BTC and ETH history back before Binance's archive begins (2017-08).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

URL = "https://api.exchange.coinbase.com/products/{product}/candles"
UA = {"User-Agent": "hermes-quant-research/2.0"}


def download_daily(product: str, start: str = "2015-01-01") -> pd.DataFrame:
    """Full daily OHLCV history for e.g. 'BTC-USD'. The API serves 300 candles per call."""
    out = []
    cursor = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    while cursor < now:
        end = min(cursor + timedelta(days=299), now)
        params = {"granularity": 86400, "start": cursor.isoformat(), "end": end.isoformat()}
        for attempt in range(5):
            r = requests.get(URL.format(product=product), params=params, headers=UA, timeout=60)
            if r.status_code == 200:
                out += r.json()
                break
            time.sleep(1.5 * 2**attempt)
        else:
            raise RuntimeError(f"coinbase {product} {cursor:%Y-%m-%d}: HTTP {r.status_code}")
        cursor = end + timedelta(days=1)
        time.sleep(0.15)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["time", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df["time"], unit="s", utc=True).dt.normalize().dt.tz_localize(None)
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["quote_volume"] = df["volume"] * df["close"]
    return df[~df.index.duplicated(keep="last")].sort_index()
