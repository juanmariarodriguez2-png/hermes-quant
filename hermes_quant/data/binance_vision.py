"""Daily OHLCV and perpetual funding history from Binance's public archive.

``data.binance.vision`` is a static, key-less archive that also keeps *delisted*
markets (LUNA, FTT, ...). That matters: a backtest that only sees the coins which
are alive today is survivorship-biased, and in crypto that bias is enormous.

Only used for research. Live trading reads prices from the venue it trades on.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ARCHIVE = "https://data.binance.vision"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
UA = {"User-Agent": "hermes-quant-research/2.0"}

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]


def _get(url: str, tries: int = 5, timeout: int = 60) -> requests.Response | None:
    """GET with exponential backoff. Returns None on a definitive 404."""
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(1.5 * 2**attempt)
    raise RuntimeError(f"giving up on {url}")


def list_keys(prefix: str) -> list[str]:
    """All object keys under an archive prefix (handles S3 pagination)."""
    keys: list[str] = []
    marker = ""
    while True:
        url = f"{LISTING}?delimiter=/&prefix={prefix}" + (f"&marker={marker}" if marker else "")
        r = _get(url)
        if r is None:
            return keys
        page = re.findall(r"<Key>([^<]+)</Key>", r.text)
        keys += page
        if "<IsTruncated>true</IsTruncated>" not in r.text or not page:
            return keys
        marker = page[-1]


def _fetch_zip(key: str, cache: Path) -> bytes | None:
    """Download one archive object, caching the raw bytes on disk."""
    dest = cache / key
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    r = _get(f"{ARCHIVE}/{key}")
    if r is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return r.content


def _read_csv_zip(blob: bytes, names: list[str] | None) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    has_header = not first.split(",")[0].strip().lstrip("-").replace(".", "").isdigit()
    if names is None:
        return pd.read_csv(io.StringIO(raw))
    return pd.read_csv(io.StringIO(raw), names=names, header=0 if has_header else None)


def _epoch_to_utc(values: pd.Series) -> pd.DatetimeIndex:
    """Binance switched spot timestamps from milliseconds to microseconds in 2025."""
    v = values.astype("int64").to_numpy()
    unit = np.where(v > 10**14, 10**6, 10**3)  # µs vs ms
    secs = v / unit
    return pd.to_datetime(secs, unit="s", utc=True)


def download_klines(symbol: str, cache: Path, workers: int = 8) -> pd.DataFrame:
    """Full daily OHLCV history for one spot symbol (monthly files + current-month dailies)."""
    monthly = [k for k in list_keys(f"data/spot/monthly/klines/{symbol}/1d/") if k.endswith(".zip")]
    if not monthly:
        return pd.DataFrame()
    last_month = re.search(r"(\d{4}-\d{2})\.zip$", monthly[-1]).group(1)
    daily = [
        k
        for k in list_keys(f"data/spot/daily/klines/{symbol}/1d/")
        if k.endswith(".zip") and re.search(r"(\d{4}-\d{2})-\d{2}\.zip$", k).group(1) > last_month
    ]
    with ThreadPoolExecutor(workers) as pool:
        blobs = list(pool.map(lambda k: _fetch_zip(k, cache), monthly + daily))
    frames = [_read_csv_zip(b, KLINE_COLS) for b in blobs if b]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.index = _epoch_to_utc(df["open_time"]).normalize().tz_localize(None)
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume", "quote_volume"]].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


def download_funding(symbol: str, cache: Path, workers: int = 8) -> pd.Series:
    """Perpetual funding rate history (one print every 8h, later 4h for some markets).

    Returned as the *daily sum* of funding rates: the fraction of notional a long
    pays (positive) or receives (negative) for holding the perpetual that day.
    """
    keys = [k for k in list_keys(f"data/futures/um/monthly/fundingRate/{symbol}/") if k.endswith(".zip")]
    if not keys:
        return pd.Series(dtype=float)
    with ThreadPoolExecutor(workers) as pool:
        blobs = list(pool.map(lambda k: _fetch_zip(k, cache), keys))
    frames = [_read_csv_zip(b, None) for b in blobs if b]
    df = pd.concat(frames, ignore_index=True)
    ts_col = "calc_time" if "calc_time" in df.columns else df.columns[0]
    rate_col = "last_funding_rate" if "last_funding_rate" in df.columns else df.columns[-1]
    idx = _epoch_to_utc(df[ts_col]).normalize().tz_localize(None)
    daily = pd.Series(df[rate_col].astype(float).to_numpy(), index=idx).groupby(level=0).sum()
    daily.index.name = "date"
    return daily.sort_index()
