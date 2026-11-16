"""Kraken market data (public, no key) and account access (private, signed).

Three product lines share one account:
  * spot crypto          pairs such as XBTEUR
  * xStocks              tokenized stocks/ETFs such as SPYxUSD; every call needs
                         ``asset_class=tokenized_asset``
  * futures / perpetuals a separate API host (futures.kraken.com)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
import urllib.parse
from datetime import datetime, timezone

import pandas as pd
import requests

SPOT = "https://api.kraken.com"
FUTURES = "https://futures.kraken.com/derivatives/api/v3"
UA = {"User-Agent": "hermes-quant/2.0"}
TOKENIZED = "tokenized_asset"


class KrakenError(RuntimeError):
    pass


def _public(endpoint: str, params: dict, tries: int = 5) -> dict:
    last = "no response"
    for attempt in range(tries):
        try:
            r = requests.get(f"{SPOT}/0/public/{endpoint}", params=params, headers=UA, timeout=30)
            body = r.json()
            if not body.get("error"):
                return body["result"]
            last = ",".join(body["error"])
            if not any(code in last for code in ("Rate limit", "Busy", "Unavailable", "Internal error")):
                raise KrakenError(f"{endpoint} {params}: {last}")
        except (requests.RequestException, ValueError) as exc:
            last = repr(exc)
        time.sleep(1.5 * 2**attempt)
    raise KrakenError(f"{endpoint} {params}: {last}")


def ohlc_daily(pair: str, tokenized: bool = False) -> pd.DataFrame:
    """Completed daily candles (UTC days), oldest first. Kraken serves about 720 of
    them; the candle still being formed is dropped."""
    params = {"pair": pair, "interval": 1440}
    if tokenized:
        params["asset_class"] = TOKENIZED
    result = _public("OHLC", params)
    rows = next(v for k, v in result.items() if k != "last")
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df.index = pd.to_datetime(df["time"].astype(int), unit="s", utc=True).dt.normalize().dt.tz_localize(None)
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    return df[df.index < today].sort_index()


def ticker(pair: str, tokenized: bool = False) -> dict:
    """Best bid/ask and last trade for one pair."""
    params = {"pair": pair}
    if tokenized:
        params["asset_class"] = TOKENIZED
    v = next(iter(_public("Ticker", params).values()))
    bid, ask = float(v["b"][0]), float(v["a"][0])
    return {"bid": bid, "ask": ask, "mid": 0.5 * (bid + ask), "last": float(v["c"][0]),
            "volume_24h": float(v["v"][1])}


def asset_pairs(tokenized: bool = False) -> dict:
    return _public("AssetPairs", {"aclass_base": TOKENIZED} if tokenized else {})


def futures_tickers() -> list[dict]:
    r = requests.get(f"{FUTURES}/tickers", headers=UA, timeout=30)
    r.raise_for_status()
    return r.json().get("tickers", [])


# --------------------------------------------------------------------------- private


def sign(uri_path: str, postdata: str, nonce: str, secret_b64: str) -> str:
    """Kraken's request signature: HMAC-SHA512 over path + SHA256(nonce + body)."""
    digest = hashlib.sha256((nonce + postdata).encode()).digest()
    mac = hmac.new(base64.b64decode(secret_b64), uri_path.encode() + digest, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


def pair_rules(tokenized: bool = False) -> dict[str, dict]:
    """Precision and minimum order size for every pair, keyed by its ``altname``."""
    out = {}
    for info in asset_pairs(tokenized).values():
        out[info["altname"]] = {"price_decimals": int(info["pair_decimals"]), "lot_decimals": int(info["lot_decimals"]),
                                "ordermin": float(info.get("ordermin") or 0.0), "costmin": float(info.get("costmin") or 0.0)}
    return out


class KrakenClient:
    """Minimal signed client. The API key must NOT have withdrawal permission."""

    def __init__(self, key: str, secret: str):
        self._key, self._secret = key, secret
        self._nonce = 0

    def _private(self, endpoint: str, data: dict | None = None) -> dict:
        path = f"/0/private/{endpoint}"
        self._nonce = max(self._nonce + 1, int(time.time() * 1000))      # strictly increasing
        payload = {"nonce": str(self._nonce), **(data or {})}
        body = urllib.parse.urlencode(payload)
        headers = {**UA, "API-Key": self._key, "API-Sign": sign(path, body, payload["nonce"], self._secret),
                   "Content-Type": "application/x-www-form-urlencoded"}
        r = requests.post(SPOT + path, data=body, headers=headers, timeout=30)
        out = r.json()
        if out.get("error"):
            message = f"{endpoint}: {','.join(out['error'])}"
            for secret in (self._key, self._secret):           # Kraken does not echo them; if it ever did, they stop here
                if secret:
                    message = message.replace(secret, "[redacted]")
            raise KrakenError(message)
        return out["result"]

    def balances(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._private("Balance").items()}

    def add_order(self, pair: str, side: str, volume: str, *, price: str, tokenized: bool = False,
                  validate: bool = True) -> dict:
        """Immediate-or-cancel limit order: it fills at ``price`` or better right now,
        and whatever does not fill is cancelled by the exchange. With validate=True
        (the default) Kraken only checks the order and discards it."""
        data = {"pair": pair, "type": side, "ordertype": "limit", "volume": volume, "price": price,
                "timeinforce": "IOC"}
        if tokenized:
            data["asset_class"] = TOKENIZED
        if validate:
            data["validate"] = "true"
        return self._private("AddOrder", data)

    def query_orders(self, txids: list[str]) -> dict:
        return self._private("QueryOrders", {"txid": ",".join(txids)})

    def open_orders(self) -> dict:
        return self._private("OpenOrders").get("open", {})

    def cancel_order(self, txid: str) -> dict:
        return self._private("CancelOrder", {"txid": txid})
