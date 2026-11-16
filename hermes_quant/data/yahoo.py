"""Reference prices from Yahoo Finance's public chart service (no key).

Trading 212's API has no quote endpoint: it tells you what you hold is worth, not what
something you do not hold yet costs. To size a first purchase the daily run needs the
price of one share of that exact listing, and this is where it gets it. The price is
only used to turn "buy 30 euros of it" into a number of shares; the order itself is a
market order and what was really paid is read back from the broker afterwards.
"""

from __future__ import annotations

import math

import requests

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
UA = {"User-Agent": "Mozilla/5.0 (compatible; hermes-quant/2.0)"}


class YahooError(RuntimeError):
    pass


def quote(symbol: str, session=None) -> tuple[float, str]:
    """Last traded price of one listing and the currency it is quoted in."""
    try:
        r = (session or requests).get(CHART.format(symbol=symbol), params={"range": "5d", "interval": "1d"}, headers=UA, timeout=20)
        meta = r.json()["chart"]["result"][0]["meta"]
        price, currency = float(meta["regularMarketPrice"]), str(meta["currency"])
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        raise YahooError(f"no quote for {symbol}: {type(exc).__name__}") from None
    if not math.isfinite(price) or price <= 0:
        raise YahooError(f"no quote for {symbol}: price {price}")
    return price, currency


def price_eur(symbol: str, eurusd: float, session=None) -> float:
    """The same quote in euros. Only euro and dollar listings are converted; anything
    else raises, because a wrong currency would size an order wrongly."""
    price, currency = quote(symbol, session)
    if currency == "EUR":
        return price
    if currency == "USD" and eurusd > 0 and math.isfinite(eurusd):
        return price / eurusd
    raise YahooError(f"no euro price for {symbol}: quoted in {currency}")
