"""Rehearsal against a Trading 212 account. Without --trade-demo nothing can be sent.

    python -m hermes_quant.live.check_trading212                 # read-only: account, instruments, market hours
    python -m hermes_quant.live.check_trading212 --trade-demo    # practice money only: buys 5 euros of one market and sells it back

It answers, before any money is at stake: does the key work, which of the model's
markets does this account really offer and under which instrument, is each one's
exchange open now, and what does the account hold. It also looks up the extra markets
the owner asked about (Spanish and European shares, Japan, China) and says which exist.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from ..data.trading212 import ReadOnlyClient, Trading212Client, Trading212Error
from .trading212 import Trading212Broker, load_markets, resolve, session_open

CONFIG = Path(__file__).resolve().parents[2] / "config" / "trading212.json"
EXAMPLE = {"SPY": {"name": "S&P 500 ETF", "isin": ["IE00B5BMR087"]},
           "NVDA": {"name": "Nvidia", "isin": ["US67066G1040"]}}
PREFERENCE = ("EUR", "USD", "GBP", "GBX")


def check(client, wanted, now, currency_preference=PREFERENCE, extra=None) -> list[tuple[str, bool | None, str]]:
    """Rows of (what was checked, did it pass, detail). ``extra`` holds markets that are only
    looked up: their rows carry ``None`` instead of a verdict, because not offering one is no failure."""
    rows = []

    def read(name, fn):
        try:
            return fn()
        except Exception as exc:  # Every failed read is a finding in a rehearsal.
            rows.append((name, False, str(exc)))
            return None

    summary = read("account", client.summary)
    instruments = read("instruments", client.instruments)
    exchanges = read("exchange schedules", client.exchanges)
    positions = read("positions", client.positions)
    pending = read("working orders", client.pending_orders)
    if summary is not None:
        cash = summary["cash"]
        currency = summary["currency"]
        rows.append(("account currency and cash", currency == "EUR",
                     f"{currency}: {cash['availableToTrade']} available, {cash['reservedForOrders']} reserved"))
    listings = {}
    if instruments is not None:
        rows.append(("instruments", bool(instruments), f"{len(instruments)} in this account's list"))
        listings, problems = resolve(wanted, instruments, tuple(currency_preference))
        for symbol in wanted:
            if symbol in listings:
                listing = listings[symbol]
                opened = exchanges is not None and session_open(exchanges, listing.schedule, now)
                detail = f"{listing.ticker}, {listing.currency}, market {'open' if opened else 'closed'}"
                rows.append((symbol, symbol not in problems, detail + (f"; {problems[symbol]}" if symbol in problems else "")))
                for isin in wanted[symbol].get("isin", []):
                    matches = sorted(i["ticker"] for i in instruments if i.get("isin") == isin)
                    if matches:
                        others = [ticker for ticker in matches if ticker != listing.ticker]
                        rows.append((f"{symbol} other listings for {isin}", True, ", ".join(others) or "none"))
            else:
                rows.append((symbol, False, problems[symbol]))
        found, _ = resolve(extra or {}, instruments, tuple(currency_preference))
        for symbol, request in (extra or {}).items():
            listing = found.get(symbol)
            opened = listing is not None and exchanges is not None and session_open(exchanges, listing.schedule, now)
            rows.append((f"also asked about: {request.get('name', symbol)}", None,
                         f"{listing.ticker}, {listing.currency}, market {'open' if opened else 'closed'}" if listing else "not offered by this account"))
    by_ticker = {listing.ticker: symbol for symbol, listing in listings.items()}
    if positions is not None:
        recognised, unknown = [], []
        for position in positions:
            ticker, quantity = position["instrument"]["ticker"], position["quantity"]
            (recognised if ticker in by_ticker else unknown).append(f"{by_ticker.get(ticker, ticker)}={quantity}")
        rows.append(("recognised positions", True, ", ".join(recognised) or "none"))
        rows.append(("unrecognised positions", True, ", ".join(unknown) or "none"))
    if pending is not None:
        rows.append(("working orders", True, json.dumps(pending)))
    return rows


class DemoClient(Trading212Client):
    """Print the actual API payloads so the owner can inspect a demo trade."""

    def market_order(self, ticker, quantity, extended_hours=False):
        result = super().market_order(ticker, quantity, extended_hours)
        print("raw order:", json.dumps(result))
        return result

    def history(self, ticker=None, limit=50):
        result = super().history(ticker, limit)
        print("raw history / fills:", json.dumps(result))
        return result


def trade_demo(client, wanted, now, price=None, currency_preference=PREFERENCE) -> int:
    if client.env != "demo":
        print("--trade-demo requires T212_ENV=demo; refusing to trade a live account.")
        return 2
    listings, problems = resolve(wanted, client.instruments(), tuple(currency_preference))
    exchanges = client.exchanges()
    listing = next((i for s, i in listings.items() if s not in problems and session_open(exchanges, i.schedule, now)), None)
    if listing is None:
        print("No resolved market has an open session.")
        return 2
    broker = Trading212Broker(client, listings, armed=True, max_order_eur=5.0)
    price = price if price is not None else broker.prices().get(listing.symbol)
    if price is None or not math.isfinite(price) or price <= 0:
        print("A finite positive --price EUR reference is needed for this market.")
        return 2
    bought = broker.execute({listing.symbol: 5.0}, {listing.symbol: price})[0]
    print("buy result:", json.dumps(bought.as_dict()))
    if bought.units <= 0:
        return 1
    # A cleanup may be tiny after a partial fill. Its cap must protect any shares held before the rehearsal.
    cleanup = Trading212Broker(client, listings, armed=True, max_order_eur=bought.units * price,
                               min_order_eur=0.0, decimals=12)
    sold = cleanup.execute({listing.symbol: -bought.units * price}, {listing.symbol: price})[0]
    print("sell result:", json.dumps(sold.as_dict()))
    return 0 if sold.units <= -bought.units * 0.999 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-demo", action="store_true")
    parser.add_argument("--price", type=float, metavar="EUR")
    args = parser.parse_args(argv)
    key, secret = os.environ.get("T212_API_KEY", ""), os.environ.get("T212_API_SECRET", "")
    env = os.environ.get("T212_ENV", "demo")
    if not key or not secret:
        print("T212_API_KEY / T212_API_SECRET are not set: nothing to check.")
        return 2
    if args.trade_demo and env != "demo":
        print("--trade-demo requires T212_ENV=demo; refusing to trade a live account.")
        return 2
    try:
        wanted, preference, extra = EXAMPLE, PREFERENCE, {}
        if CONFIG.exists():
            wanted, preference, _ = load_markets(CONFIG)
            extra = load_markets(CONFIG, section="also_check")[0]
        now = datetime.now(timezone.utc)
        print(f"Trading 212 rehearsal on the {env.upper()} account" + ("" if args.trade_demo else ": read-only, nothing can be sent.") + "\n")
        if args.trade_demo:
            return trade_demo(DemoClient(key, secret, env=env), wanted, now, args.price, preference)
        rows = check(ReadOnlyClient(key, secret, env=env), wanted, now, preference, extra=extra)
    except (Trading212Error, ValueError, OSError, KeyError) as exc:
        print(f"FAIL  rehearsal: {exc}")
        return 1
    width = max((len(row[0]) for row in rows), default=0)
    for name, ok, detail in rows:
        print(f"  {'info' if ok is None else 'ok  ' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [row for row in rows if row[1] is False]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} rows without a failure.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
