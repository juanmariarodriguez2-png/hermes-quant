"""Trading 212: real shares and funds, bought and sold by the model with no manual step.

Quantity orders sized from euro references, with execution read back from history.

Why this venue exists. Kraken refuses tokenized-stock orders by API from European
accounts, so on Kraken the stock, gold and silver orders have to be placed by hand.
Trading 212 takes them by API. A European retail account cannot buy the US-listed funds
the model was studied on (SPY, QQQ, GLD, SLV), so each of the model's markets is mapped
to the European fund that tracks the same thing (config/trading212.json); the signals
still come from the same price series as before.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

from ..data import yahoo
from ..data.trading212 import Trading212Error
from .brokers import Fill

CONFIG = Path(__file__).resolve().parents[2] / "config" / "trading212.json"


@dataclass(frozen=True)
class Listing:
    symbol: str
    ticker: str
    currency: str
    isin: str
    name: str
    schedule: int


def resolve(wanted: dict[str, dict], instruments: list[dict],
            currency_preference: tuple[str, ...] = ("EUR", "USD", "GBP", "GBX")) -> tuple[dict[str, Listing], dict[str, str]]:
    listings, problems, used = {}, {}, set()

    def rank(preference):
        def key(inst):
            currency = inst.get("currencyCode")
            return (preference.index(currency) if currency in preference else len(preference), inst["ticker"])
        return key

    for symbol, request in wanted.items():
        # A market may state its own order of currencies: a US share is best bought on its home exchange, in dollars.
        preference = tuple(request.get("currency") or currency_preference)
        chosen = next((i for i in instruments if i["ticker"] == request.get("ticker")), None)
        if chosen is None:
            for isin in request.get("isin", []):
                matches = [i for i in instruments if i.get("isin") == isin]
                ordinary = [i for i in matches if i.get("type") in ("STOCK", "ETF")]
                if matches:
                    chosen = min(ordinary or matches, key=rank(preference))
                    break
        if chosen is None:
            problems[symbol] = "not in Trading 212's list: " + ", ".join(request.get("isin", []))
            continue
        ticker = chosen["ticker"]
        if ticker in used:
            problems[symbol] = f"{ticker} is already assigned to another symbol"
            continue
        used.add(ticker)
        listings[symbol] = Listing(symbol, ticker, chosen["currencyCode"], chosen.get("isin", ""),
                                   chosen.get("name", request.get("name", symbol)), chosen["workingScheduleId"])
        if chosen.get("type") not in ("STOCK", "ETF"):
            problems[symbol] = f"resolved {ticker}, but its type is {chosen.get('type', 'unknown')}, rather than STOCK or ETF"
    return listings, problems


def session_open(exchanges: list[dict], schedule_id: int, now: datetime) -> bool:
    if now.tzinfo is None or now.utcoffset() is None:
        return False
    events = []
    for exchange in exchanges:
        for schedule in exchange.get("workingSchedules", []):
            if schedule.get("id") != schedule_id:
                continue
            for event in schedule.get("timeEvents", []):
                try:
                    date = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    if date <= now:
                        events.append((date, event["type"]))
                except (KeyError, ValueError, TypeError):
                    return False
    if not events:
        return False
    latest = max(date for date, _ in events)
    return all(kind in ("OPEN", "BREAK_END") for date, kind in events if date == latest)


def _ticker(order: dict) -> str:
    return order.get("ticker") or order.get("instrument", {}).get("ticker", "")


def _signed_quantity(order: dict) -> float:
    quantity = float(order.get("quantity") or 0)
    return -abs(quantity) if order.get("side") == "SELL" else quantity


HISTORY_TRIES = 3          # how many times a just-executed order is looked up in the history before falling back


class Trading212Broker:
    """Read balances before sizing purchases; never retry an ambiguous placement."""

    def __init__(self, client, listings: dict[str, Listing], *, armed: bool = False, max_order_eur: float = 100.0,
                 wait_seconds: float = 30.0, pause: float = 1.0, min_order_eur: float = 1.0, decimals: int = 4, sleep=time.sleep):
        if any(not math.isfinite(v) or v < 0 for v in (max_order_eur, wait_seconds, pause, min_order_eur)):
            raise ValueError("order limits and waiting times must be finite and nonnegative")
        if not isinstance(decimals, int) or not 0 <= decimals <= 12:
            raise ValueError("decimals must be between 0 and 12")
        if len({i.ticker for i in listings.values()}) != len(listings):
            raise ValueError("each listing must have a different ticker")
        self.client, self.listings, self.armed = client, listings, armed
        self.max_order_eur, self.min_order_eur, self.decimals = max_order_eur, min_order_eur, decimals
        self.wait_seconds, self.pause, self._sleep = wait_seconds, pause, sleep
        self._symbols = {i.ticker: s for s, i in listings.items()}
        self.refresh()

    def refresh(self) -> None:
        summary = self.client.summary()
        positions = self.client.positions()
        pending = self.client.pending_orders()
        self._summary, self._positions, self._pending = summary, positions, pending

    def cash(self) -> float:
        cash = self._summary["cash"]
        return float(cash["availableToTrade"]) + float(cash["reservedForOrders"])

    def available(self) -> float:
        return float(self._summary["cash"]["availableToTrade"])

    def units(self) -> dict[str, float]:
        return {self._symbols[p["instrument"]["ticker"]]: float(p["quantity"]) for p in self._positions
                if p["instrument"]["ticker"] in self._symbols and p["quantity"] > 0}

    def prices(self) -> dict[str, float]:
        return {self._symbols[p["instrument"]["ticker"]]: float(p["walletImpact"]["currentValue"]) / float(p["quantity"])
                for p in self._positions if p["instrument"]["ticker"] in self._symbols and p["quantity"] > 0}

    def pending(self) -> dict[str, float]:
        remaining = {}
        for order in self._pending:
            symbol = self._symbols.get(_ticker(order))
            if symbol is not None and order.get("status") not in ("FILLED", "CANCELLED", "REJECTED"):
                quantity = _signed_quantity(order)
                amount = max(0.0, abs(quantity) - abs(float(order.get("filledQuantity") or 0)))
                remaining[symbol] = remaining.get(symbol, 0.0) + math.copysign(amount, quantity)
        return remaining

    def unrecognised(self) -> dict[str, float]:
        return {p["instrument"]["ticker"]: float(p["quantity"]) for p in self._positions
                if p["instrument"]["ticker"] not in self._symbols}

    def _round_down(self, quantity: float) -> float:
        return float(Decimal(str(quantity)).quantize(Decimal(10) ** -self.decimals, rounding=ROUND_DOWN))

    def _adopt(self, ticker: str, quantity: float, before: set) -> dict | None:
        self._sleep(self.pause)
        pending = self.client.pending_orders()
        history = self.client.history(ticker)
        candidates = {}
        for order in pending + [h["order"] for h in history]:
            if (order.get("id") is not None and order["id"] not in before and _ticker(order) == ticker
                    and order.get("initiatedFrom") == "API" and abs(_signed_quantity(order) - quantity) <= 1e-9):
                candidates[order["id"]] = order
        return next(iter(candidates.values())) if len(candidates) == 1 else None

    def _completed(self, order: dict, ticker: str, quantity: float, price: float,
                   held_before: float | None = None) -> tuple[float, float, float, str]:
        order_id = order["id"]
        reasons = [str(order_id)]
        elapsed, started = 0.0, time.monotonic()
        done = False
        while True:
            try:
                order = self.client.order(order_id)
                done = order.get("status") in ("FILLED", "CANCELLED", "REJECTED")
            except Trading212Error as exc:
                done = exc.status == 404
                if not done:
                    reasons.append("order lookup failed")
                    break
            if done or max(elapsed, time.monotonic() - started) >= self.wait_seconds:
                break
            delay = min(max(self.pause, 0.01), self.wait_seconds - elapsed)
            self._sleep(delay)
            elapsed += delay
        if not done:
            try:
                self.client.cancel(order_id)
                reasons.append("cancel requested after waiting")
            except Trading212Error:
                reasons.append("cancel failed; order may still be working")
            try:
                order = self.client.order(order_id)
                if order.get("status") not in ("FILLED", "CANCELLED", "REJECTED"):
                    reasons.append("cancellation not yet confirmed")
            except Trading212Error:
                pass
        matches = []
        for attempt in range(HISTORY_TRIES):                  # the history can trail a fill by a moment
            try:
                history = self.client.history(ticker)
            except Trading212Error:
                history = []
            matches = [h["fill"] for h in history if h.get("order", {}).get("id") == order_id and h.get("fill")]
            if matches or attempt + 1 == HISTORY_TRIES:
                break
            self._sleep(max(self.pause, 0.01))
        if matches:
            executed = sum(abs(float(f["quantity"])) for f in matches)
            amount = sum(abs(float(f["walletImpact"]["netValue"])) for f in matches)
            fees = sum(abs(float(t["quantity"])) for f in matches for t in f["walletImpact"].get("taxes", []))
        else:
            executed = abs(float(order.get("filledQuantity") or 0))
            source = "filledQuantity"
            if executed == 0 and held_before is not None:     # a filled order vanishes from the working list: ask the account
                try:
                    now = sum(float(p["quantity"]) for p in self.client.positions() if p["instrument"]["ticker"] == ticker)
                    executed, source = abs(now - held_before), "the change in the position"
                except Trading212Error:
                    pass
            executed = min(executed, abs(quantity))            # never report more than was asked for
            amount, fees = executed * price, 0.0
            reasons.append(f"history unavailable; used {source} and reference price")
        return executed, amount, fees, "; ".join(reasons)

    def _trade(self, symbol: str, quantity: float, price: float) -> Fill:
        ticker = self.listings[symbol].ticker
        if quantity == 0 or abs(quantity) * price < self.min_order_eur:
            return Fill(symbol, 0.0, price, 0.0, 0.0, "skipped", "below the minimum order")
        if not self.armed:
            side = "buy" if quantity > 0 else "sell"
            return Fill(symbol, 0.0, price, 0.0, 0.0, "dry", f"would {side} {abs(quantity)} {ticker}")
        assert self.client.env != "live" or self.armed, "live trading requires arming"
        # Both sources are needed: a timed-out request might already be fully filled.
        try:
            pending = self.client.pending_orders()
            if any(_ticker(o) == ticker for o in pending):
                return Fill(symbol, 0.0, price, 0.0, 0.0, "skipped", "an earlier order is still working")
            before = {o["id"] for o in pending}
            before.update(h["order"]["id"] for h in self.client.history(ticker))
            held_before = sum(float(p["quantity"]) for p in self._positions if p["instrument"]["ticker"] == ticker)
        except Trading212Error as exc:
            return Fill(symbol, 0.0, price, 0.0, 0.0, "rejected", f"could not snapshot earlier orders: {exc}")
        try:
            order = self.client.market_order(ticker, quantity)
        except Trading212Error as exc:
            if not exc.uncertain:
                return Fill(symbol, 0.0, price, 0.0, 0.0, "rejected", str(exc))
            try:
                order = self._adopt(ticker, quantity, before)
            except Trading212Error:
                order = None
            if order is None:
                return Fill(symbol, 0.0, price, 0.0, 0.0, "rejected", f"the request failed and no order was found: {exc}")
        executed, amount, fees, detail = self._completed(order, ticker, quantity, price, held_before)
        status = "unfilled" if executed == 0 else "filled" if executed >= abs(quantity) * 0.999 else "partial"
        sign = 1 if quantity > 0 else -1
        return Fill(symbol, sign * executed, price, sign * amount, fees + abs(amount - executed * price), status, detail)

    def execute(self, orders: dict[str, float], price_eur: dict[str, float], instruments=None, costs=None,
                eurusd: float = 0.0) -> list[Fill]:
        fills = []
        valid = {}
        # A euro reference cannot safely size an order on a different account currency.
        currency_ok = self._summary.get("currency") == "EUR"
        for symbol, notional in sorted(orders.items(), key=lambda item: item[1] >= 0):
            price = price_eur.get(symbol, float("nan"))
            reason = ""
            if symbol not in self.listings:
                reason = "not listed on Trading 212"
            elif not math.isfinite(price) or price <= 0:
                reason = "no reference price"
            elif symbol in self.pending():
                reason = "an earlier order is still working"
            elif not math.isfinite(notional):
                reason = "invalid order notional"
            elif not currency_ok:
                reason = "account currency must be EUR"
            if reason:
                fills.append(Fill(symbol, 0.0, price, 0.0, 0.0, "skipped", reason))
            else:
                valid[symbol] = max(-self.max_order_eur, min(self.max_order_eur, notional))
        for symbol, notional in valid.items():
            if notional >= 0:
                continue
            price, listing = price_eur[symbol], self.listings[symbol]
            position = next((p for p in self._positions if p["instrument"]["ticker"] == listing.ticker), {})
            available = max(0.0, float(position.get("quantityAvailableForTrading", 0)))
            held = max(0.0, float(position.get("quantity", 0)))
            quantity = self._round_down(min(abs(notional) / price, available))
            exit_wanted = (held - quantity) * price < self.min_order_eur or abs(notional) >= held * price * 0.99
            # Removing dust must never override the hard notional cap.
            if exit_wanted and available * price <= self.max_order_eur:
                quantity = available
            fills.append(self._trade(symbol, -quantity, price))
        self.refresh()
        buys = {s: n for s, n in valid.items() if n >= 0}
        total = sum(buys.values())
        available = self.available()
        budget = max(0.0, available * 0.99) if math.isfinite(available) else 0.0
        scale = min(1.0, budget / total) if total else 0.0
        for symbol, notional in buys.items():
            price = price_eur[symbol]
            if symbol in self.pending():
                fills.append(Fill(symbol, 0.0, price, 0.0, 0.0, "skipped", "an earlier order is still working"))
                continue
            fills.append(self._trade(symbol, self._round_down(notional * scale / price), price))
        self.refresh()
        return fills


# ----------------------------------------------------------------------------- the venue, as the daily run sees it


def load_markets(path: Path | None = None, section: str = "markets") -> tuple[dict[str, dict], tuple[str, ...], dict[str, dict]]:
    """The map from the model's markets to Trading 212 instruments (config/trading212.json).
    Returns what ``resolve`` needs, the account-wide order of currencies, and for every
    market where to read a reference price: per ISIN, plus an optional default."""
    raw = json.loads(Path(path or CONFIG).read_text())
    wanted, quotes = {}, {}
    for symbol, market in (raw.get(section) or {}).items():
        listings = market.get("listings") or []
        # the short form, a bare list of ISINs, is accepted too
        wanted[symbol] = {"name": market.get("name", symbol), "isin": list(market.get("isin") or [x["isin"] for x in listings if x.get("isin")])}
        if market.get("ticker"):
            wanted[symbol]["ticker"] = market["ticker"]
        if market.get("currency"):
            wanted[symbol]["currency"] = list(market["currency"])
        quotes[symbol] = {x["isin"]: x["yahoo"] for x in listings if x.get("isin") and x.get("yahoo")}
        if market.get("yahoo"):
            quotes[symbol]["default"] = market["yahoo"]
    return wanted, tuple(raw.get("currency_preference") or ("EUR", "USD", "GBP", "GBX")), quotes


class Venue:
    """Trading 212 for one run: which of the model's markets it can trade there, whether
    each is open right now, what one share is worth in euros, and the broker itself."""

    name = "trading212"

    def __init__(self, client, *, armed: bool = False, max_order_eur: float = 100.0, path: Path | None = None,
                 quote=yahoo.price_eur, sleep=time.sleep):
        wanted, preference, self._quotes = load_markets(path)
        self.listings, self.problems = resolve(wanted, client.instruments(), preference)
        self._exchanges = client.exchanges()
        self.broker = Trading212Broker(client, self.listings, armed=armed, max_order_eur=max_order_eur, sleep=sleep)
        self.env, self._quote = client.env, quote

    def open(self, symbol: str, now: datetime) -> bool:
        listing = self.listings.get(symbol)
        return listing is not None and session_open(self._exchanges, listing.schedule, now)

    def prices(self, symbols, eurusd: float) -> tuple[dict[str, float], dict[str, str]]:
        """Euro value of one share of each market's instrument here: from the account for
        what it holds, from the reference source for the rest. Also says, per market, why
        a price is missing; a market without a price is simply not traded today."""
        prices, missing = dict(self.broker.prices()), {}
        for symbol in symbols:
            listing = self.listings.get(symbol)
            if symbol in prices or listing is None:
                continue
            source = self._quotes.get(symbol, {})
            feed = source.get(listing.isin) or source.get("default")
            if not feed:
                missing[symbol] = "no reference price configured for this listing"
                continue
            try:
                prices[symbol] = float(self._quote(feed, eurusd))
            except Exception as exc:  # noqa: BLE001 - reported; the market waits for tomorrow
                missing[symbol] = str(exc)[:120]
        return prices, missing
