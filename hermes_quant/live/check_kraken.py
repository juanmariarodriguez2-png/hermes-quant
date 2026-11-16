"""Dry run against the real Kraken account. Nothing is placed and no money moves.

    python -m hermes_quant.live.check_kraken

It answers one question before real trading is switched on: would the orders the
model is about to send be accepted? It makes the read-only calls the live broker
needs, then pushes the account's current target through the very same broker code
with Kraken's ``validate`` flag, so the exchange checks every order and discards it.

The client used here refuses, in code, to send anything but a validate-only order.
"""

from __future__ import annotations

import json
import os
import sys

from ..data import kraken
from ..data.kraken import KrakenClient
from ..portfolio.execution import CostModel
from ..universe import LIVE_INSTRUMENTS, Instrument
from .brokers import FX_PAIR, REFUSALS, KrakenBroker
from .run_daily import LIVE, _now, market_open

READ_ONLY = {"Balance", "OpenOrders", "ClosedOrders"}
CRYPTO_PROBE = ("BTC", 10.0)          # the target may hold no crypto today; check that path anyway
Row = tuple[str, bool, str]           # what was checked, did it pass, detail


class DryRunClient(KrakenClient):
    """A Kraken client that cannot trade: read-only calls and validate-only orders."""

    def _private(self, endpoint: str, data: dict | None = None) -> dict:
        if endpoint == "AddOrder":
            if (data or {}).get("validate") != "true":
                raise RuntimeError("dry run: refusing to send an order without validate=true")
        elif endpoint not in READ_ONLY:
            raise RuntimeError(f"dry run: {endpoint} is not allowed")
        return super()._private(endpoint, data)


def reference_prices(symbols, by_symbol: dict[str, Instrument], eurusd: float) -> dict[str, float]:
    """Last quoted mid in euros. Unlike the daily run this also quotes a market that is
    closed: a dry run only needs a sensible reference price to build the order."""
    out = {}
    for symbol in symbols:
        inst = by_symbol[symbol]
        try:
            mid = kraken.ticker(inst.kraken_pair, inst.tokenized)["mid"]
        except Exception:  # noqa: BLE001 - reported by the caller as "no quote"
            continue
        if inst.quote == "USD":
            mid = mid / eurusd if eurusd > 0 else 0.0
        if mid > 0:
            out[symbol] = mid
    return out


def check(client, state: dict, now=None) -> list[Row]:
    rows: list[Row] = []
    by_symbol = {i.symbol: i for i in LIVE_INSTRUMENTS}
    now = now or _now()

    def attempt(name: str, fn, describe=lambda out: ""):
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 - every failure is a finding here
            rows.append((name, False, f"{type(exc).__name__}: {exc}"))
            return None
        rows.append((name, True, describe(out)))
        return out

    attempt("read open orders", client.open_orders, lambda o: f"{len(o)} open")
    attempt("read closed orders", lambda: client._private("ClosedOrders").get("closed", {}),
            lambda o: f"{len(o)} on the first page")
    rules = attempt("trading rules for every pair", lambda: {**kraken.pair_rules(False), **kraken.pair_rules(True)},
                    lambda r: f"{len(r)} pairs")
    eurusd = attempt("euro/dollar rate", lambda: kraken.ticker(FX_PAIR)["mid"], lambda v: f"{v:.4f}")
    broker = attempt("read balances", lambda: KrakenBroker(client, by_symbol, pair_rules=rules or {}, armed=False,
                                                             tokenized_via_api=True))   # probe them on purpose
    if broker is None or rules is None or not eurusd:
        return rows

    raw = {k: v for k, v in sorted(client.balances().items()) if v > 0}
    rows.append(("balance names at Kraken", True, ", ".join(f"{k}={v:.8g}" for k, v in raw.items()) or "none"))
    held = broker.units()
    target = {s: w for s, w in state.get("target", {}).items() if s in by_symbol and w > 0}
    prices = reference_prices(sorted(set(target) | set(held) | {CRYPTO_PROBE[0]}), by_symbol, eurusd)
    value = sum(units * prices.get(s, 0.0) for s, units in held.items())
    dollars = broker.usd_cash() / eurusd
    nav = broker.cash() + dollars + value
    rows.append(("account value", nav > 0, f"{nav:.2f} EUR = {broker.cash():.2f} euro cash + {dollars:.2f} in dollars "
                                           f"+ {value:.2f} in {len(held)} position(s)"))

    orders = {s: w * nav for s, w in target.items() if w * nav >= 1.0}
    orders.setdefault(*CRYPTO_PROBE)
    for symbol in sorted(set(orders) - set(prices)):
        rows.append((f"validate buy {symbol}", False, "no quote for this pair"))
    orders = {s: n for s, n in orders.items() if s in prices}
    asked = {**orders, FX_PAIR: sum(n for s, n in orders.items() if by_symbol[s].quote == "USD")}
    # Whether the daily run is set to send tokenized stocks itself. Where it is not (the
    # default: Kraken refuses them from European accounts), a refusal here is the expected
    # answer, and an acceptance is news worth reporting.
    sends_tokenized = os.environ.get("HERMES_TOKENIZED_API", "no") == "yes"
    for f in broker.execute(orders, prices, by_symbol, CostModel(), eurusd=eurusd):
        inst = by_symbol.get(f.symbol)
        shut = inst is not None and not market_open(inst, now)
        what = "validate euros to dollars" if f.symbol == FX_PAIR else f"validate buy {f.symbol}"
        ok, detail = f.status == "validated", f"{f.status}: {f.detail}"
        if inst is not None and inst.tokenized and not sends_tokenized:
            if f.status == "rejected" and any(word in f.detail for word in REFUSALS):
                ok, detail = True, f"refused by Kraken for this account ({f.detail}): listed for you to place by hand, as configured"
            elif f.status == "validated":
                detail += "  [Kraken accepts it: HERMES_TOKENIZED_API=yes would let the model trade it]"
        rows.append((f"{what} ({asked.get(f.symbol, 0.0):.0f} EUR)", ok, detail + ("  [market closed right now]" if shut else "")))
    return rows


def main() -> int:
    key, secret = os.environ.get("KRAKEN_API_KEY", ""), os.environ.get("KRAKEN_API_SECRET", "")
    if not key or not secret:
        print("KRAKEN_API_KEY / KRAKEN_API_SECRET are not set: nothing to check.")
        return 2
    state_path = LIVE / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    rows = check(DryRunClient(key, secret), state)
    width = max(len(r[0]) for r in rows)
    print("Kraken dry run: read-only calls and validate-only orders. Nothing was placed.\n")
    for name, ok, detail in rows:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [r for r in rows if not r[1]]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
