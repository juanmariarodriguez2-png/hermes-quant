"""Order execution back-ends.

PaperBroker   simulated fills with the same cost model as the backtest. The default.
KrakenBroker  real orders on the user's Kraken account. Opt-in, and by default it
              only *validates* orders (Kraken's dry-run) until explicitly armed.

Both expose the same three calls: ``cash()``, ``units()`` and ``execute(orders, ...)``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from ..data.kraken import KrakenClient, KrakenError
from ..portfolio.execution import CostModel
from ..universe import Instrument


@dataclass
class Fill:
    symbol: str
    units: float            # signed: positive = bought
    price_eur: float
    notional_eur: float     # signed
    cost_eur: float         # fees + slippage (+ FX share), always >= 0
    status: str = "filled"  # filled | validated | rejected
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class PaperBroker:
    """Holds positions in a plain dict (persisted by the runner as JSON)."""

    def __init__(self, cash: float, units: dict[str, float]):
        self._cash = float(cash)
        self._units = {k: float(v) for k, v in units.items() if abs(v) > 0}

    def cash(self) -> float:
        return self._cash

    def units(self) -> dict[str, float]:
        return dict(self._units)

    def execute(self, orders: dict[str, float], price_eur: dict[str, float],
                instruments: dict[str, Instrument], costs: CostModel) -> list[Fill]:
        """``orders`` maps symbol -> signed euro notional. Fills at the given price;
        fees, slippage and the currency-conversion charge come out of cash."""
        fills = []
        usd_flow = 0.0
        for symbol, notional in orders.items():
            if notional == 0:
                continue
            inst = instruments[symbol]
            price = price_eur[symbol]
            units = notional / price
            cost = abs(notional) * costs.rate(inst.asset_class)
            self._units[symbol] = self._units.get(symbol, 0.0) + units
            if abs(self._units[symbol] * price) < 0.01:
                self._units.pop(symbol)
            self._cash -= notional + cost
            if inst.quote == "USD":
                usd_flow += notional
            fills.append(Fill(symbol, units, price, notional, cost))
        fx_cost = abs(usd_flow) * costs.fx_fee
        if fx_cost:
            self._cash -= fx_cost
            usd_fills = [f for f in fills if instruments[f.symbol].quote == "USD"]
            gross = sum(abs(f.notional_eur) for f in usd_fills) or 1.0
            for f in usd_fills:                       # attribute the FX charge pro rata
                f.cost_eur += fx_cost * abs(f.notional_eur) / gross
        return fills


# Kraken's balance keys for the assets we trade (legacy X/Z prefixes on the oldest ones).
_BALANCE_KEYS = {"BTC": ("XXBT", "XBT"), "ETH": ("XETH", "ETH"), "XRP": ("XXRP", "XRP"),
                 "LTC": ("XLTC", "LTC"), "DOGE": ("XXDG", "XDG")}


class KrakenBroker:
    """Real execution. Market orders, sells first, then currency conversion, then buys.

    Safety rails
    ------------
    * ``armed=False`` (default) sends every order with ``validate=true``: Kraken
      checks it and discards it. Nothing can trade until the operator arms the broker.
    * ``max_order_eur`` caps any single order.
    * The API key should carry trade permission only, never withdrawal.

    NOTE: this class has been exercised against Kraken's public endpoints and its
    documented signing example, but not against a funded account. Run it validate-only
    first and read the log before arming it.
    """

    def __init__(self, client: KrakenClient, instruments: dict[str, Instrument], *,
                 armed: bool = False, max_order_eur: float = 100.0, poll_seconds: int = 45):
        self.client = client
        self.instruments = instruments
        self.armed = armed
        self.max_order_eur = max_order_eur
        self.poll_seconds = poll_seconds
        self._balances: dict[str, float] = {}
        self.refresh()

    def refresh(self) -> None:
        self._balances = self.client.balances()

    def _balance(self, *keys: str) -> float:
        return sum(self._balances.get(k, 0.0) for k in keys)

    def cash(self) -> float:
        """Euro cash only; idle dollars are reported separately by ``usd_cash``."""
        return self._balance("ZEUR", "EUR")

    def usd_cash(self) -> float:
        return self._balance("ZUSD", "USD")

    def units(self) -> dict[str, float]:
        out = {}
        for symbol, inst in self.instruments.items():
            keys = _BALANCE_KEYS.get(symbol, (symbol,))
            if inst.tokenized:
                keys = (inst.kraken_pair[:-3],)        # "SPYxUSD" -> "SPYx"
            amount = self._balance(*keys)
            if amount > 0:
                out[symbol] = amount
        return out

    def _send(self, pair: str, side: str, volume: float, tokenized: bool) -> tuple[str, str]:
        try:
            res = self.client.add_order(pair, side, volume, tokenized=tokenized, validate=not self.armed)
        except KrakenError as exc:
            return "rejected", str(exc)
        if not self.armed:
            return "validated", res.get("descr", {}).get("order", "")
        txids = res.get("txid", [])
        deadline = time.time() + self.poll_seconds
        while txids and time.time() < deadline:
            info = self.client.query_orders(txids)
            if all(o.get("status") == "closed" for o in info.values()):
                break
            time.sleep(3)
        return "filled", ",".join(txids)

    def execute(self, orders: dict[str, float], price_eur: dict[str, float],
                instruments: dict[str, Instrument], costs: CostModel, eurusd: float = 1.0) -> list[Fill]:
        """Same signature as the paper broker, plus the EUR/USD rate for the FX leg."""
        fills: list[Fill] = []

        def trade(symbol: str, notional: float) -> None:
            inst = instruments[symbol]
            notional = max(-self.max_order_eur, min(self.max_order_eur, notional))
            volume = abs(notional) / price_eur[symbol]
            side = "buy" if notional > 0 else "sell"
            status, detail = self._send(inst.kraken_pair, side, volume, inst.tokenized)
            signed = volume if notional > 0 else -volume
            fills.append(Fill(symbol, signed, price_eur[symbol], notional,
                              abs(notional) * costs.rate(inst.asset_class), status, detail))

        for symbol, notional in orders.items():       # 1) sells free up cash
            if notional < 0:
                trade(symbol, notional)
        if self.armed:
            time.sleep(2)
            self.refresh()
        usd_needed = sum(n for s, n in orders.items() if n > 0 and instruments[s].quote == "USD") * eurusd
        shortfall_usd = usd_needed * 1.01 - self.usd_cash()
        if shortfall_usd > 5.0:                       # 2) buy the dollars the stock orders need
            status, detail = self._send("EURUSD", "sell", shortfall_usd / eurusd, False)
            fills.append(Fill("EURUSD", -shortfall_usd / eurusd, 1.0, -shortfall_usd / eurusd,
                              shortfall_usd / eurusd * costs.fx_fee, status, detail))
            if self.armed:
                time.sleep(2)
                self.refresh()
        for symbol, notional in orders.items():       # 3) buys
            if notional > 0:
                trade(symbol, notional)
        if self.armed:
            time.sleep(2)
            self.refresh()
            spare_usd = self.usd_cash()
            if spare_usd > 10.0 and usd_needed == 0:  # 4) idle dollars go back to euros
                status, detail = self._send("EURUSD", "buy", spare_usd / eurusd * 0.995, False)
                fills.append(Fill("EURUSD", spare_usd / eurusd, 1.0, spare_usd / eurusd,
                                  spare_usd / eurusd * costs.fx_fee, status, detail))
                self.refresh()
        return fills
