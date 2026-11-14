"""Order execution back-ends.

PaperBroker   simulated fills with the same cost model as the backtest. The default.
KrakenBroker  real orders on the user's Kraken account. Opt-in, and by default it
              only *validates* orders (Kraken's dry-run) until explicitly armed.

Both expose the same three calls: ``cash()``, ``units()`` and ``execute(orders, ...)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from ..data.kraken import KrakenClient, KrakenError
from ..portfolio.execution import CostModel
from ..universe import Instrument


@dataclass
class Fill:
    symbol: str
    units: float            # signed: positive = bought. What was actually executed.
    price_eur: float
    notional_eur: float     # signed, actually executed
    cost_eur: float         # fees + slippage (+ FX share), always >= 0
    status: str = "filled"  # filled | partial | unfilled | validated | skipped | rejected
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


# Kraken's balance keys for the oldest assets carry legacy X/Z prefixes; balances held
# in the flexible "earn" wallet carry an ".F" suffix.
_LEGACY = {"BTC": ("XXBT", "XBT"), "ETH": ("XETH", "ETH"), "XRP": ("XXRP", "XRP"),
           "LTC": ("XLTC", "LTC"), "DOGE": ("XXDG", "XDG")}
FX_PAIR = "EURUSD"


class KrakenBroker:
    """Real execution on Kraken.

    How an order is sent
    --------------------
    Every order is a *limit* order priced at most ``max_slippage`` through the
    reference price, with time-in-force IOC ("immediate or cancel"). It therefore
    either fills right away at an acceptable price or does not fill at all: there is
    no resting order to forget about, and a thin order book cannot produce a terrible
    fill. What actually executed is then read back from the exchange; nothing is
    ever assumed to have filled.

    Sequence: sells -> re-read balances -> buy the dollars the stock purchases need
    -> re-read balances -> buys, scaled to the cash really available -> convert idle
    dollars back to euros.

    Safety rails
    ------------
    * ``armed=False`` (default): every order goes out with ``validate=true``; Kraken
      checks it and discards it. Nothing trades until the operator arms the broker.
    * ``max_order_eur`` caps every order, the currency conversions included.
    * Orders below the exchange's minimum size are skipped, not rounded up.
    * The API key should carry trade permission only, never withdrawal.

    NOTE: exercised against Kraken's public endpoints, its documented signing example
    and a simulated exchange in the test-suite, but not against a funded account. Run
    it validate-only first and read the log before arming it.
    """

    def __init__(self, client: KrakenClient, instruments: dict[str, Instrument], *, pair_rules: dict[str, dict],
                 armed: bool = False, max_order_eur: float = 100.0, max_slippage: float = 0.005,
                 wait_seconds: float = 20.0, pause: float = 1.0):
        self.client = client
        self.instruments = instruments
        self.rules = pair_rules
        self.armed = armed
        self.max_order_eur = float(max_order_eur)
        self.max_slippage = float(max_slippage)
        self.wait_seconds = wait_seconds
        self.pause = pause
        self._balances: dict[str, float] = {}
        self.refresh()

    # ------------------------------------------------------------------ account
    def refresh(self) -> None:
        self._balances = self.client.balances()

    def _balance(self, *names: str) -> float:
        wanted = {n.upper() for n in names} | {n.upper() + ".F" for n in names}
        return sum(v for k, v in self._balances.items() if k.upper() in wanted)

    def cash(self) -> float:
        """Euro cash only; idle dollars are reported separately by ``usd_cash``."""
        return self._balance("ZEUR", "EUR")

    def usd_cash(self) -> float:
        return self._balance("ZUSD", "USD")

    def units(self) -> dict[str, float]:
        out = {}
        for symbol, inst in self.instruments.items():
            if inst.tokenized:
                base = inst.kraken_pair[:-3]                       # "SPYxUSD" -> "SPYx"
                names = (base, base[:-1] + "SPV")                  # Kraken also uses "SPYSPV"
            else:
                names = _LEGACY.get(symbol, (symbol,))
            amount = self._balance(*names)
            if amount > 0:
                out[symbol] = amount
        return out

    # ------------------------------------------------------------------ one order
    def _send(self, pair: str, side: str, volume: float, reference: float, tokenized: bool) -> dict:
        """Send one capped IOC limit order. Returns what really happened:
        {status, volume, cost, fee, detail}, with cost and fee in the pair's quote currency."""
        rule = self.rules.get(pair)
        nothing = {"volume": 0.0, "cost": 0.0, "fee": 0.0}
        if rule is None or not (reference > 0) or not math.isfinite(reference):
            return {**nothing, "status": "skipped", "detail": "no trading rules or price for this pair"}
        step = 10.0 ** -rule["lot_decimals"]
        volume = math.floor(volume / step + 1e-9) * step           # never round a size up
        limit = reference * (1.0 + self.max_slippage if side == "buy" else 1.0 - self.max_slippage)
        if volume < rule["ordermin"] or volume * reference < rule["costmin"]:
            return {**nothing, "status": "skipped", "detail": f"below exchange minimum ({rule['ordermin']})"}
        volume_s = f"{volume:.{rule['lot_decimals']}f}"
        price_s = f"{limit:.{rule['price_decimals']}f}"
        try:
            res = self.client.add_order(pair, side, volume_s, price=price_s, tokenized=tokenized,
                                        validate=not self.armed)
        except KrakenError as exc:
            return {**nothing, "status": "rejected", "detail": str(exc)}
        if not self.armed:
            return {**nothing, "status": "validated", "detail": res.get("descr", {}).get("order", "")}

        txids = res.get("txid") or []
        info = {}
        deadline = time.time() + self.wait_seconds
        while txids:
            try:
                info = self.client.query_orders(txids)
            except KrakenError:
                info = {}
            done = info and all(o.get("status") in ("closed", "canceled", "expired") for o in info.values())
            if done or time.time() >= deadline:
                break
            time.sleep(self.pause)
        if txids and not (info and all(o.get("status") in ("closed", "canceled", "expired") for o in info.values())):
            for txid in txids:                                     # should not happen with IOC; be safe
                try:
                    self.client.cancel_order(txid)
                except KrakenError:
                    pass
            try:
                info = self.client.query_orders(txids)
            except KrakenError:
                info = {}
        executed = sum(float(o.get("vol_exec") or 0.0) for o in info.values())
        cost = sum(float(o.get("cost") or 0.0) for o in info.values())
        fee = sum(float(o.get("fee") or 0.0) for o in info.values())
        status = "unfilled" if executed <= 0 else "filled" if executed >= volume * 0.999 else "partial"
        return {"status": status, "volume": executed, "cost": cost, "fee": fee, "detail": ",".join(txids)}

    # ------------------------------------------------------------------ a day's orders
    def execute(self, orders: dict[str, float], price_eur: dict[str, float],
                instruments: dict[str, Instrument], costs: CostModel, eurusd: float = 0.0) -> list[Fill]:
        """``orders`` maps symbol -> signed euro notional; ``eurusd`` is dollars per euro."""
        fills: list[Fill] = []
        has_fx = eurusd > 0 and math.isfinite(eurusd)
        if not has_fx:                                             # no rate, no dollar trades at all
            for symbol in [s for s in orders if instruments[s].quote == "USD"]:
                fills.append(Fill(symbol, 0.0, price_eur.get(symbol, float("nan")), 0.0, 0.0, "skipped", "no EUR/USD rate"))
            orders = {s: n for s, n in orders.items() if instruments[s].quote != "USD"}

        def settle() -> None:
            if self.armed:
                time.sleep(self.pause)
                self.refresh()

        def trade(symbol: str, notional: float) -> None:
            inst = instruments[symbol]
            price = price_eur.get(symbol, float("nan"))
            if inst.quote == "USD" and not has_fx:
                fills.append(Fill(symbol, 0.0, price, 0.0, 0.0, "skipped", "no EUR/USD rate"))
                return
            notional = max(-self.max_order_eur, min(self.max_order_eur, notional))
            to_quote = eurusd if inst.quote == "USD" else 1.0
            side = "buy" if notional > 0 else "sell"
            r = self._send(inst.kraken_pair, side, abs(notional) / price, price * to_quote, inst.tokenized)
            sign = 1.0 if side == "buy" else -1.0
            done_eur = r["cost"] / to_quote
            fee_eur = r["fee"] / to_quote
            slip = abs(done_eur - r["volume"] * price)             # paid versus the reference price
            fills.append(Fill(symbol, sign * r["volume"], price, sign * done_eur, fee_eur + slip, r["status"], r["detail"]))

        def convert(side: str, euros: float, why: str) -> None:
            """side='sell' turns euros into dollars; 'buy' turns dollars back into euros."""
            euros = min(euros, self.max_order_eur)                 # conversions obey the cap too
            r = self._send(FX_PAIR, side, euros, eurusd, False)
            sign = 1.0 if side == "buy" else -1.0
            fills.append(Fill(FX_PAIR, sign * r["volume"], 1.0, sign * r["volume"],
                              r["fee"] / eurusd if has_fx else 0.0, r["status"], why))

        for symbol, notional in orders.items():                    # 1) sells free up cash
            if notional < 0:
                trade(symbol, notional)
        settle()

        # An order can fill up to max_slippage through its reference price and pays a
        # fee on top, so only this fraction of a balance is safe to commit.
        headroom = 1.0 + self.max_slippage + 0.01
        usd_buys = {s: n for s, n in orders.items() if n > 0 and instruments[s].quote == "USD"}
        eur_buys = {s: n for s, n in orders.items() if n > 0 and instruments[s].quote != "USD"}
        if usd_buys and has_fx:                                    # 2) buy the dollars needed
            capped = sum(min(n, self.max_order_eur) for n in usd_buys.values())
            shortfall_eur = capped * headroom * 1.005 - self.usd_cash() / eurusd
            shortfall_eur = min(shortfall_eur, self.cash() * 0.98) if self.armed else shortfall_eur
            if shortfall_eur > 5.0:
                convert("sell", shortfall_eur, "euros to dollars for stock purchases")
                settle()

        if self.armed:                                             # 3) buys, sized to real cash
            usd_available_eur = self.usd_cash() / eurusd / headroom if has_fx else 0.0
            want = sum(min(n, self.max_order_eur) for n in usd_buys.values())
            if want > usd_available_eur > 0:
                usd_buys = {s: n * usd_available_eur / want for s, n in usd_buys.items()}
            elif usd_available_eur <= 0:
                usd_buys = {}
            eur_available = self.cash() / headroom
            want = sum(min(n, self.max_order_eur) for n in eur_buys.values())
            if want > eur_available > 0:
                eur_buys = {s: n * eur_available / want for s, n in eur_buys.items()}
            elif eur_available <= 0:
                eur_buys = {}
        for symbol, notional in {**usd_buys, **eur_buys}.items():
            trade(symbol, notional)
        settle()

        if self.armed and has_fx and not usd_buys:                 # 4) idle dollars go home
            spare_eur = self.usd_cash() / eurusd * 0.995
            if spare_eur > 10.0:
                convert("buy", spare_eur, "idle dollars back to euros")
                settle()
        return fills
