"""The production path, driven against a simulated exchange. No network, no keys."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hermes_quant.data import kraken
from hermes_quant.live import favourites, run_daily
from hermes_quant.live.brokers import KrakenBroker, PaperBroker
from hermes_quant.live.market import PriceStore, build_live_panel, venue_liquidity
from hermes_quant.portfolio.execution import CostModel
from hermes_quant.universe import LIVE_BY_SYMBOL, LIVE_INSTRUMENTS

RULES = {"XBTEUR": {"price_decimals": 1, "lot_decimals": 8, "ordermin": 0.00006, "costmin": 0.45},
         "SPYxUSD": {"price_decimals": 2, "lot_decimals": 8, "ordermin": 1e-8, "costmin": 0.5},
         "QQQxUSD": {"price_decimals": 2, "lot_decimals": 8, "ordermin": 1e-8, "costmin": 0.5},
         "EURUSD": {"price_decimals": 5, "lot_decimals": 8, "ordermin": 4.0, "costmin": 0.5}}
PRICE_EUR = {"BTC": 60000.0, "SPY": 680.0, "QQQ": 610.0}
EURUSD = 1.10


class FakeExchange:
    """Just enough of Kraken to exercise the broker: balances, IOC limit orders with
    an adjustable fill ratio, order queries, and a log of everything it was sent."""

    def __init__(self, balances: dict[str, float], fill_ratio: float = 1.0, fee: float = 0.004):
        self.bal = dict(balances)
        self.fill_ratio, self.fee = fill_ratio, fee
        self.orders, self.sent, self.cancelled = {}, [], []

    def balances(self):
        return dict(self.bal)

    def add_order(self, pair, side, volume, *, price, tokenized=False, validate=True):
        self.sent.append({"pair": pair, "side": side, "volume": float(volume), "price": float(price),
                          "tokenized": tokenized, "validate": validate})
        if validate:
            return {"descr": {"order": f"{side} {volume} {pair} @ limit {price}"}}
        vol = float(volume) * self.fill_ratio
        cost = vol * float(price)
        base = {"XBT": "XXBT", "XDG": "XXDG", "EUR": "ZEUR"}.get(pair[:-3], pair[:-3])
        quote = "Z" + pair[-3:]
        sign = 1.0 if side == "buy" else -1.0
        self.bal[base] = self.bal.get(base, 0.0) + sign * vol
        self.bal[quote] = self.bal.get(quote, 0.0) - sign * cost - cost * self.fee
        txid = f"T{len(self.orders)}"
        self.orders[txid] = {"status": "closed" if self.fill_ratio >= 1 else "canceled",
                             "vol_exec": str(vol), "cost": str(cost), "fee": str(cost * self.fee)}
        return {"txid": [txid]}

    def query_orders(self, txids):
        return {t: self.orders[t] for t in txids}

    def cancel_order(self, txid):
        self.cancelled.append(txid)
        return {}


def broker(exchange, **kw):
    return KrakenBroker(exchange, LIVE_BY_SYMBOL, pair_rules=RULES, pause=0.0, wait_seconds=0.1, **kw)


def test_signature_matches_krakens_documented_example():
    # Not a credential: this key, nonce and signature are the worked example in Kraken's API documentation.
    secret ="kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6F1huXg=="
    body = "nonce=1616492376594&ordertype=limit&pair=XBTUSD&price=37500&type=buy&volume=1.25"
    assert kraken.sign("/0/private/AddOrder", body, "1616492376594", secret) == (
        "4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS8MPtnRfp32bAb0nmbRn6H8ndwLUQ==")


def test_validate_only_is_the_default_and_moves_nothing():
    ex = FakeExchange({"ZEUR": 233.47})
    fills = broker(ex).execute({"BTC": 30.0, "SPY": 40.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)
    assert ex.sent and all(o["validate"] for o in ex.sent)          # every single order was a dry run
    assert ex.bal == {"ZEUR": 233.47}
    assert {f.status for f in fills} == {"validated"} and all(f.units == 0 for f in fills)


def test_armed_run_sells_first_converts_then_buys_and_reports_real_fills():
    ex = FakeExchange({"ZEUR": 50.0, "XXBT": 0.002})                # 120 EUR of bitcoin to sell
    b = broker(ex, armed=True)
    fills = b.execute({"BTC": -60.0, "SPY": 80.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)
    real = [(o["pair"], o["side"]) for o in ex.sent if not o["validate"]]
    assert real == [("XBTEUR", "sell"), ("EURUSD", "sell"), ("SPYxUSD", "buy")]
    rehearsals = [i for i, o in enumerate(ex.sent) if o["validate"]]
    conversion = next(i for i, o in enumerate(ex.sent) if o["pair"] == "EURUSD")
    assert len(rehearsals) == 1 and ex.sent[rehearsals[0]]["pair"] == "SPYxUSD" and rehearsals[0] < conversion
    spy = next(f for f in fills if f.symbol == "SPY")
    assert spy.status == "filled" and spy.units > 0
    assert spy.notional_eur == pytest.approx(spy.units * ex.sent[-1]["price"] / EURUSD)   # what Kraken reported
    assert ex.bal["ZEUR"] >= 0 and ex.bal["ZUSD"] >= -1e-9          # never spent money it did not have
    assert b.units()["SPY"] == pytest.approx(spy.units)


def test_every_order_including_currency_conversion_respects_the_cap():
    ex = FakeExchange({"ZEUR": 5000.0})
    b = broker(ex, armed=True, max_order_eur=25.0)
    b.execute({"BTC": 900.0, "SPY": 900.0, "QQQ": 900.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)
    for o in ex.sent:
        euros = o["volume"] if o["pair"] == "EURUSD" else o["volume"] * o["price"] / (EURUSD if "USD" in o["pair"] else 1.0)
        assert euros <= 25.0 * 1.006                                 # cap, plus the slippage allowance on price


def test_limit_price_is_capped_and_sizes_are_never_rounded_up():
    ex = FakeExchange({"ZEUR": 500.0})
    b = broker(ex, armed=True, max_slippage=0.005)
    b.execute({"BTC": 50.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)
    o = ex.sent[0]
    assert o["price"] == pytest.approx(60000.0 * 1.005, abs=0.1)
    assert o["volume"] <= 50.0 / 60000.0


def test_unfilled_and_partial_orders_are_reported_as_such():
    ex = FakeExchange({"ZEUR": 500.0}, fill_ratio=0.0)
    f = broker(ex, armed=True).execute({"BTC": 50.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)[0]
    assert f.status == "unfilled" and f.units == 0 and f.notional_eur == 0
    ex = FakeExchange({"ZEUR": 500.0}, fill_ratio=0.4)
    f = broker(ex, armed=True).execute({"BTC": 50.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)[0]
    assert f.status == "partial" and 0 < f.notional_eur < 25.0


def test_orders_below_the_exchange_minimum_are_skipped():
    ex = FakeExchange({"ZEUR": 500.0})
    f = broker(ex, armed=True).execute({"BTC": 2.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)[0]
    assert f.status == "skipped" and not ex.sent


def test_dollar_assets_are_left_alone_without_an_exchange_rate():
    ex = FakeExchange({"ZEUR": 500.0})
    fills = broker(ex, armed=True).execute({"SPY": 50.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=0.0)
    assert [f.status for f in fills] == ["skipped"] and not ex.sent


def test_balances_are_read_under_krakens_various_asset_names():
    ex = FakeExchange({"ZEUR": 10.0, "XXBT": 0.001, "XBT.F": 0.002, "SPYx": 0.5, "XXDG": 100.0, "ZUSD": 7.0})
    b = broker(ex)
    assert b.units() == pytest.approx({"BTC": 0.003, "SPY": 0.5, "DOGE": 100.0})
    assert b.cash() == 10.0 and b.usd_cash() == 7.0


def test_paper_broker_charges_exactly_what_the_backtest_charges():
    costs = CostModel()
    p = PaperBroker(1000.0, {})
    fills = p.execute({"BTC": 100.0, "SPY": 200.0}, PRICE_EUR, LIVE_BY_SYMBOL, costs)
    expected = 100.0 * costs.rate("crypto") + 200.0 * costs.rate("equity") + 200.0 * costs.fx_fee
    assert sum(f.cost_eur for f in fills) == pytest.approx(expected)
    assert p.cash() == pytest.approx(1000.0 - 300.0 - expected)


# ------------------------------------------------------------------ the daily run, offline

@pytest.fixture()
def sandbox(tmp_path):
    """A private copy of the live price store so the run can be replayed safely."""
    source = Path(__file__).resolve().parents[1] / "data" / "live" / "prices"
    if not source.exists():
        pytest.skip("live price store not seeded")
    shutil.copytree(source, tmp_path / "live" / "prices")
    return tmp_path


MONDAY = datetime(2026, 11, 16, 1, 25, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 11, 14, 1, 25, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _basket_switched_off(monkeypatch):
    """Every test starts as production does: the basket of single stocks is off."""
    monkeypatch.delenv("HERMES_FAVOURITES", raising=False)


def test_daily_run_is_idempotent_and_respects_the_weekend(sandbox):
    kw = dict(mode="paper", offline=True, live_dir=sandbox / "live", site_dir=sandbox / "site")
    weekend = run_daily.run(now=SATURDAY, **kw)
    assert not [o for o in weekend["account"]["orders_today"] if LIVE_BY_SYMBOL[o["symbol"]].tokenized]
    (sandbox / "live" / "state.json").unlink()
    first = run_daily.run(now=MONDAY, **kw)
    assert first["account"]["orders_today"], "a fresh account on a weekday should buy its target"
    again = run_daily.run(now=MONDAY, **kw)
    assert again["account"]["orders_today"] == []                    # same candle: nothing more to do
    assert again["account"]["nav"] == pytest.approx(first["account"]["nav"])
    state = json.loads((sandbox / "live" / "state.json").read_text())
    assert state["runs"] == 1
    invested = first["account"]["invested_pct"]
    assert 0.0 < invested <= 1.0 and first["account"]["cash"] >= 0.0


def test_a_missing_candle_freezes_that_market_instead_of_selling_it(sandbox):
    kw = dict(mode="paper", offline=True, live_dir=sandbox / "live", site_dir=sandbox / "site")
    first = run_daily.run(now=MONDAY, **kw)
    held = [p["symbol"] for p in first["account"]["positions"] if p["value"] > 0]
    assert held
    victim = held[0]
    store = PriceStore(sandbox / "live" / "prices")
    for inst in LIVE_INSTRUMENTS:                                    # everyone gets a new candle...
        df = store.load(inst.symbol)
        nxt = df.iloc[[-1]].copy()
        nxt.index = [df.index[-1] + pd.Timedelta(days=3)]            # (Friday -> Monday)
        if inst.symbol != victim:                                    # ...except the victim
            store.save(inst.symbol, pd.concat([df, nxt]))
    fx = store.load("EURUSD")
    nxt = fx.iloc[[-1]].copy()
    nxt.index = [fx.index[-1] + pd.Timedelta(days=3)]
    store.save("EURUSD", pd.concat([fx, nxt]))
    later = run_daily.run(now=datetime(2026, 11, 17, 1, 25, tzinfo=timezone.utc), **kw)
    assert victim in later["meta"]["stale_markets"]
    assert victim not in [o["symbol"] for o in later["account"]["orders_today"]]
    still = {p["symbol"]: p for p in later["account"]["positions"]}[victim]
    assert still["value"] > 0 and still["target"] > 0                # still held, still wanted


def test_circuit_breaker_moves_the_account_to_cash(sandbox):
    kw = dict(mode="paper", offline=True, live_dir=sandbox / "live", site_dir=sandbox / "site")
    run_daily.run(now=MONDAY, **kw)
    path = sandbox / "live" / "state.json"
    state = json.loads(path.read_text())
    state["peak_nav"] = state["nav"] * 1.5                           # pretend the account has fallen by a third
    state["last_bar"] = None
    path.write_text(json.dumps(state))
    out = run_daily.run(now=MONDAY, **kw)
    assert out["account"]["halted"]
    assert out["account"]["invested_pct"] == pytest.approx(0.0, abs=0.02)


def test_illiquid_tokens_are_kept_out_of_the_portfolio():
    idx = pd.date_range("2026-01-01", periods=120, freq="D")
    busy = pd.DataFrame({"close": 100.0, "volume": 50.0}, index=idx)
    dead = pd.DataFrame({"close": 100.0, "volume": 0.0}, index=idx)
    assert bool(venue_liquidity(busy, None).iloc[-1]) is True
    assert bool(venue_liquidity(dead, None).iloc[-1]) is False
    store = Path(__file__).resolve().parents[1] / "data" / "live" / "prices"
    if store.exists():
        panel = build_live_panel(PriceStore(store), LIVE_INSTRUMENTS)
        gate = panel.investable.iloc[-1]
        assert gate["SPY"] and gate["QQQ"] and gate["GLD"]
        assert not gate["VGK"] and not gate["IWM"]                   # listed on Kraken, but nobody trades them


def test_live_mode_end_to_end_against_the_simulated_exchange(sandbox, monkeypatch):
    """The whole daily run in live mode: starts from the exchange's balances, converts
    euros to dollars, buys the target, and does not trade again on a second run."""
    monkeypatch.delenv("HERMES_REBASE", raising=False)
    monkeypatch.setenv("HERMES_TOKENIZED_API", "yes")                 # an account Kraken lets trade them by API
    rules = {i.kraken_pair: {"price_decimals": 2, "lot_decimals": 8, "ordermin": 1e-8, "costmin": 0.5}
             for i in LIVE_INSTRUMENTS}
    rules["EURUSD"] = RULES["EURUSD"]
    ex = FakeExchange({"ZEUR": 233.47}, fee=0.001)
    kw = dict(mode="live", offline=True, live_dir=sandbox / "live", site_dir=sandbox / "site")

    dry = run_daily.run(now=MONDAY, broker=KrakenBroker(ex, LIVE_BY_SYMBOL, pair_rules=rules, pause=0.0), **kw)
    assert dry["account"]["orders_today"] and all(o["status"] == "validated" for o in dry["account"]["orders_today"])
    assert ex.bal == {"ZEUR": 233.47}                                 # unarmed: the account is untouched
    assert dry["account"]["initial_capital"] == pytest.approx(233.47)

    (sandbox / "live" / "state.json").unlink()
    armed = KrakenBroker(ex, LIVE_BY_SYMBOL, pair_rules=rules, armed=True, pause=0.0)
    out = run_daily.run(now=MONDAY, broker=armed, **kw)
    done = [o for o in out["account"]["orders_today"] if o["status"] == "filled"]
    assert {o["symbol"] for o in done} >= {"SPY", "GLD"} and any(o["symbol"] == "EURUSD" for o in done)
    assert all(v >= -1e-6 for v in ex.bal.values())                   # no balance ever went negative
    assert 0.3 < out["account"]["invested_pct"] < 1.0
    assert out["account"]["nav"] == pytest.approx(233.47, rel=0.02)   # only fees and slippage were lost

    sent = len(ex.sent)
    state = json.loads((sandbox / "live" / "state.json").read_text())
    state["last_bar"] = None                                          # as if the state write had been lost
    (sandbox / "live" / "state.json").write_text(json.dumps(state))
    again = run_daily.run(now=MONDAY, broker=armed, **kw)
    assert len(ex.sent) == sent and again["account"]["orders_today"] == []   # the exchange is the truth


# ----------------------------------------------------------------------------- trading hours


def test_tokenized_stocks_follow_new_york_hours_and_crypto_never_closes():
    spy, btc = LIVE_BY_SYMBOL["SPY"], LIVE_BY_SYMBOL["BTC"]
    utc = lambda *a: datetime(*a, tzinfo=timezone.utc)               # noqa: E731
    assert not run_daily.market_open(spy, utc(2026, 11, 16, 0, 25))  # Sunday 7:25 pm in New York: still shut
    assert run_daily.market_open(spy, utc(2026, 11, 16, 1, 25))      # Sunday 8:25 pm: the week has opened
    assert run_daily.market_open(spy, utc(2026, 11, 18, 14, 0))      # midweek
    assert run_daily.market_open(spy, utc(2026, 11, 14, 0, 25))      # Friday 7:25 pm: last half hour
    assert not run_daily.market_open(spy, utc(2026, 11, 14, 1, 25))  # Friday 8:25 pm: closed for the weekend
    assert not run_daily.market_open(spy, utc(2026, 11, 15, 12, 0))  # Sunday noon
    assert run_daily.market_open(spy, utc(2026, 7, 13, 0, 25))       # summer time: Sunday 8:25 pm is 00:25 UTC
    assert all(run_daily.market_open(btc, utc(2026, 11, d, 12, 0)) for d in (14, 15, 16))


# ----------------------------------------------------------------------------- dry run


def test_dry_run_client_can_only_read_and_validate(monkeypatch):
    from hermes_quant.live.check_kraken import DryRunClient

    reached = []
    monkeypatch.setattr(kraken.KrakenClient, "_private", lambda self, endpoint, data=None: reached.append(endpoint) or {})
    client = DryRunClient("key", "c2VjcmV0")
    client.balances()
    client.open_orders()
    client.add_order("XBTEUR", "buy", "0.001", price="50000.0")                       # validate-only by default
    with pytest.raises(RuntimeError):
        client.add_order("XBTEUR", "buy", "0.001", price="50000.0", validate=False)   # a real order
    with pytest.raises(RuntimeError):
        client.cancel_order("T1")
    with pytest.raises(RuntimeError):
        client._private("Withdraw", {"asset": "ZEUR"})
    assert reached == ["Balance", "OpenOrders", "AddOrder"]                           # nothing else got through


def test_dry_run_pushes_the_target_through_the_broker_without_trading(monkeypatch):
    from hermes_quant.live import check_kraken

    class Exchange(FakeExchange):
        def open_orders(self):
            return {}

        def _private(self, endpoint, data=None):
            assert endpoint == "ClosedOrders"
            return {"closed": {}}

    rules = {**RULES, "GLDxUSD": RULES["SPYxUSD"]}
    quotes = {"XBTEUR": 60000.0, "SPYxUSD": 748.0, "GLDxUSD": 430.0, "EURUSD": EURUSD}
    monkeypatch.setattr(kraken, "pair_rules", lambda tokenized=False: rules)
    monkeypatch.setattr(kraken, "ticker", lambda pair, tokenized=False: {"mid": quotes[pair]})
    ex = Exchange({"ZEUR": 233.47})
    rows = check_kraken.check(ex, {"target": {"SPY": 0.16, "GLD": 0.28}}, now=MONDAY)
    assert all(ok for _, ok, _ in rows), rows
    names = " | ".join(name for name, _, _ in rows)
    for wanted in ("read balances", "read closed orders", "account value", "validate euros to dollars",
                   "validate buy SPY", "validate buy GLD", "validate buy BTC"):
        assert wanted in names
    assert ex.sent and all(o["validate"] for o in ex.sent)            # every order was a dry run
    assert ex.bal == {"ZEUR": 233.47}                                 # and the account is untouched


def test_dry_run_reports_a_rejection_with_the_exchanges_own_words(monkeypatch):
    from hermes_quant.live import check_kraken

    class Exchange(FakeExchange):
        def open_orders(self):
            return {}

        def _private(self, endpoint, data=None):
            return {"closed": {}}

        def add_order(self, pair, side, volume, *, price, tokenized=False, validate=True):
            if tokenized:
                raise kraken.KrakenError("AddOrder: EGeneral:Permission denied")
            return super().add_order(pair, side, volume, price=price, tokenized=tokenized, validate=validate)

    monkeypatch.setattr(kraken, "pair_rules", lambda tokenized=False: RULES)
    monkeypatch.setattr(kraken, "ticker", lambda pair, tokenized=False: {"mid": {"XBTEUR": 60000.0, "SPYxUSD": 748.0, "EURUSD": EURUSD}[pair]})
    monkeypatch.setenv("HERMES_TOKENIZED_API", "yes")                 # the run is set to send them: a refusal is a failure
    rows = check_kraken.check(Exchange({"ZEUR": 233.47}), {"target": {"SPY": 0.16}}, now=MONDAY)
    failed = [(name, detail) for name, ok, detail in rows if not ok]
    assert len(failed) == 1 and "SPY" in failed[0][0] and "Permission denied" in failed[0][1]

    monkeypatch.delenv("HERMES_TOKENIZED_API")                        # the default: they are placed by hand anyway
    rows = check_kraken.check(Exchange({"ZEUR": 233.47}), {"target": {"SPY": 0.16}}, now=MONDAY)
    assert all(ok for _, ok, _ in rows)
    spy = next(detail for name, _, detail in rows if "SPY" in name)
    assert "Permission denied" in spy and "by hand" in spy            # still says what Kraken answered


def test_a_conversion_larger_than_the_order_cap_is_split_so_the_purchases_are_fully_funded():
    ex = FakeExchange({"ZEUR": 233.47}, fee=0.002)
    b = broker(ex, armed=True, max_order_eur=100.0)
    fills = b.execute({"SPY": 70.0, "QQQ": 58.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)
    conversions = [o for o in ex.sent if o["pair"] == "EURUSD"]
    assert len(conversions) == 2 and all(o["volume"] <= 100.0 + 1e-9 for o in conversions)    # 100 EUR, then the rest
    bought = {f.symbol: f.notional_eur for f in fills if f.symbol in ("SPY", "QQQ")}
    assert bought["SPY"] == pytest.approx(70.0, rel=0.02) and bought["QQQ"] == pytest.approx(58.0, rel=0.02)
    assert ex.bal["ZEUR"] > 0 and ex.bal.get("ZUSD", 0.0) > -1e-6                              # nothing overdrawn


def test_a_conversion_that_does_not_fill_is_not_retried():
    ex = FakeExchange({"ZEUR": 233.47}, fill_ratio=0.0)
    b = broker(ex, armed=True, max_order_eur=100.0)
    b.execute({"SPY": 70.0, "QQQ": 58.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)
    assert len([o for o in ex.sent if o["pair"] == "EURUSD"]) == 1
    assert not [o for o in ex.sent if o["pair"] != "EURUSD" and not o["validate"]]             # no dollars, no purchases


# ----------------------------------------------------------------------------- accounts Kraken restricts


class RefusesTokenized(FakeExchange):
    """Kraken's answer to a tokenized-stock order sent by API from a European account."""

    def add_order(self, pair, side, volume, *, price, tokenized=False, validate=True):
        if tokenized:
            self.sent.append({"pair": pair, "side": side, "volume": float(volume), "price": float(price),
                              "tokenized": tokenized, "validate": validate})
            raise kraken.KrakenError("AddOrder: EGeneral:Permission denied")
        return super().add_order(pair, side, volume, price=price, tokenized=tokenized, validate=validate)


def test_no_dollars_are_bought_for_purchases_the_exchange_refuses():
    ex = RefusesTokenized({"ZEUR": 233.47})
    fills = broker(ex, armed=True).execute({"SPY": 40.0, "QQQ": 30.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)
    assert not [o for o in ex.sent if o["pair"] == "EURUSD"]          # not one euro was converted
    assert not [o for o in ex.sent if not o["validate"]]              # only the rehearsal reached the exchange
    assert {f.symbol: f.status for f in fills} == {"SPY": "rejected", "QQQ": "rejected"}
    assert all("Permission denied" in f.detail for f in fills)
    assert ex.bal == {"ZEUR": 233.47}


def test_tokenized_orders_are_left_for_the_owner_when_the_account_cannot_send_them():
    ex = RefusesTokenized({"ZEUR": 233.47})
    b = broker(ex, armed=True, tokenized_via_api=False)
    fills = b.execute({"BTC": 30.0, "SPY": 40.0}, PRICE_EUR, LIVE_BY_SYMBOL, CostModel(), eurusd=EURUSD)
    assert [o["pair"] for o in ex.sent] == ["XBTEUR"]                 # crypto still trades by itself
    by = {f.symbol: f for f in fills}
    assert by["BTC"].status == "filled" and by["SPY"].status == "manual" and by["SPY"].units == 0.0
    assert "ZUSD" not in ex.bal


def test_live_run_on_a_restricted_account_lists_the_stock_orders_instead_of_sending_them(sandbox, monkeypatch):
    monkeypatch.delenv("HERMES_REBASE", raising=False)
    monkeypatch.delenv("HERMES_TOKENIZED_API", raising=False)         # the default: a European account
    rules = {i.kraken_pair: {"price_decimals": 2, "lot_decimals": 8, "ordermin": 1e-8, "costmin": 0.5}
             for i in LIVE_INSTRUMENTS}
    rules["EURUSD"] = RULES["EURUSD"]
    shutil.copytree(sandbox / "live", sandbox / "paper")              # the same prices, for a simulated account
    ex = RefusesTokenized({"ZEUR": 233.47})
    armed = KrakenBroker(ex, LIVE_BY_SYMBOL, pair_rules=rules, armed=True, pause=0.0, tokenized_via_api=False)
    kw = dict(mode="live", offline=True, live_dir=sandbox / "live", site_dir=sandbox / "site")
    out = run_daily.run(now=MONDAY, broker=armed, **kw)
    a = out["account"]
    assert not [o for o in ex.sent if o["tokenized"] or o["pair"] == "EURUSD"]     # nothing refused, nothing converted
    assert ex.bal.get("ZUSD", 0.0) == 0.0
    by_hand = {o["symbol"]: o["notional_eur"] for o in a["manual_orders"]}
    assert {"SPY", "GLD"} <= set(by_hand) and all(n > 0 for n in by_hand.values())
    assert sum(by_hand.values()) <= a["cash"]                                       # never asks for money that is not there
    assert all(LIVE_BY_SYMBOL[s].tokenized for s in by_hand)
    again = run_daily.run(now=MONDAY, broker=armed, **kw)                           # a standing instruction, not a one-off
    assert {o["symbol"] for o in again["account"]["manual_orders"]} == set(by_hand)
    paper = run_daily.run(mode="paper", offline=True, now=MONDAY, live_dir=sandbox / "paper", site_dir=sandbox / "site2")
    assert paper["account"]["manual_orders"] == []                                  # a simulated account fills everything


def test_the_published_schedule_matches_the_workflow_and_assets_say_how_to_follow_them(sandbox):
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "daily.yml").read_text()
    minute, hour = workflow.split('cron: "')[1].split('"')[0].split()[:2]
    assert run_daily.SCHEDULE_UTC == f"{int(hour):02d}:{int(minute):02d}"
    out = run_daily.run(mode="paper", offline=True, now=MONDAY, live_dir=sandbox / "live", site_dir=sandbox / "site")
    assert out["meta"]["schedule_utc"] == run_daily.SCHEDULE_UTC
    btc = next(a for a in out["model"]["assets"] if a["symbol"] == "BTC")
    spy = next(a for a in out["model"]["assets"] if a["symbol"] == "SPY")
    assert (btc["pair"], btc["quote"], btc["tokenized"]) == ("XBTEUR", "EUR", False)
    assert (spy["pair"], spy["quote"], spy["tokenized"]) == ("SPYxUSD", "USD", True)


def test_tokenized_holdings_are_found_under_the_names_kraken_really_uses():
    """Taken from the account on 16 Nov 2026: xStocks bought by hand sit under "<name>.T".
    Missed, they looked like money gone and tripped the circuit breaker."""
    ex = FakeExchange({"GLDx.T": 0.17549, "QQQx.T": 0.02303, "SLVx.T": 0.205367, "SPYx.T": 0.048593, "XETH": 1.4409e-06, "ZEUR": 121.4709})
    b = KrakenBroker(ex, LIVE_BY_SYMBOL, pair_rules=RULES, pause=0.0)
    assert b.units() == {"ETH": 1.4409e-06, "SPY": 0.048593, "QQQ": 0.02303, "GLD": 0.17549, "SLV": 0.205367}
    assert b.cash() == pytest.approx(121.4709) and b.usd_cash() == 0.0


# ----------------------------------------------------------------------------- money that moves without the model


def _restricted_live(sandbox, monkeypatch, balances):
    monkeypatch.delenv("HERMES_REBASE", raising=False)
    monkeypatch.delenv("HERMES_TOKENIZED_API", raising=False)
    rules = {i.kraken_pair: {"price_decimals": 2, "lot_decimals": 8, "ordermin": 1e-8, "costmin": 0.5} for i in LIVE_INSTRUMENTS}
    rules["EURUSD"] = RULES["EURUSD"]
    ex = RefusesTokenized(balances)
    broker_ = KrakenBroker(ex, LIVE_BY_SYMBOL, pair_rules=rules, armed=True, pause=0.0, tokenized_via_api=False)
    return ex, broker_, dict(mode="live", offline=True, live_dir=sandbox / "live", site_dir=sandbox / "site")


def test_money_moved_without_the_model_is_a_flow_not_a_loss(sandbox, monkeypatch):
    ex, b, kw = _restricted_live(sandbox, monkeypatch, {"ZEUR": 233.47})
    first = run_daily.run(now=MONDAY, broker=b, **kw)["account"]
    assert first["flow_today"] == 0.0 and first["unrecognised"] == {}

    ex.bal["ZEUR"] -= 112.0                                          # half the cash buys something the model has no market for
    ex.bal["PEPE"] = 5e6
    b.refresh()
    a = run_daily.run(now=datetime(2026, 11, 17, 1, 25, tzinfo=timezone.utc), broker=b, **kw)["account"]
    assert a["flow_today"] == pytest.approx(-112.0, abs=0.01)
    assert not a["halted"]                                           # a 48% "fall" that is not a loss
    assert a["initial_capital"] == pytest.approx(121.47, abs=0.01)
    assert a["return_since_inception"] == pytest.approx(0.0, abs=1e-6) and a["drawdown"] == pytest.approx(0.0, abs=1e-6)
    assert a["unrecognised"] == {"PEPE": 5e6}

    ex.bal["ZEUR"] += 500.0                                          # and a deposit is no profit
    b.refresh()
    a = run_daily.run(now=datetime(2026, 11, 18, 1, 25, tzinfo=timezone.utc), broker=b, **kw)["account"]
    assert a["flow_today"] == pytest.approx(500.0, abs=0.01)
    assert a["return_since_inception"] == pytest.approx(0.0, abs=1e-6)
    assert [f["eur"] for f in a["flows"]] == [-112.0, 500.0]


def test_buying_the_listed_funds_by_hand_is_seen_as_holdings_and_costs_not_as_a_flow(sandbox, monkeypatch):
    ex, b, kw = _restricted_live(sandbox, monkeypatch, {"ZEUR": 233.47})
    asked = {o["symbol"]: o["notional_eur"] for o in run_daily.run(now=MONDAY, broker=b, **kw)["account"]["manual_orders"]}
    state = json.loads((sandbox / "live" / "state.json").read_text())
    live = json.loads((sandbox / "site" / "live.json").read_text())
    price = {p["symbol"]: p["price"] for p in live["account"]["positions"]}
    for symbol, euros in asked.items():                              # the owner places them, paying a 1% margin
        ex.bal["ZEUR"] -= euros
        ex.bal[LIVE_BY_SYMBOL[symbol].kraken_pair[:-3] + ".T"] = euros / (price[symbol] * 1.01)
    b.refresh()
    a = run_daily.run(now=datetime(2026, 11, 17, 1, 25, tzinfo=timezone.utc), broker=b, **kw)["account"]
    assert a["flow_today"] == 0.0 and not a["halted"]
    assert {p["symbol"] for p in a["positions"] if p["value"] > 0.01} == set(asked)
    assert a["nav"] == pytest.approx(233.47 - 0.01 * sum(asked.values()), rel=2e-3)     # the margin is the only thing lost
    assert state["initial_capital"] == pytest.approx(233.47) == pytest.approx(a["initial_capital"])


def test_a_real_fall_still_trips_the_breaker_in_live_mode(sandbox, monkeypatch):
    ex, b, kw = _restricted_live(sandbox, monkeypatch, {"ZEUR": 233.47})
    run_daily.run(now=MONDAY, broker=b, **kw)
    path = sandbox / "live" / "state.json"
    state = json.loads(path.read_text())
    state["peak_nav"] = state["nav"] * 1.5                           # the account has fallen by a third, holdings unchanged
    path.write_text(json.dumps(state))
    a = run_daily.run(now=datetime(2026, 11, 17, 1, 25, tzinfo=timezone.utc), broker=b, **kw)["account"]
    assert a["flow_today"] == 0.0 and a["halted"]


# ----------------------------------------------------------------------------- the owner's basket of single stocks

STOCKS = {"NVDA": "Nvidia", "TSLA": "Tesla", "AAPL": "Apple", "GOOGL": "Alphabet", "MSFT": "Microsoft", "AMZN": "Amazon"}
ANY_TREND = favourites.Basket(stocks=STOCKS, enter=0.001, leave=0.001)      # holds four of them whatever the market did
STOCK_SYMBOLS = {**LIVE_BY_SYMBOL, **{i.symbol: i for i in favourites.instruments(ANY_TREND)}}


def test_the_basket_rule_has_few_seats_and_does_not_flip_at_the_line():
    b = favourites.Basket(stocks={s: s for s in "ABCDEF"}, per_stock=0.025, max_stocks=2)
    ok = dict.fromkeys("ABCDEF", True)
    nan = float("nan")
    # nothing held: the two strongest trends at 55% or more come in, each at the same size
    assert favourites.select({"A": 0.9, "B": 0.6, "C": 0.56, "D": 0.54, "E": 0.2, "F": nan}, ok, set(), b) == {"A": 0.025, "B": 0.025}
    # between 45% and 55% a holding stays and a newcomer waits; a stronger newcomer cannot take a seat that is in use
    assert set(favourites.select({"A": 0.5, "B": 0.46, "C": 0.95}, ok, {"A", "B"}, b)) == {"A", "B"}
    assert set(favourites.select({"A": 0.5, "C": 0.5}, ok, {"A"}, b)) == {"A"}
    # below 45% it goes, and the seat goes to the strongest of those waiting
    assert set(favourites.select({"A": 0.44, "B": 0.46, "C": 0.7, "D": 0.95}, ok, {"A", "B"}, b)) == {"B", "D"}
    # a stock that may not be held (too new, or a market nobody trades) is never held, whatever its trend
    assert favourites.select({"A": 0.99, "B": 0.6}, {"A": False, "B": True}, {"A"}, b) == {"B": 0.025}
    assert favourites.select({"Z": 0.99}, {"Z": True}, set(), b) == {}             # not on the list


def test_the_favourites_file_is_small_by_construction_and_typos_are_refused(tmp_path):
    shipped = favourites.load()
    assert shipped.budget <= 0.10 + 1e-12 and shipped.per_stock <= 0.03
    assert not set(shipped.stocks) & set(LIVE_BY_SYMBOL)

    def refused(**settings) -> bool:
        path = tmp_path / "f.json"
        path.write_text(json.dumps({"stocks": {"NVDA": "Nvidia"}, **settings}))
        try:
            favourites.load(path)
        except ValueError:
            return True
        return False

    assert not refused(per_stock=0.03, max_stocks=5)
    assert refused(per_stock=2.5)                                    # "2.5" meant as a percentage
    assert refused(per_stock=0.10, max_stocks=4)                     # 40% of the account
    assert refused(max_stocks=0) and refused(per_stock=0.0)
    assert refused(enter=0.4, leave=0.6) and refused(min_history_days=10)
    assert refused(stocks={"SPY": "already the model's"}) and refused(stocks={"nvda corp": "not a ticker"})
    assert refused(budget=0.5)                                       # a setting that does not exist


def test_a_stock_too_new_for_the_trend_filter_is_not_held(sandbox):
    store = PriceStore(sandbox / "live" / "prices")
    bar = store.load("BTC").index[-1]
    old_enough = favourites.evaluate(store, ANY_TREND, run_daily.PROFILES["balanced"], bar, set())
    assert len(old_enough["weights"]) == 4 and set(old_enough["weights"].values()) == {0.025}
    young = favourites.Basket(stocks=STOCKS, enter=0.001, leave=0.001, min_history_days=10**6)
    view = favourites.evaluate(store, young, run_daily.PROFILES["balanced"], bar, set())
    assert view["weights"] == {} and {r["status"] for r in view["rows"]} == {"too_new"}
    unlisted = favourites.Basket(stocks={"ZZZZ": "Not listed anywhere"})
    assert favourites.evaluate(store, unlisted, run_daily.PROFILES["balanced"], bar, set())["rows"][0]["status"] == "no_data"


def test_the_basket_is_off_until_switched_on_and_never_changes_what_the_model_does(sandbox, monkeypatch):
    shutil.copytree(sandbox / "live", sandbox / "with")
    off = run_daily.run(mode="paper", offline=True, now=MONDAY, basket=ANY_TREND, live_dir=sandbox / "live", site_dir=sandbox / "site")
    state_off = json.loads((sandbox / "live" / "state.json").read_text())
    assert not off["favourites"]["enabled"] and "basket_target" not in state_off
    assert {r["symbol"] for r in off["favourites"]["rows"]} == set(STOCKS)           # scored and shown all the same
    assert not [o for o in off["account"]["orders_today"] if o["symbol"] in STOCKS]
    assert not [p for p in off["account"]["positions"] if p["class"] == "stock"]

    monkeypatch.setenv("HERMES_FAVOURITES", "yes")
    on = run_daily.run(mode="paper", offline=True, now=MONDAY, basket=ANY_TREND, live_dir=sandbox / "with", site_dir=sandbox / "site2")
    state_on = json.loads((sandbox / "with" / "state.json").read_text())
    assert state_on["target"] == state_off["target"] and state_on["model"] == state_off["model"]
    basket = state_on["basket_target"]
    assert len(basket) == 4 and set(basket) <= set(STOCKS)
    assert sum(basket.values()) == pytest.approx(ANY_TREND.budget) and max(basket.values()) <= ANY_TREND.per_stock + 1e-12
    orders = lambda out: {o["symbol"]: o["notional_eur"] for o in out["account"]["orders_today"]}      # noqa: E731
    for symbol, euros in orders(off).items():                                         # the model's own orders are the same
        assert orders(on)[symbol] == pytest.approx(euros)
    bought = {s: n for s, n in orders(on).items() if s in STOCKS}
    assert set(bought) == set(basket) and all(0 < n <= ANY_TREND.per_stock * on["account"]["nav"] for n in bought.values())
    held = [p for p in on["account"]["positions"] if p["class"] == "stock"]
    assert len(held) == 4 and all(p["managed"] and p["value"] > 0 for p in held)
    assert on["favourites"]["invested"] <= ANY_TREND.budget


def test_the_basket_only_uses_money_the_model_leaves_idle(sandbox, monkeypatch):
    monkeypatch.setenv("HERMES_FAVOURITES", "yes")
    real = run_daily.target_weights

    def nearly_fully_invested(features, cfg, t):
        allocation = real(features, cfg, t)
        allocation.weights = allocation.weights * (0.96 / allocation.weights.sum())
        return allocation

    monkeypatch.setattr(run_daily, "target_weights", nearly_fully_invested)
    run_daily.run(mode="paper", offline=True, now=MONDAY, basket=ANY_TREND, live_dir=sandbox / "live", site_dir=sandbox / "site")
    state = json.loads((sandbox / "live" / "state.json").read_text())
    assert sum(state["target"].values()) == pytest.approx(0.96)
    assert sum(state["basket_target"].values()) == pytest.approx(0.04)                # not the 10% it would like
    assert sum(state["target"].values()) + sum(state["basket_target"].values()) <= 1.0 + 1e-9


def _stock_rules():
    rules = {i.kraken_pair: {"price_decimals": 2, "lot_decimals": 8, "ordermin": 1e-8, "costmin": 0.5} for i in STOCK_SYMBOLS.values()}
    return {**rules, "EURUSD": RULES["EURUSD"]}


def test_on_a_european_account_the_basket_is_listed_for_the_owner_not_sent(sandbox, monkeypatch):
    monkeypatch.delenv("HERMES_REBASE", raising=False)
    monkeypatch.delenv("HERMES_TOKENIZED_API", raising=False)
    monkeypatch.setenv("HERMES_FAVOURITES", "yes")
    ex = RefusesTokenized({"ZEUR": 233.47})
    b = KrakenBroker(ex, STOCK_SYMBOLS, pair_rules=_stock_rules(), armed=True, pause=0.0, tokenized_via_api=False)
    out = run_daily.run(mode="live", offline=True, now=MONDAY, broker=b, basket=ANY_TREND, live_dir=sandbox / "live", site_dir=sandbox / "site")
    assert not [o for o in ex.sent if o["tokenized"] or o["pair"] == "EURUSD"]         # no stock order, no dollars bought
    by_hand = {o["symbol"]: o for o in out["account"]["manual_orders"]}
    stocks = {s: o for s, o in by_hand.items() if o["class"] == "stock"}
    assert len(stocks) == 4 and all(o["notional_eur"] > 0 and o["pair"] == f"{s}xUSD" for s, o in stocks.items())
    assert sum(o["notional_eur"] for o in by_hand.values()) <= out["account"]["cash"]


def test_a_stock_the_owner_bought_himself_is_valued_but_left_alone_while_the_basket_is_off(sandbox, monkeypatch):
    monkeypatch.delenv("HERMES_REBASE", raising=False)
    monkeypatch.delenv("HERMES_TOKENIZED_API", raising=False)
    ex = RefusesTokenized({"ZEUR": 200.0, "NVDAx.T": 0.2})
    b = KrakenBroker(ex, STOCK_SYMBOLS, pair_rules=_stock_rules(), armed=True, pause=0.0, tokenized_via_api=False)
    a = run_daily.run(mode="live", offline=True, now=MONDAY, broker=b, basket=ANY_TREND, live_dir=sandbox / "live", site_dir=sandbox / "site")["account"]
    nvda = next(p for p in a["positions"] if p["symbol"] == "NVDA")
    assert nvda["value"] > 5.0 and nvda["class"] == "stock" and not nvda["managed"] and nvda["target"] == 0.0
    assert a["nav"] == pytest.approx(200.0 + nvda["value"])                            # counted in the account's value
    assert a["unrecognised"] == {} and a["flow_today"] == 0.0
    assert "NVDA" not in {o["symbol"] for o in a["manual_orders"]}                     # and nobody is told to sell it


def test_switching_the_basket_off_sells_nothing(sandbox, monkeypatch):
    kw = dict(mode="paper", offline=True, now=MONDAY, basket=ANY_TREND, live_dir=sandbox / "live", site_dir=sandbox / "site")
    monkeypatch.setenv("HERMES_FAVOURITES", "yes")
    on = run_daily.run(**kw)
    held = {p["symbol"]: p["value"] for p in on["account"]["positions"] if p["class"] == "stock"}
    assert len(held) == 4

    monkeypatch.setenv("HERMES_FAVOURITES", "no")
    path = sandbox / "live" / "state.json"
    state = json.loads(path.read_text())
    state["last_bar"] = None                                         # a new day
    path.write_text(json.dumps(state))
    off = run_daily.run(**kw)
    assert "basket_target" not in json.loads(path.read_text())
    assert not [o for o in off["account"]["orders_today"] if o["symbol"] in STOCKS]
    after = {p["symbol"]: p for p in off["account"]["positions"] if p["class"] == "stock"}
    assert set(after) == set(held) and all(not p["managed"] and p["target"] == 0.0 for p in after.values())


def test_the_circuit_breaker_sells_the_basket_too(sandbox, monkeypatch):
    monkeypatch.setenv("HERMES_FAVOURITES", "yes")
    kw = dict(mode="paper", offline=True, now=MONDAY, basket=ANY_TREND, live_dir=sandbox / "live", site_dir=sandbox / "site")
    assert [p for p in run_daily.run(**kw)["account"]["positions"] if p["class"] == "stock" and p["value"] > 0]
    path = sandbox / "live" / "state.json"
    state = json.loads(path.read_text())
    state.update(peak_nav=state["nav"] * 1.5, last_bar=None)
    path.write_text(json.dumps(state))
    out = run_daily.run(**kw)
    assert out["account"]["halted"] and out["account"]["invested_pct"] == pytest.approx(0.0, abs=0.02)


def test_a_stock_whose_candle_is_missing_today_is_not_traded(sandbox, monkeypatch):
    monkeypatch.setenv("HERMES_FAVOURITES", "yes")
    store = PriceStore(sandbox / "live" / "prices")
    for symbol in STOCKS:                                            # every stock's download failed tonight
        store.save(symbol, store.load(symbol).iloc[:-1])
    out = run_daily.run(mode="paper", offline=True, now=MONDAY, basket=ANY_TREND, live_dir=sandbox / "live", site_dir=sandbox / "site")
    assert all(r["stale"] for r in out["favourites"]["rows"])
    traded = {o["symbol"] for o in out["account"]["orders_today"]}
    assert not traded & set(STOCKS) and {"SPY", "GLD"} <= traded     # the model's own markets trade as usual
    assert out["meta"]["stale_markets"] == [] and not out["meta"]["data_outage"]


def test_a_broken_favourites_file_is_reported_and_the_model_carries_on(sandbox, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_FAVOURITES", "yes")
    bad = tmp_path / "favourites.json"
    bad.write_text('{"stocks": {"NVDA": "Nvidia"}, "per_stock": 25}')
    monkeypatch.setattr(favourites, "CONFIG", bad)
    out = run_daily.run(mode="paper", offline=True, now=MONDAY, live_dir=sandbox / "live", site_dir=sandbox / "site")
    assert "per_stock" in out["favourites"]["error"] and not out["favourites"]["enabled"]
    assert {"SPY", "GLD"} <= {o["symbol"] for o in out["account"]["orders_today"]}
    assert not [p for p in out["account"]["positions"] if p["class"] == "stock"]


def test_the_nightly_workflow_passes_the_basket_switch_and_defaults_to_off():
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "daily.yml").read_text()
    assert "HERMES_FAVOURITES: ${{ vars.HERMES_FAVOURITES || 'no' }}" in workflow
    assert not favourites.enabled()
