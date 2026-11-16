"""Account and execution rehearsals against an in-memory API; no keys or network."""

from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone

import pytest
import requests

from hermes_quant.data import trading212 as api
from hermes_quant.data.trading212 import ReadOnlyClient, Trading212Client, Trading212Error
from hermes_quant.live import check_trading212 as rehearsal
from hermes_quant.live.trading212 import Trading212Broker, resolve, session_open

NOW = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)
WANTED = {"SPY": {"isin": ["IE00B5BMR087"]}, "NVDA": {"isin": ["US67066G1040"]}}
INSTRUMENTS = [
    {"ticker": "SPY_US_EQ", "isin": "IE00B5BMR087", "currencyCode": "USD", "type": "ETF",
     "name": "S&P 500", "workingScheduleId": 10},
    {"ticker": "SPY_EU_EQ", "isin": "IE00B5BMR087", "currencyCode": "EUR", "type": "ETF",
     "name": "S&P 500", "workingScheduleId": 10},
    {"ticker": "NVDA_US_EQ", "isin": "US67066G1040", "currencyCode": "USD", "type": "STOCK",
     "name": "Nvidia", "workingScheduleId": 10},
]
EXCHANGES = [{"id": 1, "name": "Test exchange", "workingSchedules": [{"id": 10, "timeEvents": [
    {"date": "2026-11-16T09:00:00Z", "type": "OPEN"},
    {"date": "2026-11-16T13:00:00+00:00", "type": "BREAK_START"},
    {"date": "2026-11-16T13:30:00+00:00", "type": "BREAK_END"},
    {"date": "2026-11-16T17:00:00Z", "type": "CLOSE"},
]}]}]
PRICES = {"SPY": 10.0, "NVDA": 20.0}
LISTINGS = resolve(WANTED, INSTRUMENTS)[0]


@pytest.fixture(autouse=True)
def no_network_or_real_sleep(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("this test must not use the network or sleep")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(api.time, "sleep", forbidden)


class FakeTrading212:
    """Model cash reservations, fractional fills and ambiguous responses without a server."""

    def __init__(self, cash=100.0, positions=None, *, fill_ratio=1.0, env="demo"):
        self.env, self.cash_balance, self.reserved = env, cash, 0.0
        self.holdings = dict(positions or {})
        self.fill_ratio, self.working = fill_ratio, False
        self.reject, self.uncertain = {}, None
        self.instrument_list, self.exchange_list = deepcopy(INSTRUMENTS), deepcopy(EXCHANGES)
        self.orders, self.history_list = {}, []
        self.sent, self.cancelled, self.log = [], [], []
        self.fill_prices = {"SPY_EU_EQ": 10.0, "SPY_US_EQ": 11.0, "NVDA_US_EQ": 22.0}
        self.fx = {"EUR": 1.0, "USD": 1 / 1.1}
        self.hide_history, self.completed_404 = False, False
        self.signed_fills = True
        self.currency = "EUR"

    def summary(self):
        self.log.append(("summary",))
        reserved = self.reserved_cash()
        return {"currency": self.currency, "cash": {"availableToTrade": self.cash_balance - reserved,
                                                    "reservedForOrders": reserved}}

    def reserved_cash(self):
        reserved = self.reserved
        for order in self.pending_orders_data():
            if order["quantity"] > 0:
                inst = next(i for i in self.instrument_list if i["ticker"] == order["ticker"])
                price = self.fill_prices[order["ticker"]] * self.fx[inst["currencyCode"]]
                fee = 0.0015 if inst["currencyCode"] != "EUR" else 0.0
                reserved += (order["quantity"] - abs(order["filledQuantity"])) * price * (1 + fee)
        return reserved

    def positions(self):
        self.log.append(("positions",))
        result = []
        for ticker, quantity in self.holdings.items():
            inst = next((i for i in self.instrument_list if i["ticker"] == ticker), {"currencyCode": "EUR"})
            value = quantity * self.fill_prices.get(ticker, 7.0) * self.fx[inst["currencyCode"]]
            reserved = sum(max(0, abs(o["quantity"]) - abs(o["filledQuantity"])) for o in self.pending_orders_data()
                           if o["ticker"] == ticker and o["quantity"] < 0)
            result.append({"instrument": {"ticker": ticker}, "quantity": quantity,
                           "quantityAvailableForTrading": max(0.0, quantity - reserved),
                           "currentPrice": self.fill_prices.get(ticker, 7.0), "walletImpact": {"currentValue": value}})
        return deepcopy(result)

    def instruments(self):
        self.log.append(("instruments",))
        return deepcopy(self.instrument_list)

    def exchanges(self):
        self.log.append(("exchanges",))
        return deepcopy(self.exchange_list)

    def pending_orders_data(self):
        return [o for o in self.orders.values() if o["status"] not in ("FILLED", "CANCELLED", "REJECTED")]

    def pending_orders(self):
        self.log.append(("pending_orders",))
        return deepcopy(self.pending_orders_data())

    def order(self, order_id):
        self.log.append(("order", order_id))
        order = self.orders.get(order_id)
        if order is None or (self.completed_404 and order["status"] == "FILLED"):
            raise Trading212Error("Order not found", status=404)
        return deepcopy(order)

    def history(self, ticker=None, limit=50):
        self.log.append(("history", ticker, limit))
        if self.hide_history:
            return []
        return deepcopy([h for h in self.history_list if ticker is None or h["order"]["ticker"] == ticker][:limit])

    def market_order(self, ticker, quantity, extended_hours=False):
        self.log.append(("market_order", ticker, quantity))
        self.sent.append({"ticker": ticker, "quantity": quantity, "extendedHours": extended_hours})
        if ticker in self.reject:
            raise Trading212Error(self.reject[ticker], status=400)
        failure, self.uncertain = self.uncertain, None
        if failure == "before":
            raise Trading212Error("connection lost", uncertain=True)
        inst = next(i for i in self.instrument_list if i["ticker"] == ticker)
        unit_price = self.fill_prices[ticker] * self.fx[inst["currencyCode"]]
        executed = quantity * (0 if self.working else self.fill_ratio)
        amount = abs(executed) * unit_price
        fee = amount * (0.0015 if inst["currencyCode"] != "EUR" else 0)
        if quantity > 0:
            assert amount + fee <= self.cash_balance - self.reserved_cash() + 1e-8, "overspent available cash"
        else:
            assert abs(executed) <= self.holdings.get(ticker, 0) + 1e-8, "sold unavailable shares"
        self.cash_balance -= (amount if quantity > 0 else -amount) + fee
        self.holdings[ticker] = self.holdings.get(ticker, 0) + executed
        order_id = len(self.orders) + 1
        order = {"id": order_id, "ticker": ticker, "instrument": {"ticker": ticker}, "initiatedFrom": "API",
                 "quantity": quantity, "filledQuantity": executed,
                 "status": "FILLED" if self.fill_ratio == 1 and not self.working else "PARTIALLY_FILLED" if executed else "NEW"}
        self.orders[order_id] = order
        if executed:
            fill = {"id": order_id * 100, "quantity": executed if self.signed_fills else abs(executed),
                    "walletImpact": {"netValue": -amount if quantity > 0 else amount,
                                     "taxes": [{"name": "CURRENCY_CONVERSION_FEE", "quantity": -fee}]}}
            self.history_list.insert(0, {"order": deepcopy(order), "fill": fill})
        if failure == "after":
            raise Trading212Error("response lost", uncertain=True)
        return deepcopy(order)

    def cancel(self, order_id):
        self.log.append(("cancel", order_id))
        self.cancelled.append(order_id)
        self.orders[order_id]["status"] = "CANCELLED"


class FakeClock:
    def __init__(self):
        self.now, self.waits = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.waits.append(seconds)
        self.now += seconds


class Response:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code, self.payload, self.text, self.headers = status, payload, text, headers or {}

    def json(self):
        return deepcopy(self.payload)


class FakeSession:
    def __init__(self, responses):
        self.responses, self.sent = list(responses), []

    def request(self, method, url, **kwargs):
        self.sent.append({"method": method, "url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def http(responses, env="demo", cls=Trading212Client):
    session, clock = FakeSession(responses), FakeClock()
    client = cls("test-key", "test-secret", env=env, session=session, sleep=clock.sleep, clock=clock)
    return client, session, clock


def broker(exchange, **kwargs):
    return Trading212Broker(exchange, LISTINGS, pause=0.01, wait_seconds=0.03, sleep=lambda _: None, **kwargs)


@pytest.mark.parametrize("env", ["demo", "live"])
def test_basic_auth_and_environment_use_the_documented_header_and_url(env):
    client, session, _ = http([Response(payload={})], env)
    client.summary()
    request = session.sent[0]
    assert request["url"] == f"https://{env}.trading212.com/api/v0/equity/account/summary"
    assert request["headers"]["Authorization"] == "Basic " + base64.b64encode(b"test-key:test-secret").decode()
    assert request["timeout"] == 30 and request["allow_redirects"] is False
    assert client.env == env


def test_an_unknown_environment_is_rejected_before_connecting():
    with pytest.raises(ValueError):
        Trading212Client("k", "s", env="production")


def test_errors_do_not_disclose_credentials_even_if_the_transport_echoes_them():
    client, _, _ = http([requests.ConnectionError("test-key test-secret")] * 3)
    with pytest.raises(Trading212Error) as caught:
        client.summary()
    assert "test-secret" not in str(caught.value) and "test-key" not in str(caught.value)
    assert caught.value.status is None and not caught.value.uncertain


def test_a_server_echo_is_redacted_from_the_message_but_the_original_body_is_preserved():
    client, _, _ = http([Response(400, text="test-secret")])
    with pytest.raises(Trading212Error) as caught:
        client.summary()
    assert "test-secret" not in str(caught.value) and caught.value.body == "test-secret"


@pytest.mark.parametrize("method,args,spacing", [("summary", (), 5), ("positions", (), 1), ("instruments", (), 50),
    ("exchanges", (), 30), ("pending_orders", (), 5), ("order", (1,), 1), ("history", (), 3),
    ("market_order", ("SPY_EU_EQ", 1), 1.2), ("cancel", (1,), 1.2)])
def test_each_endpoint_waits_its_documented_spacing(method, args, spacing):
    client, _, clock = http([Response(payload={"items": []}), Response(payload={"items": []})])
    getattr(client, method)(*args)
    getattr(client, method)(*args)
    assert clock.waits == [spacing]


def test_different_endpoints_have_independent_pacing_and_order_ids_share_one_limit():
    client, _, clock = http([Response()] * 4)
    client.summary()
    client.positions()
    client.order(1)
    assert clock.waits == []
    client.order(2)
    assert clock.waits == [1]


def test_a_rate_limit_retry_waits_until_the_unix_reset_timestamp(monkeypatch):
    monkeypatch.setattr(api.time, "time", lambda: 1000)
    client, session, clock = http([Response(429, headers={"x-ratelimit-reset": "1012"}), Response(payload={"ok": True})])
    assert client.summary() == {"ok": True}
    assert clock.waits == [12] and len(session.sent) == 2


@pytest.mark.parametrize("headers", [{}, {"x-ratelimit-reset": "1"}, {"x-ratelimit-reset": "invalid"}])
def test_a_rate_limit_without_a_future_reset_uses_the_endpoint_spacing_and_stops_after_three_attempts(headers):
    client, session, clock = http([Response(429, text="Too many requests", headers=headers)] * 3)
    with pytest.raises(Trading212Error) as caught:
        client.summary()
    assert caught.value.status == 429 and len(session.sent) == 3
    assert clock.waits == [5, 5]


@pytest.mark.parametrize("failure", [Response(408), Response(500), Response(503), requests.Timeout(), requests.ConnectionError()])
def test_gets_retry_transient_failures_with_a_bounded_number_of_attempts(failure):
    client, session, _ = http([failure] * 3)
    with pytest.raises(Trading212Error):
        client.positions()
    assert len(session.sent) == 3


@pytest.mark.parametrize("failure,uncertain", [(requests.Timeout(), True), (requests.ConnectionError(), True),
    (Response(408), True), (Response(500), True), (Response(429), False), (Response(400, text="insufficient funds"), False)])
@pytest.mark.parametrize("write", ["market_order", "cancel"])
def test_writes_are_never_retried_and_report_whether_the_outcome_is_uncertain(failure, uncertain, write):
    client, session, _ = http([failure])
    with pytest.raises(Trading212Error) as caught:
        client.market_order("SPY_EU_EQ", -0.125) if write == "market_order" else client.cancel(1)
    assert len(session.sent) == 1 and caught.value.uncertain is uncertain


def test_a_validation_error_preserves_the_servers_own_words_verbatim():
    text = '{"error": "This instrument cannot be traded by this account."}\n'
    client, _, _ = http([Response(400, text=text)])
    with pytest.raises(Trading212Error) as caught:
        client.market_order("SPY_EU_EQ", 1)
    assert caught.value.body == text == str(caught.value) and caught.value.status == 400


def test_the_market_request_is_signed_and_history_returns_only_one_page_of_items():
    client, session, _ = http([Response(payload={"id": 7}), Response(payload={"items": [{"order": {"id": 7}}]})])
    client.market_order("SPY_EU_EQ", -0.25)
    assert session.sent[0]["json"] == {"ticker": "SPY_EU_EQ", "quantity": -0.25, "extendedHours": False}
    assert client.history("SPY_EU_EQ", 25) == [{"order": {"id": 7}}]
    assert session.sent[1]["params"] == {"ticker": "SPY_EU_EQ", "limit": 25}


@pytest.mark.parametrize("method", ["POST", "DELETE", "post", "delete"])
def test_the_read_only_client_refuses_writes_before_touching_the_network(method):
    client, session, _ = http([], cls=ReadOnlyClient)
    with pytest.raises(RuntimeError, match="read-only"):
        client._request(method, "/equity/orders/market")
    assert session.sent == []


def test_resolution_prefers_euros_and_is_deterministic_when_input_order_changes():
    alternate = {**INSTRUMENTS[1], "ticker": "AAA_EU_EQ"}
    first = resolve(WANTED, INSTRUMENTS + [alternate])
    second = resolve(WANTED, list(reversed(INSTRUMENTS + [alternate])))
    assert first == second and first[0]["SPY"].ticker == "AAA_EU_EQ" and first[1] == {}


def test_an_explicit_ticker_wins_over_the_isin_and_currency_preferences():
    wanted = {"SPY": {"isin": ["missing"], "ticker": "SPY_US_EQ"}}
    assert resolve(wanted, INSTRUMENTS)[0]["SPY"].ticker == "SPY_US_EQ"


def test_missing_isins_and_duplicate_tickers_are_reported_without_duplicate_listings():
    wanted = {**WANTED, "OTHER": WANTED["SPY"], "BTC": {"isin": ["missing"]}}
    listings, problems = resolve(wanted, INSTRUMENTS)
    assert set(listings) == {"SPY", "NVDA"}
    assert "already assigned" in problems["OTHER"]
    assert problems["BTC"] == "not in Trading 212's list: missing"


def test_isin_candidate_order_precedes_currency_preference_and_unknown_types_are_disclosed():
    wanted = {"ONE": {"isin": ["US67066G1040", "IE00B5BMR087"]}}
    assert resolve(wanted, INSTRUMENTS)[0]["ONE"].ticker == "NVDA_US_EQ"
    unusual = [{**INSTRUMENTS[1], "type": "CRYPTO"}]
    listings, problems = resolve({"SPY": WANTED["SPY"]}, unusual)
    assert listings["SPY"].ticker == "SPY_EU_EQ" and "CRYPTO" in problems["SPY"]
    assert resolve({"SPY": WANTED["SPY"]}, unusual + [INSTRUMENTS[0]])[0]["SPY"].currency == "USD"


@pytest.mark.parametrize("hour,minute,opened", [(8, 59, False), (9, 0, True), (12, 0, True), (13, 0, False),
                                              (13, 30, True), (17, 0, False), (18, 0, False)])
def test_the_latest_schedule_event_determines_whether_the_regular_session_is_open(hour, minute, opened):
    assert session_open(EXCHANGES, 10, NOW.replace(hour=hour, minute=minute)) is opened


@pytest.mark.parametrize("event", ["CLOSE", "BREAK_START", "PRE_MARKET_OPEN", "AFTER_HOURS_OPEN", "AFTER_HOURS_CLOSE",
                                   "OVERNIGHT_OPEN", "UNKNOWN"])
def test_non_regular_session_events_keep_the_market_closed(event):
    exchanges = deepcopy(EXCHANGES)
    exchanges[0]["workingSchedules"][0]["timeEvents"] = [{"date": "2026-11-16T12:00:00+01:00", "type": event}]
    assert not session_open(exchanges, 10, NOW)


def test_unknown_schedules_and_naive_times_are_closed():
    assert not session_open(EXCHANGES, 999, NOW)
    assert not session_open(EXCHANGES, 10, NOW.replace(tzinfo=None))


def test_refresh_reads_each_account_endpoint_once_and_account_views_use_wallet_values():
    exchange = FakeTrading212(100, {"SPY_EU_EQ": 2, "NVDA_US_EQ": 3, "UNKNOWN": 4})
    exchange.reserved = 12
    b = broker(exchange)
    # working orders are read before holdings, so an order from an earlier run shows up in one list or the other
    assert exchange.log == [("pending_orders",), ("positions",), ("summary",)]
    assert b.cash() == 100 and b.available() == 88
    assert b.units() == {"SPY": 2, "NVDA": 3} and b.prices() == pytest.approx(PRICES)
    assert b.unrecognised() == {"UNKNOWN": 4}


@pytest.mark.parametrize("env", ["demo", "live"])
def test_an_unarmed_broker_sends_nothing_and_describes_the_quantity_it_would_trade(env):
    exchange = FakeTrading212(100, {"SPY_EU_EQ": 2}, env=env)
    fills = broker(exchange).execute({"SPY": -10, "NVDA": 10}, PRICES)
    assert [f.status for f in fills] == ["dry", "dry"] and exchange.sent == []
    assert fills[0].detail == "would sell 1.0 SPY_EU_EQ" and "would buy 0.5" in fills[1].detail
    assert all(f.units == f.notional_eur == f.cost_eur == 0 for f in fills)


def test_sells_settle_before_buys_and_all_buys_share_only_the_available_cash():
    exchange = FakeTrading212(10, {"SPY_EU_EQ": 5})
    exchange.reserved = 5
    fills = broker(exchange, armed=True).execute({"NVDA": 100, "SPY": -40}, PRICES)
    assert [o["ticker"] for o in exchange.sent] == ["SPY_EU_EQ", "NVDA_US_EQ"]
    assert fills[0].units == -4 and fills[1].notional_eur <= (10 - 5 + 40) * 0.99
    assert exchange.cash_balance >= exchange.reserved
    names = [call[0] for call in exchange.log]
    first, second = [i for i, name in enumerate(names) if name == "market_order"]
    assert "summary" in names[first:second]


def test_buy_scaling_is_proportional_and_happens_after_each_order_is_capped():
    exchange = FakeTrading212(50)
    fills = broker(exchange, armed=True, max_order_eur=40).execute({"SPY": 1000, "NVDA": 1000}, PRICES)
    assert [f.notional_eur for f in fills] == pytest.approx([24.75, 24.75])
    assert exchange.cash_balance > 0


def test_every_order_respects_the_cap_even_when_a_dust_exit_would_exceed_it():
    exchange = FakeTrading212(1000, {"SPY_EU_EQ": 10.01})
    broker(exchange, armed=True, max_order_eur=100).execute({"SPY": -1000, "NVDA": 1000}, PRICES)
    for order in exchange.sent:
        price = 10 if order["ticker"] == "SPY_EU_EQ" else 20
        assert abs(order["quantity"]) * price <= 100
    assert exchange.holdings["SPY_EU_EQ"] == pytest.approx(0.01)


@pytest.mark.parametrize("notional", [12.345678, -12.345678])
def test_quantities_are_rounded_down_and_never_increased_to_the_next_step(notional):
    exchange = FakeTrading212(100, {"SPY_EU_EQ": 10})
    broker(exchange, armed=True).execute({"SPY": notional}, PRICES)
    quantity = abs(exchange.sent[0]["quantity"])
    assert quantity == 1.2345 and quantity <= abs(notional) / 10


@pytest.mark.parametrize("notional", [-12.3, -12.22222])
def test_a_full_exit_uses_the_exact_available_quantity_so_no_dust_remains(notional):
    exchange = FakeTrading212(100, {"SPY_EU_EQ": 1.23456789})
    fill = broker(exchange, armed=True).execute({"SPY": notional}, PRICES)[0]
    assert fill.units == -1.23456789 and exchange.holdings["SPY_EU_EQ"] == 0


@pytest.mark.parametrize("ratio,status", [(1, "filled"), (0.4, "partial"), (0, "unfilled")])
def test_actual_fills_and_fees_come_from_history_and_any_remainder_is_cancelled(ratio, status):
    exchange = FakeTrading212(fill_ratio=ratio)
    exchange.fill_prices["NVDA_US_EQ"] = 23.1  # 21 EUR rather than the 20 EUR reference
    fill = broker(exchange, armed=True).execute({"NVDA": 20}, PRICES)[0]
    assert fill.status == status and fill.units == ratio
    assert fill.notional_eur == pytest.approx(ratio * 21)
    assert fill.cost_eur == pytest.approx(ratio * (1 + 21 * 0.0015))
    assert exchange.pending_orders() == []
    assert bool(exchange.cancelled) is (ratio < 1)


def test_an_order_that_never_fills_is_cancelled_after_the_wait_budget():
    exchange = FakeTrading212()
    exchange.working = True
    fill = broker(exchange, armed=True).execute({"SPY": 20}, PRICES)[0]
    assert fill.status == "unfilled" and exchange.cancelled == [1]
    assert "cancel requested" in fill.detail and exchange.pending_orders() == []


def test_a_completed_order_missing_from_pending_lookup_is_read_from_history():
    exchange = FakeTrading212()
    exchange.completed_404 = True
    fill = broker(exchange, armed=True).execute({"SPY": 20}, PRICES)[0]
    assert fill.status == "filled" and fill.units == 2 and exchange.cancelled == []


def test_missing_history_falls_back_only_to_reported_filled_quantity_and_says_so():
    exchange = FakeTrading212(fill_ratio=0.25)
    exchange.hide_history = True
    fill = broker(exchange, armed=True).execute({"SPY": 20}, PRICES)[0]
    assert fill.units == 0.5 and fill.notional_eur == 5 and fill.status == "partial"
    assert "used filledQuantity and reference price" in fill.detail


def test_an_exchange_rejection_keeps_the_exchanges_own_words():
    exchange = FakeTrading212()
    exchange.reject["SPY_EU_EQ"] = "This ETF is restricted in your region."
    fill = broker(exchange, armed=True).execute({"SPY": 20}, PRICES)[0]
    assert fill.status == "rejected" and fill.detail == exchange.reject["SPY_EU_EQ"]
    assert len(exchange.sent) == 1


@pytest.mark.parametrize("working", [False, True])
def test_an_uncertain_request_that_created_an_order_is_adopted_without_a_second_post(working):
    exchange = FakeTrading212()
    exchange.uncertain, exchange.working = "after", working
    fill = broker(exchange, armed=True).execute({"SPY": 20}, PRICES)[0]
    assert len(exchange.sent) == 1 and len(exchange.orders) == 1
    assert fill.status == ("unfilled" if working else "filled") and exchange.pending_orders() == []


def test_an_uncertain_request_that_created_nothing_is_reported_without_retrying():
    exchange = FakeTrading212()
    exchange.uncertain = "before"
    fill = broker(exchange, armed=True).execute({"SPY": 20}, PRICES)[0]
    assert fill.status == "rejected" and "the request failed and no order was found" in fill.detail
    assert len(exchange.sent) == 1 and not exchange.orders


def test_an_older_identical_api_order_is_never_adopted_after_an_uncertain_failure():
    exchange = FakeTrading212()
    exchange.market_order("SPY_EU_EQ", 2)
    exchange.sent.clear()
    exchange.uncertain = "before"
    fill = broker(exchange, armed=True).execute({"SPY": 20}, PRICES)[0]
    assert fill.status == "rejected" and len(exchange.sent) == 1


def test_a_symbol_with_a_working_order_is_skipped_and_pending_uses_signed_remaining_quantity():
    exchange = FakeTrading212(100, {"SPY_EU_EQ": 5}, fill_ratio=0.25)
    exchange.market_order("SPY_EU_EQ", -2)
    exchange.sent.clear()
    b = broker(exchange, armed=True)
    assert b.pending() == {"SPY": -1.5}
    fill = b.execute({"SPY": 10}, PRICES)[0]
    assert fill.status == "skipped" and fill.detail == "an earlier order is still working" and not exchange.sent


@pytest.mark.parametrize("symbol,price,detail", [("BTC", 10, "not listed on Trading 212"),
    ("SPY", 0, "no reference price"), ("SPY", -1, "no reference price"),
    ("SPY", float("nan"), "no reference price"), ("SPY", float("inf"), "no reference price")])
def test_missing_listings_and_invalid_prices_are_skipped_without_sending(symbol, price, detail):
    exchange = FakeTrading212()
    fill = broker(exchange, armed=True).execute({symbol: 10}, {symbol: price})[0]
    assert fill.status == "skipped" and fill.detail == detail and exchange.sent == []


@pytest.mark.parametrize("notional", [0, 0.99, -0.1])
def test_orders_below_the_minimum_are_skipped_without_rounding_up(notional):
    exchange = FakeTrading212(100, {"SPY_EU_EQ": 10})
    fill = broker(exchange, armed=True).execute({"SPY": notional}, PRICES)[0]
    assert fill.status == "skipped" and fill.detail == "below the minimum order" and not exchange.sent


def test_non_euro_accounts_are_not_sized_using_euro_references():
    exchange = FakeTrading212()
    exchange.currency = "USD"
    fill = broker(exchange, armed=True).execute({"SPY": 10}, PRICES)[0]
    assert fill.status == "skipped" and "account currency" in fill.detail and not exchange.sent


def test_a_healthy_read_only_rehearsal_reports_listings_positions_and_alternatives():
    exchange = FakeTrading212(100, {"SPY_EU_EQ": 1, "UNKNOWN": 2})
    rows = rehearsal.check(exchange, WANTED, NOW)
    assert rows and all(ok for _, ok, _ in rows) and exchange.sent == []
    assert any("SPY_US_EQ" in detail for name, _, detail in rows if "other listings" in name)
    assert any("UNKNOWN=2" in detail for _, _, detail in rows)
    assert any("market open" in detail for _, _, detail in rows)


def test_an_unresolved_market_is_a_failed_rehearsal_check():
    rows = rehearsal.check(FakeTrading212(), {"BTC": {"isin": ["missing"]}}, NOW)
    assert any(name == "BTC" and not ok and "missing" in detail for name, ok, detail in rows)


def test_a_closed_session_is_reported_without_failing_a_read_only_account_check():
    rows = rehearsal.check(FakeTrading212(), WANTED, NOW.replace(hour=18))
    assert all(ok for _, ok, _ in rows) and any("market closed" in detail for _, _, detail in rows)


def test_the_demo_trade_refuses_a_live_environment_before_making_any_api_call(capsys):
    exchange = FakeTrading212(env="live")
    assert rehearsal.trade_demo(exchange, WANTED, NOW, 10) == 2
    assert exchange.log == [] and exchange.sent == [] and "refusing" in capsys.readouterr().out


def test_the_command_refuses_live_demo_trading_before_constructing_a_client(monkeypatch, capsys):
    monkeypatch.setenv("T212_API_KEY", "unused")
    monkeypatch.setenv("T212_API_SECRET", "unused")
    monkeypatch.setenv("T212_ENV", "live")
    assert rehearsal.main(["--trade-demo"]) == 2 and "T212_ENV=demo" in capsys.readouterr().out


def test_the_command_without_keys_prints_one_line_and_exits_two(monkeypatch, capsys):
    monkeypatch.delenv("T212_API_KEY", raising=False)
    monkeypatch.delenv("T212_API_SECRET", raising=False)
    assert rehearsal.main([]) == 2
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_a_demo_round_trip_requires_a_price_for_an_unheld_market_and_sends_nothing(monkeypatch, capsys):
    monkeypatch.setattr(rehearsal, "Trading212Broker", lambda client, listings, **kw: broker(client, **kw))
    exchange = FakeTrading212()
    assert rehearsal.trade_demo(exchange, WANTED, NOW) == 2
    assert not exchange.sent and "--price" in capsys.readouterr().out


def test_a_demo_round_trip_buys_then_sells_only_the_demo_quantity(monkeypatch):
    monkeypatch.setattr(rehearsal, "Trading212Broker", lambda client, listings, **kw: broker(client, **kw))
    exchange = FakeTrading212()
    assert rehearsal.trade_demo(exchange, WANTED, NOW, 10) == 0
    assert [o["quantity"] for o in exchange.sent] == [0.5, -0.5]
    assert exchange.holdings["SPY_EU_EQ"] == 0


def test_demo_cleanup_preserves_a_small_position_held_before_the_rehearsal(monkeypatch):
    monkeypatch.setattr(rehearsal, "Trading212Broker", lambda client, listings, **kw: broker(client, **kw))
    exchange = FakeTrading212(100, {"SPY_EU_EQ": 0.05})
    assert rehearsal.trade_demo(exchange, WANTED, NOW, 10) == 0
    assert exchange.sent[-1]["quantity"] == -0.5
    assert exchange.holdings["SPY_EU_EQ"] == pytest.approx(0.05)


def test_demo_cleanup_attempts_to_sell_even_a_fill_below_one_euro(monkeypatch):
    monkeypatch.setattr(rehearsal, "Trading212Broker", lambda client, listings, **kw: broker(client, **kw))
    exchange = FakeTrading212(fill_ratio=0.1)
    rehearsal.trade_demo(exchange, WANTED, NOW, 10)
    assert len(exchange.sent) == 2 and exchange.sent[-1]["quantity"] == -0.05


def test_multiple_history_fills_for_one_order_are_added_together():
    class SplitFills(FakeTrading212):
        def history(self, ticker=None, limit=50):
            result = super().history(ticker, limit)
            if not result:
                return result
            first, second = deepcopy(result[0]), deepcopy(result[0])
            for row in (first, second):
                row["fill"]["quantity"] /= 2
                row["fill"]["walletImpact"]["netValue"] /= 2
                for tax in row["fill"]["walletImpact"]["taxes"]:
                    tax["quantity"] /= 2
            second["fill"]["id"] += 1
            return [first, second]

    exchange = SplitFills()
    fill = broker(exchange, armed=True).execute({"NVDA": 20}, PRICES)[0]
    assert fill.units == 1 and fill.status == "filled"
    assert fill.cost_eur == pytest.approx(20 * 0.0015)


def test_a_working_order_that_appears_after_refresh_is_caught_before_placement():
    exchange = FakeTrading212(100, {"SPY_EU_EQ": 5})
    b = broker(exchange, armed=True)
    exchange.working = True
    exchange.market_order("SPY_EU_EQ", -1)
    exchange.sent.clear()
    fill = b.execute({"SPY": -10}, PRICES)[0]
    assert fill.detail == "an earlier order is still working" and exchange.sent == []
    fill = b.execute({"SPY": 10}, PRICES)[0]
    assert fill.detail == "an earlier order is still working" and exchange.sent == []


@pytest.mark.parametrize("origin,duplicates", [("WEB", False), ("API", True)])
def test_uncertain_adoption_refuses_non_api_orders_and_ambiguous_matches(origin, duplicates):
    class Ambiguous(FakeTrading212):
        def market_order(self, ticker, quantity, extended_hours=False):
            order = super().market_order(ticker, quantity, extended_hours)
            self.orders[order["id"]]["initiatedFrom"] = origin
            if duplicates:
                self.orders[order["id"] + 1] = {**self.orders[order["id"]], "id": order["id"] + 1}
            raise Trading212Error("response lost", uncertain=True)

    exchange = Ambiguous()
    exchange.working = True
    fill = broker(exchange, armed=True).execute({"SPY": 10}, PRICES)[0]
    assert fill.status == "rejected" and len(exchange.sent) == 1 and "no order was found" in fill.detail


def test_cancel_errors_are_reported_without_assuming_the_order_was_filled():
    class CancelFails(FakeTrading212):
        def cancel(self, order_id):
            raise Trading212Error("cancel unavailable", status=503, uncertain=True)

    exchange = CancelFails()
    exchange.working = True
    fill = broker(exchange, armed=True).execute({"SPY": 10}, PRICES)[0]
    assert fill.status == "unfilled" and fill.units == 0
    assert "cancel failed; order may still be working" in fill.detail
    assert len(exchange.sent) == 1


def test_a_buy_with_no_cash_still_returns_a_skipped_fill_for_each_symbol():
    exchange = FakeTrading212(0)
    fills = broker(exchange, armed=True).execute({"SPY": 10, "NVDA": 20}, PRICES)
    assert len(fills) == 2 and all(f.status == "skipped" for f in fills) and not exchange.sent


def test_the_read_only_command_uses_the_configured_currency_preference(monkeypatch, tmp_path, capsys):
    path = tmp_path / "trading212.json"
    path.write_text('{"currency_preference": ["USD", "EUR"], "markets": {"SPY": {"isin": ["IE00B5BMR087"]}}}')
    exchange = FakeTrading212()
    monkeypatch.setattr(rehearsal, "CONFIG", path)
    monkeypatch.setattr(rehearsal, "ReadOnlyClient", lambda *args, **kwargs: exchange)
    monkeypatch.setenv("T212_API_KEY", "unused")
    monkeypatch.setenv("T212_API_SECRET", "unused")
    assert rehearsal.main([]) == 0
    assert "SPY_US_EQ, USD" in capsys.readouterr().out and exchange.sent == []


def test_transient_get_errors_can_succeed_on_the_last_allowed_attempt():
    client, session, _ = http([requests.Timeout(), Response(503), Response(payload={"currency": "EUR"})])
    assert client.summary() == {"currency": "EUR"} and len(session.sent) == 3


def test_unavailable_account_reads_are_failed_rehearsal_rows():
    class Broken(FakeTrading212):
        def summary(self):
            raise Trading212Error("read permission missing", status=403)

    rows = rehearsal.check(Broken(), WANTED, NOW)
    assert ("account", False, "read permission missing") in rows


def test_schedule_dates_without_a_timezone_suffix_are_interpreted_as_utc():
    exchanges = deepcopy(EXCHANGES)
    exchanges[0]["workingSchedules"][0]["timeEvents"] = [{"date": "2026-11-16T09:00:00", "type": "OPEN"}]
    assert session_open(exchanges, 10, NOW)
