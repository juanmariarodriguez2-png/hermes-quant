"""The daily run on Trading 212, against the simulated API. No network, no keys."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_quant.live import favourites, run_daily
from hermes_quant.live import trading212 as t212
from test_trading212 import FakeTrading212

EUR, USD = 1.0, 1 / 1.1                                   # the simulated account converts dollars at 1.10
# What the simulated exchange fills at, per instrument, in the instrument's own currency.
FILL = {"SXR8d_EQ": 734.0, "CSPXl_EQ": 807.4, "SXRVd_EQ": 1437.0, "PPFBd_EQ": 68.0, "SSLNm_EQ": 40.5,
        "NVDA_US_EQ": 156.0, "NVDd_EQ": 141.8, "TSLA_US_EQ": 380.0, "AAPL_US_EQ": 352.0, "GOOGL_US_EQ": 284.0,
        "MSFT_US_EQ": 518.0, "AMZN_US_EQ": 258.0}
XETRA, NASDAQ = 1, 2
INSTRUMENTS = [
    {"ticker": "SXR8d_EQ", "isin": "IE00B5BMR087", "currencyCode": "EUR", "type": "ETF", "name": "iShares Core S&P 500", "workingScheduleId": XETRA},
    {"ticker": "CSPXl_EQ", "isin": "IE00B5BMR087", "currencyCode": "USD", "type": "ETF", "name": "iShares Core S&P 500", "workingScheduleId": XETRA},
    {"ticker": "SXRVd_EQ", "isin": "IE00B53SZB19", "currencyCode": "EUR", "type": "ETF", "name": "iShares Nasdaq 100", "workingScheduleId": XETRA},
    {"ticker": "PPFBd_EQ", "isin": "IE00B4ND3602", "currencyCode": "EUR", "type": "ETF", "name": "iShares Physical Gold", "workingScheduleId": XETRA},
    {"ticker": "SSLNm_EQ", "isin": "IE00B4NCWG09", "currencyCode": "EUR", "type": "ETF", "name": "iShares Physical Silver", "workingScheduleId": XETRA},
    {"ticker": "NVDd_EQ", "isin": "US67066G1040", "currencyCode": "EUR", "type": "STOCK", "name": "Nvidia (Xetra)", "workingScheduleId": XETRA},
] + [{"ticker": f"{s}_US_EQ", "isin": isin, "currencyCode": "USD", "type": "STOCK", "name": s, "workingScheduleId": NASDAQ}
     for s, isin in (("NVDA", "US67066G1040"), ("TSLA", "US88160R1014"), ("AAPL", "US0378331005"), ("GOOGL", "US02079K3059"),
                     ("MSFT", "US5949181045"), ("AMZN", "US0231351067"))]
EXCHANGES = [{"id": 1, "name": "XETRA", "workingSchedules": [{"id": XETRA, "timeEvents": [
                 {"date": "2026-11-16T08:00:00Z", "type": "OPEN"}, {"date": "2026-11-16T16:30:00Z", "type": "CLOSE"}]}]},
             {"id": 2, "name": "NASDAQ", "workingSchedules": [{"id": NASDAQ, "timeEvents": [
                 {"date": "2026-11-16T14:30:00Z", "type": "OPEN"}, {"date": "2026-11-16T21:00:00Z", "type": "CLOSE"}]}]}]
# The reference source, keyed as config/trading212.json names each listing: euros per share.
REFERENCE = {"SXR8.DE": 734.0, "SXRV.DE": 1437.0, "PPFB.DE": 68.0, "SSLN.MI": 40.5, "NVDA": 156.0 * USD, "TSLA": 380.0 * USD,
             "AAPL": 352.0 * USD, "GOOGL": 284.0 * USD, "MSFT": 518.0 * USD, "AMZN": 258.0 * USD}
NIGHT = datetime(2026, 11, 16, 1, 25, tzinfo=timezone.utc)         # every exchange shut
MORNING = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc)       # Europe open, New York not yet
AFTERNOON = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)     # both open
STOCKS = {"NVDA": "Nvidia", "TSLA": "Tesla", "AAPL": "Apple", "GOOGL": "Alphabet", "MSFT": "Microsoft", "AMZN": "Amazon"}
ANY_TREND = favourites.Basket(stocks=STOCKS, enter=0.001, leave=0.001)


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for name in ("HERMES_FAVOURITES", "HERMES_REBASE", "HERMES_VENUE", "HERMES_TOKENIZED_API"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def sandbox(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data" / "live" / "prices"
    if not source.exists():
        pytest.skip("live price store not seeded")
    shutil.copytree(source, tmp_path / "live" / "prices")
    return tmp_path


class Account(FakeTrading212):
    """The simulated API, with the account number a real summary carries."""

    number = 4001

    def summary(self):
        return {**super().summary(), "id": self.number}


def account(cash=1000.0, positions=None, *, armed=True, quotes=REFERENCE, instruments=INSTRUMENTS, env="demo", number=4001, kind=Account):
    api = kind(cash=cash, positions=positions, env=env)
    api.number = number
    api.instrument_list, api.exchange_list = [dict(i) for i in instruments], EXCHANGES
    api.fill_prices, api.fx = dict(FILL), {"EUR": EUR, "USD": USD}

    def quote(feed, eurusd):
        return quotes[feed]

    return api, t212.Venue(api, armed=armed, max_order_eur=500.0, quote=quote, sleep=lambda seconds: None)


def run(sandbox, venue, now, clock=None, **kw):
    venue.clock = lambda: clock or now                         # the exchange's clock at the moment each order goes out
    return run_daily.run(mode="live", offline=True, now=now, venue=venue, basket=kw.pop("basket", favourites.Basket()),
                         live_dir=sandbox / "live", site_dir=sandbox / "site", **kw)


def record(sandbox, venue):
    return sandbox / "live" / venue.record / "state.json"


def test_the_models_markets_resolve_to_european_funds_and_us_shares_to_their_home_listing():
    api, venue = account()
    listed = {s: i.ticker for s, i in venue.listings.items()}
    assert listed["SPY"] == "SXR8d_EQ" and listed["QQQ"] == "SXRVd_EQ" and listed["GLD"] == "PPFBd_EQ" and listed["SLV"] == "SSLNm_EQ"
    assert listed["NVDA"] == "NVDA_US_EQ"                      # in dollars on its own exchange, not the thin euro listing
    assert "BTC" not in listed and "IE00" not in venue.problems["BTC"] and "XS2940466316" in venue.problems["BTC"]


def test_nothing_is_sent_while_the_exchanges_are_shut_and_the_day_is_not_used_up(sandbox):
    api, venue = account()
    night = run(sandbox, venue, NIGHT)["account"]
    assert api.sent == [] and night["orders_today"] == [] and night["manual_orders"] == []
    afternoon = run(sandbox, venue, AFTERNOON)["account"]      # same daily candle, markets open now
    bought = {o["symbol"]: o for o in afternoon["orders_today"]}
    assert {"SPY", "GLD"} <= set(bought) and all(o["status"] == "filled" and o["notional_eur"] > 0 for o in bought.values())
    assert {s["ticker"] for s in api.sent} <= {"SXR8d_EQ", "SXRVd_EQ", "PPFBd_EQ", "SSLNm_EQ"} and all(s["quantity"] > 0 for s in api.sent)
    sent = len(api.sent)
    again = run(sandbox, venue, AFTERNOON)["account"]          # one round per market per candle
    assert len(api.sent) == sent and again["orders_today"] == []


def test_the_account_is_valued_with_the_brokers_own_prices_and_kept_apart_from_krakens(sandbox):
    api, venue = account(cash=1000.0)
    a = run(sandbox, venue, AFTERNOON)["account"]
    assert a["venue"] == "trading212" and a["venue_env"] == "demo" and a["manual_orders"] == []
    held = {p["symbol"]: p for p in a["positions"] if p["value"] > 0}
    assert held["SPY"]["instrument"] == "SXR8d_EQ" and not held["SPY"]["live_feed"]
    assert held["SPY"]["value"] == pytest.approx(api.holdings["SXR8d_EQ"] * 734.0)
    worth = api.cash_balance + sum(q * FILL[t] * (EUR if t.endswith("d_EQ") or t.endswith("m_EQ") else USD) for t, q in api.holdings.items())
    assert a["nav"] == pytest.approx(worth) and a["nav"] == pytest.approx(1000.0, rel=5e-3)     # nothing lost but costs
    assert a["flow_today"] == 0.0 and a["initial_capital"] == pytest.approx(1000.0)
    assert record(sandbox, venue).exists() and not (sandbox / "live" / "state.json").exists()
    assert "4001" not in venue.record and venue.record.startswith("trading212-demo-")          # the account number stays private
    state = json.loads(record(sandbox, venue).read_text())
    assert set(state["units"]) == set(held) and state["mode"] == "live"

    state["last_bar"] = None                                   # the next day: buying was not a flow, and nothing else is due
    record(sandbox, venue).write_text(json.dumps(state))
    sent = len(api.sent)
    later = run(sandbox, venue, AFTERNOON)["account"]
    assert later["flow_today"] == 0.0 and not later["halted"] and len(api.sent) == sent


def test_each_market_waits_for_its_own_exchange(sandbox, monkeypatch):
    monkeypatch.setenv("HERMES_FAVOURITES", "yes")
    api, venue = account()
    morning = run(sandbox, venue, MORNING, basket=ANY_TREND)["account"]
    assert {o["symbol"] for o in morning["orders_today"]} <= {"SPY", "QQQ", "GLD", "SLV"} and morning["orders_today"]
    assert not [s for s in api.sent if s["ticker"].endswith("_US_EQ")]                # New York is still shut
    afternoon = run(sandbox, venue, AFTERNOON, basket=ANY_TREND)["account"]
    stocks = [o for o in afternoon["orders_today"]]
    assert len(stocks) == 4 and {o["symbol"] for o in stocks} <= set(STOCKS)           # the funds already had their round
    assert all(s["ticker"].endswith("_US_EQ") for s in api.sent[-4:])
    assert all(p["managed"] for p in afternoon["positions"] if p["class"] == "stock")
    assert afternoon["manual_orders"] == []


def test_a_market_the_venue_does_not_list_stays_in_cash_and_is_reported(sandbox):
    api, venue = account(instruments=[i for i in INSTRUMENTS if i["isin"] != "IE00B4ND3602"])     # no gold fund at all
    a = run(sandbox, venue, AFTERNOON)["account"]
    assert "GLD" in a["unavailable"] and "IE00B4ND3602" in a["unavailable"]["GLD"]
    assert "GLD" not in {o["symbol"] for o in a["orders_today"]} and "SPY" in {o["symbol"] for o in a["orders_today"]}


def test_without_a_reference_price_a_first_purchase_waits(sandbox):
    api, venue = account(quotes={k: v for k, v in REFERENCE.items() if k != "PPFB.DE"})
    a = run(sandbox, venue, AFTERNOON)["account"]
    assert "GLD" in a["unavailable"] and "GLD" not in {o["symbol"] for o in a["orders_today"]}
    assert not [s for s in api.sent if s["ticker"] == "PPFBd_EQ"]


def test_an_unarmed_venue_sends_nothing_and_says_what_it_would_do(sandbox):
    api, venue = account(armed=False)
    a = run(sandbox, venue, AFTERNOON)["account"]
    assert api.sent == [] and a["orders_today"] and all(o["status"] == "dry" and "would buy" in o["detail"] for o in a["orders_today"])
    assert a["nav"] == pytest.approx(1000.0)


def test_the_circuit_breaker_sells_through_the_venue(sandbox):
    api, venue = account()
    run(sandbox, venue, AFTERNOON)
    path = record(sandbox, venue)
    state = json.loads(path.read_text())
    state.update(peak_nav=state["nav"] * 1.5, last_bar=None)
    path.write_text(json.dumps(state))
    out = run(sandbox, venue, AFTERNOON)["account"]
    assert out["halted"] and all(s["quantity"] < 0 for s in api.sent[-2:])
    assert sum(q * FILL[t] for t, q in api.holdings.items()) == pytest.approx(0.0, abs=1.0)


def test_a_holding_the_run_does_not_follow_is_reported_valued_and_never_touched(sandbox):
    api, venue = account(positions={"TSLA_US_EQ": 0.5, "ZZZ_US_EQ": 3.0})      # no list of stocks in this run
    a = run(sandbox, venue, AFTERNOON)["account"]
    assert api.holdings["TSLA_US_EQ"] == 0.5 and not [s for s in api.sent if s["ticker"] in ("TSLA_US_EQ", "ZZZ_US_EQ")]
    assert a["unrecognised"] == {"TSLA_US_EQ": 0.5, "ZZZ_US_EQ": 3.0}
    others = 0.5 * 380.0 * USD + 3.0 * 7.0                    # as the account itself values them
    assert a["other_value"] == pytest.approx(others)
    followed = sum(p["value"] for p in a["positions"])
    assert a["nav"] == pytest.approx(api.cash_balance + followed + others)             # every holding counts, whoever bought it
    assert a["initial_capital"] == pytest.approx(1000.0 + others, rel=5e-3)

    state = json.loads(record(sandbox, venue).read_text())
    state["last_bar"] = None
    record(sandbox, venue).write_text(json.dumps(state))
    api.holdings["ZZZ_US_EQ"] = 1.0                           # the owner sells two thirds of it for cash
    api.cash_balance += 2.0 * 7.0
    venue.broker.refresh()                                     # (a real run starts from a fresh read of the account)
    later = run(sandbox, venue, AFTERNOON)["account"]
    assert later["other_value"] == pytest.approx(others - 2.0 * 7.0)
    assert later["flow_today"] == 0.0 and not later["halted"]                          # money changed shape, none came or went


def test_practice_money_and_real_money_never_share_a_record(sandbox):
    demo_api, demo = account(env="demo", number=4001)
    assert len(run(sandbox, demo, AFTERNOON)["account"]["orders_today"]) >= 2
    live_api, live = account(env="live", number=9002)
    assert live.record != demo.record and live.record.startswith("trading212-live-")
    first = run(sandbox, live, AFTERNOON)["account"]            # same daily candle: the practice run must not have used it up
    assert len(first["orders_today"]) >= 2 and live_api.sent and first["venue_env"] == "live"
    assert record(sandbox, demo).exists() and record(sandbox, live).exists()


def test_a_market_keeps_its_round_until_an_order_is_really_sent_for_it(sandbox):
    api, venue = account(cash=3.0)                             # not enough for a single ticket yet
    morning = run(sandbox, venue, MORNING)["account"]
    assert api.sent == [] and morning["orders_today"] == []
    api.cash_balance += 1000.0                                 # the transfer arrives at midday
    venue.broker.refresh()
    afternoon = run(sandbox, venue, AFTERNOON)["account"]
    assert {"SPY", "GLD"} <= {o["symbol"] for o in afternoon["orders_today"]} and api.sent
    assert afternoon["flow_today"] == pytest.approx(1000.0) and afternoon["initial_capital"] == pytest.approx(1003.0)


def test_an_account_with_no_money_yet_does_not_break_the_run(sandbox):
    api, venue = account(cash=0.0)
    a = run(sandbox, venue, AFTERNOON)["account"]
    assert a["nav"] == 0.0 and a["orders_today"] == [] and a["drawdown"] == 0.0 and a["return_since_inception"] == 0.0 and not a["halted"]


def test_an_order_is_not_sent_once_the_exchange_has_closed_and_the_round_is_kept(sandbox):
    api, venue = account()
    after_the_close = datetime(2026, 11, 16, 21, 30, tzinfo=timezone.utc)
    late = run(sandbox, venue, AFTERNOON, clock=after_the_close)["account"]       # the run began in time; the orders would not
    assert api.sent == [] and late["orders_today"] and all(o["status"] == "skipped" and "closed" in o["detail"] for o in late["orders_today"])
    assert json.loads(record(sandbox, venue).read_text())["traded"] == {}
    again = run(sandbox, venue, AFTERNOON)["account"]
    assert {"SPY", "GLD"} <= {o["symbol"] for o in again["orders_today"] if o["status"] == "filled"}


class LosesContact(Account):
    """The account stops answering once the first order has gone through."""

    def summary(self):
        if self.sent:
            from hermes_quant.data.trading212 import Trading212Error
            raise Trading212Error("gateway timeout", status=504)
        return super().summary()


def test_what_was_sent_is_written_down_even_if_the_account_cannot_be_read_afterwards(sandbox):
    api, venue = account(kind=LosesContact)
    out = run(sandbox, venue, AFTERNOON)
    a = out["account"]
    assert len(api.sent) == 1                                  # the first purchase went; the rest were not sized blind
    sent = api.sent[0]["ticker"]
    filled = [o for o in a["orders_today"] if o["status"] == "filled"]
    assert len(filled) == 1 and all("could not be re-read" in o["detail"] for o in a["orders_today"] if o["status"] == "skipped")
    assert "broker" in out["meta"]["data_errors"]
    state = json.loads(record(sandbox, venue).read_text())
    assert list(state["traded"]) == [filled[0]["symbol"]] and venue.listings[filled[0]["symbol"]].ticker == sent
    api.sent.clear()                                           # contact is back (the simulated failure is tied to the log)
    api_orders = len(api.orders)
    again = run(sandbox, venue, AFTERNOON)["account"]
    assert filled[0]["symbol"] not in {o["symbol"] for o in again["orders_today"]}     # not bought a second time
    assert len(api.orders) > api_orders                        # the others get their round now


def test_cash_is_read_again_before_each_purchase_so_a_dear_fill_cannot_overspend():
    api = Account(cash=100.0)
    api.instrument_list, api.exchange_list, api.fx = [dict(i) for i in INSTRUMENTS], EXCHANGES, {"EUR": EUR, "USD": USD}
    api.fill_prices = {**FILL, "SXR8d_EQ": 734.0 * 1.3}       # the first purchase fills 30% above its reference price
    listings, _ = t212.resolve({"SPY": {"isin": ["IE00B5BMR087"]}, "QQQ": {"isin": ["IE00B53SZB19"]}}, api.instrument_list)
    broker = t212.Trading212Broker(api, listings, armed=True, max_order_eur=500.0, sleep=lambda seconds: None)
    fills = broker.execute({"SPY": 50.0, "QQQ": 50.0}, {"SPY": 734.0, "QQQ": 1437.0})   # the simulated exchange asserts on overspending
    assert [f.status for f in fills] == ["filled", "filled"] and api.cash_balance >= 0.0
    assert abs(fills[1].notional_eur) < 40.0                   # the second was cut to what was left


def test_a_stock_the_owner_holds_is_valued_while_the_basket_is_off(sandbox):
    api, venue = account(positions={"TSLA_US_EQ": 0.5})
    a = run(sandbox, venue, AFTERNOON, basket=ANY_TREND)["account"]            # on the list, basket switched off
    tesla = next(p for p in a["positions"] if p["symbol"] == "TSLA")
    assert tesla["value"] == pytest.approx(0.5 * 380.0 * USD) and not tesla["managed"] and a["unrecognised"] == {}
    assert api.holdings["TSLA_US_EQ"] == 0.5 and not [s for s in api.sent if s["ticker"] == "TSLA_US_EQ"]


class SlowBooks(FakeTrading212):
    """An exchange whose answer to the order says nothing filled yet and whose history trails behind."""

    def market_order(self, ticker, quantity, extended_hours=False):
        order = super().market_order(ticker, quantity, extended_hours)
        return {**order, "filledQuantity": 0.0, "status": "NEW"}


def test_a_fill_the_books_have_not_caught_up_with_is_read_from_the_position_itself():
    api = SlowBooks(cash=1000.0)
    api.instrument_list, api.exchange_list, api.fill_prices, api.fx = [dict(i) for i in INSTRUMENTS], EXCHANGES, dict(FILL), {"EUR": EUR, "USD": USD}
    api.hide_history, api.completed_404 = True, True          # filled orders vanish from the working list; no history yet
    listings, _ = t212.resolve({"SPY": {"isin": ["IE00B5BMR087"]}}, api.instrument_list)
    broker = t212.Trading212Broker(api, listings, armed=True, max_order_eur=500.0, sleep=lambda seconds: None)
    fill = broker.execute({"SPY": 200.0}, {"SPY": 734.0})[0]
    assert fill.status == "filled" and fill.units == pytest.approx(api.holdings["SXR8d_EQ"]) and fill.units > 0
    assert "the change in the position" in fill.detail and len(api.sent) == 1


def test_a_far_away_rate_limit_reset_cannot_hang_the_run():
    import time as real_time

    from hermes_quant.data import trading212 as api_module
    from test_trading212 import FakeClock, FakeSession, Response

    clock = FakeClock()
    waits = []
    session = FakeSession([Response(429, text="slow down", headers={"x-ratelimit-reset": str(real_time.time() + 86_400)}),
                           Response(200, {"currency": "EUR"})])
    client = api_module.Trading212Client("key", "secret", session=session, sleep=lambda s: (waits.append(s), clock.sleep(s))[0], clock=clock)
    assert client.summary() == {"currency": "EUR"}
    assert max(waits) <= api_module.MAX_RATE_WAIT


def test_the_venue_name_is_checked(sandbox, monkeypatch):
    monkeypatch.setenv("HERMES_VENUE", "somewhere")
    with pytest.raises(ValueError):
        run_daily.run(mode="live", offline=True, now=NIGHT, live_dir=sandbox / "live", site_dir=sandbox / "site")


def test_a_fall_in_a_holding_the_run_does_not_follow_is_a_loss_not_a_withdrawal(sandbox):
    api, venue = account(cash=1000.0, positions={"ZZZ_US_EQ": 100.0})          # 700 euros of something the owner bought
    first = run(sandbox, venue, AFTERNOON)["account"]
    assert first["nav"] == pytest.approx(1700.0, rel=5e-3)
    state = json.loads(record(sandbox, venue).read_text())
    assert state["holdings"]["ZZZ_US_EQ"] == [100.0, 700.0]
    state["last_bar"] = None
    record(sandbox, venue).write_text(json.dumps(state))
    api.fill_prices["ZZZ_US_EQ"] = 4.2                         # it falls 40% overnight
    venue.broker.refresh()
    later = run(sandbox, venue, AFTERNOON)["account"]
    assert later["flow_today"] == 0.0 and later["initial_capital"] == pytest.approx(first["initial_capital"])
    assert later["drawdown"] == pytest.approx(-280.0 / first["nav"], abs=0.01) and later["return_since_inception"] < -0.15


def test_targets_are_fractions_of_the_whole_account(sandbox):
    shutil.copytree(sandbox / "live", sandbox / "plain")
    api, venue = account(cash=1000.0, positions={"ZZZ_US_EQ": 1000.0 / 7.0})   # half the account is the owner's own holding
    mixed = {o["symbol"]: o["notional_eur"] for o in run(sandbox, venue, AFTERNOON)["account"]["orders_today"]}
    api2, venue2 = account(cash=2000.0, number=5005)
    venue2.clock = lambda: AFTERNOON
    plain = run_daily.run(mode="live", offline=True, now=AFTERNOON, venue=venue2, basket=favourites.Basket(),
                          live_dir=sandbox / "plain", site_dir=sandbox / "site2")["account"]["orders_today"]
    assert mixed and {o["symbol"] for o in plain} == set(mixed)
    for o in plain:                                            # the same sizes as an account worth the same, all in cash
        assert mixed[o["symbol"]] == pytest.approx(o["notional_eur"], rel=0.02)


def test_a_dry_run_does_not_use_up_the_day(sandbox):
    api, unarmed = account(armed=False)
    dry = run(sandbox, unarmed, AFTERNOON)["account"]
    assert api.sent == [] and dry["orders_today"] and json.loads(record(sandbox, unarmed).read_text())["traded"] == {}
    armed = t212.Venue(api, armed=True, max_order_eur=500.0, quote=lambda feed, eurusd: REFERENCE[feed], sleep=lambda seconds: None)
    assert armed.record == unarmed.record
    real = run(sandbox, armed, AFTERNOON)["account"]
    assert api.sent and {"SPY", "GLD"} <= {o["symbol"] for o in real["orders_today"] if o["status"] == "filled"}


class HistoryHiccup(Account):
    """The order history fails to answer the first time it is asked."""

    hiccups = 1

    def history(self, ticker=None, limit=50):
        if self.hiccups:
            self.hiccups -= 1
            from hermes_quant.data.trading212 import Trading212Error
            raise Trading212Error("try later", status=503)
        return super().history(ticker, limit)


def test_an_order_that_was_never_sent_keeps_its_round(sandbox):
    api, venue = account(kind=HistoryHiccup)
    first = run(sandbox, venue, AFTERNOON)["account"]
    unsent = [o for o in first["orders_today"] if o["status"] == "skipped" and "nothing was sent" in o["detail"]]
    assert len(unsent) == 1 and unsent[0]["symbol"] not in json.loads(record(sandbox, venue).read_text())["traded"]
    again = run(sandbox, venue, AFTERNOON)["account"]
    assert [o["symbol"] for o in again["orders_today"] if o["status"] == "filled"] == [unsent[0]["symbol"]]


def test_the_order_planner_counts_value_held_outside_the_plan_without_spending_it():
    import numpy as np

    from hermes_quant.portfolio.execution import CostModel, TradingRules, plan_orders

    args = (np.array([0.5]), np.array([0.0]), 1000.0, np.array([True]), ["equity"], CostModel(), TradingRules())
    alone, beside = plan_orders(*args)[0], plan_orders(*args, outside=1000.0)[0]
    assert beside == pytest.approx(2 * alone) and beside < 1000.0          # twice the target in euros, still paid from the cash
    assert plan_orders(*args, outside=0.0)[0] == alone
    assert plan_orders(np.array([0.9]), np.array([0.0]), 100.0, np.array([True]), ["equity"], CostModel(), TradingRules(), outside=5000.0)[0] <= 100.0


class FillsWhileBeingRead(Account):
    """A working order that fills at the worst moment: after the holdings were read, before the cash."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.late_fill = None

    def positions(self):
        result = super().positions()
        if self.late_fill:                                     # the fill lands right after this read
            ticker, shares, cost = self.late_fill
            self.late_fill = None
            self.holdings[ticker] = self.holdings.get(ticker, 0.0) + shares
            self.cash_balance -= cost
            self.orders[1]["status"], self.orders[1]["filledQuantity"] = "FILLED", shares
        return result


def test_the_account_is_never_pictured_with_old_shares_and_new_cash():
    api = FillsWhileBeingRead(cash=1000.0)
    api.instrument_list, api.exchange_list, api.fill_prices, api.fx = [dict(i) for i in INSTRUMENTS], EXCHANGES, dict(FILL), {"EUR": EUR, "USD": USD}
    api.orders[1] = {"id": 1, "ticker": "PPFBd_EQ", "instrument": {"ticker": "PPFBd_EQ"}, "initiatedFrom": "API",
                     "quantity": 1.0, "filledQuantity": 0.0, "status": "NEW"}       # left working by an earlier run
    api.late_fill = ("PPFBd_EQ", 1.0, 68.0)
    listings, _ = t212.resolve({"GLD": {"isin": ["IE00B4ND3602"]}}, api.instrument_list)
    broker = t212.Trading212Broker(api, listings, sleep=lambda seconds: None)
    assert broker.units() == {"GLD": 1.0} and broker.pending() == {}
    assert broker.cash() + sum(broker.units()[s] * broker.prices()[s] for s in broker.units()) == pytest.approx(1000.0)


def test_a_kraken_error_can_never_carry_the_credentials(monkeypatch):
    from hermes_quant.data import kraken

    class Echo:
        def json(self):
            return {"error": ["EAPI:Invalid key my-key-123 / c2VjcmV0"]}

    monkeypatch.setattr(kraken.requests, "post", lambda *args, **kwargs: Echo())
    client = kraken.KrakenClient("my-key-123", "c2VjcmV0")
    with pytest.raises(kraken.KrakenError) as caught:
        client.balances()
    assert "my-key-123" not in str(caught.value) and "c2VjcmV0" not in str(caught.value) and "Invalid key" in str(caught.value)
