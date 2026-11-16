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


def account(cash=1000.0, positions=None, *, armed=True, quotes=REFERENCE, instruments=INSTRUMENTS):
    api = FakeTrading212(cash=cash, positions=positions)
    api.instrument_list, api.exchange_list = [dict(i) for i in instruments], EXCHANGES
    api.fill_prices, api.fx = dict(FILL), {"EUR": EUR, "USD": USD}

    def quote(feed, eurusd):
        return quotes[feed]

    return api, t212.Venue(api, armed=armed, max_order_eur=500.0, quote=quote, sleep=lambda seconds: None)


def run(sandbox, venue, now, **kw):
    return run_daily.run(mode="live", offline=True, now=now, venue=venue, basket=kw.pop("basket", favourites.Basket()),
                         live_dir=sandbox / "live", site_dir=sandbox / "site", **kw)


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
    assert (sandbox / "live" / "trading212" / "state.json").exists() and not (sandbox / "live" / "state.json").exists()
    state = json.loads((sandbox / "live" / "trading212" / "state.json").read_text())
    assert set(state["units"]) == set(held) and state["mode"] == "live"

    state["last_bar"] = None                                   # the next day: buying was not a flow, and nothing else is due
    (sandbox / "live" / "trading212" / "state.json").write_text(json.dumps(state))
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
    path = sandbox / "live" / "trading212" / "state.json"
    state = json.loads(path.read_text())
    state.update(peak_nav=state["nav"] * 1.5, last_bar=None)
    path.write_text(json.dumps(state))
    out = run(sandbox, venue, AFTERNOON)["account"]
    assert out["halted"] and all(s["quantity"] < 0 for s in api.sent[-2:])
    assert sum(q * FILL[t] for t, q in api.holdings.items()) == pytest.approx(0.0, abs=1.0)


def test_a_holding_the_run_does_not_follow_is_reported_and_never_touched(sandbox):
    api, venue = account(positions={"TSLA_US_EQ": 0.5, "ZZZ_US_EQ": 3.0})      # no list of stocks in this run
    a = run(sandbox, venue, AFTERNOON)["account"]
    assert api.holdings["TSLA_US_EQ"] == 0.5 and not [s for s in api.sent if s["ticker"] in ("TSLA_US_EQ", "ZZZ_US_EQ")]
    assert a["unrecognised"] == {"TSLA_US_EQ": 0.5, "ZZZ_US_EQ": 3.0}


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
