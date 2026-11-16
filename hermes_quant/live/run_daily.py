"""The daily production run. One command, no human, no laptop.

    python -m hermes_quant.live.run_daily

  1. pull yesterday's completed candles from Kraken into the price store
  2. rebuild the panel and recompute every model input from scratch
  3. refresh the weekly target (or keep the current one)
  4. plan orders with the same rules the backtest used, and execute them
     - HERMES_MODE=paper (default): simulated fills, same cost model as the research
     - HERMES_MODE=live: real Kraken orders, validate-only until HERMES_LIVE_ARMED=yes
  5. write state, logs and the dashboard's data file; send a notification

Built to fail safe
------------------
* Idempotent: a second run for the same candle date trades nothing.
* In live mode the exchange, not the state file, is the source of truth for cash
  and positions; a run that died half way cannot cause a double trade.
* A market whose data failed to download is frozen for the day (no orders, target
  unchanged) instead of being mistaken for "no signal" and sold. If the currency
  rate or a third of the markets are missing, nothing trades at all.
* A 25% fall from the account's peak trips a circuit breaker: everything is sold
  and the system stays in cash until a human clears ``halted`` in state.json.
* Money that moves without the model (a deposit, a withdrawal, something bought that
  the model cannot see) is recognised as a flow: the performance base and the peak
  move with it, so it neither flatters the return nor trips the circuit breaker.
* The owner's basket of single stocks (favourites.py) is computed after the model and
  apart from it. Switched off, or broken, it changes nothing the model does.

Environment
-----------
HERMES_MODE              paper | live                         (default paper)
HERMES_PROFILE           balanced | growth                    (default balanced)
HERMES_CAPITAL           starting euros for a new paper account (default 233.47)
KRAKEN_API_KEY/SECRET    only for live mode; trade permission, never withdrawal
HERMES_LIVE_ARMED        "yes" to send real orders; anything else = validate only
HERMES_MAX_ORDER_EUR     cap per order in live mode             (default 100)
HERMES_TOKENIZED_API     "yes" if Kraken accepts tokenized-stock orders by API from the
                         account. It does not for accounts in the European Economic
                         Area, so the default is "no": in live mode those orders are
                         listed for the owner to place by hand on kraken.com
HERMES_REBASE            "yes" once after a deposit or withdrawal: restart the
                         performance and drawdown clock from today's value
HERMES_VENUE             kraken | trading212: where a live account trades   (default kraken)
                         On Trading 212 every order goes by API (real shares, and European
                         funds standing in for the US ones); nothing is left to place by hand.
                         Its record is kept apart, under data/live/trading212/.
T212_API_KEY/SECRET      only for HERMES_VENUE=trading212. The key cannot withdraw money:
                         Trading 212's API has no such call
T212_ENV                 demo | live: practice money or real money              (default demo)
HERMES_FAVOURITES        "yes" to let the basket of single stocks in config/favourites.json
                         into the target (default "no": the stocks are scored and shown,
                         and one the owner bought himself is valued, but none is traded)
TELEGRAM_BOT_TOKEN/CHAT_ID  optional daily message
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .. import __version__
from ..data import kraken
from ..data.trading212 import Trading212Client
from ..model import PROFILES, compute_features, target_weights
from ..portfolio.execution import CostModel, TradingRules, plan_orders
from ..report.snapshot import NAMES, model_snapshot
from ..universe import LIVE_INSTRUMENTS, STOCK_RADAR, Instrument
from . import favourites, monitors, notify
from . import trading212 as t212
from .brokers import KrakenBroker, PaperBroker
from .market import FX_SYMBOL, PriceStore, build_live_panel, live_regime, refresh_prices

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "live"
SITE = ROOT / "site"
MAX_DRAWDOWN_HALT = 0.25        # circuit breaker: stop and go to cash after a 25% fall from the peak
OUTAGE_SHARE = 1.0 / 3.0        # this share of markets without data = do not trade today
NEW_YORK = ZoneInfo("America/New_York")
FLOW_MIN_EUR, FLOW_SHARE = 5.0, 0.03   # an unexplained change in account value above both is money moved, not P&L
SCHEDULE_UTC = "01:23"          # when the nightly workflow runs; a test keeps it equal to the workflow's cron


def _now() -> datetime:
    return datetime.now(timezone.utc)


def load_state(path: Path, capital: float, profile: str, mode: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 2, "mode": mode, "profile": profile, "inception": _now().strftime("%Y-%m-%d"),
            "initial_capital": capital, "cash": capital, "units": {}, "peak_nav": capital, "halted": False,
            "target": {}, "target_date": None, "last_bar": None, "runs": 0, "benchmarks": {}, "new": True}


def market_open(inst: Instrument, now: datetime) -> bool:
    """Crypto never closes. Kraken's tokenized stocks trade around the clock from
    Sunday 8 pm to Friday 8 pm, New York time (01:00 UTC in winter, 00:00 in summer)."""
    if not inst.tokenized:
        return True
    ny = now.astimezone(NEW_YORK)
    day, hour = ny.weekday(), ny.hour + ny.minute / 60.0        # Monday = 0 ... Sunday = 6
    if day == 6:
        return hour >= 20.0
    if day == 5:
        return False
    if day == 4:
        return hour < 20.0
    return True


def current_prices(instruments: list[Instrument], last_close: pd.Series, fx_close: float,
                   now: datetime, offline: bool) -> tuple[dict, float]:
    """Euro price per symbol right now: the ticker mid where a market is open and
    answers, otherwise the last completed candle. Also returns dollars per euro."""
    prices = {s: float(v) for s, v in last_close.items() if np.isfinite(v)}
    eurusd = float(fx_close) if np.isfinite(fx_close) else 0.0
    if offline:
        return prices, eurusd
    try:
        eurusd = kraken.ticker("EURUSD")["mid"]
    except Exception:  # noqa: BLE001 - keep yesterday's rate
        pass
    for inst in instruments:
        if not market_open(inst, now):
            continue                                   # weekend quotes on a shut market are noise
        try:
            mid = kraken.ticker(inst.kraken_pair, inst.tokenized)["mid"]
        except Exception:  # noqa: BLE001 - keep the candle close
            continue
        if inst.quote == "USD":
            if eurusd <= 0:
                continue
            mid /= eurusd
        if mid > 0:
            prices[inst.symbol] = mid
    return prices, eurusd


def run(mode: str | None = None, profile: str | None = None, offline: bool = False,
        live_dir: Path = LIVE, site_dir: Path = SITE, broker=None, now: datetime | None = None,
        basket: favourites.Basket | None = None, venue=None) -> dict:
    mode = mode or os.environ.get("HERMES_MODE", "paper")
    profile = profile or os.environ.get("HERMES_PROFILE", "balanced")
    cfg, rules, costs = PROFILES[profile], TradingRules(), CostModel()
    now = now or _now()
    live_dir.mkdir(parents=True, exist_ok=True)
    # Where a live account trades. Each venue is a different account, so each keeps its own
    # record (state, equity, trades); the price store is shared.
    venue_name = (getattr(venue, "name", None) or os.environ.get("HERMES_VENUE", "kraken")) if mode == "live" else "kraken"
    if venue_name not in ("kraken", "trading212"):
        raise ValueError(f"HERMES_VENUE must be kraken or trading212, not {venue_name!r}")
    book_dir = live_dir if venue_name == "kraken" else live_dir / venue_name
    book_dir.mkdir(parents=True, exist_ok=True)

    # The owner's list of single stocks. A list that cannot be read is reported and
    # ignored: it must never stand between the model and its orders.
    basket_error = ""
    if basket is None:
        try:
            basket = favourites.load()
        except Exception as exc:  # noqa: BLE001 - shown on the dashboard
            basket_error = f"{type(exc).__name__}: {exc}"[:200]
    basket_on = basket is not None and favourites.enabled()

    # 1-2. data and model -------------------------------------------------------------
    store = PriceStore(live_dir / "prices")
    data_errors = {} if offline else refresh_prices(store, LIVE_INSTRUMENTS)
    stocks = favourites.instruments(basket) if basket else []
    stock_errors = {} if offline else refresh_prices(store, stocks, fx=False)
    stocks = [i for i in stocks if len(store.load(i.symbol))]
    by_symbol = {i.symbol: i for i in LIVE_INSTRUMENTS + stocks}
    names = {**NAMES, **(basket.stocks if basket else {})}
    instruments = [i for i in LIVE_INSTRUMENTS if len(store.load(i.symbol))]
    panel = build_live_panel(store, instruments)
    regime = live_regime(panel, cfg.regime_proxy, cfg.hmm_max_window) if cfg.regime else pd.DataFrame(index=panel.close.index)
    features = compute_features(panel, cfg, p_turbulent=regime)
    t = len(panel.close) - 1
    bar_date = panel.close.index[t]
    bar = bar_date.strftime("%Y-%m-%d")
    core = list(panel.close.columns)                        # the model's own markets
    class_of = {**panel.classes, **{i.symbol: i.asset_class for i in stocks}}

    # A market is stale when its newest candle is older than the newest one anywhere:
    # its download failed today. (Tokenized stocks are forward-filled in the panel, so
    # this has to be read from the store, not from the panel.)
    stale = sorted(s for s in core if store.load(s).index[-1] < bar_date)
    fx_stale = FX_SYMBOL in data_errors or not np.isfinite(panel.eurusd.iloc[t])
    outage = fx_stale or len(stale) > OUTAGE_SHARE * len(core)

    # 3. target -----------------------------------------------------------------------
    state_path = book_dir / "state.json"
    state = load_state(state_path, float(os.environ.get("HERMES_CAPITAL", "233.47")), profile, mode)
    allocation = target_weights(features, cfg, t)

    # The basket is scored every day, switched on or not, so the dashboard can show what
    # it would hold. It sees the model's filters but the model never sees it.
    view, stock_close = None, {}
    stock_stale = sorted(i.symbol for i in stocks if store.load(i.symbol).index[-1] < bar_date)
    if stocks:
        held = set(state.get("basket_target") or {})
        try:
            view = favourites.evaluate(store, basket, cfg, bar_date, held)
            stock_close = view["last_close"]
        except Exception as exc:  # noqa: BLE001 - the model carries on without the basket
            basket_error = f"{type(exc).__name__}: {exc}"[:200]
            # a stock already in the account must still be valued: its last candle, in euros
            fx_close = float(panel.eurusd.ffill().iloc[t])
            for inst in stocks:
                candles = store.load(inst.symbol)[:bar_date]
                if len(candles) and fx_close > 0:
                    stock_close[inst.symbol] = float(candles["close"].iloc[-1]) / fx_close
    symbols = core + [i.symbol for i in stocks if i.symbol in stock_close]
    classes = [class_of[s] for s in symbols]
    # The model's markets are always its to trade. A single stock is only while the
    # basket is switched on; otherwise it is valued and left alone.
    managed = np.array([class_of[s] != "stock" or basket_on for s in symbols])

    age = None if not state["target_date"] else (bar_date - pd.Timestamp(state["target_date"])).days
    fingerprint = hashlib.sha256(repr(cfg).encode()).hexdigest()[:12]      # a changed model means a new target
    due = age is None or age >= rules.rebalance_every or state.get("model") != fingerprint
    # never set a new target from incomplete data, unless there is no target at all yet
    refreshed = due and (age is None or not (stale or outage))
    if refreshed:
        state["target"] = {s: float(w) for s, w in allocation.weights.items() if abs(w) > 1e-9}
        state["target_date"] = bar
        state["profile"] = profile
        state["model"] = fingerprint

    # The basket's target is kept apart from the model's, which it never alters. It is
    # redrawn whenever the model's is (weekly) and whenever the basket itself changes:
    # switched on, a different list or limit. It only takes money the model leaves idle;
    # short of room, it shrinks.
    basket_stamp = hashlib.sha256(repr(basket).encode()).hexdigest()[:12] if basket_on else None
    if not basket_on:
        state.pop("basket_target", None)
        state.pop("basket_stamp", None)
    elif view is not None and not outage and (refreshed or state.get("basket_stamp") != basket_stamp):
        picks = view["weights"]
        room = max(0.0, 1.0 - sum(state["target"].values()))
        scale = min(1.0, room / sum(picks.values())) if picks else 0.0
        state["basket_target"] = {s: w * scale for s, w in picks.items() if w * scale > 1e-9}
        state["basket_stamp"] = basket_stamp
    wanted = {**state["target"], **(state.get("basket_target") or {})}

    # 4. account, risk checks, orders ---------------------------------------------------
    last_close = panel.close.ffill().iloc[t]                # a missing candle must not zero a holding
    last_close = pd.concat([last_close, pd.Series(stock_close, dtype=float)])
    armed = mode == "live" and os.environ.get("HERMES_LIVE_ARMED") == "yes"
    max_order = float(os.environ.get("HERMES_MAX_ORDER_EUR", "100"))
    if venue is None and broker is None and venue_name == "trading212":
        client = Trading212Client(os.environ["T212_API_KEY"], os.environ["T212_API_SECRET"], env=os.environ.get("T212_ENV", "demo"))
        venue = t212.Venue(client, armed=armed, max_order_eur=max_order)
    on_venue = venue is not None                            # Trading 212: its own instruments, prices and hours
    if on_venue:
        broker = venue.broker
    # Kraken takes no tokenized-stock orders by API from accounts in the European Economic
    # Area: there the model sends none of them and lists them for the owner instead.
    tokenized_api = os.environ.get("HERMES_TOKENIZED_API", "no") == "yes"
    by_hand = np.array([mode == "live" and not on_venue and by_symbol[s].tokenized and not tokenized_api for s in symbols])
    if broker is None:
        if mode == "live":
            client = kraken.KrakenClient(os.environ["KRAKEN_API_KEY"], os.environ["KRAKEN_API_SECRET"])
            pair_rules = {**kraken.pair_rules(False), **kraken.pair_rules(True)}
            broker = KrakenBroker(client, by_symbol, pair_rules=pair_rules, armed=armed, max_order_eur=max_order,
                                  tokenized_via_api=tokenized_api)
        else:
            broker = PaperBroker(state["cash"], state["units"])

    # Prices. On Kraken the model's own series are what is traded. On Trading 212 a market is
    # traded through a different instrument (a European fund, a real share) with its own
    # price per share: the account says what its holdings are worth, a reference source the rest.
    quoted_here = [] if on_venue else instruments + [by_symbol[s] for s in symbols[len(core):]]
    prices, eurusd = current_prices(quoted_here, last_close, panel.eurusd.ffill().iloc[t], now, offline)
    price_gaps: dict[str, str] = {}
    if on_venue:
        prices, price_gaps = venue.prices(symbols, eurusd)

    def is_open(symbol: str) -> bool:
        return venue.open(symbol, now) if on_venue else market_open(by_symbol[symbol], now)

    def account() -> tuple[dict, float, np.ndarray, float]:
        units = broker.units()
        idle_usd = broker.usd_cash() / eurusd if (hasattr(broker, "usd_cash") and eurusd > 0) else 0.0
        cash = broker.cash() + idle_usd
        value = np.nan_to_num(np.array([units.get(s, 0.0) * prices.get(s, np.nan) for s in symbols]))
        return units, cash, value, cash + float(value.sum())

    units, cash, value, nav = account()
    is_new = state.pop("new", False)
    if (mode == "live" and is_new) or os.environ.get("HERMES_REBASE") == "yes":
        # a real account starts from whatever it holds; so does one that was topped up
        state.update(initial_capital=nav, peak_nav=nav, benchmarks={}, inception=now.strftime("%Y-%m-%d"))

    # What the account should be worth if nothing but prices had moved since the last
    # run: yesterday's holdings at today's prices. A gap too large to be trading costs
    # is money that came in or went out, or an asset the model cannot see.
    flow = 0.0
    if mode == "live" and state.get("mode") == "live" and not is_new and not outage:
        held_before = state.get("units") or {}
        expected = float(state.get("cash") or 0.0) + float(np.nansum([held_before.get(s, 0.0) * prices.get(s, np.nan) for s in symbols]))
        gap = nav - expected
        if abs(gap) > max(FLOW_MIN_EUR, FLOW_SHARE * max(expected, nav)):
            flow = gap
            base = float(state["initial_capital"]) + flow
            if base > 1.0:
                state["initial_capital"] = base
                state["peak_nav"] = max(1.0, float(state.get("peak_nav") or nav) + flow)
            else:                                         # more was taken out than was ever put in: start afresh
                state.update(initial_capital=nav, peak_nav=nav, benchmarks={}, inception=now.strftime("%Y-%m-%d"))
            state["flows"] = (state.get("flows") or [])[-19:] + [{"bar": bar, "eur": round(flow, 2)}]
    unrecognised = broker.unrecognised() if hasattr(broker, "unrecognised") else {}
    if on_venue:                                            # a holding the venue knows but this run does not follow
        unrecognised.update({venue.listings[s].ticker: q for s, q in units.items() if s not in symbols})

    state["peak_nav"] = max(float(state.get("peak_nav") or nav), nav)
    drawdown = nav / state["peak_nav"] - 1.0
    if drawdown <= -MAX_DRAWDOWN_HALT and not state["halted"] and not outage:
        state["halted"] = True
        state["halt_reason"] = f"drawdown {drawdown:.1%} from peak on {bar}"
    target = np.array([0.0 if state["halted"] else wanted.get(s, 0.0) for s in symbols])

    already_ran = state.get("last_bar") == bar
    frozen = set(stale) | set(stock_stale)
    # One round of orders per market per daily candle. The round is used up only by a run
    # in which that market could trade: a run that finds an exchange shut (at night, on a
    # weekend) leaves its markets for a later run on the same candle.
    done = dict(state.get("traded") or {}) if already_ran else {}
    if already_ran and "traded" not in state:               # a state file from before this bookkeeping
        done = {s: bar for s in symbols}
    fresh = np.array([done.get(s) != bar for s in symbols])
    tradable = managed & fresh & np.array([is_open(s) and s not in frozen and np.isfinite(prices.get(s, np.nan))
                                           for s in symbols])
    if outage:
        orders = np.zeros(len(symbols))
    else:
        orders = plan_orders(target, value, cash, tradable & ~by_hand, classes, costs, rules)
        done.update({s: bar for s, ok in zip(symbols, tradable & ~by_hand) if ok})
    state["traded"] = {s: b for s, b in done.items() if b == bar}
    order_map = {s: float(o) for s, o in zip(symbols, orders) if o != 0}
    fills = []
    if order_map:
        if isinstance(broker, PaperBroker):
            fills = broker.execute(order_map, prices, by_symbol, costs)
        else:
            fills = broker.execute(order_map, prices, by_symbol, costs, eurusd=eurusd)
            broker.refresh()
    units, cash, value, nav = account()

    # What is left for the owner to place by hand, from the account as it now stands. It
    # is a standing instruction, recomputed every run until the holdings match.
    manual_orders = []
    if by_hand.any() and not outage:
        quoted = managed & np.array([s not in frozen and np.isfinite(prices.get(s, np.nan)) for s in symbols])
        pending = plan_orders(target, value, cash, by_hand & quoted, classes, costs, rules)
        manual_orders = [{"symbol": s, "name": names.get(s, s), "class": class_of[s], "notional_eur": float(n),
                          "pair": by_symbol[s].kraken_pair, "market_open": bool(is_open(s))}
                         for s, n in zip(symbols, pending) if n != 0]

    # 5. persist ----------------------------------------------------------------------
    # "What if the money had simply been left in bitcoin, or in the S&P 500": on a venue that
    # trades a market through another instrument, that is measured on the model's own series.
    level = {s: float(v) for s, v in last_close.items() if np.isfinite(v)} if on_venue else prices
    for key, symbol in (("btc", "BTC"), ("spx", "SPY")):
        if symbol in level and key not in state["benchmarks"]:
            state["benchmarks"][key] = level[symbol]
    bench = {key: state["initial_capital"] * level[sym] / state["benchmarks"][key]
             for key, sym in (("btc", "BTC"), ("spx", "SPY")) if key in state["benchmarks"] and sym in level}
    state.update(cash=cash, units=units, mode=mode, last_bar=bar, runs=int(state.get("runs", 0)) + (0 if already_ran else 1),
                 last_run=now.strftime("%Y-%m-%dT%H:%M:%SZ"), nav=nav)
    state_path.write_text(json.dumps(state, indent=2))

    equity_path = book_dir / "equity.csv"
    row = {"date": bar, "nav": round(nav, 4), "cash": round(cash, 4), "invested": round(float(value.sum()), 4),
           "btc_hold": round(bench.get("btc", np.nan), 4), "spx_hold": round(bench.get("spx", np.nan), 4),
           "positions": int(np.sum(value > 0.01))}
    equity = pd.read_csv(equity_path) if equity_path.exists() else pd.DataFrame(columns=list(row))
    equity = pd.concat([equity[equity["date"] != bar], pd.DataFrame([row])], ignore_index=True)
    equity.to_csv(equity_path, index=False)

    trades_path = book_dir / "trades.csv"
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    if fills:
        new = pd.DataFrame([{"time": state["last_run"], "bar": bar, "mode": mode, **f.as_dict()} for f in fills])
        trades = pd.concat([trades, new], ignore_index=True)
        trades.to_csv(trades_path, index=False)

    snapshot = model_snapshot(panel, features, cfg, t, allocation)
    for row in snapshot["assets"]:                        # what the live terminal needs to follow each market
        inst = by_symbol[row["symbol"]]
        row.update(pair=inst.kraken_pair, quote=inst.quote, tokenized=inst.tokenized)
    positions = [{"symbol": s, "name": names.get(s, s), "class": class_of[s], "units": units.get(s, 0.0),
                  "price": prices.get(s), "value": float(v), "weight": float(v / nav) if nav else 0.0,
                  "target": float(wanted.get(s, 0.0)), "managed": bool(m),
                  # on a venue, the instrument held (and whether Kraken's live feed still prices the holding)
                  "instrument": venue.listings[s].ticker if on_venue and s in venue.listings else by_symbol[s].kraken_pair,
                  "live_feed": not on_venue}
                 for s, v, m in zip(symbols, value, managed) if v > 0.01 or wanted.get(s, 0.0) > 0]
    held_value = dict(zip(symbols, value))
    basket_view = {"enabled": basket_on, "error": basket_error, "rows": []}
    if basket is not None:
        rows = view["rows"] if view else [{"symbol": s, "name": n, "status": "no_data"} for s, n in basket.stocks.items()]
        for row in rows:
            s = row["symbol"]
            row.update(target=float(wanted.get(s, 0.0)), units=units.get(s, 0.0), value=float(held_value.get(s, 0.0)),
                       stale=s in stock_stale, price_now=prices.get(s))
        basket_view.update(per_stock=basket.per_stock, max_stocks=basket.max_stocks, budget=basket.budget,
                           enter=basket.enter, leave=basket.leave, min_history_days=basket.min_history_days,
                           waiting=list(basket.waiting), rows=rows, errors=stock_errors,
                           target_total=float(sum(r["target"] for r in rows)),
                           invested=float(sum(r["value"] for r in rows) / nav) if nav else 0.0)
    done = trades[trades["status"].isin(["filled", "partial"])] if len(trades) else trades
    # markets the model wants that this venue cannot trade today, and why
    unavailable = {}
    if on_venue:
        for s in symbols:
            if wanted.get(s, 0.0) > 0 and s not in venue.listings:
                unavailable[s] = venue.problems.get(s, "not listed on Trading 212")
            elif wanted.get(s, 0.0) > 0 and s in price_gaps:
                unavailable[s] = price_gaps[s]
    account_view = {
        "mode": mode, "armed": armed, "profile": profile, "inception": state["inception"],
        "venue": venue_name if mode == "live" else "paper", "venue_env": getattr(venue, "env", None), "unavailable": unavailable,
        "initial_capital": state["initial_capital"], "nav": nav, "cash": cash,
        "invested_pct": float(value.sum() / nav) if nav else 0.0,
        "return_since_inception": nav / state["initial_capital"] - 1.0, "peak_nav": state["peak_nav"],
        "drawdown": nav / state["peak_nav"] - 1.0, "halted": state["halted"],
        "target_date": state["target_date"], "target_refreshed_today": bool(refreshed),
        "next_target_refresh_in_days": max(0, rules.rebalance_every - (0 if refreshed else (age or 0))),
        "positions": positions, "equity": equity.to_dict(orient="records"),
        "trades": done.tail(40).to_dict(orient="records") if len(done) else [],
        "orders_today": [f.as_dict() for f in fills], "manual_orders": manual_orders,
        "flow_today": flow, "flows": state.get("flows") or [], "unrecognised": unrecognised}
    out = {"meta": {"generated_at": state["last_run"], "bar_date": bar, "version": __version__,
                    "data_errors": data_errors, "stale_markets": stale, "data_outage": bool(outage),
                    "price_source": "kraken", "schedule_utc": SCHEDULE_UTC},
           "account": account_view, "model": snapshot, "favourites": basket_view,
           "radar": [] if offline else monitors.stock_radar(STOCK_RADAR),
           "derivatives": [] if offline else monitors.derivatives_monitor()}
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "live.json").write_text(json.dumps(out, default=float))
    (book_dir / "latest.json").write_text(json.dumps(
        {"meta": out["meta"], "account": {k: v for k, v in account_view.items() if k not in ("equity", "trades")},
         "targets": wanted}, indent=2, default=float))
    notify.send(notify.daily_message(out))
    return out


def main() -> None:
    out = run(offline="--offline" in sys.argv)
    a, meta = out["account"], out["meta"]
    print(f"[{meta['bar_date']}] mode={a['mode']}{' (ARMED)' if a['armed'] else ''} profile={a['profile']} "
          f"NAV {a['nav']:.2f} EUR ({a['return_since_inception']:+.2%} since {a['inception']}), "
          f"invested {a['invested_pct']:.0%}, {len(a['orders_today'])} order(s)")
    for f in a["orders_today"]:
        print(f"   {f['status']:9s} {f['symbol']:6s} {f['notional_eur']:+9.2f} EUR at {f['price_eur']:.6g}  cost {f['cost_eur']:.3f}  {f['detail']}")
    if a["venue"] == "trading212":
        print(f"   VENUE     Trading 212 ({a['venue_env']}): every order goes by API, nothing to place by hand")
    for s, why in a["unavailable"].items():
        print(f"   NOT TRADED here: {s}: {why}")
    for o in a["manual_orders"]:
        print(f"   BY HAND   {o['symbol']:6s} {o['notional_eur']:+9.2f} EUR  {'buy' if o['notional_eur'] > 0 else 'sell'} "
              f"{o['pair'][:-3]} on kraken.com" + ("" if o["market_open"] else "  (its market is closed right now)"))
    fav = out["favourites"]
    if fav["enabled"]:
        picked = [f"{r['symbol']} {r['target']:.1%}" for r in fav["rows"] if r.get("target", 0.0) > 0]
        print("   FAVOURITES basket on:", ", ".join(picked) if picked else "no stock qualifies today")
    if fav["error"]:
        print("   FAVOURITES ignored:", fav["error"])
    if a["flow_today"]:
        print(f"   EXTERNAL FLOW {a['flow_today']:+.2f} EUR: money moved without the model (deposit, withdrawal or an asset "
              "it cannot see). Treated as a flow, not as profit or loss.")
    if a["unrecognised"]:
        print("   NOT RECOGNISED at the exchange (left out of the account value):", ", ".join(f"{k}={v:.6g}" for k, v in a["unrecognised"].items()))
    if meta["data_errors"] or meta["stale_markets"]:
        print("   data problems:", meta["data_errors"], "stale:", meta["stale_markets"],
              "-> NO TRADING TODAY" if meta["data_outage"] else "-> those markets frozen for today")
    if a["halted"]:
        print("   CIRCUIT BREAKER ACTIVE: portfolio moved to cash; clear 'halted' in data/live/state.json to resume")


if __name__ == "__main__":
    main()
