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

Environment
-----------
HERMES_MODE              paper | live                         (default paper)
HERMES_PROFILE           balanced | growth                    (default balanced)
HERMES_CAPITAL           starting euros for a new paper account (default 233.47)
KRAKEN_API_KEY/SECRET    only for live mode; trade permission, never withdrawal
HERMES_LIVE_ARMED        "yes" to send real orders; anything else = validate only
HERMES_MAX_ORDER_EUR     cap per order in live mode             (default 100)
HERMES_REBASE            "yes" once after a deposit or withdrawal: restart the
                         performance and drawdown clock from today's value
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
from ..model import PROFILES, compute_features, target_weights
from ..portfolio.execution import CostModel, TradingRules, plan_orders
from ..report.snapshot import NAMES, model_snapshot
from ..universe import LIVE_INSTRUMENTS, STOCK_RADAR, Instrument
from . import monitors, notify
from .brokers import KrakenBroker, PaperBroker
from .market import FX_SYMBOL, PriceStore, build_live_panel, live_regime, refresh_prices

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "live"
SITE = ROOT / "site"
MAX_DRAWDOWN_HALT = 0.25        # circuit breaker: stop and go to cash after a 25% fall from the peak
OUTAGE_SHARE = 1.0 / 3.0        # this share of markets without data = do not trade today
NEW_YORK = ZoneInfo("America/New_York")


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
        live_dir: Path = LIVE, site_dir: Path = SITE, broker=None, now: datetime | None = None) -> dict:
    mode = mode or os.environ.get("HERMES_MODE", "paper")
    profile = profile or os.environ.get("HERMES_PROFILE", "balanced")
    cfg, rules, costs = PROFILES[profile], TradingRules(), CostModel()
    by_symbol = {i.symbol: i for i in LIVE_INSTRUMENTS}
    now = now or _now()
    live_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. data and model -------------------------------------------------------------
    store = PriceStore(live_dir / "prices")
    data_errors = {} if offline else refresh_prices(store, LIVE_INSTRUMENTS)
    instruments = [i for i in LIVE_INSTRUMENTS if len(store.load(i.symbol))]
    panel = build_live_panel(store, instruments)
    regime = live_regime(panel, cfg.regime_proxy, cfg.hmm_max_window) if cfg.regime else pd.DataFrame(index=panel.close.index)
    features = compute_features(panel, cfg, p_turbulent=regime)
    t = len(panel.close) - 1
    bar_date = panel.close.index[t]
    bar = bar_date.strftime("%Y-%m-%d")
    symbols = list(panel.close.columns)
    classes = [panel.classes[s] for s in symbols]

    # A market is stale when its newest candle is older than the newest one anywhere:
    # its download failed today. (Tokenized stocks are forward-filled in the panel, so
    # this has to be read from the store, not from the panel.)
    stale = sorted(s for s in symbols if store.load(s).index[-1] < bar_date)
    fx_stale = FX_SYMBOL in data_errors or not np.isfinite(panel.eurusd.iloc[t])
    outage = fx_stale or len(stale) > OUTAGE_SHARE * len(symbols)

    # 3. target -----------------------------------------------------------------------
    state_path = live_dir / "state.json"
    state = load_state(state_path, float(os.environ.get("HERMES_CAPITAL", "233.47")), profile, mode)
    allocation = target_weights(features, cfg, t)
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

    # 4. account, risk checks, orders ---------------------------------------------------
    last_close = panel.close.ffill().iloc[t]                # a missing candle must not zero a holding
    prices, eurusd = current_prices(instruments, last_close, panel.eurusd.ffill().iloc[t], now, offline)
    armed = mode == "live" and os.environ.get("HERMES_LIVE_ARMED") == "yes"
    if broker is None:
        if mode == "live":
            client = kraken.KrakenClient(os.environ["KRAKEN_API_KEY"], os.environ["KRAKEN_API_SECRET"])
            pair_rules = {**kraken.pair_rules(False), **kraken.pair_rules(True)}
            broker = KrakenBroker(client, by_symbol, pair_rules=pair_rules, armed=armed,
                                  max_order_eur=float(os.environ.get("HERMES_MAX_ORDER_EUR", "100")))
        else:
            broker = PaperBroker(state["cash"], state["units"])

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

    state["peak_nav"] = max(float(state.get("peak_nav") or nav), nav)
    drawdown = nav / state["peak_nav"] - 1.0
    if drawdown <= -MAX_DRAWDOWN_HALT and not state["halted"] and not outage:
        state["halted"] = True
        state["halt_reason"] = f"drawdown {drawdown:.1%} from peak on {bar}"
    target = np.array([0.0 if state["halted"] else state["target"].get(s, 0.0) for s in symbols])

    already_ran = state.get("last_bar") == bar
    tradable = np.array([market_open(by_symbol[s], now) and s not in stale and np.isfinite(prices.get(s, np.nan))
                         for s in symbols])
    if already_ran or outage:
        orders = np.zeros(len(symbols))
    else:
        orders = plan_orders(target, value, cash, tradable, classes, costs, rules)
    order_map = {s: float(o) for s, o in zip(symbols, orders) if o != 0}
    fills = []
    if order_map:
        if isinstance(broker, PaperBroker):
            fills = broker.execute(order_map, prices, by_symbol, costs)
        else:
            fills = broker.execute(order_map, prices, by_symbol, costs, eurusd=eurusd)
            broker.refresh()
    units, cash, value, nav = account()

    # 5. persist ----------------------------------------------------------------------
    for key, symbol in (("btc", "BTC"), ("spx", "SPY")):
        if symbol in prices and key not in state["benchmarks"]:
            state["benchmarks"][key] = prices[symbol]
    bench = {key: state["initial_capital"] * prices[sym] / state["benchmarks"][key]
             for key, sym in (("btc", "BTC"), ("spx", "SPY")) if key in state["benchmarks"] and sym in prices}
    state.update(cash=cash, units=units, mode=mode, last_bar=bar, runs=int(state.get("runs", 0)) + (0 if already_ran else 1),
                 last_run=now.strftime("%Y-%m-%dT%H:%M:%SZ"), nav=nav)
    state_path.write_text(json.dumps(state, indent=2))

    equity_path = live_dir / "equity.csv"
    row = {"date": bar, "nav": round(nav, 4), "cash": round(cash, 4), "invested": round(float(value.sum()), 4),
           "btc_hold": round(bench.get("btc", np.nan), 4), "spx_hold": round(bench.get("spx", np.nan), 4),
           "positions": int(np.sum(value > 0.01))}
    equity = pd.read_csv(equity_path) if equity_path.exists() else pd.DataFrame(columns=list(row))
    equity = pd.concat([equity[equity["date"] != bar], pd.DataFrame([row])], ignore_index=True)
    equity.to_csv(equity_path, index=False)

    trades_path = live_dir / "trades.csv"
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    if fills:
        new = pd.DataFrame([{"time": state["last_run"], "bar": bar, "mode": mode, **f.as_dict()} for f in fills])
        trades = pd.concat([trades, new], ignore_index=True)
        trades.to_csv(trades_path, index=False)

    snapshot = model_snapshot(panel, features, cfg, t, allocation)
    positions = [{"symbol": s, "name": NAMES.get(s, s), "class": panel.classes[s], "units": units.get(s, 0.0),
                  "price": prices.get(s), "value": float(v), "weight": float(v / nav) if nav else 0.0,
                  "target": float(state["target"].get(s, 0.0))}
                 for s, v in zip(symbols, value) if v > 0.01 or state["target"].get(s, 0.0) > 0]
    done = trades[trades["status"].isin(["filled", "partial"])] if len(trades) else trades
    account_view = {
        "mode": mode, "armed": armed, "profile": profile, "inception": state["inception"],
        "initial_capital": state["initial_capital"], "nav": nav, "cash": cash,
        "invested_pct": float(value.sum() / nav) if nav else 0.0,
        "return_since_inception": nav / state["initial_capital"] - 1.0, "peak_nav": state["peak_nav"],
        "drawdown": nav / state["peak_nav"] - 1.0, "halted": state["halted"],
        "target_date": state["target_date"], "target_refreshed_today": bool(refreshed),
        "next_target_refresh_in_days": max(0, rules.rebalance_every - (0 if refreshed else (age or 0))),
        "positions": positions, "equity": equity.to_dict(orient="records"),
        "trades": done.tail(40).to_dict(orient="records") if len(done) else [],
        "orders_today": [f.as_dict() for f in fills]}
    out = {"meta": {"generated_at": state["last_run"], "bar_date": bar, "version": __version__,
                    "data_errors": data_errors, "stale_markets": stale, "data_outage": bool(outage),
                    "price_source": "kraken"},
           "account": account_view, "model": snapshot,
           "radar": [] if offline else monitors.stock_radar(STOCK_RADAR),
           "derivatives": [] if offline else monitors.derivatives_monitor()}
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "live.json").write_text(json.dumps(out, default=float))
    (live_dir / "latest.json").write_text(json.dumps(
        {"meta": out["meta"], "account": {k: v for k, v in account_view.items() if k not in ("equity", "trades")},
         "targets": state["target"]}, indent=2, default=float))
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
    if meta["data_errors"] or meta["stale_markets"]:
        print("   data problems:", meta["data_errors"], "stale:", meta["stale_markets"],
              "-> NO TRADING TODAY" if meta["data_outage"] else "-> those markets frozen for today")
    if a["halted"]:
        print("   CIRCUIT BREAKER ACTIVE: portfolio moved to cash; clear 'halted' in data/live/state.json to resume")


if __name__ == "__main__":
    main()
