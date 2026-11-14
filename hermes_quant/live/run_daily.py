"""The daily production run. One command, no human, no laptop.

    python -m hermes_quant.live.run_daily

  1. pull yesterday's completed candles from Kraken into the price store
  2. rebuild the panel and recompute every model input from scratch
  3. refresh the weekly target (or keep the current one)
  4. plan orders with the same rules the backtest used, and execute them
     - HERMES_MODE=paper (default): simulated fills, same cost model as the research
     - HERMES_MODE=live: real Kraken orders, validate-only until HERMES_LIVE_ARMED=yes
  5. write state, logs and the dashboard's data file; send a notification

The run is idempotent: a second run for the same candle date trades nothing.

Environment
-----------
HERMES_MODE              paper | live                         (default paper)
HERMES_PROFILE           balanced | growth                    (default balanced)
HERMES_CAPITAL           starting euros for a new paper account (default 233.47)
KRAKEN_API_KEY/SECRET    only for live mode; trade permission, never withdrawal
HERMES_LIVE_ARMED        "yes" to send real orders; anything else = validate only
HERMES_MAX_ORDER_EUR     cap per order in live mode             (default 100)
TELEGRAM_BOT_TOKEN/CHAT_ID  optional daily message
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..data import kraken
from ..model import PROFILES, compute_features, target_weights
from ..portfolio.execution import CostModel, TradingRules, plan_orders
from ..report.snapshot import NAMES, model_snapshot
from ..universe import LIVE_INSTRUMENTS, Instrument
from . import notify
from .brokers import KrakenBroker, PaperBroker
from .market import FX_SYMBOL, PriceStore, build_live_panel, live_regime, refresh_prices

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "live"
SITE = ROOT / "site"
MAX_DRAWDOWN_HALT = 0.25        # circuit breaker: stop and go to cash after a 25% fall from the peak


def _now() -> datetime:
    return datetime.now(timezone.utc)


def load_state(path: Path, capital: float, profile: str, mode: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 2, "mode": mode, "profile": profile, "inception": _now().strftime("%Y-%m-%d"),
            "initial_capital": capital, "cash": capital, "units": {}, "peak_nav": capital, "halted": False,
            "target": {}, "target_date": None, "last_bar": None, "runs": 0, "benchmarks": {}}


def current_prices(instruments: list[Instrument], panel_close: pd.Series, offline: bool) -> tuple[dict, float, dict]:
    """Euro price per symbol right now (ticker mid), falling back to the last candle."""
    prices = {s: float(v) for s, v in panel_close.items() if np.isfinite(v)}
    eurusd, source = None, {}
    if not offline:
        try:
            eurusd = kraken.ticker("EURUSD")["mid"]
        except Exception:  # noqa: BLE001
            eurusd = None
        for inst in instruments:
            try:
                mid = kraken.ticker(inst.kraken_pair, inst.tokenized)["mid"]
                if inst.quote == "USD":
                    if not eurusd:
                        continue
                    mid /= eurusd
                if mid > 0:
                    prices[inst.symbol] = mid
                    source[inst.symbol] = "ticker"
            except Exception:  # noqa: BLE001 - keep the candle close
                pass
    return prices, (eurusd or 0.0), source


def market_open(inst: Instrument, now: datetime) -> bool:
    """Crypto never closes; tokenized stocks trade 24 hours, Monday to Friday (UTC)."""
    return (not inst.tokenized) or now.weekday() < 5


def run(mode: str | None = None, profile: str | None = None, offline: bool = False,
        live_dir: Path = LIVE, site_dir: Path = SITE) -> dict:
    mode = mode or os.environ.get("HERMES_MODE", "paper")
    profile = profile or os.environ.get("HERMES_PROFILE", "balanced")
    cfg, rules, costs = PROFILES[profile], TradingRules(), CostModel()
    instruments = LIVE_INSTRUMENTS
    by_symbol = {i.symbol: i for i in instruments}
    now = _now()
    live_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. data and model -------------------------------------------------------------
    store = PriceStore(live_dir / "prices")
    data_errors = {} if offline else refresh_prices(store, instruments)
    instruments = [i for i in instruments if len(store.load(i.symbol))]
    panel = build_live_panel(store, instruments)
    regime = live_regime(panel, cfg.regime_proxy, cfg.hmm_max_window) if cfg.regime else pd.DataFrame(index=panel.close.index)
    features = compute_features(panel, cfg, p_turbulent=regime)
    t = len(panel.close) - 1
    bar_date = panel.close.index[t]
    bar = bar_date.strftime("%Y-%m-%d")
    symbols = list(panel.close.columns)
    classes = [panel.classes[s] for s in symbols]

    # 3. target -----------------------------------------------------------------------
    state_path = live_dir / "state.json"
    state = load_state(state_path, float(os.environ.get("HERMES_CAPITAL", "233.47")), profile, mode)
    allocation = target_weights(features, cfg, t)
    age = None if not state["target_date"] else (bar_date - pd.Timestamp(state["target_date"])).days
    refreshed = age is None or age >= rules.rebalance_every or state.get("profile") != profile
    if refreshed:
        state["target"] = {s: float(w) for s, w in allocation.weights.items() if abs(w) > 1e-9}
        state["target_date"] = bar
        state["profile"] = profile

    # 4. account, risk checks, orders ---------------------------------------------------
    prices, eurusd, price_source = current_prices(instruments, panel.close.iloc[t], offline)
    if mode == "live":
        client = kraken.KrakenClient(os.environ["KRAKEN_API_KEY"], os.environ["KRAKEN_API_SECRET"])
        broker = KrakenBroker(client, by_symbol, armed=os.environ.get("HERMES_LIVE_ARMED") == "yes",
                              max_order_eur=float(os.environ.get("HERMES_MAX_ORDER_EUR", "100")))
        idle_usd_eur = broker.usd_cash() / eurusd if eurusd else 0.0
    else:
        broker = PaperBroker(state["cash"], state["units"])
        idle_usd_eur = 0.0
    units = broker.units()
    cash = broker.cash() + idle_usd_eur
    value = np.array([units.get(s, 0.0) * prices.get(s, np.nan) for s in symbols])
    value = np.nan_to_num(value)
    nav = cash + float(value.sum())

    state["peak_nav"] = max(float(state.get("peak_nav") or nav), nav)
    drawdown = nav / state["peak_nav"] - 1.0
    if drawdown <= -MAX_DRAWDOWN_HALT and not state["halted"]:
        state["halted"] = True
        state["halt_reason"] = f"drawdown {drawdown:.1%} from peak on {bar}"
    target = np.array([0.0 if state["halted"] else state["target"].get(s, 0.0) for s in symbols])

    already_ran = state.get("last_bar") == bar
    tradable = np.array([market_open(by_symbol[s], now) and np.isfinite(prices.get(s, np.nan)) for s in symbols])
    orders = np.zeros(len(symbols)) if already_ran else plan_orders(target, value, cash, tradable, classes, costs, rules)
    order_map = {s: float(o) for s, o in zip(symbols, orders) if o != 0}
    if mode == "live":
        fills = broker.execute(order_map, prices, by_symbol, costs, eurusd=eurusd) if order_map else []
        broker.refresh()
        units, cash = broker.units(), broker.cash() + (broker.usd_cash() / eurusd if eurusd else 0.0)
    else:
        fills = broker.execute(order_map, prices, by_symbol, costs) if order_map else []
        units, cash = broker.units(), broker.cash()
    value = np.nan_to_num(np.array([units.get(s, 0.0) * prices.get(s, np.nan) for s in symbols]))
    nav = cash + float(value.sum())

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
    account = {"mode": mode, "armed": mode == "live" and os.environ.get("HERMES_LIVE_ARMED") == "yes",
               "profile": profile, "inception": state["inception"], "initial_capital": state["initial_capital"],
               "nav": nav, "cash": cash, "invested_pct": float(value.sum() / nav) if nav else 0.0,
               "return_since_inception": nav / state["initial_capital"] - 1.0, "peak_nav": state["peak_nav"],
               "drawdown": nav / state["peak_nav"] - 1.0, "halted": state["halted"],
               "target_date": state["target_date"], "target_refreshed_today": bool(refreshed),
               "next_target_refresh_in_days": max(0, rules.rebalance_every - (0 if refreshed else (age or 0))),
               "positions": positions, "equity": equity.to_dict(orient="records"),
               "trades": trades.tail(40).to_dict(orient="records") if len(trades) else [],
               "orders_today": [f.as_dict() for f in fills]}
    out = {"meta": {"generated_at": state["last_run"], "bar_date": bar, "version": __version__,
                    "data_errors": data_errors, "price_source": "kraken"},
           "account": account, "model": snapshot}
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "live.json").write_text(json.dumps(out, default=float))
    (live_dir / "latest.json").write_text(json.dumps({"meta": out["meta"], "account": {k: v for k, v in account.items() if k not in ("equity", "trades")},
                                                      "targets": state["target"]}, indent=2, default=float))
    notify.send(notify.daily_message(out))
    return out


def main() -> None:
    out = run(offline="--offline" in sys.argv)
    a = out["account"]
    print(f"[{out['meta']['bar_date']}] mode={a['mode']} profile={a['profile']} NAV {a['nav']:.2f} EUR "
          f"({a['return_since_inception']:+.2%} since {a['inception']}), invested {a['invested_pct']:.0%}, "
          f"{len(a['orders_today'])} order(s)")
    for f in a["orders_today"]:
        print(f"   {f['status']:9s} {f['symbol']:6s} {f['notional_eur']:+9.2f} EUR at {f['price_eur']:.6g}  cost {f['cost_eur']:.3f}")
    if out["meta"]["data_errors"]:
        print("   data errors:", out["meta"]["data_errors"])


if __name__ == "__main__":
    main()
