"""Step 7 — derivatives: exploratory research, NOT part of the validated model.

Kraken lists perpetual futures on crypto, equity indices, gold and FX, with fees of
0.02-0.05% against 0.80% for spot crypto. Two questions worth answering before
touching them:

  A. Would the crypto sleeve be cheaper to run with perpetuals than with spot?
     (And does being able to go short add anything?)
  B. Is the classic delta-neutral "cash and carry" (own the coin, short its
     perpetual, collect the funding) still worth doing?

Funding history is Binance's (2020 onwards), used as a proxy for Kraken's. With only
two years before 2022 there is no honest way to split design from hold-out here, so
every parameter is fixed a priori and the whole period is reported as one sample.
Treat the numbers as indicative.

    python research/07_derivatives.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import RESULTS, CostModel, ModelConfig, TradingRules, load_panel, load_regime
from hermes_quant.backtest.engine import run_backtest, static_mix
from hermes_quant.backtest.metrics import summary
from hermes_quant.model import compute_features

START = "2020-03-01"
SLEEVE = dict(risk_share={"crypto": 1.0}, target_vol=0.40, max_weight={"crypto": 1.0},
              class_cap={"crypto": 1.0}, regime=False)
PERP_COSTS = CostModel(fee={"crypto": 0.0005}, slippage={"crypto": 0.0005})      # taker + slippage
SPOT_FEE, PERP_FEE = 0.0085, 0.0010                                              # per side, incl. slippage


def funding_panel(panel) -> pd.DataFrame:
    """Daily funding per coin; coins without their own history get that day's average."""
    crypto = [s for s in panel.symbols if panel.classes[s] == "crypto"]
    known = panel.funding.reindex(columns=crypto)
    average = panel.funding.mean(axis=1)
    return known.apply(lambda col: col.fillna(average)).fillna(0.0)


def part_a(panel, regime) -> list[dict]:
    carry = funding_panel(panel)
    rows = []
    variants = [
        ("spot, long only (0.80% taker)", dict(), None, None, None),
        ("perpetuals, long only (0.05% taker + funding)", dict(), PERP_COSTS, TradingRules(allow_leverage=True), carry),
        ("perpetuals, long and short (+ funding)", dict(long_only=False), PERP_COSTS, TradingRules(allow_leverage=True), carry),
    ]
    print(f"A. CRYPTO SLEEVE, {START} onwards (trend filter, top 4, 40% vol target, 10k EUR)")
    for name, extra, costs, rules, c in variants:
        cfg = ModelConfig().with_(**{**SLEEVE, **extra})
        feats = compute_features(panel, cfg, p_turbulent=regime)
        res = run_backtest(panel, feats, cfg, costs=costs, rules=rules, capital=1e4, start=START, carry=c)
        s = summary(res.equity)
        years = s["years"]
        row = {"name": name, "cagr": s["cagr"], "vol": s["vol"], "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"],
               "fees_per_year": float(res.costs.sum() / res.equity.mean() / years),
               "funding_per_year": float(res.carry.sum() / res.equity.mean() / years),
               "turnover_per_year": float(res.turnover.sum() / years)}
        rows.append(row)
        print(f"  {name:48s} CAGR {row['cagr']:7.2%}  Sharpe {row['sharpe']:5.2f}  maxDD {row['max_drawdown']:7.2%}  "
              f"fees {row['fees_per_year']:6.2%}/yr  funding {row['funding_per_year']:+7.2%}/yr")
    s = summary(static_mix(panel, {"BTC": 1.0}, START))
    rows.append({"name": "Bitcoin buy & hold", **{k: s[k] for k in ("cagr", "vol", "sharpe", "max_drawdown")}})
    print(f"  {'Bitcoin buy & hold':48s} CAGR {s['cagr']:7.2%}  Sharpe {s['sharpe']:5.2f}  maxDD {s['max_drawdown']:7.2%}")
    return rows


def part_b(panel) -> dict:
    """Delta-neutral carry: enter when the 30-day average funding exceeds 6% a year,
    leave when it turns negative. Capital = the coin plus 25% margin for the short."""
    out = {}
    print("\nB. CASH AND CARRY (long the coin, short its perpetual)")
    for coin in ("BTC", "ETH"):
        f = panel.funding[coin].dropna()
        trailing = f.rolling(30).mean() * 365
        active = pd.Series(False, index=f.index)
        on = False
        for day, level in trailing.items():
            if np.isnan(level):
                continue
            on = (level > 0.06) if not on else (level > 0.0)
            active[day] = on
        held = active.shift(1, fill_value=False)                 # decided yesterday, earned today
        switches = held.astype(int).diff().abs().fillna(0.0)
        pnl = held * f - switches * (SPOT_FEE + PERP_FEE)        # per unit of notional
        daily = pnl / 1.25                                       # per unit of capital
        years = len(daily) / 365
        equity = (1.0 + daily).cumprod()
        by_year = (1.0 + daily).groupby(daily.index.year).prod() - 1.0
        out[coin] = {
            "window": [str(f.index[0].date()), str(f.index[-1].date())],
            "funding_mean_annual": float(f.mean() * 365), "share_of_time_active": float(held.mean()),
            "round_trips": float(switches.sum() / 2), "net_return_annual": float(equity.iloc[-1] ** (1 / years) - 1.0),
            "gross_return_annual_on_capital": float((held * f).sum() / 1.25 / years),
            "vol": float(daily.std() * np.sqrt(365)), "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "by_year": {int(k): float(v) for k, v in by_year.items()},
            "last_30d_funding_annual": float(trailing.iloc[-1]),
        }
        o = out[coin]
        print(f"  {coin}: funding averaged {o['funding_mean_annual']:.1%}/yr; rule active {o['share_of_time_active']:.0%} of the time, "
              f"{o['round_trips']:.0f} round trips; net {o['net_return_annual']:.2%}/yr on capital, worst drawdown {o['max_drawdown']:.2%}")
        print("       by year:", {k: f"{v:+.1%}" for k, v in o["by_year"].items()}, f"| latest 30-day funding {o['last_30d_funding_annual']:+.1%}/yr")
    return out


def main() -> None:
    panel = load_panel()
    regime = load_regime(panel, ModelConfig())
    report = {"start": START, "sleeve": part_a(panel, regime), "carry": part_b(panel)}
    (RESULTS / "derivatives.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
