"""Step 4 — every design decision, taken on data up to 2021-12-31 only.

The last five years (2022 onwards) are locked away as a hold-out and are not
touched by anything in this file.

  a) structure      static risk parity vs pure trend vs strategic core + trend
  b) crypto sleeve  in isolation: is the trend filter worth its commissions?
  c) ablations      what each component adds, and how fragile the result is

Selection rule, fixed before running (a): highest net Sharpe among structures whose
maximum drawdown is no worse than -20%; ties go to the simpler one.

    python research/04_design_sample.py
"""

from __future__ import annotations

import json

import pandas as pd

from common import RESULTS, CostModel, ModelConfig, TradingRules, evaluate, load_panel, load_regime, summary
from hermes_quant.backtest.engine import static_mix

D0, D1 = "2007-01-01", "2021-12-31"
MAKER = CostModel(fee={"crypto": 0.0040, "equity": 0.0008, "real": 0.0008})
DOUBLE = CostModel(fee={"crypto": 0.0160, "equity": 0.0016, "real": 0.0016},
                   slippage={"crypto": 0.0010, "equity": 0.0020, "real": 0.0020}, fx_fee=0.0040)
FREE = CostModel(fee={}, slippage={}, fx_fee=0.0)

STRUCTURES = {
    "A static risk parity (no trend)": {"equity": 1.0, "real": 1.0, "crypto": 1.0},
    "B pure trend (floor 0)": {"equity": 0.0, "real": 0.0, "crypto": 0.0},
    "C core+trend (floor .5, crypto 0)": {"equity": 0.5, "real": 0.5, "crypto": 0.0},
}

# name -> (model changes, trading-rule changes, evaluate kwargs)
ABLATIONS = [
    ("C baseline (frozen candidate)", {}, {}, {}),
    ("no RMT cleaning", dict(rmt_clean=False), {}, {}),
    ("inverse-vol instead of HRP", dict(allocator="inverse_vol"), {}, {}),
    ("classic TSMOM instead of Kalman", dict(signal="tsmom"), {}, {}),
    ("close-to-close vol instead of range", dict(range_vol=False), {}, {}),
    ("regime overlay off", dict(regime=False), {}, {}),
    ("crypto top-2", dict(crypto_top_k=2), {}, {}),
    ("crypto top-6", dict(crypto_top_k=6), {}, {}),
    ("crypto top-8", dict(crypto_top_k=8), {}, {}),
    ("half-lives 20/40/80/160", dict(half_lives=(20.0, 40.0, 80.0, 160.0)), {}, {}),
    ("half-lives 60/120/240/480", dict(half_lives=(60.0, 120.0, 240.0, 480.0)), {}, {}),
    ("score_full 0.35", dict(score_full=0.35), {}, {}),
    ("score_full 0.70", dict(score_full=0.70), {}, {}),
    ("target vol 10%", dict(target_vol=0.10), {}, {}),
    ("target vol 20%", dict(target_vol=0.20), {}, {}),
    ("growth profile (50% crypto risk, 20% vol)",
     dict(target_vol=0.20, risk_share={"equity": 0.30, "real": 0.20, "crypto": 0.50},
          max_weight={"crypto": 0.25, "equity": 0.35, "real": 0.30},
          class_cap={"crypto": 0.50, "equity": 1.00, "real": 0.50}), {}, {}),
    ("execution lag 0", {}, dict(lag=0), {}),
    ("execution lag 2", {}, dict(lag=2), {}),
    ("decide daily", {}, dict(rebalance_every=1), {}),
    ("decide monthly", {}, dict(rebalance_every=30), {}),
    ("no partial rebalance (trade to target)", {}, dict(edge_fraction=0.0), {}),
    ("crypto at maker fee 0.40%", {}, {}, dict(costs=MAKER)),
    ("all costs doubled", {}, {}, dict(costs=DOUBLE)),
    ("zero costs", {}, {}, dict(costs=FREE)),
    ("capital 100k EUR", {}, {}, dict(capital=1e5)),
]


def main() -> None:
    panel = load_panel()
    cfg = ModelConfig()
    regime = load_regime(panel, cfg)
    report = {"window": [D0, D1], "structures": [], "crypto_isolated": [], "ablations": [], "benchmarks": []}
    variant_returns = {}          # genuine model variants only (not cost or timing sensitivities)

    print(f"(a) STRUCTURE   {D0} -> {D1}   236 EUR, Kraken costs, one-day execution delay")
    for name, floor in STRUCTURES.items():
        for on in (False, True):
            label = f"{name} | regime {'on' if on else 'off'}"
            res, row = evaluate("design: " + label, panel, cfg.with_(strategic_floor=floor, regime=on),
                                regime, start=D0, end=D1)
            report["structures"].append({**row, "name": label})
            variant_returns[label] = res.returns
    for name, weights in (("SPY buy & hold (EUR)", {"SPY": 1.0}), ("60% SPY / 40% GLD", {"SPY": 0.6, "GLD": 0.4})):
        s = summary(static_mix(panel, weights, D0, D1, rebalance_days=30))
        report["benchmarks"].append({"name": name, **{k: s[k] for k in ("cagr", "vol", "sharpe", "max_drawdown")}})
        print(f"{name:42s} CAGR {s['cagr']:7.2%} vol {s['vol']:6.2%} SR {s['sharpe']:5.2f} maxDD {s['max_drawdown']:7.2%}")

    c0 = "2016-09-01"
    iso = dict(risk_share={"crypto": 1.0}, target_vol=0.40, max_weight={"crypto": 1.0},
               class_cap={"crypto": 1.0}, regime=False)
    print(f"\n(b) CRYPTO SLEEVE ALONE   {c0} -> {D1}   target vol 40%, taker fee 0.80%")
    for name, extra, costs in (
        ("hold top-4, vol-targeted (no trend)", dict(strategic_floor={"crypto": 1.0}), None),
        ("trend top-4 (Kalman bank)", dict(strategic_floor={"crypto": 0.0}), None),
        ("trend top-1 (BTC only)", dict(strategic_floor={"crypto": 0.0}, crypto_top_k=1), None),
        ("trend top-2", dict(strategic_floor={"crypto": 0.0}, crypto_top_k=2), None),
        ("trend top-8", dict(strategic_floor={"crypto": 0.0}, crypto_top_k=8), None),
        ("trend top-4, classic TSMOM signal", dict(strategic_floor={"crypto": 0.0}, signal="tsmom"), None),
        ("trend top-4, maker fee 0.40%", dict(strategic_floor={"crypto": 0.0}), MAKER),
    ):
        _, row = evaluate("crypto-iso: " + name, panel, cfg.with_(**{**iso, **extra}), regime,
                          start=c0, end=D1, capital=1e4, costs=costs)
        report["crypto_isolated"].append({**row, "name": name})
    s = summary(static_mix(panel, {"BTC": 1.0}, c0, D1))
    report["crypto_isolated"].append({"name": "BTC buy & hold", **{k: s[k] for k in ("cagr", "vol", "sharpe", "max_drawdown")}})
    print(f"{'BTC buy & hold':54s} CAGR {s['cagr']:7.2%} vol {s['vol']:6.2%} SR {s['sharpe']:5.2f} maxDD {s['max_drawdown']:7.2%}")

    print(f"\n(c) ABLATIONS   {D0} -> {D1}")
    for name, model_changes, rule_changes, kwargs in ABLATIONS:
        rules = TradingRules(**rule_changes) if rule_changes else None
        res, row = evaluate("ablation: " + name, panel, cfg.with_(**model_changes), regime,
                            start=D0, end=D1, rules=rules, **kwargs)
        report["ablations"].append({**row, "name": name})
        if model_changes and "target_vol" not in model_changes:   # risk appetite is not a model variant
            variant_returns[name] = res.returns

    (RESULTS / "design_sample.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(variant_returns).to_parquet(RESULTS / "design_variant_returns.parquet")


if __name__ == "__main__":
    main()
