"""Step 4 — every design decision, taken on data up to 2021-12-31 only.

The last five years (2022 onwards) are locked away as a hold-out and are not
touched by anything in this file.

  a) structure      which sleeves use the trend filter, and which get the regime overlay
  b) crypto sleeve  in isolation: is the trend filter worth its commissions?
  c) ablations      what each component adds, and how fragile the result is

How the structure was chosen (and what was NOT done)
----------------------------------------------------
A rule was written down first: "highest net Sharpe among structures whose maximum
drawdown is no worse than -20%". On this sample that rule picks static risk parity
with the equity regime overlay (structure A). It was not adopted, and the reason is
recorded here rather than hidden: four structures finish within 0.03 Sharpe of each
other, while the standard error of a 15-year Sharpe ratio is about 0.26. The rule was
choosing on noise. So the structure was instead assembled from one test per component:

  * trend filter on the funds      pure trend (B) loses to always-held (A, D) net of
                                   costs                                   -> OFF
  * trend filter on crypto         halves the sleeve's worst drawdown at a higher
                                   Sharpe (part b); and an asset with no cash flows
                                   and -80% bear markets gets no permanent seat
                                   (a prior written into the model before any test) -> ON
  * regime overlay on equities     passes the real-time test (03) and cuts drawdown
                                   in every structure at no Sharpe cost     -> ON
  * regime overlay on crypto       lowers Sharpe in every structure         -> OFF

That is structure D with the equity overlay. It is fixed here, before the hold-out is
opened, and 05_holdout.py refuses to run unless the model's defaults match it. The
rule's own pick (A) is carried into the hold-out report as context, so the reader can
see what the alternative would have done. Every structure tried is counted in the
trial registry that the Deflated Sharpe Ratio uses.

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
    "B pure trend everywhere": {"equity": 0.0, "real": 0.0, "crypto": 0.0},
    "C half core + half trend, crypto trend-only": {"equity": 0.5, "real": 0.5, "crypto": 0.0},
    "D funds always held, crypto trend-only": {"equity": 1.0, "real": 1.0, "crypto": 0.0},
}
# Sleeves whose regime overlay passed the real-time test in 03_regime_test.py.
REGIMES = {
    "regime off": dict(regime=False),
    "regime on equities": dict(regime=True, regime_proxy={"equity": "SPY"}),
    "regime on equities+crypto": dict(regime=True, regime_proxy={"equity": "SPY", "crypto": "BTC"}),
}
MAX_DRAWDOWN_ALLOWED = -0.20
CHOSEN = "D funds always held, crypto trend-only | regime on equities"

# name -> (model changes, trading-rule changes, evaluate kwargs)
ABLATIONS = [
    ("C baseline (frozen candidate)", {}, {}, {}),          # label kept for the dashboard: "full model"
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
    all_returns = {}              # every run on the common design window (the trial registry)
    model_variants = []           # the subset that are genuinely different models

    print(f"(a) STRUCTURE   {D0} -> {D1}   236 EUR, Kraken costs, one-day execution delay")
    passed = json.loads((RESULTS / "regime_test.json").read_text())
    candidates = {}
    for name, floor in STRUCTURES.items():
        for regime_name, regime_cfg in REGIMES.items():
            if "crypto" in regime_name and not passed["crypto"]["overlay_kept"]:
                continue
            label = f"{name} | {regime_name}"
            changes = dict(strategic_floor=floor, **regime_cfg)
            res, row = evaluate("design: " + label, panel, cfg.with_(**changes), regime, start=D0, end=D1)
            report["structures"].append({**row, "name": label})
            all_returns[label] = res.returns
            model_variants.append(label)
            candidates[label] = (row, changes)
    eligible = {k: v for k, v in candidates.items() if v[0]["max_drawdown"] >= MAX_DRAWDOWN_ALLOWED}
    rule_pick = max(eligible, key=lambda k: eligible[k][0]["sharpe"])
    best = max(v[0]["sharpe"] for v in candidates.values())
    near_best = sorted(k for k, v in candidates.items() if v[0]["sharpe"] >= best - 0.05)
    selected = dict(candidates[CHOSEN][1])
    selected.setdefault("regime_proxy", cfg.regime_proxy)
    report["selected"] = {"name": CHOSEN, "config": selected,
                          "mechanical_rule_pick": rule_pick, "mechanical_rule_config": candidates[rule_pick][1],
                          "within_0.05_sharpe_of_best": near_best,
                          "chosen_metrics": candidates[CHOSEN][0], "rule_pick_metrics": candidates[rule_pick][0]}
    (RESULTS / "selected.json").write_text(json.dumps(report["selected"], indent=2))
    print(f"\n-> mechanical rule would pick: {rule_pick}")
    print(f"-> {len(near_best)} structures within 0.05 Sharpe of the best: the rule is choosing on noise")
    print(f"-> CHOSEN component by component: {CHOSEN}\n")
    cfg = cfg.with_(**selected)                    # everything below is relative to the chosen model
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
        if name.startswith("C baseline"):
            continue                                  # identical to the selected structure above
        all_returns[name] = res.returns
        if model_changes and "target_vol" not in model_changes:   # risk appetite is not a model variant
            model_variants.append(name)

    # ---- trial registry: frozen here, on design data only, before the hold-out exists
    returns = pd.DataFrame(all_returns)
    weekly = (1.0 + returns).resample("W-FRI").prod() - 1.0
    weekly_sharpe = (weekly.mean() / weekly.std(ddof=1)).to_dict()
    registry = {
        "window": [D0, D1],
        "weekly_sharpe": {k: float(v) for k, v in weekly_sharpe.items()},
        "model_variants": model_variants,
        # Attempts that are not in the matrix above still count. 4 early prototypes
        # (two allocator designs, with and without the regime overlay); a first pass of
        # 6 structures and 24 ablations on a wider fund list, discarded when four of
        # those funds turned out not to trade on Kraken; and the crypto-sleeve runs.
        "other_attempts": 4 + 6 + 24 + len(report["crypto_isolated"]) - 1,
    }
    registry["n_trials"] = len(weekly_sharpe) + registry["other_attempts"]
    (RESULTS / "design_trials.json").write_text(json.dumps(registry, indent=2))
    (RESULTS / "design_sample.json").write_text(json.dumps(report, indent=2))
    returns.to_parquet(RESULTS / "design_variant_returns.parquet")
    print(f"\ntrial registry frozen: {registry['n_trials']} attempts "
          f"({len(weekly_sharpe)} on the common window, {len(model_variants)} distinct models)")


if __name__ == "__main__":
    main()
