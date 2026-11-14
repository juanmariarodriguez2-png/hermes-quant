"""Step 5 — the exam. Opened ONCE, after every design decision was frozen.

(The script is deterministic: re-running it reproduces the same equity curve. What
"once" forbids is changing the model after seeing these numbers. The only edits made
after the first run were to reporting: a harsher Deflated Sharpe convention and a
third regression factor, both of which make the result look less impressive.)

The model is simulated as one continuous account from 2007 to the last available
day. Everything up to 2021-12-31 was used to design it (see 04_design_sample.py);
everything after is the hold-out, which no design choice has seen.

Two profiles were declared before this file was first run:
  balanced (default)  equal risk across equities / real assets / crypto, 15% vol target
  growth              50% of risk in crypto, 20% vol target (for higher risk appetite)

    python research/05_holdout.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import RESULTS, ModelConfig, evaluate, load_panel, load_regime, summary
from hermes_quant.backtest import stats
from hermes_quant.backtest.engine import static_mix
from hermes_quant.backtest.metrics import drawdown, yearly_returns
from hermes_quant.model import PROFILES

START = "2007-01-01"
SPLIT = "2021-12-31"            # last day of the design sample

CONTEXT = {  # shown for comparison only; nothing is selected from these
    "what the mechanical rule picked (crypto always held)":
        ModelConfig(strategic_floor={"equity": 1.0, "real": 1.0, "crypto": 1.0}),
    "trend filter on everything":
        ModelConfig(strategic_floor={"equity": 0.0, "real": 0.0, "crypto": 0.0}),
    "no regime overlay": ModelConfig(regime=False),
}
WIDER = "same model, if the four illiquid funds could be traded"
BENCHMARKS = {
    "Bitcoin buy & hold": {"BTC": 1.0},
    "S&P 500 buy & hold (EUR)": {"SPY": 1.0},
    "60% S&P 500 / 40% gold": {"SPY": 0.6, "GLD": 0.4},
}
STRESS = {
    "Global financial crisis": ("2008-09-01", "2009-03-09"),
    "Crypto winter 2018": ("2018-01-07", "2018-12-15"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 bear market (rates, Terra, FTX)": ("2021-11-10", "2022-11-21"),
    "April 2025 tariff shock": ("2025-02-19", "2025-04-08"),
    "Crypto slide 2025-26": ("2025-10-06", "2026-11-13"),
}


def window_stats(equity: pd.Series, lo: str | None, hi: str | None) -> dict:
    part = equity.loc[lo:hi]
    s = summary(part)
    return {k: s[k] for k in ("start", "end", "years", "total_return", "cagr", "vol", "sharpe",
                              "sharpe_excess", "sortino", "max_drawdown", "calmar", "worst_month",
                              "best_month", "positive_months")}


def weekly(equity: pd.Series) -> pd.Series:
    return equity.resample("W-FRI").last().pct_change().dropna()


def trial_stats() -> tuple[int, float]:
    """Attempts made while designing the model and the dispersion of their weekly
    Sharpe ratios. Read from the registry frozen by 04_design_sample.py, which was
    built on design data only: nothing from the hold-out can leak into the hurdle."""
    registry = json.loads((RESULTS / "design_trials.json").read_text())
    sharpes = np.array(list(registry["weekly_sharpe"].values()))
    return int(registry["n_trials"]), float(sharpes.std(ddof=1))


def check_frozen() -> None:
    """The model that sits the exam must be exactly the one the design rule chose."""
    chosen = json.loads((RESULTS / "selected.json").read_text())["config"]
    default = ModelConfig()
    for key, value in chosen.items():
        if getattr(default, key) != value:
            raise SystemExit(f"model default {key}={getattr(default, key)!r} differs from the selected "
                             f"configuration {value!r}; freeze the model before running the hold-out")


def main() -> None:
    check_frozen()
    panel = load_panel()
    regime = load_regime(panel, ModelConfig())
    last = str(panel.close.index[-1].date())
    report: dict = {"start": START, "split": SPLIT, "end": last, "profiles": {}, "context": {}, "benchmarks": {}}
    curves: dict[str, pd.Series] = {}
    results = {}

    for name, cfg in {**PROFILES, **CONTEXT}.items():
        res, row = evaluate(f"final: {name}", panel, cfg, regime, start=START, log=False)
        results[name] = res
        curves[name] = res.equity / res.equity.iloc[0]
        entry = {"design": window_stats(res.equity, START, SPLIT),
                 "holdout": window_stats(res.equity, SPLIT, None),
                 "full": window_stats(res.equity, START, None),
                 "since_2017": window_stats(res.equity, "2016-12-31", None),
                 "yearly": {int(k): float(v) for k, v in yearly_returns(res.equity).items()},
                 "turnover_per_year": row["turnover_per_year"], "cost_per_year_pct": row["cost_per_year_pct"],
                 "avg_gross": row["gross"]}
        (report["profiles"] if name in PROFILES else report["context"])[name] = entry

    # Context: the wider fund list (Europe, small caps, emerging markets, energy). Those
    # tokens exist on Kraken but do not trade, so this is NOT an investable result.
    wide_panel = load_panel(extended_funds=True)
    wide_res, wide_row = evaluate(f"final: {WIDER}", wide_panel, ModelConfig(),
                                  load_regime(wide_panel, ModelConfig()), start=START, log=False)
    curves[WIDER] = wide_res.equity / wide_res.equity.iloc[0]
    report["context"][WIDER] = {
        "design": window_stats(wide_res.equity, START, SPLIT), "holdout": window_stats(wide_res.equity, SPLIT, None),
        "full": window_stats(wide_res.equity, START, None), "since_2017": window_stats(wide_res.equity, "2016-12-31", None),
        "yearly": {int(k): float(v) for k, v in yearly_returns(wide_res.equity).items()},
        "turnover_per_year": wide_row["turnover_per_year"], "cost_per_year_pct": wide_row["cost_per_year_pct"],
        "avg_gross": wide_row["gross"]}

    for name, weights in BENCHMARKS.items():
        first = max(pd.Timestamp(START), panel.close[list(weights)].dropna().index[0])
        eq = static_mix(panel, weights, str(first.date()), rebalance_days=30)
        curves[name] = eq
        report["benchmarks"][name] = {
            "holdout": window_stats(eq, SPLIT, None),
            "since_2017": window_stats(eq, "2016-12-31", None) if first <= pd.Timestamp("2016-12-31") else None,
            "full": window_stats(eq, None, None),
            "yearly": {int(k): float(v) for k, v in yearly_returns(eq).items()}}

    # ---- inference on the default profile ------------------------------------------
    eq = results["balanced"].equity
    hold_daily = eq.loc[SPLIT:].pct_change().dropna().to_numpy()
    hold_weekly = weekly(eq.loc[SPLIT:]).to_numpy()
    design_weekly = weekly(eq.loc[START:SPLIT]).to_numpy()
    n_trials, sr_dispersion = trial_stats()
    report["inference"] = {
        # primary: block bootstrap, which respects volatility clustering and serial dependence
        "holdout_bootstrap": stats.bootstrap_ci(hold_daily, 365, n_boot=5000, mean_block=20.0),
        "holdout_bootstrap_block_sensitivity": {
            str(int(block)): stats.bootstrap_ci(hold_daily, 365, n_boot=2000, mean_block=block)["sharpe"]
            for block in (5.0, 10.0, 40.0, 80.0)},
        # secondary: closed-form, with a conservative autocorrelation correction
        "holdout_psr_vs_zero": stats.probabilistic_sharpe(hold_weekly, 0.0),
        # headline: every attempt treated as an independent no-edge strategy (harshest)
        "design_deflated_sharpe": stats.deflated_sharpe(design_weekly, n_trials),
        # original recipe: dispersion observed across the (closely related) attempts
        "design_deflated_sharpe_observed_dispersion": stats.deflated_sharpe(design_weekly, n_trials, sr_dispersion),
        "full_bootstrap": stats.bootstrap_ci(eq.pct_change().dropna().to_numpy(), 365, n_boot=5000, mean_block=20.0),
    }
    registry = json.loads((RESULTS / "design_trials.json").read_text())
    variants = pd.read_parquet(RESULTS / "design_variant_returns.parquet")[registry["model_variants"]]
    # Diagnostic only: how fragile would "pick the best Sharpe" be on these candidates?
    report["inference"]["design_pbo_diagnostic"] = stats.probability_of_backtest_overfitting(variants, n_blocks=10)

    # ---- what drives the returns: regression on the two obvious benchmarks ---------
    gold = static_mix(panel, {"GLD": 1.0}, START)
    factors = pd.DataFrame({"sp500": weekly(curves["S&P 500 buy & hold (EUR)"]),
                            "bitcoin": weekly(curves["Bitcoin buy & hold"]), "gold": weekly(gold)})
    two = factors[["sp500", "bitcoin"]]
    report["market_model"] = {
        # three factors: is there anything left once the three asset classes are accounted for?
        "holdout": stats.market_model(weekly(eq.loc[SPLIT:]), factors),
        "since_2017": stats.market_model(weekly(eq.loc["2016-12-31":]), factors),
        # two factors (no gold): the "alpha" here is largely the metals' own return
        "holdout_without_gold": stats.market_model(weekly(eq.loc[SPLIT:]), two),
    }

    # ---- stress episodes -----------------------------------------------------------
    report["stress"] = {}
    for label, (lo, hi) in STRESS.items():
        row = {}
        for name in ("balanced", "growth", *BENCHMARKS):
            part = curves[name].loc[lo:hi].dropna()
            if len(part) > 5:
                row[name] = {"return": float(part.iloc[-1] / part.iloc[0] - 1.0),
                             "max_drawdown": float(drawdown(part).min())}
        report["stress"][label] = {"window": [lo, hi], **row}

    # ---- artefacts for the dashboard ----------------------------------------------
    pd.DataFrame(curves).to_csv(RESULTS / "equity_curves.csv", float_format="%.6f")
    w = results["balanced"].weights
    by_class = w.T.groupby(pd.Series(panel.classes)).sum().T
    by_class["cash"] = 1.0 - by_class.sum(axis=1)
    by_class.resample("W-FRI").last().to_csv(RESULTS / "weights_by_sleeve.csv", float_format="%.4f")
    results["balanced"].trades.to_csv(RESULTS / "backtest_trades.csv", index=False, float_format="%.6f")
    (RESULTS / "holdout.json").write_text(json.dumps(report, indent=2, default=float))

    # ---- console summary -----------------------------------------------------------
    def line(name, s):
        return (f"{name[:52]:52s} CAGR {s['cagr']:7.2%}  vol {s['vol']:6.2%}  Sharpe {s['sharpe']:5.2f} "
                f"(excess {s['sharpe_excess']:5.2f})  maxDD {s['max_drawdown']:7.2%}  total {s['total_return']:8.1%}")

    for label, key in (("DESIGN SAMPLE (2007-2021)", "design"), (f"HOLD-OUT ({SPLIT} -> {last})", "holdout"),
                       ("FULL HISTORY", "full")):
        print(f"\n{label}")
        for name in (*PROFILES, *CONTEXT, WIDER):
            group = report["profiles"] if name in PROFILES else report["context"]
            print("  " + line(name, group[name][key]))
        if key != "design":
            for name in BENCHMARKS:
                print("  " + line(name, report["benchmarks"][name][key]))
    inf = report["inference"]
    b = inf["holdout_bootstrap"]
    print(f"\nHold-out Sharpe, 95% bootstrap interval: [{b['sharpe']['p2.5']:.2f}, {b['sharpe']['p97.5']:.2f}]   "
          f"resamples with Sharpe <= 0: {b['share_resamples_sharpe_below_zero']:.1%}   "
          f"PSR (autocorrelation-adjusted) = {inf['holdout_psr_vs_zero']:.1%}")
    d = inf["design_deflated_sharpe"]
    print(f"Design-sample Deflated Sharpe: {d['dsr']:.1%} after {d['n_trials']} attempts treated as independent "
          f"({inf['design_deflated_sharpe_observed_dispersion']['dsr']:.1%} with observed dispersion)   "
          f"PBO diagnostic = {inf['design_pbo_diagnostic']['pbo']:.1%}")
    print("Market model, 3 factors (hold-out):", {k: round(v, 3) for k, v in report["market_model"]["holdout"].items()})
    print("Market model, no gold (hold-out):", {k: round(v, 3) for k, v in report["market_model"]["holdout_without_gold"].items()})
    print("\nYearly, balanced:", {k: f"{v:+.1%}" for k, v in report["profiles"]["balanced"]["yearly"].items()})


if __name__ == "__main__":
    main()
