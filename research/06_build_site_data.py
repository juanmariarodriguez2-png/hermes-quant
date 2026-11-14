"""Step 6 — pack the research results into one file for the public dashboard.

    python research/06_build_site_data.py      ->  site/research.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import RESULTS, ROOT
from hermes_quant.filters import volatility
from hermes_quant.live.market import PriceStore, build_live_panel
from hermes_quant.universe import LIVE_INSTRUMENTS

SITE = ROOT / "site"
CURVES = {
    "balanced": "hermes", "growth": "growth", "Bitcoin buy & hold": "bitcoin",
    "S&P 500 buy & hold (EUR)": "sp500", "60% S&P 500 / 40% gold": "mix6040",
}

# What the previous, LLM-driven bot actually did, reconstructed from its own e-mail log.
LEGACY = {
    "name": "Hermes Trader v1",
    "period": ["2026-10-29", "2026-11-10"],
    "capital": 235.98,
    "final_value": 233.47,
    "return": -0.0106,
    "api_cost_eur": 2.40,
    "net_after_api_eur": -4.91,
    "trades": [
        {"asset": "ETH", "entry": "2026-11-04", "exit": "2026-11-05", "size_eur": 35.01,
         "stop_distance": 0.0285, "result_eur": -1.56, "held_hours": 23},
        {"asset": "BTC", "entry": "2026-11-05", "exit": "2026-11-05", "size_eur": 29.99,
         "stop_distance": 0.0157, "result_eur": -0.95, "held_hours": 0.3},
    ],
    "fee_per_side": 0.0080,
    "api_cost_per_day": 0.185,
}


def sample(series: pd.Series, rule: str = "W-FRI") -> list:
    s = series.dropna().resample(rule).last().dropna()
    return [[d.strftime("%Y-%m-%d"), round(float(v), 5)] for d, v in s.items()]


def stop_noise(legacy: dict) -> None:
    """How likely was each of the old bot's stops to be hit by noise alone?

    For a driftless random walk the chance of touching a level d away within time T
    is 2 * Phi(-d / (sigma * sqrt(T))) (reflection principle). sigma is this model's
    own volatility forecast for that asset on the day before the trade."""
    panel = build_live_panel(PriceStore(ROOT / "data" / "live" / "prices"), LIVE_INSTRUMENTS)
    proxy = volatility.daily_variance_proxy(panel.open, panel.high, panel.low, panel.close)
    vol = volatility.ewma_vol(proxy)
    for trade in legacy["trades"]:
        day = pd.Timestamp(trade["entry"]) - pd.Timedelta(days=1)
        daily = float(vol[trade["asset"]].loc[:day].iloc[-1]) / np.sqrt(365.0)
        trade["daily_vol"] = daily
        trade["noise_hit_prob"] = float(2.0 * norm.cdf(-trade["stop_distance"] / daily))


def main() -> None:
    stop_noise(LEGACY)
    hold = json.loads((RESULTS / "holdout.json").read_text())
    design = json.loads((RESULTS / "design_sample.json").read_text())
    curves = pd.read_csv(RESULTS / "equity_curves.csv", index_col=0, parse_dates=True)
    weights = pd.read_csv(RESULTS / "weights_by_sleeve.csv", index_col=0, parse_dates=True)

    since = "2016-12-30"                       # common start for the comparison chart
    chart = {}
    for column, key in CURVES.items():
        s = curves[column].loc[since:].dropna()
        chart[key] = sample(s / s.iloc[0])
    drawdowns = {}
    for column, key in CURVES.items():
        s = curves[column].loc[since:].dropna()
        drawdowns[key] = sample(s / s.cummax() - 1.0)
    long_run = sample(curves["balanced"].dropna() / curves["balanced"].dropna().iloc[0], "ME")

    out = {
        "generated_from": {"start": hold["start"], "split": hold["split"], "end": hold["end"]},
        "profiles": hold["profiles"], "context": hold["context"], "benchmarks": hold["benchmarks"],
        "inference": hold["inference"], "market_model": hold["market_model"], "stress": hold["stress"],
        "equity_since_2017": chart, "drawdown_since_2017": drawdowns, "hermes_full_history": long_run,
        "weights_by_sleeve": [[d.strftime("%Y-%m-%d")] + [round(float(v), 4) for v in row]
                              for d, row in zip(weights.index, weights.to_numpy())][::2],
        "weights_columns": list(weights.columns),
        "design": {"structures": design["structures"], "ablations": design["ablations"],
                   "crypto_isolated": design["crypto_isolated"], "benchmarks": design["benchmarks"],
                   "window": design["window"]},
        "regime_test": json.loads((RESULTS / "regime_test.json").read_text()),
        "volatility_forecast": json.loads((RESULTS / "volatility_forecast.json").read_text()),
        "trials": json.loads((RESULTS / "design_trials.json").read_text())["n_trials"],
        "selected": design.get("selected"),
        "n_tests": sum(f.read_text().count("\ndef test_") for f in (ROOT / "tests").glob("test_*.py")),
        "legacy": LEGACY,
    }
    SITE.mkdir(exist_ok=True)
    text = json.dumps(out, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else float(x))
    (SITE / "research.json").write_text(text.replace("NaN", "null"))
    print(f"site/research.json  {len(text) // 1024} KB")


if __name__ == "__main__":
    main()
