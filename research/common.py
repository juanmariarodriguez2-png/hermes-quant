"""Shared helpers for the research scripts: load the panel, cache slow features."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes_quant.data.panel import Panel, build_panel  # noqa: E402
from hermes_quant.model import ModelConfig, regime_probabilities  # noqa: E402

RESEARCH = ROOT / "data" / "research"
RESULTS = ROOT / "research" / "results"
FIGURES = ROOT / "docs" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

PANEL_START = "2005-01-03"
# First day the strategy is allowed to trade in the headline test. BTC (2015-07) is
# the only coin with enough history at that point; the others join as they qualify.
TEST_START = "2017-01-01"


def load_panel() -> Panel:
    return build_panel(RESEARCH, start=PANEL_START)


ALL_PROXIES = {"equity": "SPY", "crypto": "BTC"}


def load_regime(panel: Panel, cfg: ModelConfig, refresh: bool = False) -> pd.DataFrame:
    """Walk-forward HMM probabilities are expensive: compute once, reuse everywhere.

    Both proxies are always computed; the model then uses only the sleeves named in
    its own ``regime_proxy``. The probabilities depend on the proxies' prices and
    the HMM settings alone, never on the allocation parameters being compared, so
    sharing them across variants is not a leak."""
    path = RESULTS / f"regime_{cfg.hmm_min_obs}_{cfg.hmm_refit_every}_{cfg.hmm_max_window}.parquet"
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if (len(cached) == len(panel.close) and cached.index[-1] == panel.close.index[-1]
                and set(ALL_PROXIES) <= set(cached.columns)):
            return cached
    probs = regime_probabilities(panel, cfg.with_(regime_proxy=ALL_PROXIES))
    probs.to_parquet(path)
    return probs


# --------------------------------------------------------------------------- trial log

import json  # noqa: E402
import time  # noqa: E402

from hermes_quant.backtest.engine import CostModel, TradingRules, run_backtest  # noqa: E402
from hermes_quant.backtest.metrics import summary  # noqa: E402
from hermes_quant.model import compute_features  # noqa: E402

TRIALS = RESULTS / "trials.jsonl"


def evaluate(name: str, panel: Panel, cfg: ModelConfig, regime: pd.DataFrame, *,
             costs: CostModel | None = None, rules: TradingRules | None = None,
             capital: float = 236.0, start: str = TEST_START, end: str | None = None,
             log: bool = True, verbose: bool = True):
    """Run one variant and append it to the trial log.

    Every configuration that is ever looked at gets recorded. The count of distinct
    trials is what the Deflated Sharpe Ratio later charges us for."""
    feats = compute_features(panel, cfg, p_turbulent=regime)
    res = run_backtest(panel, feats, cfg, costs=costs, rules=rules, capital=capital, start=start, end=end)
    s = summary(res.equity)
    years = s["years"]
    row = {"name": name, "cagr": s["cagr"], "vol": s["vol"], "sharpe": s["sharpe"],
           "max_drawdown": s["max_drawdown"], "gross": float(res.weights.abs().sum(axis=1).mean()),
           "turnover_per_year": float(res.turnover.sum() / years),
           "cost_per_year_pct": float(res.costs.sum() / res.equity.mean() / years),
           "trades": int(len(res.trades)), "start": s["start"], "end": s["end"], "capital": capital}
    if log:
        with TRIALS.open("a") as fh:
            fh.write(json.dumps({**row, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    if verbose:
        print(f"{name:42s} CAGR {row['cagr']:7.2%} vol {row['vol']:6.2%} SR {row['sharpe']:5.2f} "
              f"maxDD {row['max_drawdown']:7.2%} gross {row['gross']:.2f} "
              f"turn/yr {row['turnover_per_year']:5.2f} cost/yr {row['cost_per_year_pct']:6.2%} trades {row['trades']}")
    return res, row
