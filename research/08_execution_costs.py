"""What the venue really charges, and what that does to the result.

The backtest prices tokenized funds as Kraken's order book would: 0.08% fee, 0.10%
slippage and 0.20% to convert euros to dollars. An account in the European Economic
Area cannot use that order book for them, nor the API: it has to buy by hand with
Kraken's "Convert", which shows no fee and puts its margin in the price.

On 16 Nov 2026 a real quote was compared with the market at the same moment: 57 EUR
bought 0.1754081 GLDx, i.e. 324.96 EUR a unit against a mid of 321.51 EUR. That is
1.07% all in, per side.

This script changes no parameter and selects nothing. It re-costs the frozen model
with that figure (and a more pessimistic one) and writes the difference down.

    python research/08_execution_costs.py
"""

from __future__ import annotations

import json

from common import RESULTS, evaluate, load_panel, load_regime
from hermes_quant.backtest.metrics import summary
from hermes_quant.model import ModelConfig
from hermes_quant.portfolio.execution import CostModel

START, SPLIT = "2007-01-01", "2021-12-31"      # the same windows as 05_holdout.py
OBSERVED = {"date": "2026-11-16", "asset": "GLDx", "paid_eur": 57.0, "units": 0.1754081,
            "market_mid_eur": 321.51}
OBSERVED["implied_eur"] = round(OBSERVED["paid_eur"] / OBSERVED["units"], 2)
OBSERVED["all_in_cost"] = round(OBSERVED["implied_eur"] / OBSERVED["market_mid_eur"] - 1.0, 4)
KEYS = ("cagr", "vol", "sharpe", "sharpe_excess", "max_drawdown")


def convert(cost: float) -> CostModel:
    """Funds bought with Convert: one all-in margin, currency conversion included."""
    return CostModel(fee={"crypto": 0.0080, "equity": cost, "real": cost},
                     slippage={"crypto": 0.0005, "equity": 0.0, "real": 0.0}, fx_fee=0.0)


def main() -> None:
    panel = load_panel()
    cfg = ModelConfig()
    regime = load_regime(panel, cfg)
    book = CostModel()
    cases = {"order_book": (book.rate("equity") + book.fx_fee, book),
             "convert": (OBSERVED["all_in_cost"], convert(OBSERVED["all_in_cost"])),
             "convert_pessimistic": (0.015, convert(0.015))}
    out = {"observed": OBSERVED, "cases": {}}
    for name, (per_side, costs) in cases.items():
        res, row = evaluate(name, panel, cfg, regime, start=START, costs=costs, log=False)
        eq = res.equity
        stats = lambda part: {k: float(summary(part)[k]) for k in KEYS}      # noqa: E731
        out["cases"][name] = {"fund_cost_per_side": per_side, "cost_per_year": float(row["cost_per_year_pct"]),
                              "turnover_per_year": float(row["turnover_per_year"]),
                              "design": stats(eq.loc[START:SPLIT]), "holdout": stats(eq.loc[SPLIT:])}
        h = out["cases"][name]["holdout"]
        print(f"{name:20s} funds {per_side:.2%} a side | hold-out {h['cagr']:.2%} a year, Sharpe {h['sharpe']:.2f}, "
              f"worst fall {h['max_drawdown']:.1%} | costs {row['cost_per_year_pct']:.2%} a year")
    (RESULTS / "execution_costs.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
