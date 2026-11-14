"""Step 3 — does the regime model know anything *in real time*?

A hidden Markov model fitted on the full sample always looks clever: its turbulent
state has terrible average returns. That is hindsight. The honest question is
whether the walk-forward (out-of-sample) probability, as it stood each evening,
says anything about returns the strategy could still capture.

Timing matches production exactly: the probability is smoothed as in the model, a
signal formed at the close of session s is acted on at the close of session s+1,
and so the first return it can earn is the one from s+1 to s+2.

Rule fixed before looking at the numbers: a sleeve gets the regime overlay only if
that capturable Sharpe ratio in the turbulent bucket (P > 0.8) is less than half of
the calm bucket's (P < 0.2). Design sample only; data is cut at the boundary first.

    python research/03_regime_test.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import ALL_PROXIES, RESULTS, load_panel, load_regime
from hermes_quant.model import ModelConfig

DESIGN_END = "2021-12-31"
BUCKETS = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]


def main() -> None:
    panel = load_panel()
    cfg = ModelConfig()
    probs = load_regime(panel, cfg).loc[:DESIGN_END]
    probs = probs.ewm(halflife=cfg.regime_smooth_halflife, ignore_na=True).mean().where(probs.notna())
    usd = panel.close_usd.loc[:DESIGN_END]
    tradable = panel.tradable.loc[:DESIGN_END]
    out = {}
    for sleeve, proxy in ALL_PROXIES.items():
        per_year = 252 if sleeve == "equity" else 365
        sessions = usd[proxy].where(tradable[proxy]).dropna()
        r = np.log(sessions / sessions.shift(1))
        capturable = r.shift(-2)                              # return from session s+1 to s+2
        d = pd.DataFrame({"p": probs[sleeve].reindex(sessions.index), "next": capturable}).dropna()
        rows = []
        print(f"\n{sleeve} ({proxy}), {d.index[0].date()} .. {d.index[-1].date()}, n = {len(d)}")
        for lo, hi in BUCKETS:
            g = d[(d.p >= lo) & (d.p < hi)]["next"]
            sr = float(g.mean() / g.std() * np.sqrt(per_year))
            rows.append({"bucket": f"[{lo:.1f}, {min(hi, 1.0):.1f})", "n": int(len(g)),
                         "mean_daily_pct": float(g.mean() * 100), "vol_daily_pct": float(g.std() * 100),
                         "annual_sharpe": sr})
            print(f"  P(turbulent) in [{lo:.1f}, {min(hi, 1.0):.1f}): n = {len(g):5d}  "
                  f"mean {g.mean() * 100:7.3f}%/day  vol {g.std() * 100:5.2f}%/day  Sharpe {sr:5.2f}")
        calm, turbulent = rows[0]["annual_sharpe"], rows[-1]["annual_sharpe"]
        keep = bool(turbulent < 0.5 * calm)
        print(f"  -> overlay {'KEPT' if keep else 'DROPPED'} for this sleeve "
              f"(turbulent Sharpe {turbulent:.2f} vs calm {calm:.2f})")
        out[sleeve] = {"proxy": proxy, "buckets": rows, "overlay_kept": keep}
    (RESULTS / "regime_test.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
