"""Step 2 — choose the volatility estimator by forecast accuracy, not by backtest profit.

Target: realised variance over the following 7 days. Loss: QLIKE, the standard
loss for variance forecasts (robust to noise in the realised proxy; Patton 2011).
Only data up to the end of the design sample is used.

    python research/02_volatility_forecast.py
"""

from __future__ import annotations

import json

import numpy as np

from common import RESULTS, load_panel
from hermes_quant.filters import volatility as V

DESIGN_END = "2021-12-31"
CRYPTO = ["BTC", "ETH", "XRP", "LTC", "BNB", "ADA"]
FUNDS = ["SPY", "QQQ", "IWM", "VGK", "GLD", "SLV", "XLE"]


def qlike(daily_var, realised, asset, lo):
    f = (daily_var[asset] * 7).loc[lo:DESIGN_END]
    y = realised[asset].loc[lo:DESIGN_END]
    ok = f.notna() & y.notna() & (y > 0) & (f > 0)
    x = y[ok] / f[ok]
    return float((x - np.log(x) - 1).mean()), float(np.sqrt(y[ok].mean() / f[ok].mean()))


def main() -> None:
    panel = load_panel()
    # Cut the raw data at the design boundary BEFORE building anything: a forecast
    # made on 28 Dec 2021 must not be scored against returns from January 2022.
    cut = slice(None, DESIGN_END)
    c = panel.close.loc[cut]
    cc = np.log(c / c.shift(1)) ** 2
    realised = cc.rolling(7).sum().shift(-7)             # NaN for the last 7 days by construction
    proxy = V.daily_variance_proxy(panel.open.loc[cut], panel.high.loc[cut], panel.low.loc[cut], c)
    ew = lambda x, hl: x.ewm(halflife=hl, min_periods=60).mean()  # noqa: E731
    candidates = {
        "close-to-close, EWMA hl 21 (RiskMetrics baseline)": ew(cc, 21),
        "close-to-close, EWMA hl 10": ew(cc, 10),
        "range blend, EWMA hl 21": ew(proxy, 21),
        "range blend, EWMA hl 10": ew(proxy, 10),
        "range blend, 0.7 x hl 21 + 0.3 x hl 126": 0.7 * ew(proxy, 21) + 0.3 * ew(proxy, 126),
        "range blend, 0.7 x hl 10 + 0.3 x hl 126 (adopted)": 0.7 * ew(proxy, 10) + 0.3 * ew(proxy, 126),
    }
    table = {}
    print(f"{'estimator':52s} {'crypto QLIKE':>12s} {'bias':>5s} | {'fund QLIKE':>10s} {'bias':>5s}")
    for name, dv in candidates.items():
        qc = [qlike(dv, realised, a, "2016") for a in CRYPTO]
        qf = [qlike(dv, realised, a, "2007") for a in FUNDS]
        row = {"crypto_qlike": float(np.mean([q[0] for q in qc])), "crypto_bias": float(np.mean([q[1] for q in qc])),
               "fund_qlike": float(np.mean([q[0] for q in qf])), "fund_bias": float(np.mean([q[1] for q in qf]))}
        table[name] = row
        print(f"{name:52s} {row['crypto_qlike']:12.3f} {row['crypto_bias']:5.2f} | "
              f"{row['fund_qlike']:10.3f} {row['fund_bias']:5.2f}")
    (RESULTS / "volatility_forecast.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
