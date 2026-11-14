"""Turns the model's internal state into a plain dict for the dashboard and the logs.

Nothing here feeds back into trading: it is the model explaining itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..filters import kalman
from ..model import Allocation, Features, ModelConfig, risk_model
from ..regime import fragility

NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "XRP": "XRP", "LINK": "Chainlink",
    "ADA": "Cardano", "DOGE": "Dogecoin", "LTC": "Litecoin", "AVAX": "Avalanche", "BNB": "BNB",
    "TRX": "TRON", "BCH": "Bitcoin Cash",
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "US small caps", "VGK": "Europe equities",
    "IEMG": "Emerging markets", "GLD": "Gold", "SLV": "Silver", "XLE": "Energy",
}


def _round(x, digits=4):
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


def _series(s: pd.Series, digits: int = 4) -> list:
    return [[d.strftime("%Y-%m-%d"), _round(v, digits)] for d, v in s.items() if np.isfinite(v)]


def model_snapshot(panel: Panel, features: Features, cfg: ModelConfig, t: int,
                   allocation: Allocation, history_days: int = 365) -> dict:
    """Everything the model believes on day ``t``."""
    index = panel.close.index
    lo = max(0, t - history_days + 1)
    window = slice(lo, t + 1)
    close = panel.close
    conviction = allocation.diagnostics.get("conviction", {})
    assets = []
    for symbol in close.columns:
        px = close[symbol].iloc[window]
        last = close[symbol].iloc[t]
        ret = lambda days: _round(last / close[symbol].iloc[t - days] - 1.0) if t >= days else None  # noqa: E731
        logp = np.log(close[symbol].iloc[: t + 1].to_numpy())
        vol_d = features.vol[symbol].iloc[t] / np.sqrt(365.0)
        trend = kalman.filtered_price_trend(logp[-800:], level_var=(0.15 * vol_d) ** 2,
                                            slope_var=(0.004 * vol_d) ** 2, obs_var=vol_d**2)
        trend_line = pd.Series(np.exp(trend[-len(px):]), index=px.index)
        assets.append({
            "symbol": symbol, "name": NAMES.get(symbol, symbol), "class": panel.classes[symbol],
            "price": _round(last, 6), "prob_up": _round(features.prob_up[symbol].iloc[t]),
            "score": _round(features.score[symbol].iloc[t]),
            "latent_sharpe": _round(features.latent_sharpe[symbol].iloc[t], 2),
            "vol": _round(features.vol[symbol].iloc[t]), "eligible": bool(features.eligible[symbol].iloc[t]),
            "venue_liquid": True if panel.investable is None else bool(panel.investable[symbol].iloc[t]),
            "conviction": _round(conviction.get(symbol, 0.0)),
            "target": _round(allocation.weights.get(symbol, 0.0)),
            "ret_30d": ret(30), "ret_90d": ret(90), "ret_365d": ret(365),
            "prob_history": _series(features.prob_up[symbol].iloc[window].iloc[::3], 3),
            "price_history": _series(px.iloc[::3], 6), "trend_history": _series(trend_line.iloc[::3], 6),
        })

    regime = {}
    for sleeve in features.p_turbulent.columns:
        p = features.p_turbulent[sleeve]
        regime[sleeve] = {"p_turbulent": _round(p.iloc[t]), "proxy": cfg.regime_proxy.get(sleeve),
                          "history": _series(p.iloc[max(lo, t - 180): t + 1], 3)}

    names = allocation.diagnostics.get("names") or [s for s in close.columns if features.eligible[s].iloc[t]]
    risk = {"target_vol": cfg.target_vol, "predicted_vol": _round(allocation.diagnostics.get("predicted_vol", 0.0)),
            "gross": _round(allocation.diagnostics.get("gross", 0.0)),
            "sleeves": allocation.diagnostics.get("sleeves", {})}
    correlation = {}
    if len(names) >= 3:
        raw_cfg = cfg.with_(rmt_clean=False)
        clean, info = risk_model(features, cfg, t, names)
        raw, _ = risk_model(features, raw_cfg, t, names)
        risk.update(n_signal=info.get("n_signal"), noise_share=info.get("noise_share"),
                    lambda_max=info.get("lambda_max"), eigenvalues=info.get("eigenvalues"),
                    n_assets=len(names))
        # which assets make up each real factor (largest loadings of its eigenvector)
        vals, vecs = np.linalg.eigh(raw)
        order = np.argsort(vals)[::-1]
        factors = []
        for k in order[: int(info.get("n_signal") or 0)]:
            load = np.abs(vecs[:, k])
            members = [names[i] for i in np.argsort(load)[::-1] if load[i] >= 0.6 * load.max()]
            factors.append({"share": _round(vals[k] / len(names), 3), "assets": members})
        risk["factors"] = factors
        correlation = {"names": names, "cleaned": np.round(clean, 3).tolist(), "raw": np.round(raw, 3).tolist()}

    elig = [s for s in close.columns if features.eligible[s].iloc[t]]
    frag = {}
    if len(elig) >= 3:
        r = features.returns[elig].iloc[max(0, t - 1100): t + 1]
        turb = fragility.turbulence(r)
        pct = fragility.rolling_percentile(turb)
        absorb = fragility.absorption_ratio(r)
        frag = {"turbulence": _round(turb.iloc[-1], 2), "turbulence_percentile": _round(pct.iloc[-1], 3),
                "absorption_ratio": _round(absorb.iloc[-1], 3),
                "turbulence_history": _series(turb.rolling(5).mean().iloc[-180:], 2),
                "absorption_history": _series(absorb.iloc[-180:], 3)}

    return {"bar_date": index[t].strftime("%Y-%m-%d"), "assets": assets, "regime": regime, "risk": risk,
            "correlation": correlation, "fragility": frag}
