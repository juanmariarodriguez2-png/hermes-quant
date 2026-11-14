"""Cross-asset stress gauges: things a single price chart cannot show.

* Turbulence (Kritzman & Li, 2010): the Mahalanobis distance of today's vector of
  asset returns from its recent history. It is large not only when moves are big,
  but when assets move in *unusual combinations* (correlations breaking down).
* Absorption ratio (Kritzman, Li, Page & Rigobon, 2011): the share of total variance
  captured by the first few principal components. When it is high, markets are
  tightly coupled and a shock travels everywhere at once.

Both are causal: day t is scored against statistics estimated from days before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def turbulence(returns: pd.DataFrame, window: int = 365, min_obs: int = 180,
               ridge: float = 0.05) -> pd.Series:
    """Daily turbulence index, scaled so that its long-run average is about 1."""
    values = returns.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for t in range(min_obs, len(values)):
        today = values[t]
        ok = np.isfinite(today)
        hist = values[max(0, t - window):t][:, ok]
        hist = hist[np.isfinite(hist).all(axis=1)]
        n = hist.shape[1]
        if n < 2 or hist.shape[0] < max(min_obs // 2, 3 * n):
            continue
        mu = hist.mean(axis=0)
        cov = np.cov(hist, rowvar=False)
        cov = (1.0 - ridge) * cov + ridge * np.diag(np.diag(cov))  # shrink for stability
        d = today[ok] - mu
        out[t] = float(d @ np.linalg.solve(cov, d)) / n
    return pd.Series(out, index=returns.index, name="turbulence")


def absorption_ratio(returns: pd.DataFrame, window: int = 365, min_obs: int = 180,
                     top_fraction: float = 0.2) -> pd.Series:
    """Fraction of variance explained by the top ``top_fraction`` of eigenvectors."""
    values = returns.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for t in range(min_obs, len(values) + 1):
        hist = values[max(0, t - window):t]
        ok = np.isfinite(hist).mean(axis=0) > 0.9
        hist = hist[:, ok]
        hist = hist[np.isfinite(hist).all(axis=1)]
        n = hist.shape[1]
        if n < 3 or hist.shape[0] < max(min_obs // 2, 3 * n):
            continue
        corr = np.corrcoef(hist, rowvar=False)
        vals = np.sort(np.linalg.eigvalsh(corr))[::-1]
        k = max(1, int(round(top_fraction * n)))
        out[t - 1] = float(vals[:k].sum() / vals.sum())
    return pd.Series(out, index=returns.index, name="absorption_ratio")


def rolling_percentile(series: pd.Series, window: int = 730, min_obs: int = 180) -> pd.Series:
    """Where today's reading sits within its own trailing history (0..1), causally."""
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for t in range(len(values)):
        if not np.isfinite(values[t]):
            continue
        hist = values[max(0, t - window):t]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= min_obs:
            out[t] = float(np.mean(hist <= values[t]))
    return pd.Series(out, index=series.index, name=f"{series.name}_pct")
