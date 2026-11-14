"""Hierarchical Risk Parity (Lopez de Prado, 2016).

Mean-variance optimisation inverts the covariance matrix, and inverting a noisy
matrix amplifies the noise: tiny estimation errors become huge, unstable bets.
HRP never inverts anything. It

  1. clusters assets by correlation distance d = sqrt((1 - rho) / 2),
  2. orders them so that similar assets sit next to each other,
  3. splits capital top-down, giving each branch a share inversely proportional
     to its risk.

The result is diversified across *groups of behaviour* (equities, crypto, metals),
not across tickers, and it is far more stable out of sample.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(corr: np.ndarray, vol: np.ndarray, method: str = "average") -> np.ndarray:
    """Long-only weights summing to 1.

    Parameters
    ----------
    corr : (N, N) correlation matrix (ideally already cleaned).
    vol : (N,) volatilities.
    """
    n = len(vol)
    if n == 1:
        return np.ones(1)
    cov = corr * np.outer(vol, vol)
    if n == 2:
        ivp = 1.0 / np.diag(cov)
        return ivp / ivp.sum()
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method=method)))

    weights = np.ones(n)
    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]
        var_left = _cluster_variance(cov, left)
        var_right = _cluster_variance(cov, right)
        alpha = 1.0 - var_left / (var_left + var_right)
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        stack += [left, right]
    return weights / weights.sum()


def inverse_vol_weights(vol: np.ndarray) -> np.ndarray:
    """Naive risk parity (ignores correlations); the baseline HRP has to beat."""
    w = 1.0 / np.asarray(vol, dtype=float)
    return w / w.sum()
