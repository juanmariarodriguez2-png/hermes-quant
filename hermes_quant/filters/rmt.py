"""Random-matrix cleaning of correlation matrices (Marchenko-Pastur).

With N assets and only T observations, most of an empirical correlation matrix is
sampling noise. Random matrix theory says what pure noise looks like: if returns
were independent, the eigenvalues of the correlation matrix would fall inside

    [ sigma^2 (1 - sqrt(q))^2 ,  sigma^2 (1 + sqrt(q))^2 ],      q = N / T.

Eigenvalues above the upper edge carry real structure (a market mode, a sector);
those inside the band are indistinguishable from noise. We keep the former and
flatten the latter (Laloux et al. 1999; Bouchaud & Potters; Lopez de Prado 2020).
An optimiser fed the raw matrix happily "diversifies" across noise; fed the cleaned
one it cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CleanResult:
    corr: np.ndarray          # cleaned correlation matrix (unit diagonal, PSD)
    eigenvalues: np.ndarray   # raw eigenvalues, descending
    lambda_max: float         # Marchenko-Pastur upper edge used
    n_signal: int             # eigenvalues above the edge
    noise_share: float        # fraction of total variance attributed to noise


def marchenko_pastur_edges(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Lower and upper edge of the Marchenko-Pastur bulk for aspect ratio q = N/T."""
    root = np.sqrt(q)
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def clean_correlation(corr: np.ndarray, n_obs: float, max_iter: int = 50) -> CleanResult:
    """Eigenvalue clipping with a self-consistent noise variance.

    The noise level sigma^2 is not 1 once real factors exist (they soak up variance),
    so it is solved for iteratively: sigma^2 = mean of the eigenvalues currently
    classified as noise. Noise eigenvalues are replaced by their average, which
    preserves the trace; the result is rescaled to a unit diagonal.
    """
    c = np.asarray(corr, dtype=float)
    n = c.shape[0]
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    q = n / max(float(n_obs), n + 1.0)

    n_signal = 0
    lam_max = marchenko_pastur_edges(q)[1]
    for _ in range(max_iter):
        noise = vals[n_signal:]
        sigma2 = float(noise.mean()) if len(noise) else 1.0
        lam_max = marchenko_pastur_edges(q, sigma2)[1]
        k = int(np.sum(vals > lam_max))
        k = min(k, n - 1)                 # always leave at least one noise direction
        if k == n_signal:
            break
        n_signal = k

    cleaned_vals = vals.copy()
    if n_signal < n:
        cleaned_vals[n_signal:] = vals[n_signal:].mean()
    cleaned = (vecs * cleaned_vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(cleaned), 1e-12, None))
    cleaned = cleaned / np.outer(d, d)
    np.fill_diagonal(cleaned, 1.0)
    return CleanResult(
        corr=cleaned,
        eigenvalues=vals,
        lambda_max=float(lam_max),
        n_signal=n_signal,
        noise_share=float(vals[n_signal:].sum() / vals.sum()),
    )


def ew_correlation(returns: np.ndarray, halflife: float) -> tuple[np.ndarray, float]:
    """Exponentially weighted correlation of the rows of ``returns`` (T x N, no NaN).

    Also returns Kish's effective sample size of the weights, which is the T that
    enters q = N/T.
    """
    t = returns.shape[0]
    w = 0.5 ** (np.arange(t - 1, -1, -1) / halflife)
    w /= w.sum()
    mean = w @ returns
    x = returns - mean
    cov = (x * w[:, None]).T @ x
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    n_eff = 1.0 / float(np.sum(w**2))
    return corr, n_eff
