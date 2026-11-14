"""Statistical tests that separate a real edge from a lucky backtest.

A Sharpe ratio is an estimate with an error bar, and every extra variant you try
before picking "the best" inflates the winner. These tools account for both.

References
----------
Lo (2002), "The Statistics of Sharpe Ratios"
Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR)
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" (DSR)
Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest Overfitting"
Politis & Romano (1994), "The Stationary Bootstrap"
Newey & West (1987), HAC covariance
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------- Sharpe inference

def sharpe(r: np.ndarray) -> float:
    """Per-period (not annualised) Sharpe ratio."""
    r = np.asarray(r, dtype=float)
    s = r.std(ddof=1)
    return float(r.mean() / s) if s > 0 else 0.0


def sharpe_std_error(sr: float, n: int, skew: float, kurt: float) -> float:
    """Standard error of a per-period Sharpe estimate under non-normal returns.
    ``kurt`` is the raw kurtosis (3 for a normal distribution)."""
    return float(np.sqrt(max(1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr * sr, 1e-12) / (n - 1)))


def autocorrelation_inflation(r: np.ndarray, lags: int = 4) -> float:
    """Factor by which serial correlation widens the Sharpe standard error (Lo 2002,
    Newey-West weights). 1.0 for independent returns; never below 1 here, so the
    adjustment can only make a result look *less* significant."""
    r = np.asarray(r, dtype=float)
    x = r - r.mean()
    denom = float(x @ x)
    if denom <= 0:
        return 1.0
    total = 1.0
    for k in range(1, min(lags, len(r) - 2) + 1):
        rho = float(x[k:] @ x[:-k]) / denom
        total += 2.0 * (1.0 - k / (lags + 1.0)) * rho
    return float(np.sqrt(max(total, 1.0)))


def probabilistic_sharpe(r: np.ndarray, benchmark_sr: float = 0.0, autocorr_lags: int = 4) -> float:
    """P(true Sharpe > benchmark_sr), both per period, allowing for fat tails, skew
    and (conservatively) serial correlation."""
    r = np.asarray(r, dtype=float)
    sr = sharpe(r)
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt()) + 3.0
    se = sharpe_std_error(sr, len(r), skew, kurt) * autocorrelation_inflation(r, autocorr_lags)
    return float(norm.cdf((sr - benchmark_sr) / se))


def expected_max_sharpe(n_trials: int, trials_sr_std: float) -> float:
    """Expected best Sharpe among ``n_trials`` strategies that all have zero true
    Sharpe (the 'false strategy' benchmark of the Deflated Sharpe Ratio)."""
    if n_trials <= 1:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(trials_sr_std * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(r: np.ndarray, n_trials: int, trials_sr_std: float) -> dict:
    """Deflated Sharpe Ratio: the probability that the selected strategy's true Sharpe
    is positive *after* discounting for having been picked out of ``n_trials``."""
    hurdle = expected_max_sharpe(n_trials, trials_sr_std)
    return {"dsr": probabilistic_sharpe(r, hurdle), "hurdle_sr_per_period": hurdle,
            "psr_vs_zero": probabilistic_sharpe(r, 0.0), "n_trials": n_trials}


# ----------------------------------------------------------------- bootstrap

def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of 0..n-1 made of blocks with geometric lengths (wraps around)."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / mean_block
    i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return idx


def bootstrap_ci(r: np.ndarray, periods: int, n_boot: int = 2000, mean_block: float = 20.0,
                 seed: int = 7) -> dict:
    """Confidence intervals for Sharpe, CAGR and maximum drawdown by resampling
    blocks of returns (which preserves volatility clustering and autocorrelation)."""
    r = np.asarray(r, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(r)
    sr, cagr, mdd = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        x = r[stationary_bootstrap_indices(n, mean_block, rng)]
        sr[b] = sharpe(x) * np.sqrt(periods)
        curve = np.concatenate([[1.0], np.cumprod(1.0 + x)])   # include the starting capital
        cagr[b] = curve[-1] ** (periods / n) - 1.0
        mdd[b] = float((curve / np.maximum.accumulate(curve) - 1.0).min())

    def band(a):
        lo, med, hi = np.quantile(a, [0.025, 0.5, 0.975])
        return {"p2.5": float(lo), "p50": float(med), "p97.5": float(hi)}

    # These are frequencies across resamples of the observed history, not literal
    # probabilities about the true parameter.
    return {"sharpe": band(sr), "cagr": band(cagr), "max_drawdown": band(mdd),
            "share_resamples_sharpe_below_zero": float(np.mean(sr <= 0.0)),
            "share_resamples_negative_cagr": float(np.mean(cagr <= 0.0)),
            "mean_block": mean_block, "n_boot": n_boot}


# ----------------------------------------------------------------- regressions

def ols_hac(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS of y on [1, x] with Newey-West (Bartlett) standard errors."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(y)
    design = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    scores = design * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += w * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv * n / (n - design.shape[1])
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"coef": beta, "se": se, "t": beta / se,
            "r2": 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan, "n": n, "lags": lags}


def market_model(strategy: pd.Series, factors: pd.DataFrame, periods_per_year: float = 52.0) -> dict:
    """CAPM-style regression of strategy returns on benchmark returns (weekly data
    recommended, which sidesteps the different closing times of the markets)."""
    data = pd.concat([strategy.rename("y"), factors], axis=1).dropna()
    fit = ols_hac(data["y"].to_numpy(), data[factors.columns].to_numpy())
    out = {"alpha_annual": float(fit["coef"][0] * periods_per_year), "alpha_t": float(fit["t"][0]),
           "r2": float(fit["r2"]), "n": int(fit["n"])}
    for j, name in enumerate(factors.columns, start=1):
        out[f"beta_{name}"] = float(fit["coef"][j])
        out[f"beta_{name}_t"] = float(fit["t"][j])
    return out


# ----------------------------------------------------------------- overfitting

def probability_of_backtest_overfitting(returns: pd.DataFrame, n_blocks: int = 10) -> dict:
    """Combinatorially symmetric cross-validation (CSCV).

    ``returns`` holds one column per strategy variant. The sample is cut into
    ``n_blocks`` blocks; for every way of choosing half of them as "in sample" we
    pick the variant with the best in-sample Sharpe and look at where it ranks out
    of sample. PBO is how often the in-sample winner lands in the bottom half."""
    x = returns.dropna().to_numpy(dtype=float)
    t, n = x.shape
    blocks = np.array_split(np.arange(t), n_blocks)
    logits, oos_of_best, is_of_best = [], [], []
    for chosen in combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[b] for b in chosen])
        oos_idx = np.concatenate([blocks[b] for b in range(n_blocks) if b not in chosen])
        is_sr = x[is_idx].mean(axis=0) / x[is_idx].std(axis=0, ddof=1)
        oos_sr = x[oos_idx].mean(axis=0) / x[oos_idx].std(axis=0, ddof=1)
        best = int(np.argmax(is_sr))
        rank = (np.sum(oos_sr < oos_sr[best]) + 0.5 * np.sum(oos_sr == oos_sr[best])) / n
        rank = min(max(rank, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))
        logits.append(np.log(rank / (1.0 - rank)))
        oos_of_best.append(oos_sr[best])
        is_of_best.append(is_sr[best])
    logits = np.array(logits)
    # Note: this ranks by plain Sharpe. It is a diagnostic of how fragile "pick the
    # best-looking variant" would be on this candidate set, not a replay of any
    # more elaborate selection rule.
    return {"pbo": float(np.mean(logits <= 0.0)), "n_splits": len(logits), "n_variants": n,
            "median_oos_rank": float(1.0 / (1.0 + np.exp(-np.median(logits)))),
            "mean_is_sharpe_of_best": float(np.mean(is_of_best)),
            "mean_oos_sharpe_of_best": float(np.mean(oos_of_best))}
