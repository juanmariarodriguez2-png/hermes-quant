"""Kalman filtering of the latent trend.

Model, on volatility-normalised daily returns z_t = r_t / sigma_{t-1}:

    z_t = s_t + e_t,        e_t ~ N(0, 1)        what we observe: mostly noise
    s_t = s_{t-1} + w_t,    w_t ~ N(0, lambda)   what we want: a slowly drifting edge

s_t is the asset's *daily Sharpe ratio right now*. It cannot be observed; the filter
keeps a full posterior N(s_hat_t, P_t) for it. From that posterior we read off

    P(s_t > 0 | data up to t) = Phi(s_hat_t / sqrt(P_t))

the probability that the underlying trend is up. Unlike a moving-average crossover
this says *how sure* we are, and it copes natively with gaps and short histories.

lambda sets the horizon. In steady state the filter is an exponential average of
returns with half-life h, lambda = K^2/(1-K) and K = 1 - 2^(-1/h); we run a bank of
filters at several horizons so that no single parameter carries the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

DEFAULT_HALF_LIVES = (30.0, 60.0, 120.0, 240.0)


def halflife_to_lambda(half_life: float) -> float:
    """Signal-to-noise ratio whose steady-state Kalman gain matches an EMA half-life."""
    k = 1.0 - 2.0 ** (-1.0 / half_life)
    return k * k / (1.0 - k)


def filter_drift(z: np.ndarray, lam: float, loading: np.ndarray | None = None,
                 prior_var: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Scalar Kalman filter for the local-level model above.

    Parameters
    ----------
    z : 1-D array of normalised returns; NaN means "no observation that day".
    lam : state innovation variance per calendar day (signal-to-noise ratio).
    loading : observation loading h_t in z_t = h_t * s_t + e_t. A return that spans
        d calendar days (a Monday close for a fund) carries d days of drift and,
        once normalised to unit variance, loads on the daily state with sqrt(d).
        Defaults to 1 everywhere.
    prior_var : variance of the vague prior on s_0.

    Returns the filtered posterior mean and variance at every step. Before the first
    observation both are NaN.
    """
    n = len(z)
    mean = np.full(n, np.nan)
    var = np.full(n, np.nan)
    s, p = 0.0, prior_var
    seen = False
    for t in range(n):
        zt = z[t]
        if seen:
            p += lam                      # predict: the edge may have drifted
        if zt == zt:                      # update only on a real observation
            h = 1.0 if loading is None else loading[t]
            k = p * h / (h * h * p + 1.0)
            s += k * (zt - h * s)
            p *= 1.0 - k * h
            seen = True
        if seen:
            mean[t] = s
            var[t] = p
    return mean, var


def trend_probability(returns: pd.DataFrame, vol_annual: pd.DataFrame,
                      half_lives: tuple[float, ...] = DEFAULT_HALF_LIVES,
                      winsor: float = 4.0, periods_per_year: int = 365) -> dict[str, pd.DataFrame]:
    """Run the filter bank on every column.

    ``returns`` are log returns on a calendar-day index with NaN on days the asset
    did not trade; the return on the next session then spans several days and is
    treated as such. ``vol_annual`` is the contemporaneous volatility estimate; it
    is lagged one day here so each return is scaled by a forecast made *before* it.

    Returns a dict with
      prob    mean over horizons of P(trend > 0), in (0, 1)
      score   2*prob - 1, in (-1, 1): the trading signal
      sharpe  horizon-averaged posterior mean of the annualised latent Sharpe ratio
    """
    daily_vol = (vol_annual.shift(1) / np.sqrt(periods_per_year)).to_numpy(dtype=float)
    r = returns.to_numpy(dtype=float)
    n_rows, n_cols = r.shape
    probs = np.full((n_rows, n_cols), np.nan)
    sharpe = np.full((n_rows, n_cols), np.nan)
    for j in range(n_cols):
        rj = r[:, j]
        observed = np.isfinite(rj) & np.isfinite(daily_vol[:, j]) & (daily_vol[:, j] > 0)
        # calendar days covered by each observed return
        span = np.ones(n_rows)
        last = -1
        for t in range(n_rows):
            if np.isfinite(rj[t]):
                span[t] = t - last if last >= 0 else 1.0
                last = t
        span = np.clip(span, 1.0, 10.0)
        z = np.full(n_rows, np.nan)
        z[observed] = rj[observed] / (daily_vol[observed, j] * np.sqrt(span[observed]))
        z = np.clip(z, -winsor, winsor)
        load = np.sqrt(span)
        pj = np.zeros(n_rows)
        sj = np.zeros(n_rows)
        for hl in half_lives:
            m, v = filter_drift(z, halflife_to_lambda(hl), load)
            pj += ndtr(m / np.sqrt(v))
            sj += m
        probs[:, j] = pj / len(half_lives)
        sharpe[:, j] = sj / len(half_lives) * np.sqrt(periods_per_year)
    prob = pd.DataFrame(probs, index=returns.index, columns=returns.columns)
    return {
        "prob": prob,
        "score": 2.0 * prob - 1.0,
        "sharpe": pd.DataFrame(sharpe, index=returns.index, columns=returns.columns),
    }


def filtered_price_trend(log_price: np.ndarray, level_var: float, slope_var: float,
                         obs_var: float) -> np.ndarray:
    """Causal local-linear-trend filter of a log price (for charts: the 'denoised' line).

    State [level, slope]; returns the filtered level. Not used for trading decisions.
    """
    n = len(log_price)
    out = np.full(n, np.nan)
    x = np.zeros(2)
    P = np.eye(2) * 1e3
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([level_var, slope_var])
    started = False
    for t in range(n):
        y = log_price[t]
        if not started:
            if y != y:
                continue
            x[:] = (y, 0.0)
            started = True
            out[t] = y
            continue
        x = F @ x
        P = F @ P @ F.T + Q
        if y == y:
            s = P[0, 0] + obs_var
            k = P[:, 0] / s
            x = x + k * (y - x[0])
            P = P - np.outer(k, P[0, :])
        out[t] = x[0]
    return out
