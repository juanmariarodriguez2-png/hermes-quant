"""Gaussian hidden Markov model for market regimes, written from scratch.

Markets do not announce whether they are in a calm or a turbulent state; the state
is *hidden* and has to be inferred from returns. A two-state HMM assumes each day's
return is drawn from N(mu_k, sigma_k^2), with the state k following a persistent
Markov chain. Fitting is by Baum-Welch (EM); inference uses only the *forward*
recursion, so the probability reported for day t depends on data up to t and never
on what came after (the backward/smoothed pass would be look-ahead in a backtest).

States are sorted by variance: state 0 is "calm", the last state is "turbulent".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class HMMParams:
    start: np.ndarray   # (K,) initial state distribution
    trans: np.ndarray   # (K, K) row-stochastic transition matrix
    mean: np.ndarray    # (K,)
    var: np.ndarray     # (K,)
    loglik: float = float("nan")

    @property
    def n_states(self) -> int:
        return len(self.mean)


def _emission(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """(T, K) Gaussian densities, floored to keep the recursions finite."""
    d = x[:, None] - mean[None, :]
    logp = -0.5 * (_LOG_2PI + np.log(var)[None, :] + d * d / var[None, :])
    return np.exp(np.clip(logp, -700.0, None)) + 1e-300


def _forward(b: np.ndarray, start: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward pass. Returns filtered probabilities and per-step scale factors."""
    t_len, k = b.shape
    alpha = np.empty((t_len, k))
    scale = np.empty(t_len)
    a = start * b[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, t_len):
        a = (alpha[t - 1] @ trans) * b[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]
    return alpha, scale


def _backward(b: np.ndarray, trans: np.ndarray, scale: np.ndarray) -> np.ndarray:
    t_len, k = b.shape
    beta = np.empty((t_len, k))
    beta[-1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta[t] = (trans @ (b[t + 1] * beta[t + 1])) / scale[t + 1]
    return beta


def _sorted(p: HMMParams) -> HMMParams:
    order = np.argsort(p.var)
    return HMMParams(p.start[order], p.trans[np.ix_(order, order)], p.mean[order], p.var[order], p.loglik)


def initial_guess(x: np.ndarray, n_states: int = 2, persistence: float = 0.97) -> HMMParams:
    """Deterministic start: split observations by |x| quantiles."""
    order = np.argsort(np.abs(x - np.median(x)))
    chunks = np.array_split(order, n_states)
    mean = np.array([x[c].mean() for c in chunks])
    var = np.array([max(x[c].var(), 1e-12) for c in chunks])
    off = (1.0 - persistence) / max(n_states - 1, 1)
    trans = np.full((n_states, n_states), off)
    np.fill_diagonal(trans, persistence)
    return HMMParams(np.full(n_states, 1.0 / n_states), trans, mean, var)


def fit(x: np.ndarray, n_states: int = 2, init: HMMParams | None = None, max_iter: int = 200,
        tol: float = 1e-7, var_floor: float | None = None) -> HMMParams:
    """Baum-Welch maximum likelihood. ``init`` allows warm starts in walk-forward use."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    p = init if init is not None else initial_guess(x, n_states)
    spread = float(np.var(x))
    if len(x) < 10 or not np.isfinite(spread) or spread <= 0:
        raise ValueError("HMM needs a non-constant series with at least 10 observations")
    floor = var_floor if var_floor is not None else max(1e-4 * spread, 1e-14)
    start, trans, mean, var = p.start.copy(), p.trans.copy(), p.mean.copy(), p.var.copy()
    prev = -np.inf
    loglik = prev
    for _ in range(max_iter):
        b = _emission(x, mean, var)
        alpha, scale = _forward(b, start, trans)
        loglik = float(np.log(scale).sum())
        beta = _backward(b, trans, scale)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        # expected transitions, summed over time
        xi = (alpha[:-1, :, None] * trans[None, :, :] * (b[1:] * beta[1:])[:, None, :]) / scale[1:, None, None]
        xi_sum = xi.sum(axis=0)
        start = gamma[0] / gamma[0].sum()
        trans = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        weight = gamma.sum(axis=0)
        mean = (gamma * x[:, None]).sum(axis=0) / weight
        var = np.maximum((gamma * (x[:, None] - mean[None, :]) ** 2).sum(axis=0) / weight, floor)
        if abs(loglik - prev) < tol * (1.0 + abs(prev)):
            break
        prev = loglik
    return _sorted(HMMParams(start, trans, mean, var, loglik))


def filtered_probabilities(x: np.ndarray, params: HMMParams) -> np.ndarray:
    """P(state_t = k | x_1..x_t).

    The chain advances once per *observation*, exactly as during fitting (where
    missing days are dropped). On a day without an observation (a weekend for an
    exchange-traded fund) the last belief is simply carried forward: applying the
    transition matrix there would let three steps pass between Friday and Monday
    although the model was estimated with one."""
    x = np.asarray(x, dtype=float)
    k = params.n_states
    out = np.full((len(x), k), np.nan)
    belief = params.start.copy()
    seen = False
    for t, xt in enumerate(x):
        if xt == xt:
            if seen:
                belief = belief @ params.trans
            lik = _emission(np.array([xt]), params.mean, params.var)[0]
            belief = belief * lik
            belief /= belief.sum()
            seen = True
        out[t] = belief
    return out


def walk_forward_turbulent_prob(x: np.ndarray, min_obs: int = 500, refit_every: int = 30,
                                n_states: int = 2, max_window: int | None = None) -> np.ndarray:
    """Out-of-sample probability of the turbulent state for every observation.

    At each refit point tau the model is estimated on data strictly before tau. The
    probabilities for [tau, next refit) are then produced by the forward filter with
    those frozen parameters. Nothing at time t ever depends on data after t.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    params: HMMParams | None = None
    tau = min_obs
    while tau < n:
        lo = 0 if max_window is None else max(0, tau - max_window)
        train = x[lo:tau]
        train = train[np.isfinite(train)]
        if len(train) >= min_obs // 2 and np.var(train) > 0:
            cold = fit(train, n_states)
            if params is not None:
                warm = fit(train, n_states, init=params)
                params = warm if warm.loglik >= cold.loglik else cold
            else:
                params = cold
            if not (np.isfinite(params.mean).all() and np.isfinite(params.var).all()):
                params = None                        # numerical failure: report NaN, not a guess
        end = min(tau + refit_every, n)
        if params is not None:
            probs = filtered_probabilities(x[lo:end], params)
            out[tau:end] = probs[tau - lo:end - lo, -1]
        tau = end
    return out
