"""The Hermes Quant model: from a price panel to target portfolio weights.

One code path serves research and production. The backtester and the live runner
both call :func:`compute_features` and :func:`target_weights`; there is no second
"research version" of the logic that could drift away from what actually trades.

Pipeline (every step is causal: day t uses data up to t only)

  1. volatility      range-based EWMA forecast per asset
  2. trend           Kalman filter bank -> P(latent trend > 0) per asset
  3. regime          walk-forward HMM on each sleeve's market proxy -> P(turbulent)
  4. risk model      EW correlation of weekly returns, cleaned with Marchenko-Pastur
  5. allocation      hierarchical risk parity across the trending assets
  6. sizing          conviction x regime, scaled to a volatility target, capped
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .data.panel import Panel
from .filters import kalman, rmt, volatility
from .portfolio.hrp import hrp_weights, inverse_vol_weights
from .regime import hmm


@dataclass(frozen=True)
class ModelConfig:
    # --- trend signal
    signal: str = "kalman"                       # "kalman" | "tsmom" (ablation baseline)
    half_lives: tuple[float, ...] = (30.0, 60.0, 120.0, 240.0)
    tsmom_lookbacks: tuple[int, ...] = (30, 90, 180, 365)
    long_only: bool = True
    min_score: float = 0.05                      # below this trend score: no tactical position
    score_full: float = 0.50                     # full position once P(trend up) >= 75%
    # Strategic floor: the share of a sleeve's budget held regardless of trend.
    # Equities and real assets carry a long-run risk premium, so half is held as a
    # core; crypto has no such anchor (and -80% bear markets), so it is trend-only.
    strategic_floor: dict = field(default_factory=lambda: {"equity": 0.5, "real": 0.5, "crypto": 0.0})
    # --- volatility
    range_vol: bool = True
    vol_fast_halflife: float = 10.0
    vol_slow_halflife: float = 126.0
    vol_fast_weight: float = 0.7
    # --- correlation / risk model
    corr_window: int = 365
    corr_halflife: float = 180.0
    corr_return_days: int = 7
    rmt_clean: bool = True
    # --- allocation
    allocator: str = "hrp"                       # "hrp" | "inverse_vol"
    target_vol: float = 0.15
    # share of portfolio *variance* each sleeve may use at full conviction
    risk_share: dict = field(default_factory=lambda: {"equity": 1 / 3, "real": 1 / 3, "crypto": 1 / 3})
    max_gross: float = 1.0
    max_weight: dict = field(default_factory=lambda: {"crypto": 0.20, "equity": 0.35, "real": 0.30})
    class_cap: dict = field(default_factory=lambda: {"crypto": 0.40, "equity": 1.00, "real": 0.50})
    # --- universe
    min_history_days: int = 400
    crypto_top_k: int = 4
    liquidity_window: int = 90
    # --- regime overlay
    regime: bool = True
    regime_strength: float = 0.75                # exposure multiplier = 1 - strength * P(turbulent)
    regime_smooth_halflife: float = 5.0          # days; damps day-to-day flicker of the probability
    # Only sleeves whose turbulent state had clearly worse risk-adjusted returns in
    # real time get the overlay. For crypto it did not (see research/03_regime_test.py).
    regime_proxy: dict = field(default_factory=lambda: {"equity": "SPY"})
    hmm_min_obs: int = 730
    hmm_refit_every: int = 30
    hmm_max_window: int = 3650

    def with_(self, **changes) -> "ModelConfig":
        return replace(self, **changes)


@dataclass
class Features:
    """Everything the allocator needs, precomputed once per panel."""

    returns: pd.DataFrame        # log returns, calendar days (0 when a fund is shut)
    vol: pd.DataFrame            # annualised volatility forecast
    score: pd.DataFrame          # trend score in (-1, 1)
    prob_up: pd.DataFrame        # P(latent trend > 0)
    latent_sharpe: pd.DataFrame  # posterior mean of the annualised latent Sharpe
    eligible: pd.DataFrame       # bool: in the investable universe that day
    p_turbulent: pd.DataFrame    # per sleeve, P(turbulent regime)
    classes: dict[str, str]


def _tsmom_score(close: pd.DataFrame, vol: pd.DataFrame, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    """Classic time-series momentum: average sign-like score of past k-day returns."""
    logp = np.log(close)
    daily_vol = vol.shift(1) / np.sqrt(volatility.DAYS_PER_YEAR)
    total = None
    for k in lookbacks:
        z = (logp - logp.shift(k)) / (daily_vol * np.sqrt(k))
        s = np.tanh(z)           # bounded, monotone
        total = s if total is None else total + s
    return total / len(lookbacks)


def regime_probabilities(panel: Panel, cfg: ModelConfig) -> pd.DataFrame:
    """Walk-forward P(turbulent) for each sleeve's proxy (slow: cache the result)."""
    out = {}
    usd = panel.close_usd
    for sleeve, proxy in cfg.regime_proxy.items():
        if proxy not in usd.columns:
            continue
        r = np.log(usd[proxy] / usd[proxy].shift(1))
        r = r.where(panel.tradable[proxy])          # sessions only; gaps are carried forward
        p = hmm.walk_forward_turbulent_prob(r.to_numpy(), min_obs=cfg.hmm_min_obs,
                                            refit_every=cfg.hmm_refit_every,
                                            max_window=cfg.hmm_max_window)
        out[sleeve] = pd.Series(p, index=usd.index)
    return pd.DataFrame(out)


def compute_features(panel: Panel, cfg: ModelConfig,
                     p_turbulent: pd.DataFrame | None = None) -> Features:
    close = panel.close
    returns = np.log(close / close.shift(1))

    if cfg.range_vol:
        proxy = volatility.daily_variance_proxy(panel.open, panel.high, panel.low, close)
    else:
        proxy = returns**2
    vol = volatility.ewma_vol(proxy, cfg.vol_fast_halflife, cfg.vol_slow_halflife, cfg.vol_fast_weight)

    if cfg.signal == "kalman":
        observed = returns.where(panel.tradable)    # NaN when shut -> multi-day spans handled
        k = kalman.trend_probability(observed, vol, cfg.half_lives)
        score, prob_up, latent = k["score"], k["prob"], k["sharpe"]
    elif cfg.signal == "tsmom":
        score = _tsmom_score(close, vol, cfg.tsmom_lookbacks)
        prob_up, latent = 0.5 * (score + 1.0), score * np.nan
    else:
        raise ValueError(cfg.signal)

    history = close.notna().cumsum()
    base = (history >= cfg.min_history_days) & close.notna() & vol.notna() & score.notna()
    eligible = base.copy()
    crypto = [s for s in close.columns if panel.classes[s] == "crypto"]
    if crypto:
        liq = panel.dollar_volume[crypto].rolling(cfg.liquidity_window,
                                                  min_periods=cfg.liquidity_window // 2).median()
        rank = liq.where(base[crypto]).rank(axis=1, ascending=False, method="first")
        eligible[crypto] = base[crypto] & (rank <= cfg.crypto_top_k)

    if p_turbulent is None:
        p_turbulent = regime_probabilities(panel, cfg) if cfg.regime else pd.DataFrame(index=close.index)
    p_turbulent = p_turbulent[[c for c in p_turbulent.columns if c in cfg.regime_proxy]]
    if cfg.regime_smooth_halflife > 0 and len(p_turbulent.columns):
        p_turbulent = p_turbulent.ewm(halflife=cfg.regime_smooth_halflife, ignore_na=True).mean().where(p_turbulent.notna())
    return Features(returns, vol, score, prob_up, latent, eligible, p_turbulent, panel.classes)


@dataclass
class Allocation:
    """Target weights for one day plus the diagnostics that explain them."""

    weights: pd.Series
    diagnostics: dict


def risk_model(features: Features, cfg: ModelConfig, t: int, names: list[str]) -> tuple[np.ndarray, dict]:
    """Cleaned correlation matrix for ``names`` using returns up to row t (inclusive)."""
    k = cfg.corr_return_days
    lo = max(0, t - cfg.corr_window + 1)
    window = features.returns[names].iloc[lo:t + 1].fillna(0.0)
    multi = window.rolling(k).sum().dropna().to_numpy()     # overlapping k-day returns
    if multi.shape[0] < 30 or len(names) == 1:
        return np.eye(len(names)), {"n_signal": 0, "noise_share": 1.0, "lambda_max": float("nan"),
                                    "eigenvalues": [1.0] * len(names)}
    corr, n_eff = rmt.ew_correlation(multi, cfg.corr_halflife)
    n_independent = n_eff / k                                # overlap shrinks the sample
    info = {"n_signal": None, "noise_share": None, "lambda_max": None,
            "eigenvalues": np.sort(np.linalg.eigvalsh(corr))[::-1].round(4).tolist()}
    if cfg.rmt_clean and len(names) >= 3:
        cleaned = rmt.clean_correlation(corr, n_independent)
        corr = cleaned.corr
        info.update(n_signal=cleaned.n_signal, noise_share=round(cleaned.noise_share, 4),
                    lambda_max=round(cleaned.lambda_max, 4))
    return corr, info


def target_weights(features: Features, cfg: ModelConfig, t: int) -> Allocation:
    """Target portfolio for the close of row ``t`` (uses rows <= t only).

    Risk is budgeted top-down. Each sleeve (equities, real assets, crypto) owns a
    share of portfolio variance. Inside a sleeve, hierarchical risk parity on the
    cleaned correlation matrix splits that budget across the eligible members, and
    each member is then scaled by the model's conviction in its trend. A member with
    no trend leaves its slice in cash: weak breadth means less exposure, not a
    bigger bet on whatever is still rising.
    """
    symbols = features.score.columns
    zero = pd.Series(0.0, index=symbols)
    eligible = features.eligible.iloc[t].to_numpy()
    score = features.score.iloc[t].to_numpy(dtype=float)
    vol = features.vol.iloc[t].to_numpy(dtype=float)

    idx = [i for i in range(len(symbols)) if eligible[i] and vol[i] > 0 and np.isfinite(score[i])]
    diag = {"n_eligible": len(idx)}
    if not idx:
        diag.update(predicted_vol=0.0, gross=0.0, n_active=0)
        return Allocation(zero, diag)

    names = [symbols[i] for i in idx]
    v = vol[idx]
    lower = 0.0 if cfg.long_only else -1.0
    raw_score = np.where(np.abs(score[idx]) >= cfg.min_score, score[idx], 0.0)
    tactical = np.clip(raw_score / cfg.score_full, lower, 1.0)
    floor = np.array([cfg.strategic_floor.get(features.classes[n], 0.0) for n in names])
    conviction = floor + (1.0 - floor) * tactical if cfg.long_only else tactical

    corr, info = risk_model(features, cfg, t, names)
    cov = corr * np.outer(v, v)
    sleeve_of = [features.classes[n] for n in names]
    sleeves = [s for s in cfg.risk_share if s in sleeve_of]
    total_share = sum(cfg.risk_share[s] for s in sleeves)

    p_row = features.p_turbulent.iloc[t] if (cfg.regime and len(features.p_turbulent.columns)) else {}
    w = np.zeros(len(names))
    sleeve_diag = {}
    for s in sleeves:
        members = [j for j, c in enumerate(sleeve_of) if c == s]
        sub_corr = corr[np.ix_(members, members)]
        sub_vol = v[members]
        base = hrp_weights(sub_corr, sub_vol) if cfg.allocator == "hrp" else inverse_vol_weights(sub_vol)
        full_vol = float(np.sqrt(base @ cov[np.ix_(members, members)] @ base))
        p = p_row.get(s, np.nan) if len(p_row) else np.nan
        regime_mult = 1.0 - cfg.regime_strength * p if np.isfinite(p) else 1.0
        budget = np.sqrt(cfg.risk_share[s] / total_share) * cfg.target_vol   # sleeve vol at full conviction
        w[members] = budget / full_vol * base * conviction[members] * regime_mult
        sleeve_diag[s] = {"full_vol": full_vol, "regime_mult": float(regime_mult),
                          "fill": float(base @ np.abs(conviction[members]))}

    # caps: single name, then asset class, then predicted volatility, then gross exposure
    for j, name in enumerate(names):
        cap = cfg.max_weight.get(features.classes[name], 1.0)
        w[j] = float(np.clip(w[j], -cap, cap))
    for s, cap in cfg.class_cap.items():
        members = [j for j, c in enumerate(sleeve_of) if c == s]
        gross_s = float(np.abs(w[members]).sum()) if members else 0.0
        if gross_s > cap:
            w[members] *= cap / gross_s
    predicted = float(np.sqrt(max(w @ cov @ w, 0.0)))
    if predicted > cfg.target_vol:
        w *= cfg.target_vol / predicted
    gross = float(np.abs(w).sum())
    if gross > cfg.max_gross:
        w *= cfg.max_gross / gross

    out = zero.copy()
    out[names] = w
    diag.update(info)
    diag.update(predicted_vol=float(np.sqrt(max(w @ cov @ w, 0.0))), gross=float(np.abs(w).sum()),
                n_active=int(np.sum(np.abs(w) > 1e-9)), names=names, sleeves=sleeve_diag,
                conviction={n: float(c) for n, c in zip(names, conviction)})
    return Allocation(out, diag)
