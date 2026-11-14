import numpy as np
import pandas as pd

from hermes_quant.filters import kalman, rmt, volatility


# ------------------------------------------------------------------ Kalman

def test_steady_state_gain_matches_ema_half_life():
    for hl in (30.0, 120.0):
        lam = kalman.halflife_to_lambda(hl)
        p = 1.0
        for _ in range(5000):          # iterate the Riccati recursion to its fixed point
            p = p + lam
            k = p / (p + 1.0)
            p = p * (1.0 - k)
        assert abs(k - (1.0 - 2.0 ** (-1.0 / hl))) < 1e-9


def test_filter_recovers_sign_of_a_real_drift():
    rng = np.random.default_rng(1)
    z = rng.normal(0.1, 1.0, 2000)     # daily Sharpe 0.1, about 1.9 annualised
    mean, var = kalman.filter_drift(z, kalman.halflife_to_lambda(120.0))
    assert mean[-500:].mean() > 0.05
    assert np.all(var[10:] > 0)


def test_pure_noise_gives_no_conviction_on_average():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-01-01", periods=3000, freq="D")
    r = pd.DataFrame(rng.normal(0.0, 0.01, (3000, 8)), index=idx)
    vol = pd.DataFrame(0.01 * np.sqrt(365), index=idx, columns=r.columns)
    prob = kalman.trend_probability(r, vol)["prob"].iloc[500:]
    assert abs(float(prob.mean().mean()) - 0.5) < 0.05


def test_trend_probability_is_causal():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=600, freq="D")
    r = pd.DataFrame(rng.normal(0.0005, 0.01, (600, 3)), index=idx)
    vol = pd.DataFrame(0.2, index=idx, columns=r.columns)
    base = kalman.trend_probability(r, vol)["prob"]
    r2, vol2 = r.copy(), vol.copy()
    r2.iloc[400:] = 0.5                # change the future violently
    vol2.iloc[400:] = 3.0
    changed = kalman.trend_probability(r2, vol2)["prob"]
    pd.testing.assert_frame_equal(base.iloc[:400], changed.iloc[:400])


def test_multi_day_returns_are_handled_like_daily_ones():
    """A fund observed only on weekdays must not look more (or less) trendy than the
    same process observed every day."""
    rng = np.random.default_rng(4)
    n = 4000
    idx = pd.date_range("2010-01-01", periods=n, freq="D")
    daily = rng.normal(0.0006, 0.01, n)
    full = pd.DataFrame({"a": daily}, index=idx)
    gappy = full.copy()
    weekday = idx.dayofweek < 5
    carried = np.zeros(n)
    acc = 0.0
    for t in range(n):                 # weekend moves show up in Monday's return
        acc += daily[t]
        if weekday[t]:
            carried[t] = acc
            acc = 0.0
    gappy["a"] = np.where(weekday, carried, np.nan)
    vol = pd.DataFrame(0.01 * np.sqrt(365), index=idx, columns=["a"])
    p_full = kalman.trend_probability(full, vol)["prob"]["a"].iloc[500:]
    p_gappy = kalman.trend_probability(gappy, vol)["prob"]["a"].iloc[500:]
    assert abs(float(p_full.mean() - p_gappy.mean())) < 0.03
    assert float(np.corrcoef(p_full, p_gappy)[0, 1]) > 0.95


# ------------------------------------------------------------------ volatility

def _gbm_ohlc(n, sigma_daily, seed, steps=96):
    rng = np.random.default_rng(seed)
    paths = np.cumsum(rng.normal(0.0, sigma_daily / np.sqrt(steps), (n, steps)), axis=1)
    close_log = np.cumsum(paths[:, -1])
    start = np.concatenate([[0.0], close_log[:-1]])
    intraday = start[:, None] + paths
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    frame = lambda x: pd.DataFrame({"a": 100.0 * np.exp(x)}, index=idx)  # noqa: E731
    return (frame(start), frame(np.maximum(intraday.max(axis=1), start)),
            frame(np.minimum(intraday.min(axis=1), start)), frame(close_log))


def test_volatility_estimate_is_unbiased_on_a_random_walk():
    sigma = 0.03
    o, h, l, c = _gbm_ohlc(3000, sigma, seed=5)
    vol = volatility.ewma_vol(volatility.daily_variance_proxy(o, h, l, c))["a"].iloc[500:]
    assert abs(float(np.sqrt((vol**2).mean())) / (sigma * np.sqrt(365)) - 1.0) < 0.08


def test_a_flash_crash_wick_does_not_poison_the_estimate():
    o, h, l, c = _gbm_ohlc(1500, 0.03, seed=6)
    clean = volatility.ewma_vol(volatility.daily_variance_proxy(o, h, l, c))["a"]
    l2 = l.copy()
    l2.iloc[1000] = 0.01               # a 99.99% wick that closed back where it started
    dirty = volatility.ewma_vol(volatility.daily_variance_proxy(o, h, l2, c))["a"]
    assert float((dirty / clean).iloc[1000:1100].max()) < 1.6


def test_a_real_crash_is_not_filtered_out():
    o, h, l, c = _gbm_ohlc(1500, 0.02, seed=7)
    shock = np.ones(1500)
    shock[1000:] = np.exp(-0.5)        # the close itself falls 40%: this is real risk
    scale = pd.DataFrame({"a": shock}, index=c.index)
    o2, h2, l2, c2 = o * scale, h * scale, l * scale, c * scale
    o2.iloc[1000] = o.iloc[1000]       # the crash day opens at the old level
    h2.iloc[1000] = np.maximum(h2.iloc[1000], o2.iloc[1000])
    before = volatility.ewma_vol(volatility.daily_variance_proxy(o, h, l, c))["a"].iloc[1001]
    after = volatility.ewma_vol(volatility.daily_variance_proxy(o2, h2, l2, c2))["a"].iloc[1001]
    assert after > 2.0 * before


def test_volatility_is_causal():
    o, h, l, c = _gbm_ohlc(800, 0.03, seed=8)
    base = volatility.ewma_vol(volatility.daily_variance_proxy(o, h, l, c))
    c2, h2 = c.copy(), h.copy()
    c2.iloc[600:] *= 3.0
    h2.iloc[600:] *= 3.0
    other = volatility.ewma_vol(volatility.daily_variance_proxy(o, h2, l, c2))
    pd.testing.assert_frame_equal(base.iloc[:600], other.iloc[:600])


# ------------------------------------------------------------------ random matrix theory

def test_pure_noise_has_no_signal_eigenvalues():
    rng = np.random.default_rng(9)
    hits = 0
    for _ in range(40):
        x = rng.normal(size=(250, 20))
        res = rmt.clean_correlation(np.corrcoef(x, rowvar=False), 250)
        hits += res.n_signal
        assert np.allclose(np.diag(res.corr), 1.0)
    assert hits <= 12                   # occasional edge fluctuation, not structure


def test_one_real_factor_is_found_and_matrix_stays_valid():
    rng = np.random.default_rng(10)
    factor = rng.normal(size=(400, 1))
    x = 0.7 * factor + rng.normal(size=(400, 15))
    res = rmt.clean_correlation(np.corrcoef(x, rowvar=False), 400)
    assert res.n_signal == 1
    assert np.linalg.eigvalsh(res.corr).min() > 0
    assert 0.0 < res.noise_share < 1.0


def test_cleaning_brings_the_estimate_closer_to_the_truth():
    rng = np.random.default_rng(11)
    n, t = 20, 120
    load = rng.uniform(0.3, 0.8, n)
    true = np.outer(load, load)
    np.fill_diagonal(true, 1.0)
    chol = np.linalg.cholesky(true)
    raw_err, clean_err = [], []
    for _ in range(30):
        x = rng.normal(size=(t, n)) @ chol.T
        raw = np.corrcoef(x, rowvar=False)
        clean = rmt.clean_correlation(raw, t).corr
        raw_err.append(np.linalg.norm(raw - true))
        clean_err.append(np.linalg.norm(clean - true))
    assert np.mean(clean_err) < np.mean(raw_err)


def test_effective_sample_size_of_ew_weights():
    x = np.random.default_rng(12).normal(size=(500, 4))
    _, n_eff = rmt.ew_correlation(x, halflife=1e9)      # flat weights -> all 500 count
    assert abs(n_eff - 500) < 1.0
    _, n_eff_fast = rmt.ew_correlation(x, halflife=20.0)
    assert n_eff_fast < 120
