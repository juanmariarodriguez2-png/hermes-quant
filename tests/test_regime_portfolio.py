import numpy as np
import pandas as pd

from hermes_quant.portfolio.hrp import hrp_weights, inverse_vol_weights
from hermes_quant.regime import fragility, hmm


def _two_state_series(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.98 if state[t - 1] == 0 else 0.95
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    x = np.where(state == 0, rng.normal(0.0006, 0.007, n), rng.normal(-0.001, 0.025, n))
    return x, state


def test_hmm_recovers_the_generating_parameters():
    x, state = _two_state_series()
    p = hmm.fit(x, 2)
    assert abs(np.sqrt(p.var[0]) - 0.007) < 0.0015
    assert abs(np.sqrt(p.var[1]) - 0.025) < 0.004
    assert p.trans[0, 0] > 0.95 and p.trans[1, 1] > 0.88
    probs = hmm.filtered_probabilities(x, p)[:, 1]
    assert np.mean((probs > 0.5) == (state == 1)) > 0.9


def test_filtered_probabilities_never_use_the_future():
    x, _ = _two_state_series(1500, seed=1)
    p = hmm.fit(x[:1000], 2)
    base = hmm.filtered_probabilities(x, p)
    x2 = x.copy()
    x2[1200:] = 0.3                     # absurd future
    other = hmm.filtered_probabilities(x2, p)
    assert np.allclose(base[:1200], other[:1200])


def test_walk_forward_probabilities_are_out_of_sample():
    x, _ = _two_state_series(1600, seed=2)
    base = hmm.walk_forward_turbulent_prob(x, min_obs=600, refit_every=100)
    x2 = x.copy()
    x2[1300:] *= 25.0                   # nothing before row 1300 may change
    other = hmm.walk_forward_turbulent_prob(x2, min_obs=600, refit_every=100)
    assert np.isnan(base[:600]).all()
    assert np.allclose(base[600:1300], other[600:1300], equal_nan=True)


def test_missing_observations_are_carried_forward():
    x, _ = _two_state_series(800, seed=3)
    p = hmm.fit(x, 2)
    x2 = x.copy()
    x2[400:403] = np.nan
    probs = hmm.filtered_probabilities(x2, p)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0)


# ------------------------------------------------------------------ HRP

def test_hrp_basic_properties():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(500, 6))
    corr = np.corrcoef(x, rowvar=False)
    w = hrp_weights(corr, np.full(6, 0.2))
    assert abs(w.sum() - 1.0) < 1e-12 and (w > 0).all()
    assert w.max() < 0.35                # roughly equal when nothing distinguishes the assets


def test_hrp_gives_less_to_riskier_and_to_redundant_assets():
    corr = np.eye(3)
    w = hrp_weights(corr, np.array([0.1, 0.2, 0.4]))
    assert w[0] > w[1] > w[2]
    # two clones plus one independent asset: the clones must share one slot
    corr = np.array([[1.0, 0.98, 0.0], [0.98, 1.0, 0.0], [0.0, 0.0, 1.0]])
    w = hrp_weights(corr, np.full(3, 0.2))
    assert w[2] > 0.45 and abs(w[0] - w[1]) < 0.05


def test_hrp_small_cases():
    assert hrp_weights(np.eye(1), np.array([0.3]))[0] == 1.0
    w = hrp_weights(np.eye(2), np.array([0.1, 0.3]))
    assert abs(w.sum() - 1.0) < 1e-12 and w[0] > w[1]
    assert np.allclose(inverse_vol_weights(np.array([0.1, 0.1])), [0.5, 0.5])


# ------------------------------------------------------------------ fragility

def test_turbulence_spikes_on_an_unusual_day_and_is_causal():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-01", periods=700, freq="D")
    r = pd.DataFrame(rng.normal(0.0, 0.01, (700, 5)), index=idx)
    r.iloc[600] = [0.08, -0.08, 0.08, -0.08, 0.08]
    t = fragility.turbulence(r)
    assert t.iloc[600] > 10 * t.iloc[300:600].median()
    r2 = r.copy()
    r2.iloc[650:] = 0.2
    pd.testing.assert_series_equal(t.iloc[:650], fragility.turbulence(r2).iloc[:650])


def test_absorption_ratio_rises_when_assets_move_together():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2020-01-01", periods=800, freq="D")
    independent = pd.DataFrame(rng.normal(size=(800, 10)), index=idx)
    common = rng.normal(size=(800, 1))
    coupled = pd.DataFrame(common * 2.0 + rng.normal(size=(800, 10)), index=idx)
    assert fragility.absorption_ratio(coupled).iloc[-1] > fragility.absorption_ratio(independent).iloc[-1] + 0.3


def test_filter_advances_once_per_observation_not_per_calendar_day():
    """Friday -> Monday is one step of the chain, however many days lie between."""
    x, _ = _two_state_series(600, seed=7)
    p = hmm.fit(x, 2)
    dense = hmm.filtered_probabilities(x, p)
    gappy = np.full(3 * len(x), np.nan)
    gappy[::3] = x                                          # same observations, two empty days between
    sparse = hmm.filtered_probabilities(gappy, p)
    assert np.allclose(sparse[::3], dense)
    assert np.allclose(sparse[1::3], dense)                 # carried forward unchanged


def test_hmm_rejects_a_constant_series():
    import pytest

    with pytest.raises(ValueError):
        hmm.fit(np.zeros(500), 2)
    out = hmm.walk_forward_turbulent_prob(np.zeros(900), min_obs=300, refit_every=100)
    assert np.isnan(out).all()                              # unavailable, not silently "calm"
