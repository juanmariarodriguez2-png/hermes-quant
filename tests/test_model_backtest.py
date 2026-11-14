"""The properties that make a backtest trustworthy: no peeking, honest accounting."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_panel
from hermes_quant.backtest import stats
from hermes_quant.backtest.engine import CostModel, TradingRules, run_backtest, static_mix
from hermes_quant.backtest.metrics import summary
from hermes_quant.model import ModelConfig, compute_features, target_weights

CFG = ModelConfig(regime=False, min_history_days=200)
FREE = CostModel(fee={}, slippage={}, fx_fee=0.0)


@pytest.fixture(scope="module")
def features(panel):
    return compute_features(panel, CFG)


def test_target_weights_respect_every_limit(panel, features):
    for t in range(450, len(panel.close), 37):
        w = target_weights(features, CFG, t).weights
        assert (w >= -1e-12).all()                      # long only
        assert w.sum() <= CFG.max_gross + 1e-9
        for name, value in w.items():
            assert value <= CFG.max_weight[panel.classes[name]] + 1e-9
        for cls, cap in CFG.class_cap.items():
            members = [s for s in w.index if panel.classes[s] == cls]
            assert w[members].sum() <= cap + 1e-9
        assert (w[~features.eligible.iloc[t]] == 0).all()


def test_predicted_volatility_never_exceeds_the_target(panel, features):
    for t in range(450, len(panel.close), 53):
        assert target_weights(features, CFG, t).diagnostics["predicted_vol"] <= CFG.target_vol + 1e-9


def test_model_has_no_look_ahead(panel):
    """Rewrite the future; nothing the model decided before it may move."""
    cut = 900
    base = compute_features(panel, CFG)
    future = make_panel(seed=99)                        # a completely different continuation
    mixed = make_panel()
    for field in ("open", "high", "low", "close", "close_usd"):
        frame = getattr(mixed, field)
        scale = frame.iloc[cut - 1] / getattr(future, field).iloc[cut - 1]
        frame.iloc[cut:] = (getattr(future, field).iloc[cut:] * scale).to_numpy()
    mixed.dollar_volume.iloc[cut:] = future.dollar_volume.iloc[cut:].to_numpy() * 3.0
    other = compute_features(mixed, CFG)
    for t in (500, 700, cut - 1):
        pd.testing.assert_series_equal(target_weights(base, CFG, t).weights,
                                       target_weights(other, CFG, t).weights)


def test_backtest_equity_is_unaffected_by_later_prices(panel):
    cut = 1000
    base = run_backtest(panel, compute_features(panel, CFG), CFG, capital=10_000.0)
    mixed = make_panel()
    rng = np.random.default_rng(5)
    shock = np.exp(np.cumsum(rng.normal(0.0, 0.05, (len(mixed.close) - cut, mixed.close.shape[1])), axis=0))
    for field in ("open", "high", "low", "close", "close_usd"):
        getattr(mixed, field).iloc[cut:] *= shock
    other = run_backtest(mixed, compute_features(mixed, CFG), CFG, capital=10_000.0)
    pd.testing.assert_series_equal(base.equity.iloc[:cut], other.equity.iloc[:cut])


def test_decisions_are_executed_with_the_configured_delay(panel):
    feats = compute_features(panel, CFG)
    for lag in (0, 1, 3):
        res = run_backtest(panel, feats, CFG, costs=FREE, rules=TradingRules(lag=lag), capital=10_000.0)
        first_trade = res.trades["date"].min()
        first_signal = next(panel.close.index[t] for t in range(len(panel.close))
                            if target_weights(feats, CFG, t).weights.sum() > 0)
        assert (first_trade - first_signal).days >= lag


def test_accounting_identity_and_no_free_money(panel):
    feats = compute_features(panel, CFG)
    capital = 5_000.0
    gross = run_backtest(panel, feats, CFG, costs=FREE, capital=capital)
    net = run_backtest(panel, feats, CFG, capital=capital)
    assert net.costs.sum() > 0
    assert net.equity.iloc[-1] < gross.equity.iloc[-1]
    assert (net.weights.sum(axis=1) <= 1.0 + 1e-6).all()        # never levered
    assert (net.weights.min(axis=1) >= -1e-9).all()             # never short
    # equity = cash + positions: cash can be recovered and must never go negative
    cash_share = 1.0 - net.weights.sum(axis=1)
    assert (cash_share >= -1e-6).all()


def test_fees_match_a_hand_calculation():
    """One buy of a known size: the charge must be exactly rate x notional + FX."""
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    flat = pd.DataFrame({"SPY": 100.0}, index=idx)
    from hermes_quant.data.panel import Panel
    from hermes_quant.model import Features

    panel = Panel(flat, flat, flat, flat, flat, flat * np.nan, flat > 0,
                  pd.Series(1.0, index=idx), {"SPY": "equity"}, pd.DataFrame(index=idx))
    ones = pd.DataFrame(1.0, index=idx, columns=["SPY"])
    feats = Features(flat * 0.0, ones * 0.15, ones, ones, ones, flat > 0,
                     pd.DataFrame(index=idx), {"SPY": "equity"})
    cfg = ModelConfig(regime=False, max_weight={"equity": 0.5}, target_vol=1.0)
    costs = CostModel(fee={"equity": 0.001}, slippage={"equity": 0.0005}, fx_fee=0.002)
    rules = TradingRules(lag=0, rebalance_every=1000, edge_fraction=0.0)
    res = run_backtest(panel, feats, cfg, costs=costs, rules=rules, capital=1000.0)
    assert len(res.trades) == 1
    notional = res.trades["notional"].iloc[0]
    assert notional == pytest.approx(500.0)
    assert res.costs.sum() == pytest.approx(500.0 * (0.0015 + 0.002))
    assert res.equity.iloc[-1] == pytest.approx(1000.0 - 500.0 * 0.0035)


def test_small_tickets_are_skipped_and_delisted_assets_are_sold(panel):
    feats = compute_features(panel, CFG)
    tiny = run_backtest(panel, feats, CFG, capital=30.0)
    assert (tiny.trades.loc[tiny.trades["reason"] == "rebalance", "notional"].abs() >= 1.0).all()
    dead = make_panel()
    for field in ("open", "high", "low", "close", "close_usd"):
        getattr(dead, field).loc[dead.close.index[1100]:, "BTC"] = np.nan
    res = run_backtest(dead, compute_features(dead, CFG), CFG, capital=10_000.0)
    assert res.weights["BTC"].iloc[1110:].abs().max() == 0.0
    assert np.isfinite(res.equity).all()


def test_static_mix_matches_closed_form(panel):
    eq = static_mix(panel, {"BTC": 1.0}, str(panel.close.index[0].date()))
    expected = panel.close["BTC"].iloc[-1] / panel.close["BTC"].iloc[0]
    assert eq.iloc[-1] == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------------ statistics

def test_summary_on_a_known_curve():
    idx = pd.date_range("2020-01-01", periods=731, freq="D")
    equity = pd.Series(100.0 * 1.0005 ** np.arange(731), index=idx)
    s = summary(equity)
    assert s["cagr"] == pytest.approx(1.0005**365 - 1.0, rel=1e-6)
    assert s["max_drawdown"] == 0.0


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1500)
    one = stats.deflated_sharpe(r, 1, 0.05)["dsr"]
    many = stats.deflated_sharpe(r, 200, 0.05)["dsr"]
    assert one > many
    assert stats.expected_max_sharpe(200, 0.05) > stats.expected_max_sharpe(10, 0.05) > 0


def test_probabilistic_sharpe_extremes():
    rng = np.random.default_rng(1)
    assert stats.probabilistic_sharpe(rng.normal(0.002, 0.01, 2000)) > 0.99
    assert 0.02 < stats.probabilistic_sharpe(rng.normal(0.0, 0.01, 2000)) < 0.98


def test_bootstrap_interval_has_the_right_coverage():
    """A 95% interval should contain the truth about 95% of the time: check the
    interval itself, over many independent samples, rather than one lucky draw."""
    true_sr = 0.05 * np.sqrt(252)
    covered = 0
    for seed in range(40):
        r = np.random.default_rng(1000 + seed).normal(0.0005, 0.01, 2500)
        ci = stats.bootstrap_ci(r, 252, n_boot=200, seed=seed)
        covered += ci["sharpe"]["p2.5"] < true_sr < ci["sharpe"]["p97.5"]
    assert covered >= 34                # >= 85% of 40, leaving room for sampling noise


def test_hac_regression_recovers_beta():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.02, 1500)
    y = 0.001 + 0.6 * x + rng.normal(0.0, 0.01, 1500)
    fit = stats.ols_hac(y, x)
    assert fit["coef"][1] == pytest.approx(0.6, abs=0.05)
    assert fit["t"][1] > 20


def test_pbo_is_high_for_noise_and_low_for_a_real_edge():
    rng = np.random.default_rng(4)
    noise = pd.DataFrame(rng.normal(0.0, 0.01, (2000, 30)))
    assert stats.probability_of_backtest_overfitting(noise, 8)["pbo"] > 0.25
    skilled = noise.copy()
    skilled[0] += 0.002                 # one variant with a genuine, persistent edge
    assert stats.probability_of_backtest_overfitting(skilled, 8)["pbo"] < 0.1
