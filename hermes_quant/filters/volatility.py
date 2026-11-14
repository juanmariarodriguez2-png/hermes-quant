"""Volatility estimation that uses the whole candle, and survives bad prints.

A daily close-to-close return is one noisy draw. The high-low range of the same day
summarises the entire intraday path, and Garman & Klass (1980) showed that a
range-based variance estimate is several times more efficient. Two real-world
problems have to be handled before it can be trusted:

* Bad prints. A single flash-crash wick (Coinbase printed BTC at $0.06 in April
  2017) makes one day's range look like a 700% move and poisons an exponential
  average for months. The range term is therefore winsorised against the asset's
  own recent typical variance.
* Structural bias. Intraday ranges overstate the close-to-close risk a daily
  investor actually bears whenever prices mean-revert within the day (very common
  in crypto). The range estimate is rescaled, causally, so that its trailing
  average matches the close-to-close variance: we keep its efficiency, not its bias.

The half-lives (10 and 126 days, 70/30) were chosen by out-of-sample forecast
accuracy (QLIKE loss on next-week realised variance, data to 2021), not by backtest
profit: against a plain 21-day EWMA of squared returns the loss falls about 9% for
crypto and 3% for funds. See research/02_volatility_forecast.py.

Everything here is causal: the value at row t uses data up to and including t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365  # the model runs on calendar days (crypto never closes)
_GK = 2.0 * np.log(2.0) - 1.0


def _winsorise(x: pd.DataFrame, window: int, multiple: float) -> pd.DataFrame:
    """Cap each value at ``multiple`` x the trailing median of the *positive* values
    seen before it (a fund's zero-variance weekend days must not drag the median)."""
    typical = x.where(x > 0).rolling(window, min_periods=window // 3).median().shift(1)
    return x.where(~(x > multiple * typical), multiple * typical)


def daily_variance_proxy(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                         close: pd.DataFrame, robust_window: int = 90, robust_multiple: float = 25.0,
                         calibration_window: int = 365) -> pd.DataFrame:
    """Per-day variance estimate: wick-filtered, bias-corrected Garman-Klass range term
    averaged 50/50 with the squared close-to-close return.

    Only the range term is winsorised. A flash-crash wick lives in the high/low, never
    in the close, so the close-to-close return is left untouched: a genuine crash day
    must hit the risk estimate in full.
    """
    prev_close = close.shift(1)
    close_to_close = np.log(close / prev_close) ** 2
    overnight = np.log(open_ / prev_close) ** 2
    intraday = 0.5 * np.log(high / low) ** 2 - _GK * np.log(close / open_) ** 2
    range_based = _winsorise(overnight + intraday.clip(lower=0.0), robust_window, robust_multiple)

    # causal level calibration: trailing mean(close-to-close) / trailing mean(range)
    num = close_to_close.rolling(calibration_window, min_periods=calibration_window // 4).mean()
    den = range_based.rolling(calibration_window, min_periods=calibration_window // 4).mean()
    ratio = (num / den).shift(1).clip(0.25, 1.5).fillna(1.0)

    proxy = 0.5 * ratio * range_based + 0.5 * close_to_close
    return proxy.where(range_based.notna(), close_to_close)


def ewma_vol(variance_proxy: pd.DataFrame, fast_halflife: float = 10.0,
             slow_halflife: float = 126.0, fast_weight: float = 0.7,
             min_periods: int = 60) -> pd.DataFrame:
    """Annualised volatility forecast: blend of a fast and a slow EWMA of daily variance.

    The slow leg stops the estimate collapsing in eerily calm markets, which is
    exactly when volatility-scaled positions would otherwise balloon.
    """
    fast = variance_proxy.ewm(halflife=fast_halflife, min_periods=min_periods).mean()
    slow = variance_proxy.ewm(halflife=slow_halflife, min_periods=min_periods).mean()
    daily_var = fast_weight * fast + (1.0 - fast_weight) * slow
    return np.sqrt(daily_var * DAYS_PER_YEAR)


def close_to_close_vol(close: pd.DataFrame, halflife: float = 21.0,
                       min_periods: int = 60) -> pd.DataFrame:
    """Classic RiskMetrics-style EWMA volatility (kept as the baseline for ablations)."""
    r2 = np.log(close / close.shift(1)) ** 2
    return np.sqrt(r2.ewm(halflife=halflife, min_periods=min_periods).mean() * DAYS_PER_YEAR)
