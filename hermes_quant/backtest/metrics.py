"""Performance and risk metrics computed from a daily equity curve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = 365  # calendar-day equity curves
_ECB = Path(__file__).resolve().parents[1] / "data" / "ecb_deposit_rate.csv"


def euro_cash_rate(index: pd.DatetimeIndex) -> pd.Series:
    """Daily euro risk-free return: the ECB deposit facility rate (a step function,
    public ECB data bundled with the package), accrued over 365 days."""
    steps = pd.read_csv(_ECB, index_col="effective_from", parse_dates=True)["rate_pct"]
    annual = steps.reindex(steps.index.union(index)).ffill().reindex(index).bfill() / 100.0
    return annual / 365.0


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def summary(equity: pd.Series, periods: int = PERIODS) -> dict:
    """Headline statistics.

    Two Sharpe ratios are reported. ``sharpe`` measures returns against idle cash on
    the exchange, which earns nothing. ``sharpe_excess`` subtracts the euro
    risk-free rate (ECB deposit facility), the textbook definition: it is lower
    when rates are positive and higher when they were negative (2014-2022)."""
    equity = equity.dropna()
    r = equity.pct_change().dropna()
    if len(r) < 2:
        return {}
    years = len(r) / periods
    total = equity.iloc[-1] / equity.iloc[0]
    cagr = total ** (1.0 / years) - 1.0
    vol = r.std(ddof=1) * np.sqrt(periods)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(periods) if r.std(ddof=1) > 0 else np.nan
    excess = r - euro_cash_rate(r.index) if periods == PERIODS else r
    sharpe_excess = excess.mean() / excess.std(ddof=1) * np.sqrt(periods) if excess.std(ddof=1) > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(periods)
    dd = drawdown(equity)
    max_dd = float(dd.min())
    underwater = (dd < 0).astype(int)
    longest = int(underwater.groupby((underwater == 0).cumsum()).sum().max())
    monthly = equity.resample("ME").last().pct_change().dropna()
    var95 = float(np.quantile(r, 0.05))
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "years": round(years, 2),
        "total_return": float(total - 1.0),
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),                    # versus idle exchange cash (0%)
        "sharpe_excess": float(sharpe_excess),      # versus the euro risk-free rate
        "sortino": float(r.mean() * periods / downside) if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "longest_underwater_days": longest,
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurt()),
        "var_95_daily": var95,
        "cvar_95_daily": float(r[r <= var95].mean()),
        "best_month": float(monthly.max()) if len(monthly) else np.nan,
        "worst_month": float(monthly.min()) if len(monthly) else np.nan,
        "positive_months": float((monthly > 0).mean()) if len(monthly) else np.nan,
    }


def yearly_returns(equity: pd.Series) -> pd.Series:
    yearly = equity.resample("YE").last()
    first = pd.Series([equity.iloc[0]], index=[equity.index[0] - pd.Timedelta(days=1)])
    return pd.concat([first, yearly]).pct_change().dropna().rename(lambda d: d.year)


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    monthly = equity.resample("ME").last().pct_change().dropna()
    frame = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    return frame.pivot(index="year", columns="month", values="ret")
