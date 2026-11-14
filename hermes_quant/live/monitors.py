"""Things the model watches but does not trade.

* Stock radar: the trend filter applied to the most liquid single-stock tokens on
  Kraken. Informational only: there is no survivorship-free history on which a
  stock-picking rule could be validated, so none is traded.
* Derivatives monitor: what a perpetual-futures position would cost or pay right now.
  The research (research/07_derivatives.py) found that funding makes perpetuals more
  expensive than spot for long positions, and that the delta-neutral carry only pays
  when funding is high. This shows where funding stands today.

Both are best-effort: any failure returns an empty list and never blocks trading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import kraken
from ..filters import kalman, volatility

STOCK_NAMES = {
    "NVDA": "Nvidia", "TSLA": "Tesla", "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta", "AMD": "AMD", "INTC": "Intel", "AVGO": "Broadcom",
    "NFLX": "Netflix", "ORCL": "Oracle", "PLTR": "Palantir", "MSTR": "Strategy", "COIN": "Coinbase",
    "HOOD": "Robinhood", "TSM": "TSMC", "ASML": "ASML", "MU": "Micron", "MRVL": "Marvell",
}
# Note: Kraken's PF_SPXUSD is the SPX6900 meme coin, not the S&P 500. The equity-index
# perpetuals are the ones written on the tokenized funds (SPYx, QQQx).
PERPETUALS = {
    "PF_XBTUSD": "Bitcoin", "PF_ETHUSD": "Ethereum", "PF_SOLUSD": "Solana",
    "PF_SPYXUSD": "S&P 500", "PF_QQQXUSD": "Nasdaq 100", "PF_XAUUSD": "Gold",
}
CARRY_THRESHOLD = 0.06      # annual funding above which the delta-neutral carry was worth doing


def stock_radar(symbols: list[str]) -> list[dict]:
    rows = []
    for symbol in symbols:
        try:
            df = kraken.ohlc_daily(f"{symbol}xUSD", tokenized=True)
        except Exception:  # noqa: BLE001
            continue
        if len(df) < 150:
            continue
        index = pd.date_range(df.index[0], df.index[-1], freq="D")
        df = df.reindex(index)
        session = pd.DataFrame({symbol: (index.dayofweek < 5) & df["close"].notna().to_numpy()}, index=index)
        close = pd.DataFrame({symbol: df["close"].where(session[symbol]).ffill()})
        flat = lambda f: pd.DataFrame({symbol: df[f].where(session[symbol], close[symbol])})  # noqa: E731
        vol = volatility.ewma_vol(volatility.daily_variance_proxy(flat("open"), flat("high"), flat("low"), close))
        returns, spans = kalman.session_returns(close, session)
        prob = kalman.trend_probability(returns, vol, spans=spans)["prob"][symbol]
        weekdays = df[index.dayofweek < 5].tail(30)
        last = close[symbol].iloc[-1]
        rows.append({
            "symbol": symbol, "name": STOCK_NAMES.get(symbol, symbol),
            "prob_up": round(float(prob.iloc[-1]), 4), "vol": round(float(vol[symbol].iloc[-1]), 4),
            "ret_30d": round(float(last / close[symbol].iloc[-31] - 1.0), 4),
            "ret_180d": round(float(last / close[symbol].iloc[-181] - 1.0), 4) if len(close) > 181 else None,
            "daily_usd": round(float((weekdays["volume"] * weekdays["close"]).median()), 0),
            "history_days": int(len(df)),
        })
    return sorted(rows, key=lambda r: -r["prob_up"])


def derivatives_monitor() -> list[dict]:
    try:
        tickers = {t["symbol"]: t for t in kraken.futures_tickers()}
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for symbol, name in PERPETUALS.items():
        t = tickers.get(symbol)
        if not t or not t.get("markPrice"):
            continue
        # Kraken quotes funding in dollars per unit per hour; divide by price for a rate.
        hourly = float(t.get("fundingRate") or 0.0) / float(t["markPrice"])
        annual = hourly * 24 * 365
        if not np.isfinite(annual):
            continue
        rows.append({"symbol": symbol, "name": name, "mark": float(t["markPrice"]),
                     "funding_annual": round(annual, 4),
                     "volume_24h_usd": round(float(t.get("volumeQuote") or 0.0), 0),
                     "carry_attractive": bool(annual > CARRY_THRESHOLD)})
    return rows
