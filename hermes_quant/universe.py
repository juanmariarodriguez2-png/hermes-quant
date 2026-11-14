"""Instrument definitions: what the model studies, and what it can actually trade.

Two universes on purpose:

* RESEARCH — long histories from public archives, *including coins that later
  collapsed or faded*, so the backtest cannot quietly benefit from hindsight.
* LIVE — the exact instruments tradable from one Kraken account: spot crypto in
  EUR, tokenized stocks/ETFs (xStocks) in USD, and perpetual futures.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- research

# Every coin that sat in the top ~20 by market value at some point in 2017-2026,
# winners and losers alike (EOS, NEO, IOTA, XEM, LUNA, FTT, ICP ... are the point).
# Selection *within* this list is done point-in-time by trailing traded volume.
CRYPTO_CANDIDATES = [
    "BTC", "ETH", "XRP", "LTC", "ADA", "XLM", "EOS", "NEO", "IOTA", "DASH", "XMR",
    "TRX", "ETC", "XEM", "VET", "BNB", "LINK", "XTZ", "ATOM", "DOT", "UNI", "SOL",
    "AVAX", "LUNA", "MATIC", "DOGE", "SHIB", "FTT", "ALGO", "ICP", "FIL", "NEAR",
    "APT", "ARB", "OP", "TON", "SUI", "HBAR", "ZEC", "AAVE", "PEPE", "INJ", "BCH",
    "TAO", "WIF", "ENA", "ONDO", "RENDER",
]

# Stablecoins, wrapped and leveraged tokens are never tradable "assets".
CRYPTO_EXCLUDE = {"USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "PAX", "EUR", "WBTC", "UST"}

# US-listed ETFs that also exist as Kraken xStocks. Broad asset-class funds only:
# picking today's popular single stocks would be hindsight in disguise.
ETF_RESEARCH = {
    "SPY": "US large caps (S&P 500)",
    "QQQ": "US growth / Nasdaq 100",
    "IWM": "US small caps (Russell 2000)",
    "VGK": "European equities",
    "IEMG": "Emerging-market equities",
    "GLD": "Gold",
    "SLV": "Silver",
    "XLE": "Energy equities",
}

# Reference series (benchmarks / diagnostics; not traded by the default portfolio).
REFERENCE = ["^GSPC", "TLT", "IEF", "BIL", "SGOV", "EURUSD=X"]

ASSET_CLASS = {
    **{k: "equity" for k in ("SPY", "QQQ", "IWM", "VGK", "IEMG")},
    **{k: "real" for k in ("GLD", "SLV", "XLE")},
}


def asset_class(symbol: str) -> str:
    """'equity', 'real' (commodities / real assets) or 'crypto'."""
    return ASSET_CLASS.get(symbol, "crypto")


# --------------------------------------------------------------------------- live


@dataclass(frozen=True)
class Instrument:
    """One tradable market on Kraken."""

    symbol: str            # model-side name, e.g. "BTC" or "SPY"
    kraken_pair: str       # REST pair name, e.g. "XBTEUR" or "SPYxUSD"
    asset_class: str       # crypto | equity | real
    quote: str             # EUR | USD
    tokenized: bool = False  # xStocks need asset_class=tokenized_asset on the API
    perp: str | None = None  # matching Kraken Futures perpetual, if any


LIVE_INSTRUMENTS: list[Instrument] = [
    # Spot crypto, quoted in EUR (no FX leg for a euro account).
    Instrument("BTC", "XBTEUR", "crypto", "EUR", perp="PF_XBTUSD"),
    Instrument("ETH", "ETHEUR", "crypto", "EUR", perp="PF_ETHUSD"),
    Instrument("SOL", "SOLEUR", "crypto", "EUR", perp="PF_SOLUSD"),
    Instrument("XRP", "XRPEUR", "crypto", "EUR", perp="PF_XRPUSD"),
    Instrument("LINK", "LINKEUR", "crypto", "EUR", perp="PF_LINKUSD"),
    Instrument("ADA", "ADAEUR", "crypto", "EUR", perp="PF_ADAUSD"),
    Instrument("DOGE", "XDGEUR", "crypto", "EUR", perp="PF_DOGEUSD"),
    Instrument("LTC", "LTCEUR", "crypto", "EUR", perp="PF_LTCUSD"),
    Instrument("AVAX", "AVAXEUR", "crypto", "EUR", perp="PF_AVAXUSD"),
    Instrument("BNB", "BNBEUR", "crypto", "EUR"),
    Instrument("TRX", "TRXEUR", "crypto", "EUR"),
    Instrument("BCH", "BCHEUR", "crypto", "EUR"),
    # Tokenized ETFs (xStocks), quoted in USD.
    Instrument("SPY", "SPYxUSD", "equity", "USD", tokenized=True, perp="PF_SPXUSD"),
    Instrument("QQQ", "QQQxUSD", "equity", "USD", tokenized=True, perp="PF_US100USD"),
    Instrument("IWM", "IWMxUSD", "equity", "USD", tokenized=True),
    Instrument("VGK", "VGKxUSD", "equity", "USD", tokenized=True),
    Instrument("IEMG", "IEMGxUSD", "equity", "USD", tokenized=True),
    Instrument("GLD", "GLDxUSD", "real", "USD", tokenized=True, perp="PF_XAUUSD"),
    Instrument("SLV", "SLVxUSD", "real", "USD", tokenized=True),
    Instrument("XLE", "XLExUSD", "real", "USD", tokenized=True),
]

LIVE_BY_SYMBOL = {i.symbol: i for i in LIVE_INSTRUMENTS}
