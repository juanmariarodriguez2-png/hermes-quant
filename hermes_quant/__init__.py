"""Hermes Quant — a systematic, multi-asset trading research and execution engine.

Layers
------
data        market data loaders (Kraken, Binance public archive, Yahoo)
filters     noise filters: Kalman trend, range-based volatility, random-matrix cleaning
regime      hidden Markov regimes, turbulence and absorption-ratio fragility gauges
signals     time-series momentum ensemble, derivatives carry
portfolio   hierarchical risk parity, volatility targeting, constraints
backtest    causal event-driven simulator with realistic costs + statistics
live        daily cloud runner, paper broker, Kraken execution adapters
"""

__version__ = "2.0.0"
