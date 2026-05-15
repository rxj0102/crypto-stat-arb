# crypto-stat-arb

A research-grade statistical arbitrage platform for cryptocurrencies,
focused on discovering and backtesting momentum and reversal strategies.

## Overview

The platform implements a complete quantitative research pipeline:

1. **Data acquisition** — fetch and store OHLCV data from crypto exchanges via ccxt
2. **Signal generation** — momentum, reversal, pairs, seasonality, activity, and theme signals
3. **Backtesting** — unconstrained cross-sectional backtesting engine with realistic costs
4. **Strategy combination** — IC-weighted, Sharpe-weighted, and minimum-correlation blending
5. **Performance evaluation** — Sharpe ratio, drawdown, alpha/beta decomposition, and plots

**Core thesis**: Crypto markets exhibit exploitable momentum patterns at medium horizons
(3–12 months) and mean-reversion at short horizons (1–5 days), especially when conditioned
on trading activity. Dollar-neutral long-short portfolios across 50+ liquid assets can
deliver Sharpe ratios of 1–2 net of realistic transaction costs.

## Repository Structure

```
crypto-stat-arb/
├── statarb/                  # Core library
│   ├── data/                 # Fetching, storage, universe, features
│   ├── signals/              # Momentum, reversal, pairs, seasonality, activity, themes
│   ├── backtest/             # Engine, execution, positions, weighting
│   └── evaluation/           # Metrics, risk, reporting, plots
├── experiments/              # Runnable research scripts + config.yaml
├── notebooks/                # Jupyter notebooks for interactive analysis
├── tests/                    # pytest test suite
├── docs/                     # Research notes and strategy summary
└── data/                     # Data directory (raw + processed, gitignored)
```

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/rxj0102/crypto-stat-arb.git
cd crypto-stat-arb

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Quickstart

### 1. Fetch data

```bash
# Fetch daily data for the default ~50-asset universe (2020-01-01 → today)
python experiments/run_data_fetch.py

# Or fetch specific symbols / timeframes
python experiments/run_data_fetch.py --timeframe 1h --symbols BTC/USDT ETH/USDT SOL/USDT
```

### 2. Run signal research

```bash
python experiments/run_momentum_research.py
python experiments/run_reversal_research.py
python experiments/run_pairs_research.py
```

### 3. Run combined strategy

```bash
python experiments/run_combined_strategy.py
```

### 4. Full evaluation with plots

```bash
python experiments/run_full_evaluation.py
# Outputs saved to output/plots/ and output/reports/
```

### 5. Programmatic usage

```python
from statarb.data.fetcher import CryptoDataFetcher
from statarb.data.storage import DataStore
from statarb.data.features import FeatureEngine
from statarb.signals.momentum import MomentumSignals
from statarb.backtest.engine import BacktestEngine
from statarb.evaluation.metrics import PerformanceMetrics

# Load data
store = DataStore()
prices = store.build_panel(symbols, exchange='binance', timeframe='1d')

# Generate and rank signal
fe = FeatureEngine()
returns = fe.log_returns(prices)
mom = MomentumSignals()
signal = fe.cross_sectional_rank(mom.momentum_6_1(returns))

# Run backtest
engine = BacktestEngine()
result = engine.run(signal, prices)

# Evaluate
pm = PerformanceMetrics(result.portfolio_returns)
print(pm.summary())
```

## Signal Library

| Module | Signals |
|--------|---------|
| `signals.momentum` | Price momentum (12-1, 6-1, 3-1, 1W), MA crossover, TS momentum, Sharpe momentum, vol-conditioned momentum, residual momentum |
| `signals.reversal` | Short-term (1D, 5D, 10D), vol-adjusted, Bollinger Band, MA distance, high-volume reversal, large-move reversal, medium-term reversal |
| `signals.pairs` | Spread z-score, relative value, cointegration scanner, cluster momentum |
| `signals.seasonality` | Day-of-week, month-of-year, January effect, Monday reversal, intraday (hourly), month-end indicator |
| `signals.activity` | Volume regime, volume z-score, activity scaling/gating, Amihud illiquidity |
| `signals.themes` | Theme momentum, within-theme relative value, low-vol factor, size factor, momentum-reversal blend |

## Execution Cost Model

| Order Type | One-way Cost | Round-trip |
|------------|-------------|------------|
| Market orders | 10 bps | 20 bps |
| Limit orders | 3.5 bps | 7 bps |

Additional market impact modelled as a function of trade size vs ADTV.

## Configuration

All experiment parameters are in `experiments/config.yaml`. Key settings:

```yaml
backtest:
  rebalance_freq: daily        # daily | weekly | monthly
  position_method: rank        # rank | signal | top_bottom | vol_target
  max_position: 0.10           # max ±10% per asset

execution:
  order_type: market           # market (20 bps) | limit (7 bps)

combination:
  method: equal                # equal | ic_weighted | sharpe_weighted | min_corr
```

## Running Tests

```bash
pytest tests/          # run all tests
pytest tests/ -v       # verbose output
pytest tests/test_signals.py  # specific module
```

## Research References

- Jegadeesh & Titman (1993) — Cross-sectional momentum
- Moskowitz, Ooi & Pedersen (2012) — Time-series momentum
- De Bondt & Thaler (1985) — Long-run reversal
- Amihud (2002) — Illiquidity ratio
- Liu, Tsyvinski & Wu (2022) — Three-factor model for crypto
- Grobys & Sapkota (2019) — Momentum in crypto markets

## License

MIT — see [LICENSE](LICENSE).
