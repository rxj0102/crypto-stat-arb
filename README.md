# crypto-stat-arb

A research-grade statistical arbitrage platform for cryptocurrencies —
cross-sectional momentum and reversal strategies, backtested on a universe of
liquid crypto assets with realistic execution costs.

## Overview

The platform implements a complete quantitative research pipeline:

1. **Data acquisition** — fetch daily OHLCV from Binance (or any ccxt exchange); synthetic fallback for offline use
2. **Signal generation** — momentum (7 variants), reversal (8 variants), pairs, seasonality, activity-gated signals
3. **Backtesting** — dollar-neutral `UnconstrainedBacktest` with realistic costs (market/limit orders)
4. **Strategy combination** — equal weight, inverse vol, Sharpe-weighted, and min-variance blending
5. **Performance evaluation** — Sharpe, Sortino, Calmar, alpha/beta, factor exposure, drawdown episodes
6. **Interactive notebooks** — 7 self-contained Jupyter notebooks covering the full research cycle

**Core thesis**: Crypto markets exhibit exploitable momentum at medium horizons (3–6 months) and
mean-reversion at short horizons (1–5 days), especially when conditioned on trading activity.
Dollar-neutral long-short portfolios targeting 50+ liquid assets can deliver Sharpe ratios of 1–2
net of realistic transaction costs.

---

## Repository Structure

```
crypto-stat-arb/
├── statarb/                        # Core library
│   ├── data/                       # Fetching, storage, universe, features
│   ├── signals/                    # 6 signal modules (momentum, reversal, pairs…)
│   ├── backtest/                   # UnconstrainedBacktest, ExecutionModel, weighting
│   └── evaluation/                 # PerformanceMetrics, RiskAnalytics, plots, reporting
├── experiments/
│   ├── config.yaml                 # Central configuration
│   ├── generate_synthetic_data.py  # Synthetic OHLCV generator (API-free testing)
│   ├── run_data_fetch.py           # Live data fetch + synthetic fallback
│   ├── run_momentum_research.py    # Momentum signal sweep
│   ├── run_reversal_research.py    # Reversal signal sweep
│   ├── run_pairs_research.py       # Pairs / cointegration research
│   ├── run_combined_strategy.py    # Signal combination + diversification
│   ├── run_full_evaluation.py      # Final strategy evaluation + all plots
│   └── results/                    # CSV + PNG outputs (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_momentum_signals.ipynb
│   ├── 03_reversal_signals.ipynb
│   ├── 04_pairs_and_correlation.ipynb
│   ├── 05_seasonality.ipynb
│   ├── 06_combined_strategy.ipynb
│   └── 07_final_report.ipynb
├── tests/                          # 325 pytest tests
├── docs/
│   ├── research_notes.md           # Signal development log + findings
│   └── strategy_summary.md         # Strategy description + performance targets
└── data/                           # Data directory (raw + processed, gitignored)
```

---

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/rxj0102/crypto-stat-arb.git
cd crypto-stat-arb

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

---

## Quickstart

### 1. Get data

```bash
# Option A: fetch live data from Binance (2020-01-01 → today)
python experiments/run_data_fetch.py

# Option B: generate synthetic data instantly (no API required)
python experiments/run_data_fetch.py --synthetic

# Option C: generate synthetic data directly
python experiments/generate_synthetic_data.py --n-assets 20 --n-days 1000
```

### 2. Run signal research

```bash
# Momentum: test all lookbacks [7,14,21,42,63,126d], skip params, signal variants
python experiments/run_momentum_research.py

# Reversal: test lookbacks [1,2,3,5,7d], vol-filtered, beta-neutral variants
python experiments/run_reversal_research.py

# Pairs: cointegration scan, spread backtest, cluster momentum
python experiments/run_pairs_research.py
```

### 3. Combine and evaluate

```bash
# Compare equal/inverse-vol/Sharpe/min-variance combination
python experiments/run_combined_strategy.py

# Full evaluation: all metrics, alpha/beta, drawdown, factor exposure, all plots
python experiments/run_full_evaluation.py
# → experiments/results/full_eval/
```

### 4. Interactive notebooks

```bash
jupyter lab notebooks/
# Start with 01_data_exploration.ipynb, each notebook is self-contained
```

### 5. Run tests

```bash
pytest tests/ -v          # 325 tests
pytest tests/ -q          # quiet
pytest tests/test_metrics.py    # specific module
```

---

## Programmatic Usage

```python
from statarb.data.storage import DataStore
from statarb.data.features import FeatureEngine
from statarb.signals.momentum import MomentumSignals
from statarb.backtest.engine import UnconstrainedBacktest
from statarb.evaluation.metrics import PerformanceMetrics

# Load data
store = DataStore()
prices = store.build_panel(symbols, exchange="binance", timeframe="1d")
volume = store.build_volume_panel(symbols, exchange="binance", timeframe="1d")

# Generate signal
fe = FeatureEngine()
returns = fe.log_returns(prices)
mom = MomentumSignals()
signal = fe.cross_sectional_rank(mom.time_series_momentum(returns, lookback=63))

# Run backtest
bt = UnconstrainedBacktest(returns, signal, dollar_neutral=True)
result = bt.run()

# Evaluate
pm = PerformanceMetrics()
print(f"Sharpe: {pm.sharpe_ratio(result['returns'], periods_per_year=365):.2f}")
print(f"MDD:    {pm.max_drawdown(result['returns'])*100:.1f}%")

# Full report
report = pm.full_report(result["returns"])
print(report)
```

---

## Signal Library

| Module | Signals Available |
|--------|------------------|
| `signals.momentum` | `time_series_momentum`, `cross_sectional_momentum`, `volume_weighted_momentum`, `momentum_with_activity_filter`, `breakout_momentum`, `acceleration`, `momentum_12_1`, `momentum_6_1`, `momentum_3_1`, `momentum_1w`, `sharpe_momentum`, `moving_average_crossover` |
| `signals.reversal` | `short_term_reversal`, `mean_reversion_zscore`, `volume_filtered_reversal`, `liquidation_reversal`, `weekly_reversal`, `biweekly_reversal`, `vol_adjusted_reversal`, `bollinger_reversal`, `large_move_reversal` |
| `signals.pairs` | `spread_zscore`, `find_cointegrated_pairs`, `relative_value_signal`, `cluster_momentum_signal` |
| `signals.seasonality` | `day_of_week_signal`, `month_of_year_signal`, `january_effect`, `monday_reversal`, `intraday_signal`, `month_end_indicator` |
| `signals.activity` | `volume_surprise`, `return_volume_interaction`, `volatility_regime`, `volume_regime`, `volume_zscore`, `activity_scale`, `activity_gate`, `market_activity_index`, `amihud_illiquidity` |
| `signals.themes` | Thematic cluster signals + `CRYPTO_THEMES` dict |

---

## Execution Cost Model

| Order Type | Round-trip Cost | When to Use |
|------------|----------------|-------------|
| Market orders | 20 bps (2 × 10 bps) | Urgent fills; large moves |
| Limit orders | 7 bps (2 × 3.5 bps) | Default for reversal strategies |

Configured in `experiments/config.yaml`:
```yaml
execution:
  order_type: market        # market | limit
  market_order_cost: 0.0020
  limit_order_cost: 0.0007
```

---

## Configuration Reference

```yaml
# experiments/config.yaml (key settings)
data:
  exchange: binance
  start_date: "2020-01-01"

universe:
  min_dollar_volume: 10_000_000  # $10M ADTV
  min_history_days: 365

signals:
  momentum:
    lookbacks: [7, 14, 21, 42, 63, 126]
    skip_days: [0, 1, 2, 3]
  reversal:
    lookbacks: [1, 2, 3, 5, 7]

backtest:
  max_position: 0.10        # ±10% per asset
  rebalance_freq: daily

evaluation:
  periods_per_year: 365     # crypto is 24/7

combination:
  method: equal             # equal | inverse_vol | sharpe | min_variance
  estimation_window: 63

synthetic:
  n_assets: 20
  n_days: 1000
  seed: 42
```

---

## Research References

- Jegadeesh & Titman (1993) — Cross-sectional momentum in equities
- Moskowitz, Ooi & Pedersen (2012) — Time-series momentum
- De Bondt & Thaler (1985) — Long-run reversal
- Amihud (2002) — Illiquidity ratio
- Liu, Tsyvinski & Wu (2022) — Three-factor model for crypto markets
- Grobys & Sapkota (2019) — Momentum effects in crypto markets

---

## License

MIT — see [LICENSE](LICENSE).
