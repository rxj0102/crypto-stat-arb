# Strategy Summary — crypto-stat-arb

## Overview

**crypto-stat-arb** is a cross-sectional statistical arbitrage strategy for cryptocurrency
markets. It combines momentum and reversal signals across a universe of liquid crypto assets,
executing a dollar-neutral long-short portfolio with daily rebalancing.

The strategy is designed for research and backtesting. It requires exchange API access
(Binance by default) for live data, or can run on synthetic data for pipeline validation.

---

## Strategy Design

### Universe

- **Size**: ~50 liquid crypto assets vs USDT on Binance (configurable)
- **Minimum volume**: $10M average daily dollar volume (30-day rolling)
- **Minimum history**: 365 days
- **Excludes**: stablecoins; wrapped tokens optional
- **Dynamic**: eligibility re-evaluated at each rebalancing date

### Signal Stack (Production Composite)

| Signal | Weight | Horizon | Type |
|--------|--------|---------|------|
| 6-month price momentum (skip 1M) | 30% | Medium | Momentum |
| 3-month price momentum (skip 1M) | 20% | Short-medium | Momentum |
| Sharpe-weighted momentum (63D) | 20% | Short-medium | Momentum + Risk |
| 5-day vol-adjusted reversal | 20% | Short | Reversal |
| Volume-weighted momentum (63D) | 10% | Medium | Momentum + Activity |

**Pre-processing pipeline** (applied to every signal):
1. Cross-sectionally rank at each date → values in [-1, +1]
2. Activity gate: set to 0 where volume/MA < 50% (low-activity filter)
3. Combine via equal-weight (or IC-weighted in research mode)

### Portfolio Construction

- **Dollar-neutral**: long top signal quartile, short bottom quartile
- **Position sizing**: proportional to ranked signal magnitude
- **Maximum weight**: ±10% per asset (before normalization)
- **Gross exposure**: $1 long + $1 short = $2 gross per $1 capital
- **Net market exposure**: ≈ 0 (dollar-neutral by construction)

### Execution

| Mode | Cost Assumption | When to Use |
|------|----------------|-------------|
| Limit orders | 7 bps round-trip (3.5 bps/side) | Live trading (preferred) |
| Market orders | 20 bps round-trip (10 bps/side) | Large size, urgent fills |

- Rebalance: daily at close prices
- Implementation shortfall: modelled via turnover × cost_rate
- No additional market impact model in base configuration

### Risk Management

- **Position cap**: max ±10% per asset
- **Activity gate**: no signal where V/MA < 50%
- **Drawdown monitoring**: reduce exposure if composite drawdown > 15% (manual)
- **Beta target**: dollar-neutral construction keeps BTC beta near zero

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Annualized return (net) | 15–25% | Achievable with Sharpe 1–2 at 10–15% vol |
| Annualized volatility | 10–15% | Dollar-neutral, diversified |
| Sharpe ratio (net) | 1.0–2.0 | Competitive with quant funds |
| Maximum drawdown | < 20% | Limits funding risk |
| Daily turnover (one-way) | 5–15% | Balances responsiveness vs costs |
| Correlation to BTC | < 0.3 | Dollar-neutral construction |

---

## Combination Methods Available

| Method | Class | Lookback | Best For |
|--------|-------|----------|----------|
| Equal weight | `StrategyWeighting` | N/A | Baseline; most robust |
| Inverse volatility | `StrategyWeighting` | 63d | Risk parity; reduces realized vol |
| Sharpe-weighted | `StrategyWeighting` | 63d | Adaptive allocation |
| Min variance | `StrategyWeighting` | 63d | Exploit low correlations |

---

## Configuration

All parameters are in `experiments/config.yaml`:

```yaml
signals:
  momentum:
    lookbacks: [7, 14, 21, 42, 63, 126]
    skip_days: [0, 1, 2, 3]
    default_lookback: 63
    default_skip: 1
  reversal:
    lookbacks: [1, 2, 3, 5, 7]
    default_lookback: 5

backtest:
  max_position: 0.10
  rebalance_freq: daily

execution:
  order_type: market
  market_order_cost: 0.0020   # 20 bps round-trip
  limit_order_cost: 0.0007    # 7 bps round-trip

evaluation:
  periods_per_year: 365       # crypto is 24/7
```

---

## Key Dependencies and Risks

- **Liquidity risk**: altcoin liquidity dries up in bear markets → costs spike beyond model
- **Correlation collapse**: all assets go to 1.0 in crashes → no diversification benefit
- **Strategy crowding**: as crypto matures, stat-arb alpha may compress
- **Data quality**: exchange data gaps, outliers from delistings / hacks need cleaning
- **Regime sensitivity**: momentum works in trending markets, reversal in choppy/ranging
- **Synthetic data**: the pipeline is validated on synthetic data; real-world performance
  requires real data and out-of-sample validation

---

## Implementation Notes

- Signal look-ahead lag: 1 day (signal computed on day t, position entered on t+1)
- Data lag: prices available at EOD; signals use prior-day close
- Rebalancing: at daily close prices (live: use VWAP or MOC limit orders)
- Costs applied at portfolio level (turnover × cost_rate per period)
- `periods_per_year=365` throughout: crypto trades continuously including weekends

---

## Quick Reproduction

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Generate synthetic data (or fetch live: remove --synthetic)
python experiments/run_data_fetch.py --synthetic

# 3. Research individual signals
python experiments/run_momentum_research.py
python experiments/run_reversal_research.py
python experiments/run_pairs_research.py

# 4. Combine and evaluate
python experiments/run_combined_strategy.py
python experiments/run_full_evaluation.py

# 5. Interactive analysis
jupyter lab notebooks/
```

Results are saved to `experiments/results/` as CSV files and PNG plots.
