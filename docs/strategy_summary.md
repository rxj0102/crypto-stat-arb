# Strategy Summary

## Overview

**crypto-stat-arb** is a cross-sectional statistical arbitrage strategy for
cryptocurrency markets. It combines momentum and reversal signals across a
universe of ~50 liquid crypto assets, executing a dollar-neutral long-short
portfolio with daily or weekly rebalancing.

## Strategy Design

### Universe
- ~50 liquid crypto assets vs USDT on Binance
- Minimum $10M average daily volume, 365-day minimum history
- Excludes stablecoins and delisted/illiquid assets
- Dynamic universe: eligibility re-evaluated at each rebalance

### Signal Stack (Production Composite)

| Signal | Weight | Horizon | Type |
|--------|--------|---------|------|
| 6-month price momentum (skip 1M) | 30% | Medium | Momentum |
| 3-month price momentum (skip 1M) | 20% | Short-medium | Momentum |
| Sharpe-weighted momentum (63D) | 20% | Short-medium | Momentum |
| 5-day reversal (vol-adjusted) | 20% | Short | Reversal |
| Volume-conditioned momentum | 10% | Medium | Momentum + Activity |

All signals are:
1. Cross-sectionally ranked at each date (prevents level contamination)
2. Activity-gated (signals suppressed for low-volume assets/days)
3. Combined via equal-weight or IC-weighted blending

### Portfolio Construction
- Dollar-neutral: long top quartile, short bottom quartile
- Rank-based position sizing (largest positions in highest-conviction signals)
- Maximum single-asset weight: ±10% of portfolio
- No leverage (gross exposure = 100% long + 100% short)

### Execution
- Primary: limit orders (3.5 bps/side assumed, 7 bps round-trip)
- Fallback: market orders (10 bps/side, 20 bps round-trip)
- Rebalance: daily or weekly (weekly preferred to reduce turnover costs)

### Risk Management
- Position cap: max 10% per asset
- Activity gate: no trading when V/MA < 50%
- Drawdown monitoring: reduce gross exposure by 50% if drawdown exceeds 15%

## Performance Targets

| Metric | Target |
|--------|--------|
| Annualised return (net) | 15-25% |
| Annualised volatility | 10-15% |
| Sharpe ratio (net) | 1.0-2.0 |
| Maximum drawdown | < 20% |
| Daily turnover (one-way) | < 10% |
| Correlation to BTC | < 0.3 |

## Key Dependencies and Risks

- **Liquidity risk**: altcoin liquidity dries up in bear markets → costs spike
- **Correlation collapse**: all assets go to 1.0 in crashes → no diversification
- **Strategy crowding**: as crypto matures, stat-arb alpha may compress
- **Data quality**: exchange data gaps, outliers from delistings / hacks
- **Regime sensitivity**: momentum works in trending markets, reversal in choppy

## Implementation Notes

- Signal look-ahead lag: 1 day (signal computed on day t, trades executed on t+1 open)
- Data lag: 1 day (use previous day's close for all signals)
- Rebalancing: at daily close prices (approximation; live would use VWAP or limit orders)
- Transaction costs applied at portfolio level (not per-asset) for simplicity
