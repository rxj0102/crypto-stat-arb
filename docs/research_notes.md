# Research Notes

## Core Hypotheses

### Momentum
Cryptocurrency markets exhibit cross-sectional price momentum over 3-12 month horizons,
consistent with the Jegadeesh-Titman (1993) findings in equities. The mechanism:
information diffusion is slow — institutional adoption of new L1s, DeFi protocol
upgrades, and regulatory changes propagate gradually across investor bases.

Key findings from academic literature on crypto momentum:
- Liu, Tsyvinski & Wu (2022): strong cross-sectional momentum in crypto (1-4 week horizon)
- Grobys & Sapkota (2019): momentum in top-100 cryptos, especially short-term (weekly)
- Momentum stronger during high-volume / high-activity regimes (informed flow)

### Reversal
Short-term (1-5 day) reversal is pervasive in crypto, driven by:
1. **Liquidity provision**: market makers rebalance inventory after large moves
2. **Bid-ask bounce**: especially visible in hourly data
3. **Retail overreaction**: weekend retail activity creates Monday reversals
4. **Cross-exchange arbitrage**: price discrepancies correct quickly

### Pairs Trading
Cointegrated pairs within thematic clusters (e.g. SOL/AVAX, ETH/BNB):
- Fundamental similarity creates structural cointegration
- Short-term deviations due to liquidity differences revert
- Best pairs identified via Engle-Granger cointegration test

## Signal Development Log

### Momentum Signals Evaluated
| Signal | Lookback | Skip | Notes |
|--------|----------|------|-------|
| Price momentum | 12M | 1M | Standard academic factor |
| Price momentum | 6M | 1M | Better in crypto (shorter cycle) |
| Price momentum | 3M | 1M | Shorter-term trend following |
| TS momentum | 63D | - | Works well in trending markets |
| Sharpe-weighted | 63D | - | Risk-adjusted, more robust |
| MA crossover | 20/60D | - | Trend signal, high turnover |
| Vol-conditioned | 63D | - | Amplified in high-activity periods |

### Reversal Signals Evaluated
| Signal | Lookback | Notes |
|--------|----------|-------|
| Short-term reversal | 1D | Strong but high turnover (limit orders needed) |
| Weekly reversal | 5D | Better turnover profile |
| Vol-adjusted reversal | 5D | More stable, vol-normalized |
| Bollinger Band | 20D | Parametric mean-reversion |
| Large-move reversal | 1D | Event-driven, sparse signal |

## Execution Cost Analysis

Assumed costs (Binance, liquid assets):
- **Taker fee**: 10 bps one-way (20 bps round-trip)
- **Maker rebate**: ~-2.5 bps, net 3.5 bps maker cost
- **Market impact**: estimated 5 bps per 1% of ADTV

Break-even analysis:
- Daily rebalance at 10% turnover → 2 bps/day needed in gross alpha
- Weekly rebalance at 10% weekly turnover → 0.4 bps/day needed
- Short-term reversal (1D, limit): ~0.7 bps/day needed

## Universe Construction

Final universe: ~50 assets meeting:
- $10M+ average daily dollar volume (30-day rolling)
- 365+ days of trading history
- Listed on Binance (primary venue)
- Not stablecoins

Dynamic universe prevents look-ahead bias: each day's tradable set
uses only volume data available up to the prior day.

## Key Risks

1. **Regime changes**: crypto correlations spike during market stress
2. **Structural breaks**: exchange failures (FTX Nov 2022) cause data gaps
3. **Low liquidity altcoins**: market impact much higher than assumed
4. **Regulatory risk**: asset delistings can cause permanent loss
5. **Overfitting**: with 50 assets and 4 years of daily data, overfitting risk is high
