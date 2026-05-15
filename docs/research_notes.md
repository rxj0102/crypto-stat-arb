# Research Notes — crypto-stat-arb

## Core Hypotheses

### Momentum
Cryptocurrency markets exhibit cross-sectional price momentum over 1-week to 4-month horizons,
consistent with Liu, Tsyvinski & Wu (2022). The mechanism: information diffusion is slow —
institutional adoption of new L1s, DeFi protocol upgrades, and regulatory changes propagate
gradually across investor bases.

Key findings from academic literature on crypto momentum:
- Liu, Tsyvinski & Wu (2022): strong cross-sectional momentum in crypto (1–4 week horizon)
- Grobys & Sapkota (2019): momentum in top-100 cryptos, especially short-term (weekly)
- Momentum stronger during high-volume / high-activity regimes (informed flow hypothesis)

**Tested configurations:**
- Lookbacks: 7, 14, 21, 42, 63, 126 days
- Skip parameters: 0, 1, 2, 3 days (to avoid microstructure reversal contamination)
- Default: 63-day lookback, 1-day skip

### Reversal
Short-term (1–7 day) reversal is pervasive in crypto, driven by:
1. **Liquidity provision**: market makers rebalance inventory after large moves
2. **Bid-ask bounce**: especially visible in hourly data
3. **Retail overreaction**: weekend retail activity creates Monday reversals
4. **Cross-exchange arbitrage**: price discrepancies correct quickly

**Key cost finding**: 1-day reversal is profitable only with limit orders (7 bps).
Market orders (20 bps) destroy the alpha. 5-day reversal is more robust.

**Tested configurations:**
- Lookbacks: 1, 2, 3, 5, 7 days
- Volume-filtered variant: active only on low-volume days (uninformed flow)
- Beta-neutral: subtract rolling market beta to isolate idiosyncratic reversion

### Pairs Trading
Cointegrated pairs within thematic clusters (e.g. SOL/AVAX, ETH/BNB):
- Fundamental similarity creates structural cointegration
- Short-term deviations due to liquidity differences revert
- Best pairs identified via Engle-Granger cointegration test (p < 0.05)

**Note**: cointegration tests require 252+ days of history and tend to find fewer
true pairs than advertised — most crypto pairs are merely correlated, not cointegrated.

---

## Signal Development Log

### Momentum Signals Evaluated

| Signal | Lookback | Skip | Notes |
|--------|----------|------|-------|
| Time-series momentum | 7–126d | 0 | Basis for lookback sweep |
| Price momentum (12-1) | ~252d | ~21d | Standard academic factor; long horizon |
| Price momentum (6-1) | ~126d | ~21d | Best risk-adjusted in crypto |
| Price momentum (3-1) | ~63d | ~21d | Shorter cycle, higher turnover |
| Price momentum (1W) | 7d | 0 | Very short-term trend |
| Sharpe-weighted | 63d | — | Risk-adjusted momentum; lower drawdown |
| MA crossover | 20/60d | — | Trend signal; high turnover |
| Volume-weighted | 63d | — | Amplified in high-activity periods |

**Key finding**: skip parameter matters. Skip=0 picks up microstructure reversal at the 1-day
horizon; skip=1 gives cleaner momentum signal. Longer skips reduce IC without reducing costs.

### Reversal Signals Evaluated

| Signal | Lookback | Notes |
|--------|----------|-------|
| Short-term reversal | 1d | Strong but needs limit orders; daily rebalance |
| Short-term reversal | 3–5d | Better turnover; robust to execution cost |
| Weekly reversal | 5d | Best Sharpe in most regimes |
| Vol-adjusted reversal | 5d | More stable; normalized by rolling vol |
| Bollinger Band | 20d | Parametric mean-reversion; works in ranging markets |
| Large-move reversal | 1d | Conditional on large move yesterday; sparse signal |
| Volume-filtered | 3d | Active only on low-volume days (uninformed flow) |
| Beta-neutral | 5d | Market beta removed; pure idiosyncratic reversal |

### Combination Results

| Method | Notes |
|--------|-------|
| Equal weight | Robust baseline; always competitive |
| Inverse vol | Favors lower-volatility signals; reduces realized vol |
| Sharpe-weighted | Adaptive; good when strategies have divergent quality |
| Min variance | Most sophisticated; requires stable covariance estimate |

**Finding**: Sharpe-weighted combination is best out-of-sample when there is genuine
quality dispersion between momentum and reversal. Equal weight wins when
lookback is short (uncertain Sharpe estimates).

---

## Execution Cost Analysis

Assumed costs (Binance, liquid assets):
- **Taker fee (market order)**: 10 bps one-way, 20 bps round-trip
- **Maker rebate (limit order)**: ~3.5 bps, net 7 bps round-trip
- **Market impact**: estimated 5 bps per 1% of ADTV

Break-even analysis per strategy type:
| Strategy | Typical Turnover | Required Gross Alpha |
|----------|-----------------|---------------------|
| 5d reversal (limit) | 10–15%/day | 0.7 bps/day |
| 63d momentum | 5–10%/day | 0.35 bps/day |
| 1d reversal (market) | 25%+/day | 5+ bps/day → difficult |
| Combined (daily rebal) | 8–12%/day | 0.8 bps/day |

---

## Universe Construction

Final universe: ~20–50 assets meeting:
- $10M+ average daily dollar volume (30-day rolling)
- 365+ days of trading history
- Listed on Binance (primary venue)
- Not stablecoins

Dynamic universe prevents look-ahead bias: each day's tradable set
uses only volume data available up to the prior day.

---

## Key Risks

1. **Regime changes**: crypto correlations spike during market stress → L/S pairs both fall
2. **Structural breaks**: exchange failures (FTX Nov 2022) cause data gaps and delistings
3. **Low liquidity altcoins**: market impact much higher than assumed for tail assets
4. **Regulatory risk**: asset delistings can cause permanent loss; needs monitoring
5. **Overfitting**: with 20 assets and 3 years of daily data, any IC > 5% is suspicious
6. **Synthetic data caveat**: results on synthetic data are illustrative only — real alpha
   requires real price discovery dynamics not captured by GARCH models

---

## Findings Summary (Synthetic Data — Illustrative)

| Finding | Observed? |
|---------|-----------|
| TS momentum at 63d lookback outperforms 7d | Yes |
| Skip=1 improves momentum (vs skip=0 on synthetic) | Mixed — synthetic has mean-reverting structure |
| Reversal more profitable with limit vs market orders | Yes (cost drag alone) |
| Activity-gating improves signal stability | Yes (less noise) |
| Combination improves over average individual | Depends on combination method |
| Dollar-neutral beta ≈ 0 | Yes — by construction |

> **Important**: these findings are based on synthetic GARCH-correlated data generated by
> `experiments/generate_synthetic_data.py`. Real crypto data will produce different magnitudes
> and potentially different rankings. The pipeline is validated for correctness, not alpha discovery.
