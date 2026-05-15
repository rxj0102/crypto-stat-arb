"""
Momentum signal generators.

Core thesis: winners over the past 1–12 months continue to outperform
losers over the next 1–3 months (Jegadeesh & Titman 1993 extended to crypto).
Crypto exhibits strong time-series and cross-sectional momentum, especially
at medium horizons (4–12 weeks) and during high-activity regimes.
"""

import numpy as np
import pandas as pd

from statarb.data.features import FeatureEngine
from statarb.utils import get_logger

logger = get_logger(__name__)
_fe = FeatureEngine()


class MomentumSignals:
    """
    Cross-sectional and time-series momentum signals.

    All signals return a DataFrame[dates × symbols] of *raw* signal values
    (not yet ranked). Pass to :meth:`FeatureEngine.cross_sectional_rank`
    before use in a portfolio.

    Conventions:
    - Positive signal value → buy (long)
    - Negative signal value → sell (short)
    - NaN → no position
    """

    # ------------------------------------------------------------------
    # Classic cross-sectional price momentum
    # ------------------------------------------------------------------

    def price_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 63,
        skip: int = 5,
    ) -> pd.DataFrame:
        """
        Standard cross-sectional price momentum.

        Cumulative log return over [t-lookback, t-skip], skipping the most
        recent ``skip`` periods to avoid short-term reversal contamination.

        Args:
            returns: log return panel (dates × symbols)
            lookback: total look-back in periods
            skip: recent periods to exclude (microstructure reversal)

        Returns:
            Signal panel (dates × symbols). Higher = stronger winner.
        """
        if skip >= lookback:
            raise ValueError("skip must be less than lookback")
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        recent = returns.rolling(window=skip, min_periods=1).sum()
        return cumret - recent

    def momentum_12_1(self, returns: pd.DataFrame) -> pd.DataFrame:
        """12-month momentum skipping last 1 month (standard academic factor)."""
        return self.price_momentum(returns, lookback=252, skip=21)

    def momentum_6_1(self, returns: pd.DataFrame) -> pd.DataFrame:
        """6-month momentum skipping last 1 month."""
        return self.price_momentum(returns, lookback=126, skip=21)

    def momentum_3_1(self, returns: pd.DataFrame) -> pd.DataFrame:
        """3-month momentum skipping last 1 month."""
        return self.price_momentum(returns, lookback=63, skip=21)

    def momentum_1w(self, returns: pd.DataFrame) -> pd.DataFrame:
        """1-week momentum (5 days), no skip — short-term continuation."""
        return returns.rolling(window=5, min_periods=3).sum()

    # ------------------------------------------------------------------
    # Trend / moving-average signals
    # ------------------------------------------------------------------

    def moving_average_crossover(
        self,
        prices: pd.DataFrame,
        fast: int = 20,
        slow: int = 60,
    ) -> pd.DataFrame:
        """
        Signal based on fast MA crossing above slow MA.

        Value = (fast_MA / slow_MA) - 1.
        Positive → fast above slow → uptrend (buy signal).
        """
        fast_ma = _fe.rolling_mean(prices, fast)
        slow_ma = _fe.rolling_mean(prices, slow)
        return fast_ma / slow_ma.replace(0, np.nan) - 1

    def dual_ma_crossover(
        self, prices: pd.DataFrame, fast: int = 10, slow: int = 30
    ) -> pd.DataFrame:
        """10/30-day dual MA crossover — shorter-term trend following."""
        return self.moving_average_crossover(prices, fast, slow)

    def price_above_ma(
        self, prices: pd.DataFrame, window: int = 200
    ) -> pd.DataFrame:
        """
        Binary-ish signal: (price / MA_window) - 1.

        Positive = price above long-run MA → bullish trend.
        """
        ma = _fe.rolling_mean(prices, window)
        return prices / ma.replace(0, np.nan) - 1

    # ------------------------------------------------------------------
    # Time-series (absolute) momentum
    # ------------------------------------------------------------------

    def time_series_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 63,
    ) -> pd.DataFrame:
        """
        Time-series (absolute) momentum: sign of cumulative return.

        Long if the asset has positive cumulative return over the lookback,
        short if negative. Signal magnitude = cumulative return.

        Reference: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum".
        """
        return returns.rolling(window=lookback, min_periods=lookback // 2).sum()

    def sharpe_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 63,
    ) -> pd.DataFrame:
        """
        Risk-adjusted time-series momentum: Sharpe ratio over lookback.

        Mean return / std dev over the window — weights recent performance
        by its consistency (low-vol momentum more reliable).
        """
        mean_ret = _fe.rolling_mean(returns, lookback)
        vol = returns.rolling(window=lookback, min_periods=lookback // 2).std()
        return mean_ret / vol.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Volume-conditioned momentum
    # ------------------------------------------------------------------

    def volume_conditioned_momentum(
        self,
        returns: pd.DataFrame,
        volume_ratio: pd.DataFrame,
        lookback: int = 63,
        vol_threshold: float = 1.2,
    ) -> pd.DataFrame:
        """
        Momentum signal amplified during high-volume regimes.

        The hypothesis: momentum is stronger when driven by informed
        (high-volume) activity. Scale the momentum signal by an indicator
        of above-average volume.

        Args:
            volume_ratio: V_t / MA(V, window) panel — see FeatureEngine
            vol_threshold: minimum volume ratio to consider "high activity"
        """
        base_signal = self.price_momentum(returns, lookback)
        high_vol_indicator = (volume_ratio >= vol_threshold).astype(float)
        high_vol_indicator = high_vol_indicator.reindex(base_signal.index).fillna(0)
        # Amplify: multiply signal by 1.5 in high-volume regime, 0.5 otherwise
        scale = high_vol_indicator.replace(0, 0.5).replace(1.0, 1.5)
        return base_signal * scale

    # ------------------------------------------------------------------
    # Residual / idiosyncratic momentum
    # ------------------------------------------------------------------

    def residual_momentum(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        lookback: int = 63,
        estimation_window: int = 126,
    ) -> pd.DataFrame:
        """
        Idiosyncratic momentum: cumulative residual return after removing
        market beta.

        More predictive than raw momentum because it strips out common-factor
        (BTC) exposure and focuses on asset-specific continuation.

        Args:
            market_returns: BTC or equal-weighted market returns (Series)
            lookback: horizon for cumulative residual return
            estimation_window: window for beta estimation
        """
        market = market_returns.reindex(returns.index)
        residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for sym in returns.columns:
            asset = returns[sym].dropna()
            mkt = market.reindex(asset.index).dropna()
            common_idx = asset.index.intersection(mkt.index)
            if len(common_idx) < estimation_window // 2:
                continue
            r = asset.loc[common_idx]
            m = mkt.loc[common_idx]

            # Rolling beta estimation
            roll_cov = r.rolling(estimation_window).cov(m)
            roll_var = m.rolling(estimation_window).var()
            beta = roll_cov / roll_var.replace(0, np.nan)
            residuals[sym] = r - beta * m

        return residuals.rolling(window=lookback, min_periods=lookback // 2).sum()
