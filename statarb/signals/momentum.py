"""
Momentum signal generators.

Momentum = assets that have gone up tend to continue going up (and vice versa).
Works at medium-to-long horizons due to information diffusion and trend persistence.

All public methods return a DataFrame of the same shape as the input, with
cross-sectionally ranked values in [-1, 1]:
  +1 = strongest buy signal (expect to outperform)
  -1 = strongest sell signal (expect to underperform)
  NaN = no signal (insufficient data or NaN input)
"""

import numpy as np
import pandas as pd

from statarb.utils import get_logger

logger = get_logger(__name__)


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally rank each row, normalising to [-1, 1].

    Rank 0-based: 0 → -1, N-1 → +1. Ties use average rank.
    NaN values are excluded from ranking and remain NaN.
    """
    def _rank_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        n = len(valid)
        if n < 2:
            return row * np.nan
        ranks = valid.rank(method="average") - 1          # 0-based
        normalised = ranks / (n - 1) * 2 - 1              # → [-1, 1]
        out = row.copy().astype(float)
        out[:] = np.nan
        out[normalised.index] = normalised
        return out

    return df.apply(_rank_row, axis=1)


class MomentumSignals:
    """
    Momentum signal generators.

    Signals are:
    1. Computed using only past data (no look-ahead)
    2. Cross-sectionally ranked to [-1, 1] at each date
    3. NaN where input is NaN or insufficient history exists
    """

    # ------------------------------------------------------------------
    # Time-series momentum
    # ------------------------------------------------------------------

    def time_series_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 21,
    ) -> pd.DataFrame:
        """
        Classic time-series momentum (TSMOM).

        signal_{i,t} = Σ_{s=t-lookback+1}^{t} r_{i,s}

        Cumulative return over the lookback window, then cross-sectionally ranked.
        Test multiple lookbacks: 7, 14, 21, 42, 63, 126 days.

        Args:
            returns: log return panel (dates × symbols)
            lookback: rolling window in periods

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        raw = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        return _rank(raw)

    # ------------------------------------------------------------------
    # Cross-sectional momentum (skip-version)
    # ------------------------------------------------------------------

    def cross_sectional_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 21,
        skip: int = 1,
    ) -> pd.DataFrame:
        """
        Cross-sectional momentum (XSMOM).

        signal_{i,t} = rank(cumret_{i, [t-lookback-skip, t-skip]})

        The ``skip`` parameter excludes the most recent days to avoid
        short-term reversal contaminating the momentum signal.
        Jegadeesh & Titman (1993) use skip=1 month in equities;
        in crypto try skip=0, 1, 2, 3 days.

        Args:
            returns: log return panel
            lookback: look-back horizon (excluding skip period)
            skip: number of recent periods to exclude

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        if skip < 0:
            raise ValueError("skip must be >= 0")
        total = returns.rolling(window=lookback + skip, min_periods=(lookback + skip) // 2).sum()
        if skip > 0:
            recent = returns.rolling(window=skip, min_periods=1).sum()
            raw = total - recent
        else:
            raw = total
        return _rank(raw)

    # ------------------------------------------------------------------
    # Volume-weighted momentum
    # ------------------------------------------------------------------

    def volume_weighted_momentum(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame,
        lookback: int = 21,
        vol_ma_window: int = 21,
    ) -> pd.DataFrame:
        """
        Momentum weighted by volume relative to its moving average.

        signal = Σ_{t-lookback}^{t} (r_s × V_s / MA(V, vol_ma_window))

        Higher-volume days receive larger weight in the cumulative sum.
        Theory: high-volume moves contain more information, so volume-weighted
        momentum is a cleaner measure of informed price discovery.

        Args:
            returns: log return panel
            volume: raw volume (or dollar volume) panel
            lookback: accumulation window
            vol_ma_window: window for volume moving average normalisation

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        vol_ma = volume.rolling(window=vol_ma_window, min_periods=max(1, vol_ma_window // 2)).mean()
        vol_ratio = volume / vol_ma.replace(0, np.nan)
        # Align columns
        common = returns.columns.intersection(vol_ratio.columns)
        weighted_ret = returns[common] * vol_ratio[common].reindex(returns.index)
        raw = weighted_ret.rolling(window=lookback, min_periods=lookback // 2).sum()
        # Reindex back to full column set (missing symbols → NaN)
        raw = raw.reindex(columns=returns.columns)
        return _rank(raw)

    # ------------------------------------------------------------------
    # Activity-filtered momentum
    # ------------------------------------------------------------------

    def momentum_with_activity_filter(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame,
        lookback: int = 21,
        activity_threshold: float = 1.5,
        vol_ma_window: int = 21,
    ) -> pd.DataFrame:
        """
        Momentum signal activated only when trading activity is high.

        Activity = V_t / MA(V, vol_ma_window).
        Signal = ranked momentum when activity > threshold, else 0.

        Theory: momentum is stronger during periods of new information
        flow (proxied by above-average volume). Low-activity periods are
        dominated by noise/liquidity, which reverses rather than continues.

        Args:
            returns: log return panel
            volume: volume panel
            lookback: momentum accumulation window
            activity_threshold: minimum V/MA ratio to activate signal
            vol_ma_window: window for volume normalisation

        Returns:
            Signal panel (0 where inactive, ranked momentum values elsewhere)
        """
        mom_raw = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        ranked_mom = _rank(mom_raw)

        vol_ma = volume.rolling(window=vol_ma_window, min_periods=max(1, vol_ma_window // 2)).mean()
        vol_ratio = volume / vol_ma.replace(0, np.nan)
        vol_ratio_aligned = vol_ratio.reindex_like(ranked_mom).fillna(0)

        high_activity = vol_ratio_aligned >= activity_threshold
        result = ranked_mom.where(high_activity, other=0.0)
        return result

    # ------------------------------------------------------------------
    # Breakout / Donchian channel momentum
    # ------------------------------------------------------------------

    def breakout_momentum(
        self,
        prices: pd.DataFrame,
        lookback: int = 21,
    ) -> pd.DataFrame:
        """
        Donchian channel breakout momentum.

        signal = (P_t - min(P, lookback)) / (max(P, lookback) - min(P, lookback))

        Values near 1 → price at top of recent range (bullish breakout).
        Values near 0 → price at bottom of recent range (bearish breakdown).
        After cross-sectional ranking: +1 = breakout leader, -1 = breakdown laggard.

        Args:
            prices: close price panel
            lookback: rolling range window

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        roll_min = prices.rolling(window=lookback, min_periods=lookback // 2).min()
        roll_max = prices.rolling(window=lookback, min_periods=lookback // 2).max()
        denom = (roll_max - roll_min).replace(0, np.nan)
        raw = (prices - roll_min) / denom
        return _rank(raw)

    # ------------------------------------------------------------------
    # Momentum acceleration
    # ------------------------------------------------------------------

    def acceleration(
        self,
        returns: pd.DataFrame,
        short_window: int = 7,
        long_window: int = 28,
    ) -> pd.DataFrame:
        """
        Momentum acceleration: is momentum speeding up or slowing down?

        signal = cumret(short_window) - (short_window/long_window) × cumret(long_window)

        Positive → recent performance better than the longer-term trend (accelerating).
        Negative → recent performance worse than longer-term trend (decelerating).

        Args:
            returns: log return panel
            short_window: short-term window (e.g. 1 week)
            long_window: long-term window (e.g. 1 month)

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        short_cum = returns.rolling(window=short_window, min_periods=max(1, short_window // 2)).sum()
        long_cum = returns.rolling(window=long_window, min_periods=max(1, long_window // 2)).sum()
        scale = short_window / long_window
        raw = short_cum - scale * long_cum
        return _rank(raw)

    # ------------------------------------------------------------------
    # Legacy convenience wrappers (from Prompt 1 — kept for compatibility)
    # ------------------------------------------------------------------

    def momentum_12_1(self, returns: pd.DataFrame) -> pd.DataFrame:
        """12-month momentum skipping 1 month (standard academic factor)."""
        return self.cross_sectional_momentum(returns, lookback=231, skip=21)

    def momentum_6_1(self, returns: pd.DataFrame) -> pd.DataFrame:
        """6-month momentum skipping 1 month."""
        return self.cross_sectional_momentum(returns, lookback=105, skip=21)

    def momentum_3_1(self, returns: pd.DataFrame) -> pd.DataFrame:
        """3-month momentum skipping 1 month."""
        return self.cross_sectional_momentum(returns, lookback=42, skip=21)

    def momentum_1w(self, returns: pd.DataFrame) -> pd.DataFrame:
        """1-week momentum (5 days), no skip."""
        return self.time_series_momentum(returns, lookback=5)

    def sharpe_momentum(
        self, returns: pd.DataFrame, lookback: int = 63
    ) -> pd.DataFrame:
        """Risk-adjusted momentum: mean / std over lookback."""
        mean_ret = returns.rolling(window=lookback, min_periods=lookback // 2).mean()
        vol = returns.rolling(window=lookback, min_periods=lookback // 2).std()
        raw = mean_ret / vol.replace(0, np.nan)
        return _rank(raw)

    def moving_average_crossover(
        self, prices: pd.DataFrame, fast: int = 20, slow: int = 60
    ) -> pd.DataFrame:
        """Signal from fast MA crossing above slow MA: (fast_MA / slow_MA) - 1."""
        fast_ma = prices.rolling(window=fast, min_periods=fast // 2).mean()
        slow_ma = prices.rolling(window=slow, min_periods=slow // 2).mean()
        raw = fast_ma / slow_ma.replace(0, np.nan) - 1
        return _rank(raw)
