"""
Activity / information-based signal filters.

These are not standalone signals but MODIFIERS that strengthen or weaken
other signals based on the information environment.

Two classes are provided:
- ActivitySignals: produces signal values (+1/-1/0) based on activity classification
- ActivityFilter: utility methods for gating/scaling other signals (Prompt 1)
"""

import numpy as np
import pandas as pd

from statarb.utils import get_logger

logger = get_logger(__name__)


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally rank each row, normalising to [-1, 1]."""
    def _rank_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        n = len(valid)
        if n < 2:
            return row * np.nan
        ranks = valid.rank(method="average") - 1
        normalised = ranks / (n - 1) * 2 - 1
        out = row.copy().astype(float)
        out[:] = np.nan
        out[normalised.index] = normalised
        return out

    return df.apply(_rank_row, axis=1)


class ActivitySignals:
    """
    Activity / information-flow signal classifiers.

    These methods classify each (date, symbol) observation as informed or
    uninformed, enabling downstream signals to be amplified or suppressed
    appropriately. Use as multipliers on momentum or reversal signals.
    """

    # ------------------------------------------------------------------
    # Volume surprise
    # ------------------------------------------------------------------

    def volume_surprise(
        self,
        volume: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """
        Volume surprise: V_t / MA(V, window).

        High values (> 1) indicate unusual activity → new information arriving.
        Low values (< 1) indicate quiet market → likely noise / liquidity.

        Args:
            volume: raw or dollar volume panel (dates × symbols)
            window: rolling window for moving average

        Returns:
            DataFrame of volume surprise ratios (same shape as volume).
            Values > 1 = above-average activity. NaN where MA is zero/NaN.
        """
        ma = volume.rolling(window=window, min_periods=max(1, window // 2)).mean()
        return volume / ma.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Return-volume interaction classifier
    # ------------------------------------------------------------------

    def return_volume_interaction(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame,
        window: int = 21,
        volume_high_threshold: float = 1.2,
        volume_low_threshold: float = 0.8,
        return_threshold_std: float = 0.5,
    ) -> pd.DataFrame:
        """
        Classify each price move as likely informed (+1) or uninformed (-1).

        Classification logic:
        - High volume + large return  → informed (+1) → favour momentum
        - Low volume  + large return  → uninformed (-1) → favour reversal
        - Small return (any volume)   → neutral (0)

        ``large return`` is defined as |z-score| >= return_threshold_std,
        where z-score = return / rolling_std(returns, window).

        Args:
            returns: log return panel
            volume: volume panel
            window: window for volume MA and return std estimation
            volume_high_threshold: V/MA above this → high volume
            volume_low_threshold:  V/MA below this → low volume
            return_threshold_std:  minimum |return z-score| to classify

        Returns:
            DataFrame with values in {-1, 0, +1} (float), NaN where data
            is insufficient. Same shape as returns.
        """
        vol_ma = volume.rolling(window=window, min_periods=max(1, window // 2)).mean()
        vol_ratio = volume / vol_ma.replace(0, np.nan)
        vol_ratio_aligned = vol_ratio.reindex_like(returns)

        ret_std = returns.rolling(window=window, min_periods=window // 2).std()
        ret_zscore = returns / ret_std.replace(0, np.nan)

        large_move = ret_zscore.abs() >= return_threshold_std
        high_vol = vol_ratio_aligned >= volume_high_threshold
        low_vol = vol_ratio_aligned <= volume_low_threshold

        result = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        result[large_move & high_vol] = 1.0    # informed  → momentum
        result[large_move & low_vol] = -1.0    # uninformed → reversal
        # Set NaN where underlying data is NaN
        na_mask = returns.isna() | vol_ratio_aligned.isna()
        result[na_mask] = np.nan
        return result

    # ------------------------------------------------------------------
    # Volatility regime classifier
    # ------------------------------------------------------------------

    def volatility_regime(
        self,
        returns: pd.DataFrame,
        window: int = 21,
        threshold_percentile: float = 75,
    ) -> pd.Series:
        """
        Classify the market regime as high-volatility or low-volatility.

        Market-wide volatility is proxied by the cross-sectional mean of
        individual asset rolling volatilities.

        High volatility / dislocation → more reversal opportunity
        Low volatility / trending     → more momentum opportunity

        Args:
            returns: log return panel
            window: rolling window for realised vol estimation
            threshold_percentile: percentile above which = high-vol regime

        Returns:
            Series['high_vol' | 'low_vol'] indexed by date.
            NaN for dates with insufficient history.
        """
        rol_vol = returns.rolling(window=window, min_periods=window // 2).std()
        market_vol = rol_vol.mean(axis=1)  # cross-sectional average vol per date

        # Expanding percentile threshold (no look-ahead)
        threshold = market_vol.expanding(min_periods=window).quantile(
            threshold_percentile / 100
        )
        regime = pd.Series("low_vol", index=returns.index, dtype=object)
        regime[market_vol >= threshold] = "high_vol"
        regime[market_vol.isna() | threshold.isna()] = np.nan
        return regime


# ---------------------------------------------------------------------------
# ActivityFilter — utility class from Prompt 1 (kept for backward compat)
# ---------------------------------------------------------------------------

class ActivityFilter:
    """
    Utility methods for gating and scaling other signals by activity level.

    Use these to condition downstream signals rather than as standalone
    directional signals.
    """

    def volume_regime(
        self,
        volume: pd.DataFrame,
        window: int = 21,
        high_threshold: float = 1.5,
        low_threshold: float = 0.7,
    ) -> pd.DataFrame:
        """Classify volume regime: high (1), normal (0), low (-1)."""
        act = ActivitySignals()
        ratio = act.volume_surprise(volume, window)
        regime = pd.DataFrame(0.0, index=ratio.index, columns=ratio.columns)
        regime[ratio >= high_threshold] = 1.0
        regime[ratio <= low_threshold] = -1.0
        regime[ratio.isna()] = np.nan
        return regime

    def volume_zscore(self, volume: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """Z-score of volume (expanding)."""
        mu = np.log1p(volume).expanding(min_periods=20).mean()
        sigma = np.log1p(volume).expanding(min_periods=20).std()
        return (np.log1p(volume) - mu) / sigma.replace(0, np.nan)

    def activity_scale(
        self,
        signal: pd.DataFrame,
        volume: pd.DataFrame,
        window: int = 21,
        power: float = 0.5,
    ) -> pd.DataFrame:
        """Scale signal by (V / MA(V))^power."""
        act = ActivitySignals()
        ratio = act.volume_surprise(volume, window).reindex(signal.index)
        scale = np.power(ratio.clip(lower=0.1), power)
        return signal * scale

    def activity_gate(
        self,
        signal: pd.DataFrame,
        volume: pd.DataFrame,
        window: int = 21,
        min_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """Set signal to NaN when V/MA < min_ratio."""
        act = ActivitySignals()
        ratio = act.volume_surprise(volume, window).reindex(signal.index)
        gated = signal.copy()
        gated[ratio < min_ratio] = np.nan
        return gated

    def market_activity_index(
        self, volume: pd.DataFrame, window: int = 21
    ) -> pd.Series:
        """Z-score of total market dollar volume."""
        total = volume.sum(axis=1)
        mu = total.rolling(window=window, min_periods=window // 2).mean()
        sigma = total.rolling(window=window, min_periods=window // 2).std()
        return (total - mu) / sigma.replace(0, np.nan)

    def high_activity_days(
        self, volume: pd.DataFrame, window: int = 21, threshold: float = 1.5
    ) -> pd.Series:
        """Boolean: True on high market-activity days."""
        return self.market_activity_index(volume, window) >= threshold

    def amihud_illiquidity(
        self,
        returns: pd.DataFrame,
        dollar_volume: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """Smoothed Amihud illiquidity: |return| / dollar_volume."""
        illiq = returns.abs() / dollar_volume.replace(0, np.nan)
        return illiq.rolling(window=window, min_periods=window // 2).mean()

    def liquidity_score(
        self,
        returns: pd.DataFrame,
        dollar_volume: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """Liquidity score (inverse Amihud), percentile-ranked cross-sectionally."""
        illiq = self.amihud_illiquidity(returns, dollar_volume, window)
        return illiq.rank(axis=1, ascending=False, pct=True)
