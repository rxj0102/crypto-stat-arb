"""
Activity and information-flow filters.

The activity filter hypothesis: momentum signals are stronger and more
reliable when accompanied by elevated trading activity (volume). High
volume indicates informed trading; low volume signals noise / illiquidity.

Use these filters to *condition* or *scale* other signals rather than
as standalone directional signals.
"""

import numpy as np
import pandas as pd

from statarb.data.features import FeatureEngine
from statarb.utils import get_logger

logger = get_logger(__name__)
_fe = FeatureEngine()


class ActivityFilter:
    """
    Activity-based signal conditioners.

    All methods return DataFrames or Series that can be used to:
    1. Scale a signal (multiply by activity weight)
    2. Gate a signal (set to NaN when activity is below threshold)
    3. Select assets (trade only high-activity assets)
    """

    # ------------------------------------------------------------------
    # Volume-based regime classification
    # ------------------------------------------------------------------

    def volume_regime(
        self,
        volume: pd.DataFrame,
        window: int = 21,
        high_threshold: float = 1.5,
        low_threshold: float = 0.7,
    ) -> pd.DataFrame:
        """
        Classify volume regime as high (1), normal (0), or low (-1).

        Args:
            volume: dollar volume panel
            window: MA window for comparison
            high_threshold: V / MA threshold for "high volume"
            low_threshold: V / MA threshold for "low volume"

        Returns:
            DataFrame with values in {-1, 0, 1}.
        """
        ratio = _fe.volume_ma_ratio(volume, window)
        regime = pd.DataFrame(0, index=ratio.index, columns=ratio.columns, dtype=float)
        regime[ratio >= high_threshold] = 1.0
        regime[ratio <= low_threshold] = -1.0
        regime[ratio.isna()] = np.nan
        return regime

    def volume_zscore(
        self, volume: pd.DataFrame, window: int = 63
    ) -> pd.DataFrame:
        """
        Z-score of volume relative to rolling window.

        Values > 2 indicate extremely high activity (potential news event).
        Used to detect unusual trading days.

        Returns:
            DataFrame of volume z-scores.
        """
        return _fe.expanding_zscore(np.log1p(volume))

    # ------------------------------------------------------------------
    # Signal scaling by activity
    # ------------------------------------------------------------------

    def activity_scale(
        self,
        signal: pd.DataFrame,
        volume: pd.DataFrame,
        window: int = 21,
        power: float = 0.5,
    ) -> pd.DataFrame:
        """
        Scale a signal by normalised volume (soft gating).

        signal_scaled = signal × (V / MA(V))^power

        The power parameter controls sensitivity:
        - power=0: no scaling (passthrough)
        - power=0.5: mild scaling (default)
        - power=1.0: full linear scaling

        Args:
            signal: raw signal panel to scale
            volume: dollar volume panel
            window: MA window
            power: exponent for volume scaling

        Returns:
            Scaled signal panel.
        """
        ratio = _fe.volume_ma_ratio(volume, window).reindex(signal.index)
        scale = np.power(ratio.clip(lower=0.1), power)
        return signal * scale

    def activity_gate(
        self,
        signal: pd.DataFrame,
        volume: pd.DataFrame,
        window: int = 21,
        min_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """
        Hard gate: set signal to NaN when volume is too low.

        Prevents trading in illiquid conditions where execution costs
        would overwhelm any signal alpha.

        Args:
            signal: raw signal panel
            volume: dollar volume panel
            window: MA window
            min_ratio: minimum V/MA ratio to allow trading

        Returns:
            Signal panel with NaN inserted for low-activity dates/symbols.
        """
        ratio = _fe.volume_ma_ratio(volume, window).reindex(signal.index)
        gated = signal.copy()
        gated[ratio < min_ratio] = np.nan
        return gated

    # ------------------------------------------------------------------
    # Market-wide activity indicators
    # ------------------------------------------------------------------

    def market_activity_index(
        self,
        volume: pd.DataFrame,
        window: int = 21,
    ) -> pd.Series:
        """
        Market-wide activity index: z-score of total market dollar volume.

        High values indicate a high-information-flow day (good for momentum).
        Low values indicate quiet market (good for mean reversion).

        Returns:
            Series of market activity z-scores (one value per date).
        """
        total_vol = volume.sum(axis=1)
        rolling_mean = total_vol.rolling(window=window, min_periods=window // 2).mean()
        rolling_std = total_vol.rolling(window=window, min_periods=window // 2).std()
        return (total_vol - rolling_mean) / rolling_std.replace(0, np.nan)

    def high_activity_days(
        self,
        volume: pd.DataFrame,
        window: int = 21,
        threshold: float = 1.5,
    ) -> pd.Series:
        """
        Boolean indicator: True on high market-wide activity days.

        Returns:
            Series[bool] — True when market activity index >= threshold.
        """
        mai = self.market_activity_index(volume, window)
        return mai >= threshold

    # ------------------------------------------------------------------
    # Amihud illiquidity (proxy)
    # ------------------------------------------------------------------

    def amihud_illiquidity(
        self,
        returns: pd.DataFrame,
        dollar_volume: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """
        Amihud (2002) illiquidity ratio: |return| / dollar_volume.

        Higher values → more illiquid (price moves more per unit of volume).
        Use to down-weight positions in less-liquid assets.

        Args:
            returns: return panel
            dollar_volume: dollar volume panel
            window: rolling window for smoothing

        Returns:
            Panel of smoothed Amihud ratios (lower = more liquid).
        """
        illiq = returns.abs() / dollar_volume.replace(0, np.nan)
        return illiq.rolling(window=window, min_periods=window // 2).mean()

    def liquidity_score(
        self,
        returns: pd.DataFrame,
        dollar_volume: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """
        Liquidity score (inverse of Amihud): higher = more liquid.

        Normalised to [0, 1] range cross-sectionally.

        Returns:
            Panel of liquidity scores (0=least liquid, 1=most liquid).
        """
        illiq = self.amihud_illiquidity(returns, dollar_volume, window)
        # Cross-sectional rank normalised to [0, 1]
        ranked = illiq.rank(axis=1, ascending=False, pct=True)
        return ranked
