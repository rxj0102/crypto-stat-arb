"""
Reversal signal generators.

Core thesis: over short horizons (1 day to ~1 week), crypto returns exhibit
mean reversion — recent losers outperform recent winners. This is driven by
liquidity provision, bid-ask bounce, and uninformed retail overreaction.
"""

import numpy as np
import pandas as pd

from statarb.data.features import FeatureEngine
from statarb.utils import get_logger

logger = get_logger(__name__)
_fe = FeatureEngine()


class ReversalSignals:
    """
    Short-term and medium-term reversal (mean-reversion) signals.

    Convention: *negative* of the recent return → buy recent losers,
    sell recent winners.
    """

    # ------------------------------------------------------------------
    # Short-term reversal (1-day, 1-week)
    # ------------------------------------------------------------------

    def short_term_reversal(
        self, returns: pd.DataFrame, lookback: int = 1
    ) -> pd.DataFrame:
        """
        Classic short-term reversal: negate the past ``lookback``-period return.

        The simplest mean-reversion signal. Highly effective intraday
        and 1-day at high turnover; best applied with low execution costs
        (limit orders).

        Args:
            returns: log return panel
            lookback: periods to reverse (1 = overnight reversal)
        """
        recent = returns.rolling(window=lookback, min_periods=1).sum()
        return -recent

    def weekly_reversal(self, returns: pd.DataFrame) -> pd.DataFrame:
        """5-day (1-week) reversal signal."""
        return self.short_term_reversal(returns, lookback=5)

    def biweekly_reversal(self, returns: pd.DataFrame) -> pd.DataFrame:
        """10-day (2-week) reversal signal."""
        return self.short_term_reversal(returns, lookback=10)

    # ------------------------------------------------------------------
    # Volatility-adjusted reversal
    # ------------------------------------------------------------------

    def vol_adjusted_reversal(
        self, returns: pd.DataFrame, lookback: int = 5, vol_window: int = 21
    ) -> pd.DataFrame:
        """
        Reversal signal scaled by recent volatility.

        Dividing by volatility normalises signal magnitude and gives
        larger positions to low-volatility reversals (more reliable).

        Signal = -cumret(lookback) / vol(vol_window)
        """
        recent = returns.rolling(window=lookback, min_periods=1).sum()
        vol = _fe.realized_volatility(returns, vol_window)
        return -recent / vol.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Distance from moving average
    # ------------------------------------------------------------------

    def ma_distance_reversal(
        self, prices: pd.DataFrame, window: int = 20
    ) -> pd.DataFrame:
        """
        Reversal based on distance from moving average.

        Signal = -(price / MA - 1): buy assets far below their MA,
        sell assets far above. Captures mean-reversion toward trend.

        Args:
            prices: price panel
            window: MA look-back window
        """
        ma = _fe.rolling_mean(prices, window)
        deviation = prices / ma.replace(0, np.nan) - 1
        return -deviation

    def bollinger_reversal(
        self,
        prices: pd.DataFrame,
        window: int = 20,
        n_std: float = 2.0,
    ) -> pd.DataFrame:
        """
        Bollinger Band reversal signal.

        Signal = -(price - MA) / (n_std * rolling_std)
        Values near +1 → price at upper band → sell signal.
        Values near -1 → price at lower band → buy signal.

        Args:
            prices: price panel
            window: look-back window for MA and std
            n_std: number of standard deviations for the bands
        """
        ma = _fe.rolling_mean(prices, window)
        std = prices.rolling(window=window, min_periods=window // 2).std()
        band_width = n_std * std.replace(0, np.nan)
        signal = (prices - ma) / band_width
        return -signal

    # ------------------------------------------------------------------
    # Liquidity-driven reversal
    # ------------------------------------------------------------------

    def high_volume_reversal(
        self,
        returns: pd.DataFrame,
        volume_ratio: pd.DataFrame,
        lookback: int = 1,
        vol_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Amplified reversal signal conditioned on high volume.

        Hypothesis: large price moves on unusually high volume are more
        likely driven by liquidity demands (noise traders) than information
        flow → stronger subsequent reversal.

        Args:
            volume_ratio: V_t / MA(V) panel
            lookback: reversal horizon
            vol_threshold: volume ratio above which to amplify signal
        """
        base = self.short_term_reversal(returns, lookback)
        high_vol = (volume_ratio >= vol_threshold).astype(float)
        high_vol = high_vol.reindex(base.index).fillna(0)
        # Amplify reversal during high-volume periods
        scale = 1.0 + high_vol * 0.5  # 1.0× normal, 1.5× high-volume
        return base * scale

    # ------------------------------------------------------------------
    # Earnings / event-driven reversal (proxy via large-move detection)
    # ------------------------------------------------------------------

    def large_move_reversal(
        self,
        returns: pd.DataFrame,
        lookback: int = 1,
        threshold_std: float = 2.0,
        vol_window: int = 21,
    ) -> pd.DataFrame:
        """
        Reversal signal activated only for abnormally large return moves.

        Only takes positions when a move exceeds ``threshold_std`` standard
        deviations. No signal (NaN) for normal return days.
        This is an event-driven mean-reversion strategy.

        Args:
            lookback: periods to look back for the large move
            threshold_std: z-score threshold for "large" move
            vol_window: window for historical vol estimation
        """
        recent = returns.rolling(window=lookback, min_periods=1).sum()
        vol = _fe.realized_volatility(returns, vol_window)
        zscore = recent / vol.replace(0, np.nan)

        # Only fire signal when |z| > threshold
        extreme_mask = zscore.abs() >= threshold_std
        signal = -zscore.where(extreme_mask, other=np.nan)
        return signal

    # ------------------------------------------------------------------
    # Medium-term reversal (contrarian / value-like)
    # ------------------------------------------------------------------

    def medium_term_reversal(
        self, returns: pd.DataFrame, lookback: int = 252
    ) -> pd.DataFrame:
        """
        Medium-term (1-year) reversal — contrarian signal.

        DeBondt & Thaler (1985) showed that long-run losers outperform
        long-run winners over the following 3-5 years. In crypto,
        this manifests as ~12-month mean reversion after extreme moves.

        Signal = negative of the 12-month cumulative return.
        """
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        return -cumret
