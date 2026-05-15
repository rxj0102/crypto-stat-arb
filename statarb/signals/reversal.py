"""
Reversal signal generators.

Reversal = assets that have gone up tend to come back down (and vice versa).
Works at short horizons. Driven by liquidity provision, bid-ask bounce,
and uninformed / forced selling (margin calls, liquidations).

Key principle: UNINFORMED trades (liquidity, forced selling) reverse more
than INFORMED trades (news-driven momentum). Volume distinguishes the two.

All public methods return DataFrames of the same shape as the input, with
cross-sectionally ranked values in [-1, 1] unless otherwise noted.
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


class ReversalSignals:
    """
    Reversal (mean-reversion) signal generators.

    Signals are:
    1. Computed using only past data (no look-ahead)
    2. Cross-sectionally ranked to [-1, 1] at each date (where noted)
    3. Zero where a conditional filter is not met (volume-filtered variants)
    """

    # ------------------------------------------------------------------
    # Simple short-term reversal
    # ------------------------------------------------------------------

    def short_term_reversal(
        self,
        returns: pd.DataFrame,
        lookback: int = 1,
    ) -> pd.DataFrame:
        """
        Classic short-term reversal: buy past losers, sell past winners.

        signal = -cumret(lookback)

        Test lookbacks: 1, 2, 3, 5, 7 days.
        For hourly data also test: 1h, 4h, 8h, 12h periods.

        Args:
            returns: log return panel (daily or sub-daily)
            lookback: number of periods to look back

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        cumret = returns.rolling(window=lookback, min_periods=1).sum()
        return _rank(-cumret)

    # ------------------------------------------------------------------
    # Z-score mean reversion
    # ------------------------------------------------------------------

    def mean_reversion_zscore(
        self,
        prices: pd.DataFrame,
        lookback: int = 21,
    ) -> pd.DataFrame:
        """
        Z-score mean reversion: buy when price is below its moving average.

        signal = -(P_t - MA(P, lookback)) / std(P, lookback)

        Positive signal (below MA) → expect reversion upward.
        Negative signal (above MA) → expect reversion downward.

        Args:
            prices: close price panel
            lookback: window for MA and std estimation

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        ma = prices.rolling(window=lookback, min_periods=lookback // 2).mean()
        std = prices.rolling(window=lookback, min_periods=lookback // 2).std()
        zscore = (prices - ma) / std.replace(0, np.nan)
        return _rank(-zscore)

    # ------------------------------------------------------------------
    # Volume-filtered reversal (low-activity condition)
    # ------------------------------------------------------------------

    def volume_filtered_reversal(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame,
        lookback: int = 3,
        activity_threshold: float = 0.7,
        vol_ma_window: int = 21,
    ) -> pd.DataFrame:
        """
        Reversal signal active only when trading activity is LOW.

        Activity = V_t / MA(V, vol_ma_window).
        Signal = ranked(-cumret(lookback)) when activity < threshold, else 0.

        Theory: low-volume moves are more likely uninformed / liquidity-driven
        and therefore more likely to reverse. High-volume moves carry information
        and are better captured by momentum signals.

        Args:
            returns: log return panel
            volume: volume panel (raw or dollar volume)
            lookback: reversal look-back window
            activity_threshold: maximum V/MA ratio to activate reversal
            vol_ma_window: window for volume normalisation

        Returns:
            Signal panel (0 where inactive, ranked reversal values elsewhere)
        """
        cumret = returns.rolling(window=lookback, min_periods=1).sum()
        ranked_rev = _rank(-cumret)

        vol_ma = volume.rolling(window=vol_ma_window, min_periods=max(1, vol_ma_window // 2)).mean()
        vol_ratio = volume / vol_ma.replace(0, np.nan)
        vol_ratio_aligned = vol_ratio.reindex_like(ranked_rev).fillna(1.0)

        low_activity = vol_ratio_aligned < activity_threshold
        return ranked_rev.where(low_activity, other=0.0)

    # ------------------------------------------------------------------
    # Overnight reversal
    # ------------------------------------------------------------------

    def overnight_reversal(
        self,
        ohlcv_data: dict,
        lookback: int = 1,
    ) -> pd.DataFrame:
        """
        Reversal based on overnight (low-retail-activity) returns.

        For hourly data: "overnight" = UTC hours 22:00–08:00 (US night),
        when institutional liquidity is thin and retail less active.
        For daily data: weekend return (Friday close → Monday open).

        Uninformed flow dominates overnight → stronger reversal signal.

        Args:
            ohlcv_data: dict mapping symbol → OHLCV DataFrame with UTC DatetimeIndex.
                        Expected columns: open, high, low, close, volume.
                        Hourly frequency recommended; daily frequency returns NaN.
            lookback: number of overnight periods to accumulate

        Returns:
            DataFrame of overnight return reversals, ranked ∈ [-1, 1].
            Returns empty DataFrame if data is daily-only.
        """
        overnight_returns: dict = {}

        for sym, df in ohlcv_data.items():
            if df.empty or "close" not in df.columns:
                continue
            df = df.sort_index()
            idx = pd.DatetimeIndex(df.index)

            # Detect frequency: if gaps > 1 day on average, it's daily data
            if len(idx) > 2:
                median_gap = pd.Series(idx).diff().median()
                if median_gap >= pd.Timedelta(days=1):
                    # Daily data: compute weekend returns (Fri → Mon gap)
                    log_ret = np.log(df["close"] / df["close"].shift(1))
                    weekend_mask = idx.dayofweek == 0  # Monday
                    overnight_ret = pd.Series(np.nan, index=df.index)
                    overnight_ret[weekend_mask] = log_ret[weekend_mask]
                    overnight_returns[sym] = overnight_ret
                    continue

            # Hourly data: overnight = hours 22-07 UTC
            log_ret = np.log(df["close"] / df["close"].shift(1))
            is_overnight = idx.hour.isin(range(22, 24)) | idx.hour.isin(range(0, 8))
            overnight_ret = log_ret.where(is_overnight, other=np.nan)
            overnight_returns[sym] = overnight_ret

        if not overnight_returns:
            return pd.DataFrame()

        panel = pd.DataFrame(overnight_returns)
        cumret = panel.rolling(window=lookback, min_periods=1).sum()
        return _rank(-cumret)

    # ------------------------------------------------------------------
    # Liquidation cascade reversal
    # ------------------------------------------------------------------

    def liquidation_reversal(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame,
        lookback: int = 1,
        volume_spike_threshold: float = 3.0,
        return_threshold_std: float = 1.5,
        vol_ma_window: int = 21,
    ) -> pd.DataFrame:
        """
        Reversal after volume spikes (potential forced liquidations).

        Triggered when both conditions hold simultaneously:
        1. Volume spike: V_t / MA(V) > volume_spike_threshold
        2. Large return: |r_t| > return_threshold_std × rolling_std(r, vol_ma_window)

        Theory: liquidation cascades (margin calls, DeFi liquidations) cause
        uninformed forced selling → prices overshoot → subsequent reversal.

        Signal = ranked(-return) where both conditions met, else 0.

        Args:
            returns: log return panel
            volume: volume panel
            lookback: return accumulation window before testing the condition
            volume_spike_threshold: V/MA threshold for "spike"
            return_threshold_std: z-score magnitude to qualify as "large return"
            vol_ma_window: window for volume MA and return std estimation

        Returns:
            Signal panel (0 where conditions not met, ranked reversal elsewhere)
        """
        cumret = returns.rolling(window=lookback, min_periods=1).sum()
        ranked_rev = _rank(-cumret)

        # Volume spike condition
        vol_ma = volume.rolling(window=vol_ma_window, min_periods=max(1, vol_ma_window // 2)).mean()
        vol_ratio = volume / vol_ma.replace(0, np.nan)

        # Large return condition (|r| > threshold_std × rolling std)
        ret_std = returns.rolling(window=vol_ma_window, min_periods=vol_ma_window // 2).std()
        ret_zscore = cumret.abs() / ret_std.replace(0, np.nan)

        common = returns.columns.intersection(volume.columns)
        vol_ratio_aligned = vol_ratio[common].reindex_like(ranked_rev[common]).fillna(0)
        ret_zscore_common = ret_zscore[common].reindex_like(ranked_rev[common]).fillna(0)

        spike_condition = vol_ratio_aligned >= volume_spike_threshold
        large_return_condition = ret_zscore_common >= return_threshold_std
        both_conditions = spike_condition & large_return_condition

        result = ranked_rev.copy()
        result[common] = ranked_rev[common].where(both_conditions, other=0.0)
        # Columns not in common get 0
        extra = [c for c in ranked_rev.columns if c not in common]
        if extra:
            result[extra] = 0.0
        return result

    # ------------------------------------------------------------------
    # Legacy methods (from Prompt 1 — kept for compatibility)
    # ------------------------------------------------------------------

    def weekly_reversal(self, returns: pd.DataFrame) -> pd.DataFrame:
        """5-day reversal signal."""
        return self.short_term_reversal(returns, lookback=5)

    def biweekly_reversal(self, returns: pd.DataFrame) -> pd.DataFrame:
        """10-day reversal signal."""
        return self.short_term_reversal(returns, lookback=10)

    def vol_adjusted_reversal(
        self, returns: pd.DataFrame, lookback: int = 5, vol_window: int = 21
    ) -> pd.DataFrame:
        """Reversal scaled by volatility: -cumret / vol."""
        cumret = returns.rolling(window=lookback, min_periods=1).sum()
        vol = returns.rolling(window=vol_window, min_periods=vol_window // 2).std()
        raw = -cumret / vol.replace(0, np.nan)
        return _rank(raw)

    def bollinger_reversal(
        self, prices: pd.DataFrame, window: int = 20, n_std: float = 2.0
    ) -> pd.DataFrame:
        """Bollinger Band reversal: -(price - MA) / (n_std × std)."""
        ma = prices.rolling(window=window, min_periods=window // 2).mean()
        std = prices.rolling(window=window, min_periods=window // 2).std()
        raw = -((prices - ma) / (n_std * std.replace(0, np.nan)))
        return _rank(raw)

    def large_move_reversal(
        self, returns: pd.DataFrame, lookback: int = 1,
        threshold_std: float = 2.0, vol_window: int = 21,
    ) -> pd.DataFrame:
        """Reversal for abnormally large moves only (others → NaN)."""
        cumret = returns.rolling(window=lookback, min_periods=1).sum()
        vol = returns.rolling(window=vol_window, min_periods=vol_window // 2).std()
        zscore = cumret / vol.replace(0, np.nan)
        extreme_mask = zscore.abs() >= threshold_std
        raw = -zscore.where(extreme_mask, other=np.nan)
        return _rank(raw)
