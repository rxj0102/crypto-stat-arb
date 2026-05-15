"""
Seasonality and calendar-based signals.

Key questions:
- Do crypto returns vary by day of week? (institutional vs retail)
- Do intraday patterns exist? (US/Europe/Asia hours)
- Is there month-end rebalancing pressure?
- Is momentum stronger during weekdays or weekends?

All historical averages are computed using only past data to prevent
look-ahead bias. The signal at time t uses only information up to t-1.
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


class SeasonalitySignals:
    """
    Calendar and seasonality signals.

    Signals represent the expected excess return based on historical
    calendar patterns. They are best used as modifiers / tilts on top of
    directional signals rather than as standalone strategies.

    All estimates are purely backward-looking: the signal at time t is
    based on the average return for the same calendar period observed
    in the rolling estimation window ending at t-1.
    """

    # ------------------------------------------------------------------
    # Day-of-week momentum / reversal
    # ------------------------------------------------------------------

    def day_of_week_momentum(
        self,
        returns: pd.DataFrame,
        estimation_window: int = 252,
        min_obs: int = 4,
    ) -> pd.DataFrame:
        """
        Weekend vs weekday seasonality signal.

        Hypothesis: institutional traders active on weekdays create
        different patterns than retail-dominated weekends.

        At each date t, the signal for asset i is the historical average
        return observed on the same weekday, computed over the prior
        ``estimation_window`` periods.

        Signal = deviation from the overall asset mean → captures the
        day-of-week effect relative to the typical return level.

        No look-ahead: signal at t uses weekday averages computed from
        data ending at t-1.

        Args:
            returns: log return panel (daily frequency expected)
            estimation_window: rolling look-back for pattern estimation
            min_obs: minimum same-weekday observations to produce a signal

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        returns = returns.copy()
        returns.index = pd.DatetimeIndex(returns.index)
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for i in range(estimation_window, len(returns)):
            date = returns.index[i]
            current_dow = date.dayofweek
            window = returns.iloc[i - estimation_window: i]
            dow_mask = window.index.dayofweek == current_dow
            if dow_mask.sum() < min_obs:
                continue
            # Expected return on this weekday vs unconditional mean
            dow_avg = window[dow_mask].mean()
            overall_avg = window.mean()
            signal.iloc[i] = dow_avg - overall_avg  # excess above unconditional

        return _rank(signal)

    # ------------------------------------------------------------------
    # Intraday (hour-of-day) seasonality
    # ------------------------------------------------------------------

    def hour_of_day_signal(
        self,
        hourly_returns: pd.DataFrame,
        estimation_window_days: int = 30,
        min_obs: int = 3,
    ) -> pd.DataFrame:
        """
        Intraday seasonality signal from hourly return patterns.

        Computes the average return by hour of day (UTC) for each asset
        over the prior ``estimation_window_days`` days and assigns the
        expected excess return as the signal for each period.

        US hours (14:00–21:00 UTC) and Asia hours (00:00–08:00 UTC)
        exhibit different dynamics due to varying market participant mix.

        No look-ahead: estimates use only past hourly data.

        Args:
            hourly_returns: hourly log return panel with UTC DatetimeIndex
            estimation_window_days: look-back in calendar days
            min_obs: minimum same-hour observations needed

        Returns:
            Signal panel (same shape as hourly_returns), ranked ∈ [-1, 1]
        """
        hourly_returns = hourly_returns.copy()
        hourly_returns.index = pd.DatetimeIndex(hourly_returns.index)
        window_periods = estimation_window_days * 24
        signal = pd.DataFrame(np.nan, index=hourly_returns.index,
                               columns=hourly_returns.columns)

        for i in range(window_periods, len(hourly_returns)):
            current_hour = hourly_returns.index[i].hour
            window = hourly_returns.iloc[i - window_periods: i]
            hour_mask = window.index.hour == current_hour
            if hour_mask.sum() < min_obs:
                continue
            hour_avg = window[hour_mask].mean()
            overall_avg = window.mean()
            signal.iloc[i] = hour_avg - overall_avg

        return _rank(signal)

    # ------------------------------------------------------------------
    # Month-end effect
    # ------------------------------------------------------------------

    def month_end_effect(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        window_days: int = 3,
        estimation_window: int = 252,
        min_obs: int = 2,
    ) -> pd.DataFrame:
        """
        Month-end rebalancing signal.

        Institutional products (ETFs, index funds, crypto funds) often
        rebalance at month-end, creating predictable buying/selling pressure.
        The last ``window_days`` trading days of each month are flagged.

        Signal = historical average month-end excess return per asset,
        computed from the prior ``estimation_window`` periods.
        Non-month-end dates receive a signal of 0.

        Args:
            returns: log return panel (daily frequency)
            prices: close price panel (used only for index alignment)
            window_days: number of days before month-end to flag
            estimation_window: look-back for historical average estimation
            min_obs: minimum month-end observations

        Returns:
            Signal panel (0 outside month-end window, ranked values within)
        """
        returns = returns.copy()
        returns.index = pd.DatetimeIndex(returns.index)
        idx = returns.index

        # Days remaining until end of month
        days_remaining = idx.days_in_month - idx.day
        is_month_end = pd.Series(days_remaining <= window_days, index=returns.index)

        signal = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

        for i in range(estimation_window, len(returns)):
            if not is_month_end.iloc[i]:
                continue
            date = returns.index[i]
            window = returns.iloc[i - estimation_window: i]
            window_month_end = is_month_end.iloc[i - estimation_window: i]
            if window_month_end.sum() < min_obs:
                continue
            # Average return during month-end periods in the window
            me_returns = window[window_month_end.values]
            signal.iloc[i] = me_returns.mean()

        # Rank only the non-zero dates; zero dates stay 0
        ranked = _rank(signal.replace(0, np.nan))
        ranked = ranked.fillna(0).where(is_month_end, other=0.0)
        return ranked

    # ------------------------------------------------------------------
    # Momentum filtered to weekday / weekend
    # ------------------------------------------------------------------

    def momentum_seasonality(
        self,
        returns: pd.DataFrame,
        lookback: int = 21,
        weekday_only: bool = False,
        weekend_only: bool = False,
    ) -> pd.DataFrame:
        """
        Momentum signal restricted to weekday-only or weekend-only returns.

        Tests whether momentum is stronger during institutional trading days
        (weekdays) or retail-dominated weekends.

        If both flags are False, computes standard momentum on all days.
        The flag filters which *input return days* contribute to the
        cumulative momentum calculation.

        Args:
            returns: log return panel (daily frequency)
            lookback: momentum accumulation window
            weekday_only: if True, zero out weekend returns before cumulation
            weekend_only: if True, zero out weekday returns before cumulation

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        if weekday_only and weekend_only:
            raise ValueError("weekday_only and weekend_only cannot both be True")

        returns = returns.copy()
        returns.index = pd.DatetimeIndex(returns.index)

        if weekday_only:
            is_weekend = returns.index.dayofweek >= 5  # Sat=5, Sun=6
            filtered = returns.copy()
            filtered.iloc[is_weekend] = 0.0
        elif weekend_only:
            is_weekday = returns.index.dayofweek < 5
            filtered = returns.copy()
            filtered.iloc[is_weekday] = 0.0
        else:
            filtered = returns

        cumret = filtered.rolling(window=lookback, min_periods=lookback // 2).sum()
        return _rank(cumret)

    # ------------------------------------------------------------------
    # Legacy methods (from Prompt 1 — kept for compatibility)
    # ------------------------------------------------------------------

    def january_effect(self, returns: pd.DataFrame) -> pd.Series:
        """January indicator: 1.0 in January, 0.0 otherwise."""
        idx = pd.DatetimeIndex(returns.index)
        return pd.Series((idx.month == 1).astype(float), index=returns.index)

    def weekend_indicator(self, index: pd.DatetimeIndex) -> pd.Series:
        """True on Friday and Saturday (crypto weekends)."""
        dow = pd.Series(index.dayofweek, index=index)
        return dow.isin([4, 5])

    def month_end_rebalancing(
        self, index: pd.DatetimeIndex, days_before_end: int = 3
    ) -> pd.Series:
        """Boolean indicator for last ``days_before_end`` days of month."""
        idx = pd.DatetimeIndex(index)
        days_in_month = idx.days_in_month
        days_remaining = days_in_month - idx.day
        return pd.Series(days_remaining <= days_before_end, index=index)
