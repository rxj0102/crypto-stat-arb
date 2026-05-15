"""
Seasonality and calendar-based signals.

Crypto markets exhibit well-documented calendar patterns:
- "Sell in May" effects
- Weekend effects (retail-driven moves Fri–Sun)
- Month-end / month-start rebalancing flows
- January effect (tax-loss selling reversal)
- Intraday patterns (hourly mean-reversion peaks)
"""

import numpy as np
import pandas as pd

from statarb.utils import get_logger

logger = get_logger(__name__)


class SeasonalitySignals:
    """
    Calendar and seasonality signals.

    These are primarily used as *regime indicators* or *signal multipliers*
    rather than standalone directional signals. E.g., amplify momentum
    signals during historically strong months, suppress during weak ones.

    All signals return DataFrames[dates × symbols] or Series[dates].
    """

    # ------------------------------------------------------------------
    # Day-of-week effects
    # ------------------------------------------------------------------

    def day_of_week_signal(
        self,
        returns: pd.DataFrame,
        estimation_window: int = 252,
    ) -> pd.DataFrame:
        """
        Signal based on estimated day-of-week return patterns.

        Compute the historical average return by weekday over
        ``estimation_window`` days, then assign the expected day return
        as a signal (buy on historically positive days).

        No look-ahead: at each date t, only uses data up to t-1.

        Args:
            returns: return panel (daily or hourly)
            estimation_window: rolling window for pattern estimation

        Returns:
            Signal panel where each day's value is the historical mean
            return for that weekday. Positive = buy, negative = sell.
        """
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        returns_copy = returns.copy()
        returns_copy.index = pd.DatetimeIndex(returns_copy.index)

        for i in range(estimation_window, len(returns_copy)):
            date = returns_copy.index[i]
            current_dow = date.dayofweek
            window = returns_copy.iloc[i - estimation_window: i]
            dow_mask = window.index.dayofweek == current_dow
            if dow_mask.sum() < 4:
                continue
            avg = window[dow_mask].mean()
            signal.iloc[i] = avg

        return signal

    def weekend_indicator(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Boolean indicator: True on Friday and Saturday (crypto weekends).

        Crypto trades 24/7; "weekend" here means Fri/Sat when retail
        activity increases and institutional liquidity decreases.

        Returns:
            Series[bool] aligned to ``index``.
        """
        dow = pd.Series(index.dayofweek, index=index)
        return dow.isin([4, 5])  # Friday=4, Saturday=5

    def monday_reversal(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Monday reversal signal: reverse Friday's return on Monday open.

        Crypto often sees retail-driven moves over weekends. Professionals
        fade these moves at Monday open.

        Returns:
            Signal panel where Monday values = negative of prior Friday return.
            Non-Monday dates are NaN.
        """
        returns = returns.copy()
        returns.index = pd.DatetimeIndex(returns.index)
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for i, date in enumerate(returns.index):
            if date.dayofweek == 0:  # Monday
                # Find the most recent Friday
                for j in range(i - 1, max(0, i - 5), -1):
                    if returns.index[j].dayofweek == 4:
                        signal.iloc[i] = -returns.iloc[j]
                        break

        return signal

    # ------------------------------------------------------------------
    # Month-of-year effects
    # ------------------------------------------------------------------

    def month_of_year_signal(
        self,
        returns: pd.DataFrame,
        estimation_window_years: int = 3,
    ) -> pd.DataFrame:
        """
        Signal based on estimated month-of-year return patterns.

        Historical average return in the same calendar month, computed
        using data from prior years only (no look-ahead).

        Args:
            returns: return panel (daily frequency recommended)
            estimation_window_years: years of history to use

        Returns:
            Signal panel — positive in historically strong months.
        """
        returns = returns.copy()
        returns.index = pd.DatetimeIndex(returns.index)
        estimation_window_days = estimation_window_years * 252
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for i in range(estimation_window_days, len(returns)):
            date = returns.index[i]
            current_month = date.month
            window = returns.iloc[max(0, i - estimation_window_days): i]
            month_mask = window.index.month == current_month
            if month_mask.sum() < 5:
                continue
            avg = window[month_mask].mean()
            signal.iloc[i] = avg

        return signal

    def january_effect(self, returns: pd.DataFrame) -> pd.Series:
        """
        January indicator: 1.0 in January, 0.0 otherwise.

        January effect: assets that fell in December tend to rebound in
        January due to tax-loss selling reversal. Use as a regime multiplier.

        Returns:
            Series[float] aligned to returns.index.
        """
        idx = pd.DatetimeIndex(returns.index)
        return pd.Series((idx.month == 1).astype(float), index=returns.index)

    # ------------------------------------------------------------------
    # Intraday seasonality (hourly data)
    # ------------------------------------------------------------------

    def hourly_seasonality_signal(
        self,
        returns: pd.DataFrame,
        estimation_window: int = 30,
    ) -> pd.DataFrame:
        """
        Signal based on intraday (hourly) return patterns.

        Certain hours of the day show systematic biases (e.g., Asian session,
        US market open). Compute historical hourly average and use as signal.

        Expects hourly-frequency ``returns`` with a UTC DatetimeIndex.

        Args:
            returns: hourly return panel
            estimation_window: rolling window in days (each day = 24 periods)

        Returns:
            Signal panel where each hour's value is the historical mean
            return for that hour of day.
        """
        returns = returns.copy()
        returns.index = pd.DatetimeIndex(returns.index)
        window_periods = estimation_window * 24
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for i in range(window_periods, len(returns)):
            current_hour = returns.index[i].hour
            window = returns.iloc[i - window_periods: i]
            hour_mask = window.index.hour == current_hour
            if hour_mask.sum() < 3:
                continue
            avg = window[hour_mask].mean()
            signal.iloc[i] = avg

        return signal

    # ------------------------------------------------------------------
    # Quarter-end / month-end effects
    # ------------------------------------------------------------------

    def month_end_rebalancing(
        self, index: pd.DatetimeIndex, days_before_end: int = 3
    ) -> pd.Series:
        """
        Indicator for the last ``days_before_end`` trading days of each month.

        Institutional rebalancing at month-end creates predictable price
        pressure. This indicator can be used to suppress other signals
        (avoid trading into rebalancing flows).

        Returns:
            Series[bool] indicating month-end rebalancing window.
        """
        idx = pd.DatetimeIndex(index)
        # Days remaining until end of month
        days_in_month = idx.days_in_month
        days_remaining = days_in_month - idx.day
        return pd.Series(days_remaining <= days_before_end, index=index)
