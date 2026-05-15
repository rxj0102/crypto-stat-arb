"""Raw feature engineering from price-volume data."""

from typing import Optional

import numpy as np
import pandas as pd

from statarb.utils import get_logger

logger = get_logger(__name__)


class FeatureEngine:
    """
    Compute raw features from price-volume panels.

    All methods operate on DataFrames where rows are dates and columns are
    symbols. Features are computed without forward-looking data — rolling
    windows only look backward.

    Cross-sectional ranking (see :meth:`cross_sectional_rank`) is the
    canonical pre-processing step before using any feature as a signal:
    it removes the level effect and focuses on relative ordering.
    """

    # ------------------------------------------------------------------
    # Return computation
    # ------------------------------------------------------------------

    def log_returns(self, prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """Compute log returns: log(P_t / P_{t-periods})."""
        return np.log(prices / prices.shift(periods))

    def simple_returns(self, prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """Compute simple (arithmetic) returns: (P_t - P_{t-periods}) / P_{t-periods}."""
        return prices.pct_change(periods)

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def realized_volatility(
        self, returns: pd.DataFrame, window: int = 21, annualize: bool = False,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """
        Rolling realized volatility (std dev of returns).

        Args:
            window: look-back window in periods
            annualize: if True, multiply by sqrt(periods_per_year)
            periods_per_year: trading periods per year for annualisation

        Returns:
            DataFrame of same shape as ``returns``
        """
        vol = returns.rolling(window=window, min_periods=max(2, window // 2)).std()
        if annualize:
            vol = vol * np.sqrt(periods_per_year)
        return vol

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------

    def dollar_volume(
        self, close: pd.DataFrame, volume: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute dollar volume: close × volume."""
        common = close.columns.intersection(volume.columns)
        return (close[common] * volume[common]).reindex(columns=close.columns)

    def volume_ma_ratio(self, volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """
        Volume relative to its rolling moving average: V_t / MA(V, window).

        Values > 1 indicate above-average activity, which can precede
        stronger momentum signals (informed trading hypothesis).
        """
        ma = volume.rolling(window=window, min_periods=max(1, window // 2)).mean()
        return volume / ma.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Cross-sectional features
    # ------------------------------------------------------------------

    def return_dispersion(
        self, returns: pd.DataFrame, window: Optional[int] = None
    ) -> pd.Series:
        """
        Cross-sectional standard deviation of returns at each date.

        High dispersion implies more opportunity for cross-sectional stat arb.

        Args:
            window: if provided, compute rolling dispersion over this window;
                    otherwise compute instantaneous cross-sectional std.

        Returns:
            Series indexed by date.
        """
        if window is None:
            return returns.std(axis=1)
        # Rolling cross-sectional std: use expanding mean-of-stacked approach
        return returns.rolling(window=window).std().mean(axis=1)

    def pairwise_correlation(
        self, returns: pd.DataFrame, window: int = 63
    ) -> pd.Series:
        """
        Rolling average pairwise correlation across all assets.

        High correlation → less diversification, potentially stronger
        mean-reversion in pairs trading.

        Args:
            window: rolling window length (periods)

        Returns:
            Series indexed by date, values in [-1, 1].
        """
        n = len(returns.columns)
        if n < 2:
            return pd.Series(np.nan, index=returns.index)

        def _mean_pairwise_corr(mat: np.ndarray) -> float:
            """Mean of upper triangle of correlation matrix."""
            corr = np.corrcoef(mat.T)
            upper = corr[np.triu_indices(corr.shape[0], k=1)]
            return float(np.nanmean(upper))

        # Apply rolling window
        result = pd.Series(index=returns.index, dtype=float)
        clean = returns.dropna(axis=1, how="all")
        for i in range(window, len(returns) + 1):
            block = clean.iloc[i - window: i].dropna(axis=1)
            if block.shape[1] >= 2:
                result.iloc[i - 1] = _mean_pairwise_corr(block.values)
        return result

    # ------------------------------------------------------------------
    # Cross-sectional ranking
    # ------------------------------------------------------------------

    def cross_sectional_rank(self, feature: pd.DataFrame) -> pd.DataFrame:
        """
        Rank the feature cross-sectionally at each date and normalise to [-1, 1].

        Formula: rank_normalised = rank / (N - 1) * 2 - 1
        where rank ∈ {0, 1, …, N-1} (0 = smallest, N-1 = largest).

        NaN values are excluded from ranking and left as NaN in the output.
        This is the standard approach in cross-sectional stat arb: it removes
        the market-level effect and focuses entirely on relative ordering.

        Returns:
            DataFrame of same shape as ``feature`` with values in [-1, 1].
        """

        def _rank_row(row: pd.Series) -> pd.Series:
            valid = row.dropna()
            if len(valid) < 2:
                return row * np.nan
            n = len(valid)
            ranks = valid.rank(method="average") - 1  # 0-based
            normalised = ranks / (n - 1) * 2 - 1
            return row.copy().where(row.isna(), other=normalised.reindex(row.index))

        return feature.apply(_rank_row, axis=1)

    # ------------------------------------------------------------------
    # Composite helpers used by signal generators
    # ------------------------------------------------------------------

    def rolling_mean(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """Rolling mean with min_periods = window // 2."""
        return data.rolling(window=window, min_periods=max(1, window // 2)).mean()

    def rolling_sum(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """Rolling sum with min_periods = window // 2."""
        return data.rolling(window=window, min_periods=max(1, window // 2)).sum()

    def expanding_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        """Expanding (historical) z-score: (x - expanding_mean) / expanding_std."""
        mu = data.expanding(min_periods=20).mean()
        sigma = data.expanding(min_periods=20).std()
        return (data - mu) / sigma.replace(0, np.nan)
