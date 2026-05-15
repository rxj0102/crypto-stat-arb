"""
Position sizing and portfolio construction.

Converts raw signal scores into portfolio weights. Supports several
construction methods appropriate for unconstrained stat-arb portfolios.
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd

from statarb.utils import get_logger

logger = get_logger(__name__)


class PositionManager:
    """
    Convert signal scores into portfolio weights.

    Available construction methods:

    - ``rank``: positions proportional to cross-sectional rank (default)
    - ``signal``: positions proportional to raw signal value
    - ``top_bottom``: binary long-top-k / short-bottom-k book
    - ``equal_weight``: equal weight long/short for non-zero signals
    - ``vol_target``: scale positions to target portfolio volatility

    All methods return dollar-neutral portfolios (sum of longs = sum of
    shorts, subject to signal availability).
    """

    def __init__(
        self,
        method: Literal["rank", "signal", "top_bottom", "equal_weight", "vol_target"] = "rank",
        max_position: float = 0.10,
        top_k: int = 10,
        vol_target: float = 0.15,
        vol_window: int = 63,
    ):
        """
        Args:
            method: weight construction method
            max_position: maximum weight per asset (absolute value)
            top_k: number of longs and shorts for top_bottom method
            vol_target: annualised portfolio vol target (for vol_target method)
            vol_window: window for portfolio vol estimation
        """
        self.method = method
        self.max_position = max_position
        self.top_k = top_k
        self.vol_target = vol_target
        self.vol_window = vol_window

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_positions(
        self,
        signal: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute portfolio weights from a signal panel.

        Args:
            signal: cross-sectional signal panel (dates × symbols)
            returns: return panel (required for vol_target method)

        Returns:
            Weight panel (dates × symbols), dollar-neutral per date.
            Values in [-max_position, +max_position].
        """
        dispatch = {
            "rank": self._rank_weights,
            "signal": self._signal_weights,
            "top_bottom": self._top_bottom_weights,
            "equal_weight": self._equal_weight_weights,
            "vol_target": self._vol_target_weights,
        }
        if self.method not in dispatch:
            raise ValueError(f"Unknown method '{self.method}'")

        if self.method == "vol_target" and returns is None:
            raise ValueError("vol_target method requires returns panel")

        weights = dispatch[self.method](signal, returns)
        return weights.clip(-self.max_position, self.max_position)

    # ------------------------------------------------------------------
    # Construction methods
    # ------------------------------------------------------------------

    def _rank_weights(
        self, signal: pd.DataFrame, _returns: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Weights proportional to cross-sectional signal rank (demeaned)."""

        def _row_rank(row: pd.Series) -> pd.Series:
            valid = row.dropna()
            if len(valid) < 2:
                return row * np.nan
            n = len(valid)
            ranks = valid.rank(method="average")  # 1 to N
            # Centre: subtract mean rank so long + short ≈ 0
            demeaned = ranks - ranks.mean()
            # Normalise so max |weight| = max_position
            scale = self.max_position / demeaned.abs().max() if demeaned.abs().max() > 0 else 1
            return row.copy().where(row.isna(), other=(demeaned * scale).reindex(row.index))

        return signal.apply(_row_rank, axis=1)

    def _signal_weights(
        self, signal: pd.DataFrame, _returns: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Weights proportional to raw signal value (demeaned, normalised)."""
        demeaned = signal.sub(signal.mean(axis=1), axis=0)
        row_max = demeaned.abs().max(axis=1).replace(0, np.nan)
        normalised = demeaned.div(row_max, axis=0) * self.max_position
        return normalised

    def _top_bottom_weights(
        self, signal: pd.DataFrame, _returns: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Equal-weight long top-k, short bottom-k by signal rank."""
        weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        k = self.top_k
        for date, row in signal.iterrows():
            valid = row.dropna()
            if len(valid) < 2 * k:
                continue
            top = valid.nlargest(k).index
            bottom = valid.nsmallest(k).index
            weights.loc[date, top] = self.max_position
            weights.loc[date, bottom] = -self.max_position
        return weights

    def _equal_weight_weights(
        self, signal: pd.DataFrame, _returns: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Equal-weight positions: +max_pos for positive signals, -max_pos for negative."""
        weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        weights[signal > 0] = self.max_position
        weights[signal < 0] = -self.max_position
        weights[signal.isna()] = 0.0
        # Dollar-neutralise each row
        for date, row in weights.iterrows():
            n_long = (row > 0).sum()
            n_short = (row < 0).sum()
            if n_long > 0:
                weights.loc[date, row > 0] = self.max_position / n_long
            if n_short > 0:
                weights.loc[date, row < 0] = -self.max_position / n_short
        return weights

    def _vol_target_weights(
        self, signal: pd.DataFrame, returns: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Scale rank weights to target annualised portfolio volatility."""
        base_weights = self._rank_weights(signal, None)
        if returns is None:
            return base_weights

        # Estimate portfolio vol on lagged returns
        port_ret = (base_weights.shift(1) * returns).sum(axis=1)
        port_vol = port_ret.rolling(window=self.vol_window, min_periods=20).std() * np.sqrt(252)

        # Daily vol target = annual vol / sqrt(252)
        daily_target = self.vol_target / np.sqrt(252)
        daily_est = port_vol / np.sqrt(252)

        scale = (daily_target / daily_est.replace(0, np.nan)).clip(0.1, 5.0)
        scaled = base_weights.mul(scale, axis=0)
        return scaled

    # ------------------------------------------------------------------
    # Portfolio diagnostics
    # ------------------------------------------------------------------

    def turnover(self, positions: pd.DataFrame) -> pd.Series:
        """
        Compute daily one-way turnover (sum of absolute weight changes / 2).

        Returns:
            Series of daily turnover rates.
        """
        daily_changes = positions.diff().abs().sum(axis=1) / 2
        return daily_changes

    def gross_exposure(self, positions: pd.DataFrame) -> pd.Series:
        """Sum of absolute weights at each date."""
        return positions.abs().sum(axis=1)

    def net_exposure(self, positions: pd.DataFrame) -> pd.Series:
        """Sum of signed weights at each date (long minus short)."""
        return positions.sum(axis=1)
