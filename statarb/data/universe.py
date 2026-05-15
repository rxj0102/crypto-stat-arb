"""Tradable universe construction and filtering."""

from typing import List, Optional

import numpy as np
import pandas as pd

from statarb.utils import get_logger

logger = get_logger(__name__)

# Stablecoin base symbols to exclude
_STABLECOINS = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD",
    "FRAX", "LUSD", "USDD", "SUSD", "CUSD", "XSGD",
}

# Wrapped / bridged token prefixes to exclude by default
_WRAPPED_PREFIXES = {"W", "CB", "ST", "R"}  # WBTC, CBETH, STETH, RETH etc.


class TradableUniverse:
    """
    Define and maintain the tradable universe of crypto assets.

    Filtering criteria applied in :meth:`apply_filters`:

    - Minimum daily dollar volume (default $10M) — ensures tradability
    - Minimum history length (default 365 days) — ensures signal validity
    - Not a stablecoin (USDT, USDC, DAI, …)
    - Optionally excludes wrapped / bridged tokens

    The universe can be **dynamic** (re-evaluated at each rebalance date
    using rolling window liquidity, preventing look-ahead bias) or
    **static** (a fixed list, simpler but ignores delistings / new listings).
    """

    def __init__(
        self,
        min_dollar_volume: float = 1e7,
        min_history_days: int = 365,
        exclude_stablecoins: bool = True,
        exclude_wrapped: bool = False,
        static_list: Optional[List[str]] = None,
    ):
        """
        Args:
            min_dollar_volume: minimum 30-day average daily dollar volume
            min_history_days: minimum number of trading days with data
            exclude_stablecoins: drop known stablecoin tickers
            exclude_wrapped: drop WBTC-style wrapped tokens
            static_list: if provided, skip dynamic filtering and use this list
        """
        self.min_dollar_volume = min_dollar_volume
        self.min_history_days = min_history_days
        self.exclude_stablecoins = exclude_stablecoins
        self.exclude_wrapped = exclude_wrapped
        self.static_list = static_list

    # ------------------------------------------------------------------
    # Universe construction
    # ------------------------------------------------------------------

    def get_universe(self, as_of_date: Optional[str] = None) -> List[str]:
        """
        Return symbol list for the tradable universe.

        If ``static_list`` was provided at init, return that directly.
        Otherwise, raises NotImplementedError — dynamic universe requires
        calling :meth:`apply_filters` with actual price/volume panels.

        Args:
            as_of_date: ignored when static_list is set; placeholder for
                        dynamic universe support

        Returns:
            List of symbol strings (e.g. ['BTC/USDT', 'ETH/USDT', ...])
        """
        if self.static_list is not None:
            return list(self.static_list)
        raise NotImplementedError(
            "Dynamic universe requires volume/price data. "
            "Call apply_filters(return_panel, volume_panel) instead, "
            "or pass static_list= at construction."
        )

    # ------------------------------------------------------------------
    # Panel filtering (prevents look-ahead bias)
    # ------------------------------------------------------------------

    def apply_filters(
        self,
        return_panel: pd.DataFrame,
        volume_panel: pd.DataFrame,
        liquidity_window: int = 30,
    ) -> pd.DataFrame:
        """
        Apply dynamic universe filters to a return panel in-place.

        For each date t, a symbol is *eligible* only if:
        1. It passes the stablecoin / wrapped-token name filter.
        2. Its rolling ``liquidity_window``-day average dollar volume
           (computed from data *prior to* t) exceeds ``min_dollar_volume``.
        3. It has at least ``min_history_days`` non-NaN observations
           up to and including t.

        Ineligible returns are set to NaN so they do not contribute to
        cross-sectional signals or portfolio construction.

        Args:
            return_panel: DataFrame[dates × symbols] of returns
            volume_panel: DataFrame[dates × symbols] of dollar volumes
            liquidity_window: look-back window for volume filter (days)

        Returns:
            Filtered return panel with ineligible cells set to NaN.
        """
        filtered = return_panel.copy()

        # ---- 1. Name-based exclusion ----
        eligible_symbols = [
            sym for sym in filtered.columns
            if self._name_eligible(sym)
        ]
        ineligible = set(filtered.columns) - set(eligible_symbols)
        if ineligible:
            logger.info("Name-filtered out: %s", sorted(ineligible))
        filtered = filtered[eligible_symbols]

        # ---- 2. Align volume panel to filtered symbols ----
        common_syms = filtered.columns.intersection(volume_panel.columns)
        vol = volume_panel[common_syms].reindex(filtered.index)

        # ---- 3. Rolling liquidity mask ----
        rolling_vol = vol.rolling(window=liquidity_window, min_periods=1).mean()
        liquidity_mask = rolling_vol >= self.min_dollar_volume

        # ---- 4. History length mask ----
        cumulative_obs = (~filtered[common_syms].isna()).cumsum()
        history_mask = cumulative_obs >= self.min_history_days

        # Combine masks — must satisfy both
        combined_mask = liquidity_mask & history_mask
        filtered[common_syms] = filtered[common_syms].where(combined_mask, other=np.nan)

        logger.debug(
            "Universe filter applied: %d symbols remain (avg eligible per date: %.0f)",
            len(eligible_symbols),
            combined_mask.sum(axis=1).mean(),
        )
        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _name_eligible(self, symbol: str) -> bool:
        """True if the symbol passes stablecoin / wrapped-token name checks."""
        base = symbol.split("/")[0].upper()

        if self.exclude_stablecoins and base in _STABLECOINS:
            return False

        if self.exclude_wrapped:
            for prefix in _WRAPPED_PREFIXES:
                if base.startswith(prefix) and len(base) > len(prefix):
                    return False

        return True
