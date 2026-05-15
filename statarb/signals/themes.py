"""
Investment theme and style signals.

Crypto assets cluster into thematic groups (DeFi, L1s, L2s, gaming, etc.).
Theme-level momentum / reversal can be stronger than single-asset signals.
Style factors (size, value-proxy, volatility) also carry cross-sectional alpha.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from statarb.data.features import FeatureEngine
from statarb.utils import get_logger

logger = get_logger(__name__)
_fe = FeatureEngine()

# Predefined thematic clusters
CRYPTO_THEMES: Dict[str, List[str]] = {
    "L1": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "AVAX/USDT",
        "DOT/USDT", "NEAR/USDT", "ATOM/USDT", "ALGO/USDT", "ETC/USDT",
        "XTZ/USDT", "EOS/USDT", "EGLD/USDT", "THETA/USDT", "FTM/USDT",
        "FLOW/USDT", "ICP/USDT", "HBAR/USDT", "VET/USDT",
    ],
    "L2": [
        "MATIC/USDT", "ARB/USDT", "OP/USDT",
    ],
    "DeFi": [
        "UNI/USDT", "AAVE/USDT", "MKR/USDT", "SNX/USDT", "COMP/USDT",
        "CRV/USDT", "GRT/USDT", "LDO/USDT", "RUNE/USDT", "INJ/USDT",
    ],
    "Gaming_Metaverse": [
        "SAND/USDT", "MANA/USDT", "AXS/USDT", "ENJ/USDT", "GALA/USDT",
        "APE/USDT", "CHZ/USDT",
    ],
    "PoW": [
        "BTC/USDT", "LTC/USDT", "BCH/USDT", "ZEC/USDT", "DASH/USDT",
        "XMR/USDT",
    ],
    "Infrastructure": [
        "LINK/USDT", "FIL/USDT", "GRT/USDT", "THETA/USDT", "HBAR/USDT",
    ],
    "Exchange_Tokens": [
        "BNB/USDT",
    ],
    "Privacy": [
        "XMR/USDT", "ZEC/USDT",
    ],
    "Payments": [
        "XRP/USDT", "LTC/USDT", "BCH/USDT", "XLM/USDT",
    ],
}


class ThemeSignals:
    """
    Theme-level and style-factor signals for crypto.

    These signals operate at two levels:
    1. **Theme momentum**: buy the winning theme, sell the losing theme
    2. **Within-theme relative value**: buy theme laggards, sell theme leaders
    3. **Style factors**: size, volatility, liquidity tilts
    """

    def __init__(self, themes: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            themes: custom theme dictionary; defaults to CRYPTO_THEMES
        """
        self.themes = themes or CRYPTO_THEMES

    # ------------------------------------------------------------------
    # Theme-level return computation
    # ------------------------------------------------------------------

    def theme_returns(
        self, returns: pd.DataFrame, lookback: int = 21
    ) -> pd.DataFrame:
        """
        Compute equal-weighted cumulative returns for each theme.

        Args:
            returns: individual asset return panel
            lookback: return horizon

        Returns:
            DataFrame[dates × themes] of cumulative returns.
        """
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        theme_rets: Dict[str, pd.Series] = {}

        for theme_name, members in self.themes.items():
            valid = [m for m in members if m in cumret.columns]
            if not valid:
                continue
            theme_rets[theme_name] = cumret[valid].mean(axis=1)

        return pd.DataFrame(theme_rets, index=returns.index)

    # ------------------------------------------------------------------
    # Theme momentum signal
    # ------------------------------------------------------------------

    def theme_momentum_signal(
        self,
        returns: pd.DataFrame,
        lookback: int = 21,
        skip: int = 0,
    ) -> pd.DataFrame:
        """
        Cross-theme momentum: assign each asset the momentum score of its theme.

        An asset inherits the cross-sectional rank of its primary theme.
        Assets in winning themes get positive scores; losing themes get negative.

        Args:
            returns: individual asset return panel
            lookback: theme momentum look-back
            skip: recent periods to skip (microstructure)

        Returns:
            Signal panel (dates × symbols) with theme momentum scores.
        """
        theme_ret = self.theme_returns(returns, lookback)
        if skip > 0:
            recent_theme_ret = self.theme_returns(returns, skip)
            theme_ret = theme_ret - recent_theme_ret

        # Cross-sectional rank across themes
        theme_rank = _fe.cross_sectional_rank(theme_ret)

        # Assign theme rank to each member asset
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        primary_theme = self._primary_theme_map()
        for sym in returns.columns:
            if sym in primary_theme:
                theme = primary_theme[sym]
                if theme in theme_rank.columns:
                    signal[sym] = theme_rank[theme]

        return signal

    # ------------------------------------------------------------------
    # Within-theme relative value
    # ------------------------------------------------------------------

    def within_theme_relative_value(
        self, returns: pd.DataFrame, lookback: int = 21
    ) -> pd.DataFrame:
        """
        Within-theme relative value: buy theme laggards, sell theme leaders.

        Signal for each asset = -(return_asset - return_theme_average).
        Mean-reversion within a thematic cluster.

        Returns:
            Signal panel (dates × symbols).
        """
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for _theme_name, members in self.themes.items():
            valid = [m for m in members if m in cumret.columns]
            if len(valid) < 2:
                continue
            theme_avg = cumret[valid].mean(axis=1)
            for sym in valid:
                relative_perf = cumret[sym] - theme_avg
                signal[sym] = -relative_perf  # revert toward theme mean

        return signal

    # ------------------------------------------------------------------
    # Style factors
    # ------------------------------------------------------------------

    def low_volatility_factor(
        self, returns: pd.DataFrame, vol_window: int = 63
    ) -> pd.DataFrame:
        """
        Low-volatility factor: overweight low-vol assets.

        Low-vol anomaly: lower-volatility assets have better risk-adjusted
        returns than theory predicts (Ang et al. 2006).

        Signal = -realized_vol (higher signal → lower volatility → buy).
        """
        vol = _fe.realized_volatility(returns, vol_window)
        return -vol

    def size_factor(self, dollar_volume: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Size factor: proxy for market cap using dollar volume.

        Small-cap crypto tends to have higher momentum but also higher risk.
        This factor can be used to tilt toward smaller assets.

        Signal = -log(avg_dollar_volume) → negative score for large-caps.
        """
        avg_vol = dollar_volume.rolling(window=window, min_periods=window // 2).mean()
        return -np.log1p(avg_vol)

    def momentum_reversal_blend(
        self,
        returns: pd.DataFrame,
        momentum_lookback: int = 63,
        reversal_lookback: int = 5,
        momentum_weight: float = 0.6,
    ) -> pd.DataFrame:
        """
        Blend of momentum (medium-term) and reversal (short-term) signals.

        At the individual stock level, momentum and reversal are complementary:
        - Medium-term momentum → directional bet
        - Short-term reversal → timing / entry improvement

        Signal = momentum_weight × momentum + (1 - momentum_weight) × reversal

        Args:
            momentum_weight: weight on the momentum component (0 to 1)
        """
        mom = returns.rolling(window=momentum_lookback, min_periods=momentum_lookback // 2).sum()
        rev = -returns.rolling(window=reversal_lookback, min_periods=1).sum()
        blend = momentum_weight * mom + (1 - momentum_weight) * rev
        return blend

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _primary_theme_map(self) -> Dict[str, str]:
        """Map each symbol to its primary (first-listed) theme."""
        seen = {}
        for theme, members in self.themes.items():
            for sym in members:
                if sym not in seen:
                    seen[sym] = theme
        return seen

    def get_theme_members(self, theme: str) -> List[str]:
        """Return the list of symbols in a given theme."""
        return self.themes.get(theme, [])

    def list_themes(self) -> List[str]:
        """Return all theme names."""
        return list(self.themes.keys())
