"""
Investment theme and style signals for crypto.

Assets cluster into thematic groups (L1s, DeFi, Gaming, etc.).
Theme-level momentum / reversal is often stronger than single-asset signals
because it captures narrative-driven flows across related assets.
"""

from typing import Dict, List, Optional

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


# Predefined thematic clusters (expanded for Prompt 2)
CRYPTO_THEMES: Dict[str, List[str]] = {
    "layer1": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "DOT/USDT",
        "ADA/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT", "ATOM/USDT",
        "ALGO/USDT", "ETC/USDT", "XTZ/USDT", "EOS/USDT", "EGLD/USDT",
        "THETA/USDT", "FTM/USDT", "FLOW/USDT", "ICP/USDT", "HBAR/USDT",
        "VET/USDT",
    ],
    "defi": [
        "UNI/USDT", "AAVE/USDT", "CRV/USDT", "MKR/USDT", "SNX/USDT",
        "COMP/USDT", "SUSHI/USDT", "LDO/USDT", "RUNE/USDT", "INJ/USDT",
        "GRT/USDT",
    ],
    "meme": [
        "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "FLOKI/USDT", "BONK/USDT",
    ],
    "ai": [
        "FET/USDT", "RNDR/USDT", "AGIX/USDT",
    ],
    "gaming": [
        "AXS/USDT", "SAND/USDT", "MANA/USDT", "IMX/USDT", "ENJ/USDT",
        "GALA/USDT", "APE/USDT", "CHZ/USDT",
    ],
    "infrastructure": [
        "LINK/USDT", "GRT/USDT", "FIL/USDT", "THETA/USDT", "HBAR/USDT",
        "ATOM/USDT",
    ],
    "l2": [
        "MATIC/USDT", "ARB/USDT", "OP/USDT",
    ],
    "pow": [
        "BTC/USDT", "LTC/USDT", "BCH/USDT", "ZEC/USDT", "DASH/USDT",
        "XMR/USDT",
    ],
    "payments": [
        "XRP/USDT", "LTC/USDT", "BCH/USDT",
    ],
}


class ThemeSignals:
    """
    Theme-level and style-factor signals for crypto.

    Operates at two levels:
    1. **Theme momentum**: buy assets in hot themes, short assets in cold themes
    2. **Within-theme relative value**: buy laggards within a theme (reversal)
    3. **Style factors**: low-vol, size, momentum-reversal blend
    """

    def __init__(self, themes: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            themes: custom theme dict; defaults to CRYPTO_THEMES
        """
        self.themes = themes or CRYPTO_THEMES

    # ------------------------------------------------------------------
    # Theme-level return computation (helper)
    # ------------------------------------------------------------------

    def theme_returns(
        self, returns: pd.DataFrame, lookback: int = 21
    ) -> pd.DataFrame:
        """
        Equal-weighted cumulative return for each theme.

        Returns:
            DataFrame[dates × theme_names]
        """
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        result: Dict[str, pd.Series] = {}
        for theme_name, members in self.themes.items():
            valid = [m for m in members if m in cumret.columns]
            if not valid:
                continue
            result[theme_name] = cumret[valid].mean(axis=1)
        return pd.DataFrame(result, index=returns.index)

    # ------------------------------------------------------------------
    # Theme momentum
    # ------------------------------------------------------------------

    def theme_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 21,
    ) -> pd.DataFrame:
        """
        Theme-level cross-sectional momentum.

        For each asset, the signal = its theme's average return over
        ``lookback`` periods, excluding the asset itself (leave-one-out).

        Long assets in hot themes, short assets in cold themes.

        Args:
            returns: log return panel (dates × symbols)
            lookback: cumulative return horizon for theme ranking

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for _theme, members in self.themes.items():
            valid = [m for m in members if m in cumret.columns]
            if len(valid) < 2:
                continue
            theme_sum = cumret[valid].sum(axis=1)
            for sym in valid:
                n = valid.index(sym)  # will use count not index
                n_others = len(valid) - 1
                if n_others < 1:
                    continue
                # Leave-one-out theme average
                sym_contrib = cumret[sym].fillna(0)
                loo_avg = (theme_sum - sym_contrib) / n_others
                signal[sym] = loo_avg

        return _rank(signal)

    # ------------------------------------------------------------------
    # Theme rotation
    # ------------------------------------------------------------------

    def theme_rotation(
        self,
        returns: pd.DataFrame,
        lookback: int = 7,
        holding: int = 7,
    ) -> pd.DataFrame:
        """
        Theme rotation: rank themes by past performance, overweight top themes.

        Within each theme, members receive equal weight.
        The signal at each date reflects the theme rank determined
        ``holding`` periods ago (signal persistence), then updated.

        Args:
            returns: log return panel
            lookback: window for ranking theme performance
            holding: holding period — how long to maintain theme positions

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        # Compute theme returns over lookback
        theme_ret_df = self.theme_returns(returns, lookback=lookback)

        for i in range(lookback, len(returns)):
            date = returns.index[i]
            theme_row = theme_ret_df.iloc[i].dropna()
            if theme_row.empty:
                continue

            # Rank themes: top → +1, bottom → -1
            n_themes = len(theme_row)
            theme_ranks = theme_row.rank(method="average") - 1
            if n_themes > 1:
                theme_ranks_norm = theme_ranks / (n_themes - 1) * 2 - 1
            else:
                theme_ranks_norm = theme_ranks * 0

            # Assign theme rank to each asset in that theme
            for theme_name, members in self.themes.items():
                if theme_name not in theme_ranks_norm.index:
                    continue
                rank_val = theme_ranks_norm[theme_name]
                valid = [m for m in members if m in returns.columns]
                for sym in valid:
                    signal.at[date, sym] = rank_val

        return signal  # Already in [-1, 1] from theme rank normalisation

    # ------------------------------------------------------------------
    # Relative theme value (contrarian)
    # ------------------------------------------------------------------

    def relative_theme_value(
        self,
        returns: pd.DataFrame,
        lookback: int = 63,
    ) -> pd.DataFrame:
        """
        Reversal at the theme level: themes that underperformed may revert.

        Signal for each asset = −theme_cumret(lookback).
        Assigns the negative of the theme's cumulative return to all members,
        capturing mean-reversion in cross-theme returns.

        Args:
            returns: log return panel
            lookback: theme performance horizon for reversal

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for _theme, members in self.themes.items():
            valid = [m for m in members if m in cumret.columns]
            if not valid:
                continue
            theme_avg = cumret[valid].mean(axis=1)
            for sym in valid:
                signal[sym] = -theme_avg  # negative → contrarian

        return _rank(signal)

    # ------------------------------------------------------------------
    # Style factors
    # ------------------------------------------------------------------

    def low_volatility_factor(
        self, returns: pd.DataFrame, vol_window: int = 63
    ) -> pd.DataFrame:
        """Low-vol factor: signal = −realized_vol (buy low-vol, sell high-vol)."""
        vol = returns.rolling(window=vol_window, min_periods=vol_window // 2).std()
        return _rank(-vol)

    def size_factor(
        self, dollar_volume: pd.DataFrame, window: int = 30
    ) -> pd.DataFrame:
        """Size factor proxy: signal = −log(avg_dollar_volume) (tilt toward small)."""
        avg_vol = dollar_volume.rolling(window=window, min_periods=window // 2).mean()
        return _rank(-np.log1p(avg_vol))

    def momentum_reversal_blend(
        self,
        returns: pd.DataFrame,
        momentum_lookback: int = 63,
        reversal_lookback: int = 5,
        momentum_weight: float = 0.6,
    ) -> pd.DataFrame:
        """Blend of medium-term momentum and short-term reversal signals."""
        mom_raw = returns.rolling(window=momentum_lookback, min_periods=momentum_lookback // 2).sum()
        rev_raw = -returns.rolling(window=reversal_lookback, min_periods=1).sum()
        blend = momentum_weight * mom_raw + (1 - momentum_weight) * rev_raw
        return _rank(blend)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_theme_members(self, theme: str) -> List[str]:
        """Return the list of symbols in a given theme."""
        return self.themes.get(theme, [])

    def list_themes(self) -> List[str]:
        """Return all theme names."""
        return list(self.themes.keys())

    def _primary_theme_map(self) -> Dict[str, str]:
        """Map each symbol to its primary (first-listed) theme."""
        seen: Dict[str, str] = {}
        for theme, members in self.themes.items():
            for sym in members:
                if sym not in seen:
                    seen[sym] = theme
        return seen

    def within_theme_relative_value(
        self, returns: pd.DataFrame, lookback: int = 21
    ) -> pd.DataFrame:
        """Within-theme reversal: buy laggards vs the theme average."""
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        for _theme_name, members in self.themes.items():
            valid = [m for m in members if m in cumret.columns]
            if len(valid) < 2:
                continue
            theme_avg = cumret[valid].mean(axis=1)
            for sym in valid:
                signal[sym] = -(cumret[sym] - theme_avg)
        return _rank(signal)

    def theme_momentum_signal(
        self,
        returns: pd.DataFrame,
        lookback: int = 21,
        skip: int = 0,
    ) -> pd.DataFrame:
        """Assign each asset the cross-sectional rank of its primary theme."""
        from statarb.data.features import FeatureEngine
        theme_ret = self.theme_returns(returns, lookback)
        if skip > 0:
            recent = self.theme_returns(returns, skip)
            theme_ret = theme_ret - recent
        fe = FeatureEngine()
        theme_rank = fe.cross_sectional_rank(theme_ret)
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        primary_theme = self._primary_theme_map()
        for sym in returns.columns:
            if sym in primary_theme:
                theme = primary_theme[sym]
                if theme in theme_rank.columns:
                    signal[sym] = theme_rank[theme]
        return signal
