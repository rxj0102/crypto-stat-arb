"""
Pairs / correlation-based trading signals.

Principle: Security A − (Something Correlated) is more mean-reverting.
Natural pairs in crypto exist within thematic clusters (L1s, DeFi, etc.)
and can be identified via cointegration testing.
"""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

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


class PairsSignals:
    """
    Pairs and correlation-based signals.

    Methods at the pair level return pd.Series; methods that produce
    portfolio-level signals return DataFrames[dates × symbols].
    """

    # ------------------------------------------------------------------
    # Cointegration scanning
    # ------------------------------------------------------------------

    def find_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
        lookback: int = 126,
        p_threshold: float = 0.05,
    ) -> List[Tuple[str, str, float, float]]:
        """
        Find cointegrated pairs using the Engle-Granger two-step test.

        For each pair (i, j):
        1. Regress log(P_i) on log(P_j): log(P_i) = α + β·log(P_j) + ε
        2. Test ε for stationarity using the ADF test
        3. If ADF p-value < p_threshold → pair is cointegrated

        Uses only the most recent ``lookback`` periods (no look-ahead).

        Args:
            prices: price panel (dates × symbols)
            lookback: number of recent periods to use for the test
            p_threshold: maximum ADF p-value to consider cointegrated

        Returns:
            List of (symbol_i, symbol_j, beta, p_value) tuples,
            sorted by p_value ascending.
        """
        from statsmodels.tsa.stattools import coint

        recent = prices.iloc[-lookback:].dropna(axis=1, how="any")
        syms = recent.columns.tolist()
        results: List[Tuple[str, str, float, float]] = []

        for sym_a, sym_b in itertools.combinations(syms, 2):
            log_a = np.log(recent[sym_a].replace(0, np.nan).dropna())
            log_b = np.log(recent[sym_b].replace(0, np.nan).dropna())
            common = log_a.index.intersection(log_b.index)
            if len(common) < lookback // 2:
                continue
            try:
                # Engle-Granger cointegration (includes OLS regression + ADF)
                _, pvalue, _ = coint(log_a.loc[common], log_b.loc[common])
                if pvalue <= p_threshold:
                    # Estimate beta from OLS
                    slope, _, _, _, _ = stats.linregress(
                        log_b.loc[common].values, log_a.loc[common].values
                    )
                    results.append((sym_a, sym_b, float(slope), float(pvalue)))
            except Exception:  # noqa: BLE001
                pass

        results.sort(key=lambda x: x[3])
        logger.info(
            "Found %d cointegrated pairs (p < %.2f) from %d candidates",
            len(results), p_threshold, len(list(itertools.combinations(syms, 2)))
        )
        return results

    # ------------------------------------------------------------------
    # Pair spread z-score signal
    # ------------------------------------------------------------------

    def pairs_spread_zscore(
        self,
        prices: pd.DataFrame,
        pair: Tuple[str, str],
        lookback: int = 63,
        zscore_window: int = 21,
    ) -> pd.Series:
        """
        Z-score of the hedge-ratio-adjusted spread for a single pair.

        spread_t = log(P_i) - β·log(P_j)
        zscore_t = (spread_t - MA(spread, zscore_window)) / std(spread, zscore_window)

        β is estimated by rolling OLS over ``lookback`` periods to
        prevent look-ahead bias.

        Trading rule:
        - z < −2 → buy i, sell j (spread likely to widen back toward mean)
        - z > +2 → sell i, buy j
        Signal is returned as a Series; negate to get reversion direction.

        Args:
            prices: price panel
            pair: (symbol_i, symbol_j) tuple
            lookback: rolling window for β estimation (OLS)
            zscore_window: window for spread mean/std normalisation

        Returns:
            Series of spread z-scores (positive → spread above mean).
        """
        sym_a, sym_b = pair
        if sym_a not in prices.columns or sym_b not in prices.columns:
            raise KeyError(f"Pair symbols not found in prices: {pair}")

        log_a = np.log(prices[sym_a].replace(0, np.nan))
        log_b = np.log(prices[sym_b].replace(0, np.nan))

        betas = pd.Series(np.nan, index=prices.index)
        for i in range(lookback, len(prices) + 1):
            win_a = log_a.iloc[i - lookback: i].dropna()
            win_b = log_b.iloc[i - lookback: i].dropna()
            common = win_a.index.intersection(win_b.index)
            if len(common) < lookback // 2:
                continue
            slope, _, _, _, _ = stats.linregress(
                win_b.loc[common].values, win_a.loc[common].values
            )
            betas.iloc[i - 1] = slope

        spread = log_a - betas * log_b
        roll_mean = spread.rolling(window=zscore_window, min_periods=zscore_window // 2).mean()
        roll_std = spread.rolling(window=zscore_window, min_periods=zscore_window // 2).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)
        return zscore

    # ------------------------------------------------------------------
    # Sector-neutral reversal
    # ------------------------------------------------------------------

    def sector_neutral_reversal(
        self,
        returns: pd.DataFrame,
        sector_map: Dict[str, List[str]],
        lookback: int = 5,
    ) -> pd.DataFrame:
        """
        Reversal after removing sector / category common-factor exposure.

        For each asset:
        residual_return = return − sector_average_return
        signal = −cumulative_residual_return(lookback)

        Categories in crypto: L1s, DeFi, Memecoins, AI tokens, Gaming, etc.
        This isolates idiosyncratic reversal from common factor moves, e.g.
        the entire DeFi sector falling does not trigger DeFi reversal signals.

        Args:
            returns: log return panel (dates × symbols)
            sector_map: dict mapping sector_name → list of symbol strings
            lookback: reversal accumulation window

        Returns:
            Ranked signal panel ∈ [-1, 1]. Symbols not in any sector are
            included with unadjusted residuals (sector average = 0).
        """
        # Build a sector-average panel aligned to returns
        sector_avg = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        for _sector, members in sector_map.items():
            valid = [m for m in members if m in returns.columns]
            if len(valid) < 2:
                continue
            avg = returns[valid].mean(axis=1)
            for sym in valid:
                sector_avg[sym] = avg

        # Residual = individual return − sector average; 0 if no sector info
        residuals = returns - sector_avg.fillna(0)
        cumresid = residuals.rolling(window=lookback, min_periods=1).sum()
        return _rank(-cumresid)

    # ------------------------------------------------------------------
    # BTC-beta-neutral reversal
    # ------------------------------------------------------------------

    def beta_neutral_reversal(
        self,
        returns: pd.DataFrame,
        btc_returns: pd.Series,
        lookback: int = 5,
        beta_window: int = 63,
    ) -> pd.DataFrame:
        """
        Reversal after hedging out BTC (market) beta.

        Most crypto assets are highly correlated with BTC. Hedging out
        BTC exposure isolates the idiosyncratic component, which tends
        to revert more reliably.

        For each asset i:
        1. β_i = cov(r_i, r_BTC) / var(r_BTC) over rolling ``beta_window``
        2. ε_i = r_i − β_i × r_BTC  (idiosyncratic residual)
        3. signal = −cumulative_ε_i(lookback)

        Args:
            returns: log return panel (dates × symbols)
            btc_returns: BTC log return Series (same index)
            lookback: residual accumulation window
            beta_window: rolling window for beta estimation

        Returns:
            Ranked signal panel ∈ [-1, 1]
        """
        btc = btc_returns.reindex(returns.index).fillna(0)
        residuals = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for sym in returns.columns:
            asset_ret = returns[sym]
            roll_cov = asset_ret.rolling(beta_window, min_periods=beta_window // 2).cov(btc)
            roll_var = btc.rolling(beta_window, min_periods=beta_window // 2).var()
            beta = roll_cov / roll_var.replace(0, np.nan)
            residuals[sym] = asset_ret - beta * btc

        cumresid = residuals.rolling(window=lookback, min_periods=1).sum()
        return _rank(-cumresid)

    # ------------------------------------------------------------------
    # Legacy convenience methods (from Prompt 1)
    # ------------------------------------------------------------------

    def compute_spread(
        self,
        prices: pd.DataFrame,
        asset_a: str,
        asset_b: str,
        estimation_window: int = 63,
        method: str = "ols",
    ) -> pd.Series:
        """Rolling OLS spread: log(P_A) − β·log(P_B)."""
        if method == "ratio":
            return np.log(prices[asset_a]) - np.log(prices[asset_b])
        # Rolling OLS
        log_a = np.log(prices[asset_a].replace(0, np.nan))
        log_b = np.log(prices[asset_b].replace(0, np.nan))
        betas = pd.Series(np.nan, index=prices.index)
        for i in range(estimation_window, len(prices) + 1):
            wa = log_a.iloc[i - estimation_window: i].dropna()
            wb = log_b.iloc[i - estimation_window: i].dropna()
            common = wa.index.intersection(wb.index)
            if len(common) < estimation_window // 2:
                continue
            slope, _, _, _, _ = stats.linregress(wb.loc[common], wa.loc[common])
            betas.iloc[i - 1] = slope
        return log_a - betas * log_b

    def relative_value_signal(
        self,
        returns: pd.DataFrame,
        correlation_window: int = 63,
        return_window: int = 21,
        min_correlation: float = 0.5,
    ) -> pd.DataFrame:
        """Buy assets that underperform their correlated peers."""
        cumret = returns.rolling(window=return_window, min_periods=return_window // 2).sum()
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        for i in range(correlation_window, len(returns)):
            date = returns.index[i]
            window_ret = returns.iloc[i - correlation_window: i].dropna(axis=1)
            if window_ret.shape[1] < 3:
                continue
            corr_matrix = window_ret.corr()
            for sym in corr_matrix.index:
                peers = corr_matrix[sym][
                    (corr_matrix[sym] >= min_correlation) & (corr_matrix[sym].index != sym)
                ].index.tolist()
                if not peers:
                    continue
                asset_ret = cumret.at[date, sym] if sym in cumret.columns else np.nan
                peer_rets = cumret.loc[date, [p for p in peers if p in cumret.columns]]
                if pd.isna(asset_ret) or peer_rets.empty:
                    continue
                signal.at[date, sym] = -(asset_ret - peer_rets.mean())
        return signal

    def cluster_momentum_signal(
        self,
        returns: pd.DataFrame,
        clusters: Dict[str, List[str]],
        lookback: int = 21,
    ) -> pd.DataFrame:
        """Momentum signal relative to thematic cluster average."""
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        for _cluster_name, members in clusters.items():
            valid = [m for m in members if m in cumret.columns]
            if len(valid) < 2:
                continue
            cluster_avg = cumret[valid].mean(axis=1)
            for sym in valid:
                signal[sym] = cumret[sym] - cluster_avg
        return signal
