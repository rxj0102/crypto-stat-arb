"""
Pairs / correlation-based trading signals.

Pairs trading exploits temporary deviations between historically
co-moving assets. In crypto, natural pairs exist within the same
narrative cluster (e.g. ETH/BTC, SOL/AVAX, UNI/SUSHI).
"""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from statarb.utils import get_logger

logger = get_logger(__name__)


class PairsSignals:
    """
    Pairs and correlation-based signals.

    All signals are returned as DataFrames[dates × symbols] where possible.
    Individual pair series are also available for targeted pair trading.
    """

    # ------------------------------------------------------------------
    # Spread computation
    # ------------------------------------------------------------------

    def compute_spread(
        self,
        prices: pd.DataFrame,
        asset_a: str,
        asset_b: str,
        estimation_window: int = 63,
        method: str = "ols",
    ) -> pd.Series:
        """
        Compute the hedge-ratio-adjusted spread between two assets.

        Spread = log(P_A) - β × log(P_B)

        The hedge ratio β is estimated via rolling OLS regression of
        log(P_A) on log(P_B) to prevent look-ahead bias.

        Args:
            prices: price panel
            asset_a: first symbol (dependent variable)
            asset_b: second symbol (independent variable)
            estimation_window: rolling window for β estimation
            method: 'ols' (rolling OLS) or 'ratio' (simple ratio)

        Returns:
            Series of spread values (positive = A expensive vs B).
        """
        if asset_a not in prices.columns or asset_b not in prices.columns:
            raise KeyError(f"Assets {asset_a} and/or {asset_b} not in prices panel")

        log_a = np.log(prices[asset_a].replace(0, np.nan))
        log_b = np.log(prices[asset_b].replace(0, np.nan))

        if method == "ratio":
            return log_a - log_b

        # Rolling OLS hedge ratio
        betas = pd.Series(index=prices.index, dtype=float)
        for i in range(estimation_window, len(prices) + 1):
            window_a = log_a.iloc[i - estimation_window: i].dropna()
            window_b = log_b.iloc[i - estimation_window: i].dropna()
            common = window_a.index.intersection(window_b.index)
            if len(common) < estimation_window // 2:
                continue
            slope, _, _, _, _ = stats.linregress(window_b.loc[common], window_a.loc[common])
            betas.iloc[i - 1] = slope

        spread = log_a - betas * log_b
        return spread

    # ------------------------------------------------------------------
    # Spread z-score signal
    # ------------------------------------------------------------------

    def spread_zscore(
        self,
        spread: pd.Series,
        zscore_window: int = 21,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
    ) -> pd.Series:
        """
        Generate a mean-reversion signal from a spread z-score.

        Signal = -zscore(spread, zscore_window)
        - z < -entry_threshold → buy A, sell B (spread expected to widen)
        - z > +entry_threshold → sell A, buy B (spread expected to narrow)
        - |z| < exit_threshold → flat

        Args:
            spread: raw spread series
            zscore_window: rolling mean/std window for z-score
            entry_threshold: z-score magnitude to open a position
            exit_threshold: z-score magnitude to close a position

        Returns:
            Series: +1 (long A / short B), -1 (short A / long B), 0 (flat), NaN
        """
        roll_mean = spread.rolling(window=zscore_window, min_periods=zscore_window // 2).mean()
        roll_std = spread.rolling(window=zscore_window, min_periods=zscore_window // 2).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)
        return -zscore

    # ------------------------------------------------------------------
    # Portfolio-level cross-sectional relative value
    # ------------------------------------------------------------------

    def relative_value_signal(
        self,
        returns: pd.DataFrame,
        correlation_window: int = 63,
        return_window: int = 21,
        min_correlation: float = 0.5,
    ) -> pd.DataFrame:
        """
        Cross-sectional relative value signal for correlated asset pairs.

        For each asset, compute the weighted-average underperformance vs
        its correlated peers. Assets that underperform their peers are
        expected to revert upward (mean reversion across correlated clusters).

        Algorithm:
        1. Compute rolling pairwise correlations.
        2. For each asset A, find peers with correlation >= min_correlation.
        3. Signal_A = -(return_A - mean_return_of_peers) → buy laggards.

        Args:
            returns: return panel
            correlation_window: window for rolling pairwise correlation
            return_window: return horizon for performance comparison
            min_correlation: minimum correlation to consider a pair

        Returns:
            Signal panel (dates × symbols).
        """
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
                relative_perf = asset_ret - peer_rets.mean()
                signal.at[date, sym] = -relative_perf  # reversal

        return signal

    # ------------------------------------------------------------------
    # Statistical cointegration scanner
    # ------------------------------------------------------------------

    def find_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
        lookback: int = 252,
        pvalue_threshold: float = 0.05,
    ) -> List[Tuple[str, str, float]]:
        """
        Scan all pairs for cointegration using Engle-Granger test.

        Only uses data up to the last available date (no look-ahead).
        Useful for pair selection in strategy construction.

        Args:
            prices: price panel
            lookback: number of recent periods to use for the test
            pvalue_threshold: maximum p-value to consider a pair cointegrated

        Returns:
            List of (asset_a, asset_b, p_value) tuples, sorted by p_value.
        """
        from statsmodels.tsa.stattools import coint

        syms = prices.columns.tolist()
        recent = prices.iloc[-lookback:].dropna(axis=1, how="any")
        available = [s for s in syms if s in recent.columns]

        results = []
        for a, b in itertools.combinations(available, 2):
            try:
                _, pvalue, _ = coint(np.log(recent[a]), np.log(recent[b]))
                if pvalue <= pvalue_threshold:
                    results.append((a, b, float(pvalue)))
            except Exception:  # noqa: BLE001
                pass

        results.sort(key=lambda x: x[2])
        logger.info("Found %d cointegrated pairs (p < %.2f)", len(results), pvalue_threshold)
        return results

    # ------------------------------------------------------------------
    # Intra-cluster momentum signal
    # ------------------------------------------------------------------

    def cluster_momentum_signal(
        self,
        returns: pd.DataFrame,
        clusters: Dict[str, List[str]],
        lookback: int = 21,
    ) -> pd.DataFrame:
        """
        Momentum signal relative to thematic cluster performance.

        For each cluster (e.g. DeFi, L1, Gaming), compute the cumulative
        return vs the cluster equal-weight return. Positive deviation
        indicates within-cluster winner (momentum) or winner for short.

        Args:
            clusters: dict mapping cluster_name → list of symbols
            lookback: momentum lookback period

        Returns:
            Signal panel: positive = outperforming cluster (momentum).
        """
        cumret = returns.rolling(window=lookback, min_periods=lookback // 2).sum()
        signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for _cluster_name, members in clusters.items():
            valid_members = [m for m in members if m in cumret.columns]
            if len(valid_members) < 2:
                continue
            cluster_avg = cumret[valid_members].mean(axis=1)
            for sym in valid_members:
                signal[sym] = cumret[sym] - cluster_avg

        return signal
