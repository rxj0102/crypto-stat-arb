"""
Strategy combination / signal weighting framework.

Combines multiple signals into a single composite signal or blends
multiple strategy return streams into a combined portfolio.
"""

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from statarb.utils import get_logger

logger = get_logger(__name__)


class StrategyWeighter:
    """
    Combine multiple signals or strategy return streams.

    Supports several weighting schemes:

    - ``equal``: simple equal-weight average of signals
    - ``ic_weighted``: weight by information coefficient (signal-return correlation)
    - ``sharpe_weighted``: weight by recent Sharpe ratio of strategy returns
    - ``min_corr``: minimum-correlation portfolio (maximises diversification)
    - ``mean_variance``: mean-variance optimisation (requires return estimates)
    - ``custom``: user-supplied fixed weights

    All methods work on *signal panels* (dates × symbols) or *strategy
    return series* (dates), depending on the use case.
    """

    def __init__(
        self,
        method: Literal[
            "equal", "ic_weighted", "sharpe_weighted",
            "min_corr", "mean_variance", "custom"
        ] = "equal",
        estimation_window: int = 63,
        custom_weights: Optional[Dict[str, float]] = None,
        regularisation: float = 1e-4,
    ):
        """
        Args:
            method: weighting method
            estimation_window: rolling window for weight estimation
            custom_weights: dict of signal_name → weight (for 'custom' method)
            regularisation: ridge regularisation for MV optimisation
        """
        self.method = method
        self.estimation_window = estimation_window
        self.custom_weights = custom_weights or {}
        self.regularisation = regularisation

    # ------------------------------------------------------------------
    # Signal combination (returns DataFrames)
    # ------------------------------------------------------------------

    def combine_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        forward_returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Combine multiple signal panels into a single composite signal.

        Args:
            signals: dict of signal_name → signal panel (dates × symbols)
            forward_returns: required for IC-weighted method

        Returns:
            Composite signal panel (dates × symbols).
        """
        if not signals:
            raise ValueError("No signals provided")

        if self.method == "equal":
            return self._equal_combine(signals)
        elif self.method == "ic_weighted":
            if forward_returns is None:
                raise ValueError("ic_weighted requires forward_returns")
            return self._ic_weighted_combine(signals, forward_returns)
        elif self.method == "custom":
            return self._custom_combine(signals)
        else:
            # For sharpe/min_corr/mean_variance, combine via strategy returns
            raise ValueError(
                f"Method '{self.method}' should be used with combine_strategy_returns(), "
                "not combine_signals(). Convert signals to returns first."
            )

    def _equal_combine(self, signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Simple equal-weight average across signals."""
        aligned = self._align_signals(signals)
        stacked = np.stack([df.values for df in aligned.values()], axis=2)
        composite = np.nanmean(stacked, axis=2)
        first = next(iter(aligned.values()))
        return pd.DataFrame(composite, index=first.index, columns=first.columns)

    def _ic_weighted_combine(
        self,
        signals: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Weight signals by their rolling Information Coefficient (rank IC)."""
        aligned = self._align_signals(signals)
        first = next(iter(aligned.values()))
        composite = pd.DataFrame(0.0, index=first.index, columns=first.columns)

        for date_idx in range(self.estimation_window, len(first)):
            date = first.index[date_idx]
            window_slice = slice(max(0, date_idx - self.estimation_window), date_idx)

            # Compute IC for each signal over the estimation window
            ics = {}
            for name, sig_df in aligned.items():
                sig_window = sig_df.iloc[window_slice]
                ret_window = forward_returns.iloc[window_slice].reindex(
                    columns=sig_df.columns
                )
                ic = self._rolling_ic(sig_window, ret_window)
                ics[name] = max(ic, 0.0)  # only use positive-IC signals

            total_ic = sum(ics.values())
            if total_ic <= 0:
                continue

            row = pd.Series(0.0, index=first.columns)
            for name, ic in ics.items():
                w = ic / total_ic
                signal_row = aligned[name].iloc[date_idx]
                row = row.add(signal_row * w, fill_value=0)

            composite.loc[date] = row

        return composite

    def _custom_combine(self, signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Weighted average using pre-specified custom weights."""
        aligned = self._align_signals(signals)
        first = next(iter(aligned.values()))
        composite = pd.DataFrame(0.0, index=first.index, columns=first.columns)

        total_w = sum(self.custom_weights.get(name, 0) for name in aligned)
        if total_w == 0:
            return self._equal_combine(signals)

        for name, sig_df in aligned.items():
            w = self.custom_weights.get(name, 0) / total_w
            composite += sig_df.fillna(0) * w

        return composite

    # ------------------------------------------------------------------
    # Strategy return combination (returns Series)
    # ------------------------------------------------------------------

    def combine_strategy_returns(
        self,
        strategy_returns: Dict[str, pd.Series],
    ) -> pd.Series:
        """
        Combine multiple strategy return streams into one portfolio.

        Args:
            strategy_returns: dict of strategy_name → returns Series

        Returns:
            Combined portfolio return Series.
        """
        returns_df = pd.DataFrame(strategy_returns).dropna(how="all")

        if self.method == "equal":
            return returns_df.mean(axis=1)

        elif self.method == "sharpe_weighted":
            return self._sharpe_weighted_returns(returns_df)

        elif self.method == "min_corr":
            return self._min_corr_returns(returns_df)

        elif self.method == "mean_variance":
            return self._mean_variance_returns(returns_df)

        elif self.method == "custom":
            names = list(strategy_returns.keys())
            weights = np.array([self.custom_weights.get(n, 1.0) for n in names])
            weights = weights / weights.sum()
            return returns_df[names].mul(weights, axis=1).sum(axis=1)

        elif self.method == "ic_weighted":
            return returns_df.mean(axis=1)  # fallback to equal for returns

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _sharpe_weighted_returns(self, returns_df: pd.DataFrame) -> pd.Series:
        """Weight strategies by rolling Sharpe ratio (positive SR only)."""
        combined = pd.Series(0.0, index=returns_df.index)
        w = self.estimation_window

        for i in range(w, len(returns_df)):
            date = returns_df.index[i]
            window = returns_df.iloc[i - w: i]
            sharpes = {}
            for col in window.columns:
                s = window[col].dropna()
                if len(s) < 10:
                    continue
                sr = s.mean() / s.std() if s.std() > 0 else 0.0
                sharpes[col] = max(sr, 0.0)
            total = sum(sharpes.values())
            if total == 0:
                row_val = returns_df.iloc[i].mean()
            else:
                row_val = sum(
                    (sr / total) * returns_df.at[date, col]
                    for col, sr in sharpes.items()
                    if col in returns_df.columns
                )
            combined.iloc[i] = row_val

        return combined

    def _min_corr_returns(self, returns_df: pd.DataFrame) -> pd.Series:
        """Minimum-correlation portfolio (maximises diversification)."""
        combined = pd.Series(0.0, index=returns_df.index)
        w = self.estimation_window
        n = returns_df.shape[1]

        for i in range(w, len(returns_df)):
            date = returns_df.index[i]
            window = returns_df.iloc[i - w: i].dropna()
            if len(window) < 20:
                combined.iloc[i] = returns_df.iloc[i].mean()
                continue

            corr = window.corr().values
            # Solve: min w'Cw subject to sum(w)=1, w>=0
            result = minimize(
                fun=lambda wts: wts @ corr @ wts,
                x0=np.ones(n) / n,
                method="SLSQP",
                bounds=[(0, 1)] * n,
                constraints={"type": "eq", "fun": lambda wts: wts.sum() - 1},
            )
            weights = result.x if result.success else np.ones(n) / n
            row_ret = sum(weights[j] * returns_df.iloc[i, j] for j in range(n))
            combined.iloc[i] = row_ret

        return combined

    def _mean_variance_returns(self, returns_df: pd.DataFrame) -> pd.Series:
        """Mean-variance optimal weights (with ridge regularisation)."""
        combined = pd.Series(0.0, index=returns_df.index)
        w = self.estimation_window
        n = returns_df.shape[1]

        for i in range(w, len(returns_df)):
            window = returns_df.iloc[i - w: i].dropna()
            if len(window) < 20:
                combined.iloc[i] = returns_df.iloc[i].mean()
                continue

            mu = window.mean().values
            cov = window.cov().values + np.eye(n) * self.regularisation

            try:
                cov_inv = np.linalg.inv(cov)
                raw_weights = cov_inv @ mu
                # Long-only constraint + normalise
                raw_weights = np.maximum(raw_weights, 0)
                total = raw_weights.sum()
                weights = raw_weights / total if total > 0 else np.ones(n) / n
            except np.linalg.LinAlgError:
                weights = np.ones(n) / n

            combined.iloc[i] = (weights * returns_df.iloc[i].values).sum()

        return combined

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _align_signals(signals: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align all signal DataFrames to a common index and columns."""
        all_dfs = list(signals.values())
        common_idx = all_dfs[0].index
        common_cols = all_dfs[0].columns
        for df in all_dfs[1:]:
            common_idx = common_idx.union(df.index)
            common_cols = common_cols.union(df.columns)

        return {
            name: df.reindex(index=common_idx, columns=common_cols)
            for name, df in signals.items()
        }

    @staticmethod
    def _rolling_ic(signal_window: pd.DataFrame, return_window: pd.DataFrame) -> float:
        """Mean rank correlation (IC) between signal and subsequent returns."""
        ics = []
        for _, (sig_row, ret_row) in enumerate(
            zip(signal_window.itertuples(index=False), return_window.itertuples(index=False))
        ):
            sig = pd.Series(sig_row)
            ret = pd.Series(ret_row)
            valid = sig.notna() & ret.notna()
            if valid.sum() < 5:
                continue
            ic = sig[valid].corr(ret[valid], method="spearman")
            if not np.isnan(ic):
                ics.append(ic)
        return float(np.mean(ics)) if ics else 0.0
