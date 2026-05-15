"""
Strategy combination / signal weighting framework.

Two classes are provided:

StrategyWeighting — the Prompt 3 API.  Takes a DataFrame of strategy returns
    (dates × strategy_names) and combines them into a single return series.

StrategyWeighter — the legacy Prompt 1 class.  Combines signal panels or
    strategy return streams; kept for experiment-script compatibility.
"""

from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from statarb.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# StrategyWeighting  (Prompt 3)
# ---------------------------------------------------------------------------

class StrategyWeighting:
    """
    Combine multiple strategies into a single portfolio.

    Given K strategy return streams (as a DataFrame), find weights to combine
    them into one return series.
    """

    # ------------------------------------------------------------------
    # Equal weight
    # ------------------------------------------------------------------

    def equal_weight(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """
        Simple equal weighting: w_k = 1/K.
        Combined return = (1/K) Σ r_k.
        """
        return strategy_returns.mean(axis=1)

    # ------------------------------------------------------------------
    # Inverse-vol weight
    # ------------------------------------------------------------------

    def inverse_vol_weight(
        self,
        strategy_returns: pd.DataFrame,
        lookback: int = 63,
    ) -> pd.Series:
        """
        Weight inversely proportional to rolling volatility:
            w_k ∝ 1 / σ_k

        Lower-vol strategies get more weight (risk-parity lite).
        Signal is NaN during the initial warmup period.
        """
        combined = pd.Series(np.nan, index=strategy_returns.index)

        for i in range(lookback, len(strategy_returns)):
            window = strategy_returns.iloc[i - lookback: i]
            vols = window.std()
            inv_vols = 1.0 / vols.replace(0, np.nan)
            total = inv_vols.sum()
            if total == 0 or np.isnan(total):
                combined.iloc[i] = strategy_returns.iloc[i].mean()
                continue
            weights = inv_vols / total
            combined.iloc[i] = float((weights * strategy_returns.iloc[i]).sum())

        return combined

    # ------------------------------------------------------------------
    # Sharpe-weighted
    # ------------------------------------------------------------------

    def sharpe_weighted(
        self,
        strategy_returns: pd.DataFrame,
        lookback: int = 126,
    ) -> pd.Series:
        """
        Weight proportional to rolling Sharpe ratio:
            w_k ∝ max(Sharpe_k, 0)

        Only allocate to strategies with positive recent Sharpe.
        Zero weight for negative-Sharpe strategies.
        Signal is NaN during the initial warmup period.
        """
        combined = pd.Series(np.nan, index=strategy_returns.index)

        for i in range(lookback, len(strategy_returns)):
            window = strategy_returns.iloc[i - lookback: i]
            sharpes = {}
            for col in window.columns:
                s = window[col].dropna()
                if len(s) < 10:
                    continue
                sr = s.mean() / s.std() if s.std() > 0 else 0.0
                sharpes[col] = max(sr, 0.0)

            total = sum(sharpes.values())
            if total == 0:
                combined.iloc[i] = strategy_returns.iloc[i].mean()
            else:
                row = strategy_returns.iloc[i]
                combined.iloc[i] = float(
                    sum((sr / total) * row[col] for col, sr in sharpes.items())
                )

        return combined

    # ------------------------------------------------------------------
    # Minimum-variance
    # ------------------------------------------------------------------

    def min_variance(
        self,
        strategy_returns: pd.DataFrame,
        lookback: int = 126,
    ) -> pd.Series:
        """
        Minimum variance portfolio of strategies.

            w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)

        where Σ is the covariance matrix of strategy returns.
        Uses Ledoit-Wolf shrinkage estimator for Σ.
        Signal is NaN during the initial warmup period.
        """
        from sklearn.covariance import LedoitWolf

        n = strategy_returns.shape[1]
        ones = np.ones(n)
        combined = pd.Series(np.nan, index=strategy_returns.index)

        for i in range(lookback, len(strategy_returns)):
            window = strategy_returns.iloc[i - lookback: i].dropna()
            if len(window) < max(20, n + 1):
                combined.iloc[i] = strategy_returns.iloc[i].mean()
                continue

            lw = LedoitWolf()
            lw.fit(window.values)
            cov = lw.covariance_

            try:
                cov_inv = np.linalg.inv(cov)
                w = cov_inv @ ones
                denom = ones @ w
                if denom == 0:
                    raise np.linalg.LinAlgError("singular")
                weights = w / denom  # normalise: sum to 1
            except np.linalg.LinAlgError:
                weights = np.ones(n) / n

            combined.iloc[i] = float((weights * strategy_returns.iloc[i].values).sum())

        return combined

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def combine(
        self,
        strategy_returns: pd.DataFrame,
        method: str = "equal",
        lookback: int = 126,
    ) -> pd.Series:
        """
        Combine strategies using the specified method.

        Args:
            strategy_returns: DataFrame of strategy returns (dates × strategies)
            method:           'equal' | 'inverse_vol' | 'sharpe' | 'min_variance'
            lookback:         rolling estimation window (ignored for 'equal')

        Returns:
            Combined strategy return series.
        """
        dispatch = {
            "equal": lambda: self.equal_weight(strategy_returns),
            "inverse_vol": lambda: self.inverse_vol_weight(strategy_returns, lookback),
            "sharpe": lambda: self.sharpe_weighted(strategy_returns, lookback),
            "min_variance": lambda: self.min_variance(strategy_returns, lookback),
        }
        if method not in dispatch:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {list(dispatch)}"
            )
        return dispatch[method]()


# ---------------------------------------------------------------------------
# StrategyWeighter  (Prompt 1 — kept for compatibility)
# ---------------------------------------------------------------------------

class StrategyWeighter:
    """
    Combine multiple signals or strategy return streams.

    Supports several weighting schemes:
    - ``equal``: simple equal-weight average of signals
    - ``ic_weighted``: weight by information coefficient (signal-return correlation)
    - ``sharpe_weighted``: weight by recent Sharpe ratio of strategy returns
    - ``min_corr``: minimum-correlation portfolio
    - ``mean_variance``: mean-variance optimisation
    - ``custom``: user-supplied fixed weights
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
        """Combine multiple signal panels into a single composite signal."""
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
            raise ValueError(
                f"Method '{self.method}' should be used with combine_strategy_returns(), "
                "not combine_signals(). Convert signals to returns first."
            )

    def _equal_combine(self, signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        aligned = self._align_signals(signals)
        first = next(iter(aligned.values()))
        composite = pd.DataFrame(0.0, index=first.index, columns=first.columns)

        for date_idx in range(self.estimation_window, len(first)):
            date = first.index[date_idx]
            window_slice = slice(max(0, date_idx - self.estimation_window), date_idx)

            ics = {}
            for name, sig_df in aligned.items():
                sig_window = sig_df.iloc[window_slice]
                ret_window = forward_returns.iloc[window_slice].reindex(columns=sig_df.columns)
                ic = self._rolling_ic(sig_window, ret_window)
                ics[name] = max(ic, 0.0)

            total_ic = sum(ics.values())
            if total_ic <= 0:
                continue

            row = pd.Series(0.0, index=first.columns)
            for name, ic in ics.items():
                w = ic / total_ic
                row = row.add(aligned[name].iloc[date_idx] * w, fill_value=0)
            composite.loc[date] = row

        return composite

    def _custom_combine(self, signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        """Combine multiple strategy return streams into one portfolio."""
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
            return returns_df.mean(axis=1)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _sharpe_weighted_returns(self, returns_df: pd.DataFrame) -> pd.Series:
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
                )
            combined.iloc[i] = row_val
        return combined

    def _min_corr_returns(self, returns_df: pd.DataFrame) -> pd.Series:
        combined = pd.Series(0.0, index=returns_df.index)
        w = self.estimation_window
        n = returns_df.shape[1]
        for i in range(w, len(returns_df)):
            window = returns_df.iloc[i - w: i].dropna()
            if len(window) < 20:
                combined.iloc[i] = returns_df.iloc[i].mean()
                continue
            corr = window.corr().values
            result = minimize(
                fun=lambda wts: wts @ corr @ wts,
                x0=np.ones(n) / n,
                method="SLSQP",
                bounds=[(0, 1)] * n,
                constraints={"type": "eq", "fun": lambda wts: wts.sum() - 1},
            )
            weights = result.x if result.success else np.ones(n) / n
            combined.iloc[i] = sum(weights[j] * returns_df.iloc[i, j] for j in range(n))
        return combined

    def _mean_variance_returns(self, returns_df: pd.DataFrame) -> pd.Series:
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
                raw_weights = np.maximum(cov_inv @ mu, 0)
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
        ics = []
        for sig_row, ret_row in zip(
            signal_window.itertuples(index=False),
            return_window.itertuples(index=False),
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
