"""
Risk analytics: alpha/beta decomposition, factor exposure, drawdown analysis.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from statarb.utils import get_logger

logger = get_logger(__name__)


class RiskAnalytics:
    """
    Risk analytics: alpha, beta, factor exposure.

    Stateless API: every method takes strategy_returns (and other inputs)
    as explicit arguments so one instance can be reused across strategies.

    Example::

        ra = RiskAnalytics()
        result = ra.alpha_beta(strategy_returns, benchmark_returns)
    """

    # ------------------------------------------------------------------
    # Alpha / Beta
    # ------------------------------------------------------------------

    def alpha_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict:
        """
        CAPM regression:
            r_strategy = α + β × r_benchmark + ε

        For crypto stat arb the benchmark is typically BTC or an
        equal-weighted crypto index.

        Returns:
            dict with:
            - 'alpha'       : annualised daily intercept (× 365)
            - 'beta'        : slope coefficient
            - 'r_squared'   : R² of the regression
            - 'alpha_tstat' : t-statistic for alpha
            - 'beta_tstat'  : t-statistic for beta
        """
        clean_s = strategy_returns.dropna()
        bench = benchmark_returns.reindex(clean_s.index).dropna()
        y = clean_s.reindex(bench.index).dropna().values
        x = bench.reindex(clean_s.reindex(bench.index).dropna().index).values

        if len(y) < 10:
            return {
                "alpha": 0.0, "beta": 0.0, "r_squared": 0.0,
                "alpha_tstat": 0.0, "beta_tstat": 0.0,
            }

        slope, intercept, r_value, _, _ = stats.linregress(x, y)

        # Manual SE computation for t-stats
        n = len(y)
        resid = y - (slope * x + intercept)
        s2 = float((resid ** 2).sum() / (n - 2))
        x_demean_sq = float(((x - x.mean()) ** 2).sum())
        if x_demean_sq > 0 and s2 > 0:
            se_slope = float(np.sqrt(s2 / x_demean_sq))
            se_intercept = float(np.sqrt(s2 * (1.0 / n + x.mean() ** 2 / x_demean_sq)))
        else:
            se_slope = se_intercept = 1.0

        return {
            "alpha":       float(intercept * 365),
            "beta":        float(slope),
            "r_squared":   float(r_value ** 2),
            "alpha_tstat": float(intercept / se_intercept) if se_intercept > 0 else 0.0,
            "beta_tstat":  float(slope / se_slope) if se_slope > 0 else 0.0,
        }

    def rolling_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 63,
    ) -> pd.Series:
        """
        Rolling beta over time.
        Should be near 0 for a dollar-neutral stat-arb strategy.
        """
        clean_s = strategy_returns.dropna()
        bench = benchmark_returns.reindex(clean_s.index).fillna(0)
        result = pd.Series(np.nan, index=clean_s.index)

        for i in range(window, len(clean_s) + 1):
            y = clean_s.iloc[i - window: i].values
            x = bench.iloc[i - window: i].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < window // 2:
                continue
            slope, *_ = stats.linregress(x[mask], y[mask])
            result.iloc[i - 1] = float(slope)

        return result

    # ------------------------------------------------------------------
    # Multi-factor regression
    # ------------------------------------------------------------------

    def factor_exposure(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> dict:
        """
        Multi-factor regression:
            r_strategy = α + Σ β_k × f_k + ε

        Common crypto factors:
        - Market (BTC return)
        - Size (large cap vs small cap)
        - Momentum (winners minus losers)
        - Volatility (high vol vs low vol)

        Returns dict with factor betas, t-stats, alpha, and R².
        """
        import statsmodels.api as sm

        common = strategy_returns.index.intersection(factor_returns.index)
        y = strategy_returns.loc[common].dropna()
        X = factor_returns.loc[y.index]
        X_c = sm.add_constant(X)

        try:
            model = sm.OLS(y, X_c, missing="drop").fit()
        except Exception as exc:
            logger.error("Factor regression failed: %s", exc)
            return {}

        result: dict = dict(model.params)
        result["alpha_tstat"] = float(model.tvalues.get("const", 0.0))
        for col in factor_returns.columns:
            result[f"{col}_tstat"] = float(model.tvalues.get(col, 0.0))
        if "const" in result:
            result["alpha"] = result.pop("const") * 365
        result["r_squared"] = float(model.rsquared)
        return result

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------

    def drawdown_analysis(self, returns: pd.Series) -> pd.DataFrame:
        """
        Detailed drawdown analysis.
        For each drawdown exceeding 5% depth returns a row with:
        - start:         first date below previous peak
        - trough:        date of maximum drawdown
        - recovery:      date of full recovery (NaT if not recovered)
        - max_depth:     maximum decline from peak (negative fraction)
        - duration:      total number of periods in the drawdown episode
        - recovery_time: periods from trough to recovery (NaN if unrecovered)
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return pd.DataFrame(
                columns=["start", "trough", "recovery",
                         "max_depth", "duration", "recovery_time"]
            )

        cum = (1 + clean).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak

        records: List[dict] = []
        in_dd = False
        start_idx: Optional[int] = None

        for i in range(len(dd)):
            val = float(dd.iloc[i])
            if not in_dd and val < 0:
                in_dd = True
                start_idx = i
            elif in_dd and val == 0:
                # Recovery reached
                episode = dd.iloc[start_idx: i + 1]
                depth = float(episode.min())
                if depth <= -0.05:
                    trough_idx = int(episode.argmin())
                    records.append({
                        "start":         clean.index[start_idx],
                        "trough":        clean.index[start_idx + trough_idx],
                        "recovery":      clean.index[i],
                        "max_depth":     depth,
                        "duration":      i - start_idx + 1,
                        "recovery_time": i - (start_idx + trough_idx),
                    })
                in_dd = False

        # Open drawdown at end of series
        if in_dd and start_idx is not None:
            episode = dd.iloc[start_idx:]
            depth = float(episode.min())
            if depth <= -0.05:
                trough_idx = int(episode.argmin())
                records.append({
                    "start":         clean.index[start_idx],
                    "trough":        clean.index[start_idx + trough_idx],
                    "recovery":      None,
                    "max_depth":     depth,
                    "duration":      len(episode),
                    "recovery_time": None,
                })

        return pd.DataFrame(
            records,
            columns=["start", "trough", "recovery", "max_depth", "duration", "recovery_time"],
        )

    # ------------------------------------------------------------------
    # Rolling Sharpe
    # ------------------------------------------------------------------

    def rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 126,
    ) -> pd.Series:
        """Rolling annualized Sharpe ratio."""
        clean = returns.dropna()
        roll_mean = clean.rolling(window=window, min_periods=window // 2).mean()
        roll_std = clean.rolling(window=window, min_periods=window // 2).std()
        return (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(365)

    # ------------------------------------------------------------------
    # Return attribution
    # ------------------------------------------------------------------

    def return_attribution(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> dict:
        """
        Attribute strategy returns to:
        - Long book contribution
        - Short book contribution
        - Selection (cross-sectional picking)
        - Timing (from rebalancing)

        Args:
            positions: weight panel (dates × symbols)
            returns:   forward return panel (dates × symbols)

        Returns:
            dict with annualised contribution estimates.
        """
        common_idx = positions.index.intersection(returns.index)
        common_col = positions.columns.intersection(returns.columns)
        pos = positions.loc[common_idx, common_col]
        ret = returns.loc[common_idx, common_col]

        long_pos = pos.clip(lower=0)
        short_pos = pos.clip(upper=0)
        long_contrib = (long_pos * ret).sum(axis=1)
        short_contrib = (short_pos * ret).sum(axis=1)
        total_contrib = long_contrib + short_contrib

        # Timing: contribution from position changes vs a static average position
        avg_pos = pos.mean(axis=0)
        static_daily = (avg_pos * ret).sum(axis=1)
        timing_contrib = total_contrib - static_daily
        selection_contrib = static_daily  # cross-sectional component

        scale = 365
        return {
            "long_contribution":       float(long_contrib.mean() * scale),
            "short_contribution":      float(short_contrib.mean() * scale),
            "total_contribution":      float(total_contrib.mean() * scale),
            "selection_contribution":  float(selection_contrib.mean() * scale),
            "timing_contribution":     float(timing_contrib.mean() * scale),
        }

    # ------------------------------------------------------------------
    # Legacy helpers (kept for reporting.py compatibility)
    # ------------------------------------------------------------------

    def rolling_alpha_beta(
        self,
        benchmark_returns: pd.Series,
        window: int = 63,
        strategy_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Rolling alpha, beta, and R² (legacy interface)."""
        if strategy_returns is None:
            raise ValueError("strategy_returns required")
        clean_s = strategy_returns.dropna()
        bench = benchmark_returns.reindex(clean_s.index).fillna(0)
        result = pd.DataFrame(
            np.nan,
            index=clean_s.index,
            columns=["alpha", "beta", "r_squared"],
        )
        for i in range(window, len(clean_s) + 1):
            y = clean_s.iloc[i - window: i].values
            x = bench.iloc[i - window: i].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < window // 2:
                continue
            slope, intercept, r_value, _, _ = stats.linregress(x[mask], y[mask])
            result.iloc[i - 1] = [intercept * 365, slope, r_value ** 2]
        return result
