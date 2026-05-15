"""
Performance metrics for backtested strategies.

All metrics operate on a return Series (one value per trading period).
Conventions:
- Annualisation uses 365 periods for daily crypto (24/7 markets) or
  252 for comparison with traditional assets. Default: 252.
- Sharpe ratio: excess return / volatility (risk-free rate = 0)
- All metrics handle NaN values gracefully.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from statarb.utils import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    Compute and summarise strategy performance metrics.

    Instantiate with a return series, then call :meth:`summary` for
    a complete metrics dictionary, or call individual metric methods.
    """

    def __init__(self, returns: pd.Series, periods_per_year: int = 252):
        """
        Args:
            returns: daily (or sub-daily) strategy return series
            periods_per_year: annualisation factor (252 for daily)
        """
        self.returns = returns.dropna()
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        """Compound total return over the full period."""
        return float(np.exp(self.returns.sum()) - 1)

    def annualised_return(self) -> float:
        """Compound annualised return (CAGR)."""
        n_years = len(self.returns) / self.periods_per_year
        if n_years <= 0:
            return 0.0
        return float((1 + self.total_return()) ** (1 / n_years) - 1)

    def annualised_volatility(self) -> float:
        """Annualised standard deviation of returns."""
        return float(self.returns.std() * np.sqrt(self.periods_per_year))

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Annualised Sharpe ratio.

        Args:
            risk_free_rate: annualised risk-free rate (default 0%)

        Returns:
            Sharpe ratio. Higher is better.
        """
        daily_rf = (1 + risk_free_rate) ** (1 / self.periods_per_year) - 1
        excess = self.returns - daily_rf
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(self.periods_per_year))

    def sortino_ratio(self, risk_free_rate: float = 0.0, target: float = 0.0) -> float:
        """
        Sortino ratio: excess return / downside deviation.

        Uses only negative returns for risk estimation, avoiding
        penalising upside volatility.
        """
        daily_rf = (1 + risk_free_rate) ** (1 / self.periods_per_year) - 1
        excess = self.returns - daily_rf
        downside = excess[excess < target]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(self.periods_per_year)
        return float(excess.mean() * self.periods_per_year / downside_std)

    def calmar_ratio(self) -> float:
        """Calmar ratio: annualised return / maximum drawdown."""
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0.0
        return float(self.annualised_return() / mdd)

    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Information ratio: annualised active return / tracking error.

        Args:
            benchmark_returns: benchmark (e.g. BTC) return series

        Returns:
            Information ratio.
        """
        bench = benchmark_returns.reindex(self.returns.index).dropna()
        active = self.returns.reindex(bench.index) - bench
        active = active.dropna()
        if active.std() == 0:
            return 0.0
        return float(active.mean() / active.std() * np.sqrt(self.periods_per_year))

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    def drawdown_series(self) -> pd.Series:
        """
        Compute the drawdown series (fraction below running peak).

        Returns:
            Series with values ≤ 0 (0 at new highs, negative during drawdowns).
        """
        cum_ret = np.exp(self.returns.cumsum())
        running_max = cum_ret.cummax()
        return (cum_ret / running_max) - 1

    def max_drawdown(self) -> float:
        """Maximum drawdown (most negative drawdown value)."""
        return float(self.drawdown_series().min())

    def max_drawdown_duration(self) -> int:
        """
        Maximum drawdown duration in periods (peak to recovery).

        Returns:
            Number of periods of the longest drawdown episode.
        """
        dd = self.drawdown_series()
        in_dd = dd < 0
        max_dur = 0
        current_dur = 0
        for val in in_dd:
            if val:
                current_dur += 1
                max_dur = max(max_dur, current_dur)
            else:
                current_dur = 0
        return max_dur

    def average_drawdown(self) -> float:
        """Average drawdown across all drawdown episodes."""
        dd = self.drawdown_series()
        return float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    # ------------------------------------------------------------------
    # Distribution metrics
    # ------------------------------------------------------------------

    def skewness(self) -> float:
        """Return skewness (positive = right-skewed, desirable)."""
        return float(stats.skew(self.returns))

    def kurtosis(self) -> float:
        """Excess kurtosis (>0 = fat tails)."""
        return float(stats.kurtosis(self.returns))

    def var(self, confidence: float = 0.95) -> float:
        """
        Historical Value at Risk (VaR) at given confidence level.

        Returns:
            VaR as a positive fraction (loss amount).
        """
        return float(-np.percentile(self.returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        """
        Conditional VaR (Expected Shortfall) — mean of worst (1-conf) returns.

        Returns:
            CVaR as a positive fraction.
        """
        threshold = -self.var(confidence)
        tail = self.returns[self.returns <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else self.var(confidence)

    def hit_rate(self) -> float:
        """Fraction of periods with positive returns."""
        return float((self.returns > 0).mean())

    def profit_factor(self) -> float:
        """Sum of gains / sum of losses (gross)."""
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return float(gains / losses) if losses > 0 else float("inf")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Compute all performance metrics and return as a dictionary.

        Args:
            benchmark_returns: optional benchmark for IR calculation

        Returns:
            Dict of metric_name → value.
        """
        metrics = {
            "total_return": self.total_return(),
            "annualised_return": self.annualised_return(),
            "annualised_volatility": self.annualised_volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_duration_days": self.max_drawdown_duration(),
            "average_drawdown": self.average_drawdown(),
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
            "var_95": self.var(0.95),
            "cvar_95": self.cvar(0.95),
            "hit_rate": self.hit_rate(),
            "profit_factor": self.profit_factor(),
            "n_periods": len(self.returns),
        }
        if benchmark_returns is not None:
            metrics["information_ratio"] = self.information_ratio(benchmark_returns)

        return metrics

    @staticmethod
    def compare(
        results: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """
        Compare multiple strategy return series side-by-side.

        Args:
            results: dict of strategy_name → return Series
            benchmark_returns: optional benchmark for IR
            periods_per_year: annualisation factor

        Returns:
            DataFrame with strategies as columns and metrics as rows.
        """
        summaries = {}
        for name, rets in results.items():
            pm = PerformanceMetrics(rets, periods_per_year)
            summaries[name] = pm.summary(benchmark_returns)
        return pd.DataFrame(summaries)
