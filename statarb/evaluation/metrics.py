"""
Performance metrics for backtested strategies.

All methods are stateless: pass the return Series explicitly to each call.
Default periods_per_year=365 for crypto (24/7 markets).
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from statarb.utils import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    Compute all key performance metrics for strategy evaluation.

    Stateless API: every method takes ``returns`` as its first argument.

    Example::

        pm = PerformanceMetrics()
        sharpe = pm.sharpe_ratio(returns)
        report = pm.full_report(returns, benchmark_returns=btc_returns)
    """

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    def annualized_return(
        self,
        returns: pd.Series,
        periods_per_year: int = 365,
    ) -> float:
        """Annualized compound return (CAGR). Use periods_per_year=365 for daily crypto."""
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        n_years = len(clean) / periods_per_year
        if n_years <= 0:
            return 0.0
        cum = float((1 + clean).prod())
        return float(cum ** (1.0 / n_years) - 1)

    def annualized_volatility(
        self,
        returns: pd.Series,
        periods_per_year: int = 365,
    ) -> float:
        """Annualized volatility = std(r) × √(periods_per_year)."""
        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0
        vol = float(clean.std())
        if vol < 1e-14:
            return 0.0
        return float(vol * np.sqrt(periods_per_year))

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365,
    ) -> float:
        """
        Annualized Sharpe ratio = (ann_return − rf) / ann_vol.
        Returns ±np.inf when volatility is zero and mean is non-zero.
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0
        daily_rf = (1 + risk_free_rate) ** (1.0 / periods_per_year) - 1
        excess = clean - daily_rf
        vol = float(excess.std())
        if vol < 1e-14:
            m = float(excess.mean())
            return np.inf if m > 0 else (-np.inf if m < 0 else 0.0)
        return float(excess.mean() / vol * np.sqrt(periods_per_year))

    def sortino_ratio(
        self,
        returns: pd.Series,
        target_return: float = 0.0,
        periods_per_year: int = 365,
    ) -> float:
        """
        Sortino = (ann_return − target) / downside_deviation.
        Penalizes only downside volatility. Returns np.inf when no downside exists.
        """
        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0
        daily_target = target_return / periods_per_year
        excess = clean - daily_target
        ann_excess = float(excess.mean() * periods_per_year)
        downside = excess[excess < 0]
        if len(downside) == 0:
            return np.inf if ann_excess > 0 else 0.0
        downside_std = float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))
        if downside_std == 0:
            return 0.0
        return float(ann_excess / downside_std)

    def calmar_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 365,
    ) -> float:
        """Calmar = annualized_return / |max_drawdown|. Returns 0 when MDD = 0."""
        mdd = abs(self.max_drawdown(returns))
        if mdd == 0:
            return 0.0
        return float(self.annualized_return(returns, periods_per_year) / mdd)

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    def max_drawdown(self, returns: pd.Series) -> float:
        """
        Maximum drawdown: largest peak-to-trough decline.

        DD_t = (cum_t − cummax_t) / cummax_t
        MaxDD = min(DD_t)  (always ≤ 0)
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        cum = (1 + clean).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())

    def max_drawdown_duration(self, returns: pd.Series) -> int:
        """Number of periods in the longest drawdown episode."""
        clean = returns.dropna()
        if len(clean) == 0:
            return 0
        cum = (1 + clean).cumprod()
        peak = cum.cummax()
        in_dd = cum < peak
        max_dur = 0
        cur_dur = 0
        for val in in_dd:
            if val:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        return max_dur

    # ------------------------------------------------------------------
    # Distribution metrics
    # ------------------------------------------------------------------

    def win_rate(self, returns: pd.Series) -> float:
        """Fraction of positive-return periods."""
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        return float((clean > 0).mean())

    def profit_factor(self, returns: pd.Series) -> float:
        """Sum of positive returns / |sum of negative returns|."""
        clean = returns.dropna()
        gains = float(clean[clean > 0].sum())
        losses = abs(float(clean[clean < 0].sum()))
        if losses == 0:
            return float("inf")
        return float(gains / losses)

    def skewness(self, returns: pd.Series) -> float:
        """Return distribution skewness."""
        clean = returns.dropna()
        if len(clean) < 3:
            return 0.0
        return float(stats.skew(clean))

    def kurtosis(self, returns: pd.Series) -> float:
        """Return distribution excess kurtosis."""
        clean = returns.dropna()
        if len(clean) < 4:
            return 0.0
        return float(stats.kurtosis(clean))

    def tail_ratio(self, returns: pd.Series, percentile: float = 95) -> float:
        """
        P(percentile-th) / |P((100−percentile)-th)|.
        >1 = positive-skew distribution (more upside tail than downside).
        """
        clean = returns.dropna()
        if len(clean) < 10:
            return 1.0
        p_high = float(np.percentile(clean, percentile))
        p_low = abs(float(np.percentile(clean, 100 - percentile)))
        if p_low == 0:
            return float("inf") if p_high > 0 else 1.0
        return float(p_high / p_low)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate a comprehensive performance report.

        Metrics always computed:
            annualized_return, annualized_volatility, sharpe_ratio,
            sortino_ratio, max_drawdown, max_drawdown_duration, calmar_ratio,
            win_rate, profit_factor, skewness, kurtosis, tail_ratio

        If benchmark provided, also computes:
            alpha, beta, information_ratio, tracking_error

        Returns:
            DataFrame indexed by metric name with a single 'value' column.
        """
        metrics: dict = {
            "annualized_return":     self.annualized_return(returns),
            "annualized_volatility": self.annualized_volatility(returns),
            "sharpe_ratio":          self.sharpe_ratio(returns),
            "sortino_ratio":         self.sortino_ratio(returns),
            "max_drawdown":          self.max_drawdown(returns),
            "max_drawdown_duration": self.max_drawdown_duration(returns),
            "calmar_ratio":          self.calmar_ratio(returns),
            "win_rate":              self.win_rate(returns),
            "profit_factor":         self.profit_factor(returns),
            "skewness":              self.skewness(returns),
            "kurtosis":              self.kurtosis(returns),
            "tail_ratio":            self.tail_ratio(returns),
        }

        if benchmark_returns is not None:
            from statarb.evaluation.risk import RiskAnalytics
            ra = RiskAnalytics()
            ab = ra.alpha_beta(returns, benchmark_returns)
            metrics["alpha"] = ab["alpha"]
            metrics["beta"] = ab["beta"]
            bench_aligned = benchmark_returns.reindex(returns.dropna().index).fillna(0)
            active = returns.dropna() - bench_aligned
            te = float(active.std() * np.sqrt(365))
            ar = float(
                (returns.dropna().mean() - benchmark_returns.dropna().mean()) * 365
            )
            metrics["information_ratio"] = ar / te if te > 0 else 0.0
            metrics["tracking_error"] = te

        return pd.DataFrame({"value": metrics})

    # ------------------------------------------------------------------
    # Static helpers for backward compatibility
    # ------------------------------------------------------------------

    @staticmethod
    def compare(
        results: dict,
        benchmark_returns: Optional[pd.Series] = None,
        periods_per_year: int = 365,
    ) -> pd.DataFrame:
        """Compare multiple strategy return series side-by-side."""
        pm = PerformanceMetrics()
        summaries = {}
        for name, rets in results.items():
            report = pm.full_report(rets, benchmark_returns)
            summaries[name] = report["value"]
        return pd.DataFrame(summaries)
