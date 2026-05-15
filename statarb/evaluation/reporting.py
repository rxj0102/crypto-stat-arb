"""
Performance report generation.

Produces structured summaries suitable for display, logging,
or export to CSV/HTML.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from statarb.evaluation.metrics import PerformanceMetrics
from statarb.evaluation.risk import RiskAnalytics
from statarb.utils import get_logger

logger = get_logger(__name__)


class PerformanceReporter:
    """
    Generate structured performance reports for one or more strategies.

    Reports include:
    - Core return and risk metrics
    - Alpha / beta decomposition vs benchmark
    - Drawdown analysis
    - Regime breakdown (optional)
    """

    def __init__(
        self,
        strategy_returns: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
        periods_per_year: int = 252,
    ):
        """
        Args:
            strategy_returns: dict of strategy_name → return Series
            benchmark_returns: benchmark return series (e.g. BTC)
            periods_per_year: annualisation factor
        """
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # Core report
    # ------------------------------------------------------------------

    def metrics_table(self) -> pd.DataFrame:
        """
        Full metrics table: strategies as columns, metrics as rows.

        Returns:
            DataFrame suitable for display or CSV export.
        """
        tables = {}
        for name, rets in self.strategy_returns.items():
            pm = PerformanceMetrics(rets, self.periods_per_year)
            metrics = pm.summary(self.benchmark_returns)
            tables[name] = metrics
        return pd.DataFrame(tables)

    def alpha_beta_table(self) -> pd.DataFrame:
        """
        Alpha / beta decomposition for each strategy vs the benchmark.

        Returns:
            DataFrame with alpha, beta, r_squared, p_value columns.
        """
        if self.benchmark_returns is None:
            raise ValueError("benchmark_returns required for alpha/beta decomposition")

        rows = {}
        for name, rets in self.strategy_returns.items():
            ra = RiskAnalytics(rets, self.periods_per_year)
            alpha, beta, r2, pval = ra.alpha_beta(self.benchmark_returns)
            rows[name] = {
                "alpha_ann": alpha,
                "beta": beta,
                "r_squared": r2,
                "p_value_alpha": pval,
            }
        return pd.DataFrame(rows).T

    def drawdown_table(self) -> pd.DataFrame:
        """
        Drawdown summary table: max drawdown, duration, and average.

        Returns:
            DataFrame indexed by strategy name.
        """
        rows = {}
        for name, rets in self.strategy_returns.items():
            pm = PerformanceMetrics(rets, self.periods_per_year)
            rows[name] = {
                "max_drawdown": pm.max_drawdown(),
                "max_drawdown_duration_days": pm.max_drawdown_duration(),
                "average_drawdown": pm.average_drawdown(),
                "calmar_ratio": pm.calmar_ratio(),
            }
        return pd.DataFrame(rows).T

    def correlation_table(self) -> pd.DataFrame:
        """Pairwise correlation matrix of all strategy returns."""
        return pd.DataFrame(self.strategy_returns).corr()

    # ------------------------------------------------------------------
    # Formatted report
    # ------------------------------------------------------------------

    def print_report(self, title: str = "Strategy Performance Report") -> None:
        """Print a formatted performance report to stdout."""
        separator = "=" * 70
        print(f"\n{separator}")
        print(f"  {title}")
        print(separator)

        metrics = self.metrics_table()
        print("\n[ CORE METRICS ]")
        fmt_metrics = metrics.copy()
        pct_rows = [
            "total_return", "annualised_return", "annualised_volatility",
            "max_drawdown", "average_drawdown", "hit_rate", "var_95", "cvar_95",
        ]
        for row in pct_rows:
            if row in fmt_metrics.index:
                fmt_metrics.loc[row] = (fmt_metrics.loc[row] * 100).round(2).astype(str) + "%"

        float_rows = ["sharpe_ratio", "sortino_ratio", "calmar_ratio",
                      "skewness", "kurtosis", "profit_factor"]
        for row in float_rows:
            if row in fmt_metrics.index:
                fmt_metrics.loc[row] = fmt_metrics.loc[row].round(3)

        print(fmt_metrics.to_string())

        if self.benchmark_returns is not None:
            print("\n[ ALPHA / BETA ]")
            print(self.alpha_beta_table().round(4).to_string())

        print("\n[ DRAWDOWN ]")
        print(self.drawdown_table().round(4).to_string())

        print("\n[ CORRELATION ]")
        print(self.correlation_table().round(3).to_string())
        print(f"\n{separator}\n")

    def to_csv(self, path: str) -> None:
        """Export full metrics table to CSV."""
        self.metrics_table().to_csv(path)
        logger.info("Performance report saved to %s", path)

    def to_html(self, path: str) -> None:
        """Export full metrics table to styled HTML."""
        html = self.metrics_table().to_html(classes="performance-table", border=0)
        with open(path, "w") as f:
            f.write(f"<html><body>{html}</body></html>")
        logger.info("HTML report saved to %s", path)

    # ------------------------------------------------------------------
    # Single-strategy summary string
    # ------------------------------------------------------------------

    @staticmethod
    def one_liner(returns: pd.Series, name: str = "strategy",
                  periods_per_year: int = 252) -> str:
        """
        One-line summary string for quick inspection.

        Returns:
            e.g. "momentum | Sharpe 1.23 | Ann.Ret 24.5% | MDD -18.3% | Turnover N/A"
        """
        pm = PerformanceMetrics(returns.dropna(), periods_per_year)
        return (
            f"{name:20s} | "
            f"Sharpe {pm.sharpe_ratio():5.2f} | "
            f"Ann.Ret {pm.annualised_return()*100:6.1f}% | "
            f"Vol {pm.annualised_volatility()*100:5.1f}% | "
            f"MDD {pm.max_drawdown()*100:6.1f}%"
        )
