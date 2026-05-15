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

_pm = PerformanceMetrics()
_ra = RiskAnalytics()


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
        periods_per_year: int = 365,
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
            report = _pm.full_report(rets, self.benchmark_returns)
            tables[name] = report["value"]
        return pd.DataFrame(tables)

    def alpha_beta_table(self) -> pd.DataFrame:
        """
        Alpha / beta decomposition for each strategy vs the benchmark.

        Returns:
            DataFrame with alpha_ann, beta, r_squared columns.
        """
        if self.benchmark_returns is None:
            raise ValueError("benchmark_returns required for alpha/beta decomposition")

        rows = {}
        for name, rets in self.strategy_returns.items():
            ab = _ra.alpha_beta(rets, self.benchmark_returns)
            rows[name] = {
                "alpha_ann":   ab["alpha"],
                "beta":        ab["beta"],
                "r_squared":   ab["r_squared"],
                "alpha_tstat": ab["alpha_tstat"],
                "beta_tstat":  ab["beta_tstat"],
            }
        return pd.DataFrame(rows).T

    def drawdown_table(self) -> pd.DataFrame:
        """
        Drawdown summary table: max drawdown, duration, calmar.

        Returns:
            DataFrame indexed by strategy name.
        """
        rows = {}
        for name, rets in self.strategy_returns.items():
            rows[name] = {
                "max_drawdown":              _pm.max_drawdown(rets),
                "max_drawdown_duration_days": _pm.max_drawdown_duration(rets),
                "calmar_ratio":              _pm.calmar_ratio(rets, self.periods_per_year),
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
        fmt_metrics = metrics.copy().astype(object)
        pct_rows = [
            "annualized_return", "annualized_volatility",
            "max_drawdown", "win_rate", "tracking_error",
        ]
        for row in pct_rows:
            if row in fmt_metrics.index:
                fmt_metrics.loc[row] = (
                    fmt_metrics.loc[row].astype(float) * 100
                ).round(2).astype(str) + "%"

        float_rows = [
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "skewness", "kurtosis", "profit_factor", "tail_ratio",
            "alpha", "beta", "information_ratio",
        ]
        for row in float_rows:
            if row in fmt_metrics.index:
                fmt_metrics.loc[row] = fmt_metrics.loc[row].astype(float).round(3)

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
    def one_liner(
        returns: pd.Series,
        name: str = "strategy",
        periods_per_year: int = 365,
    ) -> str:
        """
        One-line summary string for quick inspection.

        Returns:
            e.g. "momentum | Sharpe 1.23 | Ann.Ret 24.5% | MDD -18.3%"
        """
        pm = PerformanceMetrics()
        return (
            f"{name:20s} | "
            f"Sharpe {pm.sharpe_ratio(returns, periods_per_year=periods_per_year):5.2f} | "
            f"Ann.Ret {pm.annualized_return(returns, periods_per_year)*100:6.1f}% | "
            f"Vol {pm.annualized_volatility(returns, periods_per_year)*100:5.1f}% | "
            f"MDD {pm.max_drawdown(returns)*100:6.1f}%"
        )
