"""
Visualization utilities for strategy performance analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from statarb.evaluation.metrics import PerformanceMetrics
from statarb.utils import get_logger

logger = get_logger(__name__)

# Default plot style
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


class Plotter:
    """
    Visualization utilities for backtesting and performance analysis.

    All plot methods return the matplotlib Figure so callers can
    save, display, or further customise them.
    """

    # ------------------------------------------------------------------
    # Cumulative return chart
    # ------------------------------------------------------------------

    def plot_cumulative_returns(
        self,
        returns: Dict[str, pd.Series],
        title: str = "Cumulative Returns",
        log_scale: bool = False,
        figsize: tuple = (12, 5),
    ) -> plt.Figure:
        """
        Plot cumulative return curves for multiple strategies.

        Args:
            returns: dict of strategy_name → return Series
            title: chart title
            log_scale: use log y-axis
            figsize: figure size tuple

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        for name, rets in returns.items():
            cum = np.exp(rets.dropna().cumsum()) - 1
            ax.plot(cum.index, cum.values * 100, label=name, linewidth=1.5)

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(frameon=False)
        if log_scale:
            ax.set_yscale("log")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Drawdown chart
    # ------------------------------------------------------------------

    def plot_drawdowns(
        self,
        returns: Dict[str, pd.Series],
        title: str = "Drawdowns",
        figsize: tuple = (12, 5),
    ) -> plt.Figure:
        """Plot drawdown series for multiple strategies."""
        fig, ax = plt.subplots(figsize=figsize)
        for name, rets in returns.items():
            pm = PerformanceMetrics(rets.dropna())
            dd = pm.drawdown_series()
            ax.fill_between(dd.index, dd.values * 100, 0,
                            alpha=0.3, label=name)
            ax.plot(dd.index, dd.values * 100, linewidth=0.8)

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Rolling Sharpe ratio
    # ------------------------------------------------------------------

    def plot_rolling_sharpe(
        self,
        returns: Dict[str, pd.Series],
        window: int = 63,
        title: str = "Rolling Sharpe Ratio (63-day)",
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """Plot rolling annualised Sharpe ratio."""
        fig, ax = plt.subplots(figsize=figsize)
        for name, rets in returns.items():
            from statarb.evaluation.risk import RiskAnalytics
            ra = RiskAnalytics(rets.dropna())
            rolling = ra.rolling_sharpe(window)
            ax.plot(rolling.index, rolling.values, label=name, linewidth=1.2)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(1, color="green", linewidth=0.6, linestyle=":", alpha=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Sharpe Ratio (annualised)")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Return distribution
    # ------------------------------------------------------------------

    def plot_return_distribution(
        self,
        returns: Dict[str, pd.Series],
        bins: int = 60,
        figsize: tuple = (10, 5),
    ) -> plt.Figure:
        """Histogram of return distributions with KDE overlay."""
        fig, ax = plt.subplots(figsize=figsize)
        for name, rets in returns.items():
            clean = rets.dropna()
            sns.histplot(clean * 100, bins=bins, kde=True, label=name,
                         alpha=0.4, ax=ax, stat="density")

        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.set_title("Return Distribution", fontweight="bold")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Heatmap: monthly returns
    # ------------------------------------------------------------------

    def plot_monthly_returns_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns (%)",
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """
        Plot a calendar heatmap of monthly returns.

        Args:
            returns: daily return Series for one strategy
            title: chart title

        Returns:
            matplotlib Figure
        """
        monthly = returns.resample("ME").sum() * 100  # approximate monthly
        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][:len(pivot.columns)]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot, annot=True, fmt=".1f", center=0,
            cmap="RdYlGn", linewidths=0.5,
            cbar_kws={"label": "Return (%)"}, ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Signal IC over time
    # ------------------------------------------------------------------

    def plot_ic_series(
        self,
        signal: pd.DataFrame,
        forward_returns: pd.DataFrame,
        window: int = 21,
        title: str = "Rolling Information Coefficient",
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """
        Plot the rolling rank IC (Spearman correlation) between signal and returns.

        Args:
            signal: signal panel (dates × symbols)
            forward_returns: forward return panel (dates × symbols)
            window: rolling window for IC smoothing
            title: chart title

        Returns:
            matplotlib Figure
        """
        common_idx = signal.index.intersection(forward_returns.index)
        sig_aligned = signal.loc[common_idx]
        ret_aligned = forward_returns.loc[common_idx]

        ics = []
        for date in common_idx:
            s = sig_aligned.loc[date].dropna()
            r = ret_aligned.loc[date].reindex(s.index).dropna()
            common_syms = s.index.intersection(r.index)
            if len(common_syms) < 5:
                ics.append(np.nan)
                continue
            ic = s.loc[common_syms].corr(r.loc[common_syms], method="spearman")
            ics.append(ic)

        ic_series = pd.Series(ics, index=common_idx)
        rolling_ic = ic_series.rolling(window, min_periods=5).mean()

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(ic_series.index, ic_series.values, alpha=0.3, color="steelblue",
               label="Daily IC")
        ax.plot(rolling_ic.index, rolling_ic.values, color="navy",
                linewidth=1.5, label=f"{window}-day avg IC")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Rank IC (Spearman)")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        returns: Dict[str, pd.Series],
        title: str = "Strategy Correlation Matrix",
        figsize: tuple = (8, 6),
    ) -> plt.Figure:
        """Plot pairwise correlation heatmap for multiple strategies."""
        corr = pd.DataFrame(returns).corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr, annot=True, fmt=".2f", center=0,
            cmap="coolwarm", square=True, linewidths=0.5,
            vmin=-1, vmax=1, ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------

    @staticmethod
    def save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
        """Save figure to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved to %s", path)
