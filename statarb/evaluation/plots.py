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

_pm = PerformanceMetrics()


class PerformancePlots:
    """
    Visualization utilities for single-strategy performance analysis.

    All plot methods return the matplotlib Figure.

    Example::

        pp = PerformancePlots()
        fig = pp.cumulative_return_plot(returns, benchmark=btc_returns)
    """

    # ------------------------------------------------------------------
    # Cumulative return
    # ------------------------------------------------------------------

    def cumulative_return_plot(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: Optional[str] = None,
        figsize: tuple = (12, 5),
    ) -> plt.Figure:
        """Cumulative return curve, optionally with benchmark overlay."""
        fig, ax = plt.subplots(figsize=figsize)
        cum = (1 + returns.dropna()).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, label="Strategy", linewidth=1.5)

        if benchmark is not None:
            bench_cum = (1 + benchmark.dropna()).cumprod() - 1
            ax.plot(bench_cum.index, bench_cum.values * 100,
                    label="Benchmark", linewidth=1.2, linestyle="--", alpha=0.7)

        ax.set_title(title or "Cumulative Return", fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(frameon=False)
        ax.axhline(0, color="black", linewidth=0.6)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def drawdown_plot(
        self,
        returns: pd.Series,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """Underwater (drawdown) plot."""
        clean = returns.dropna()
        cum = (1 + clean).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak * 100

        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="crimson")
        ax.plot(dd.index, dd.values, color="crimson", linewidth=0.8)
        ax.set_title("Drawdown", fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Monthly returns heatmap
    # ------------------------------------------------------------------

    def monthly_returns_heatmap(
        self,
        returns: pd.Series,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """Calendar heatmap of monthly returns."""
        monthly = returns.resample("ME").sum() * 100
        df = pd.DataFrame({
            "year":   monthly.index.year,
            "month":  monthly.index.month,
            "return": monthly.values,
        })
        pivot = df.pivot(index="year", columns="month", values="return")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot, annot=True, fmt=".1f", center=0,
            cmap="RdYlGn", linewidths=0.5,
            cbar_kws={"label": "Return (%)"}, ax=ax,
        )
        ax.set_title("Monthly Returns (%)", fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Rolling Sharpe
    # ------------------------------------------------------------------

    def rolling_sharpe_plot(
        self,
        returns: pd.Series,
        window: int = 126,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """Rolling annualized Sharpe ratio."""
        from statarb.evaluation.risk import RiskAnalytics
        ra = RiskAnalytics()
        rolling = ra.rolling_sharpe(returns, window=window)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rolling.index, rolling.values, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(1, color="green", linewidth=0.6, linestyle=":", alpha=0.7)
        ax.set_title(f"Rolling Sharpe ({window}-day)", fontweight="bold")
        ax.set_ylabel("Sharpe Ratio (annualized)")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Return distribution
    # ------------------------------------------------------------------

    def return_distribution(
        self,
        returns: pd.Series,
        bins: int = 60,
        figsize: tuple = (10, 5),
    ) -> plt.Figure:
        """Histogram of return distribution with KDE overlay."""
        clean = returns.dropna()
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(clean * 100, bins=bins, kde=True, alpha=0.4,
                     color="steelblue", stat="density", ax=ax)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.set_title("Return Distribution", fontweight="bold")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Strategy comparison table
    # ------------------------------------------------------------------

    def strategy_comparison_table(
        self,
        strategy_dict: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Build a side-by-side metrics comparison DataFrame.

        Args:
            strategy_dict: {name: return_series}

        Returns:
            DataFrame with metrics as rows and strategies as columns.
        """
        return PerformanceMetrics.compare(strategy_dict)

    # ------------------------------------------------------------------
    # Turnover plot
    # ------------------------------------------------------------------

    def turnover_plot(
        self,
        turnover: pd.Series,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """Bar chart of daily portfolio turnover."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(turnover.index, turnover.values * 100, width=1, alpha=0.7)
        roll = turnover.rolling(21, min_periods=5).mean()
        ax.plot(roll.index, roll.values * 100, color="navy",
                linewidth=1.5, label="21-day avg")
        ax.set_title("Portfolio Turnover", fontweight="bold")
        ax.set_ylabel("Turnover (%)")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Signal decay (IC by horizon)
    # ------------------------------------------------------------------

    def signal_decay_plot(
        self,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
        max_horizon: int = 20,
        figsize: tuple = (10, 4),
    ) -> plt.Figure:
        """
        Plot rank IC (Spearman) between signal and forward returns at each horizon.

        Args:
            signal:      signal panel (dates × symbols)
            returns:     daily return panel (dates × symbols)
            max_horizon: max number of days forward to compute IC
        """
        horizons = range(1, max_horizon + 1)
        ics = []
        for h in horizons:
            fwd = returns.shift(-h)
            common = signal.index.intersection(fwd.index)
            sig_flat = signal.loc[common].stack().dropna()
            ret_flat = fwd.loc[common].stack().reindex(sig_flat.index).dropna()
            sig_aligned = sig_flat.reindex(ret_flat.index).dropna()
            ret_aligned = ret_flat.reindex(sig_aligned.index)
            if len(sig_aligned) < 20:
                ics.append(np.nan)
                continue
            ic = float(sig_aligned.corr(ret_aligned, method="spearman"))
            ics.append(ic)

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(list(horizons), ics, alpha=0.7, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Forward Horizon (days)")
        ax.set_ylabel("Rank IC (Spearman)")
        ax.set_title("Signal Decay", fontweight="bold")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Full report: grid of plots
    # ------------------------------------------------------------------

    def full_report_plots(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        positions: Optional[pd.DataFrame] = None,
        turnover: Optional[pd.Series] = None,
        save_dir: Optional[str] = None,
    ) -> List[plt.Figure]:
        """
        Generate and optionally save the standard set of diagnostic plots.

        Returns list of Figures: [cumulative, drawdown, distribution, rolling_sharpe,
                                   monthly_heatmap, turnover (if provided)].
        """
        figs = []

        figs.append(self.cumulative_return_plot(returns, benchmark=benchmark))
        figs.append(self.drawdown_plot(returns))
        figs.append(self.return_distribution(returns))
        figs.append(self.rolling_sharpe_plot(returns))
        figs.append(self.monthly_returns_heatmap(returns))

        if turnover is not None:
            figs.append(self.turnover_plot(turnover))

        if save_dir is not None:
            names = [
                "cumulative_return", "drawdown", "return_distribution",
                "rolling_sharpe", "monthly_heatmap", "turnover",
            ]
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            for fig, name in zip(figs, names):
                fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved %s/%s.png", save_dir, name)

        return figs


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
            cum = (1 + rets.dropna()).cumprod() - 1
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
            clean = rets.dropna()
            cum = (1 + clean).cumprod()
            peak = cum.cummax()
            dd = (cum - peak) / peak
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
        from statarb.evaluation.risk import RiskAnalytics
        ra = RiskAnalytics()
        for name, rets in returns.items():
            rolling = ra.rolling_sharpe(rets.dropna(), window=window)
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
        monthly = returns.resample("ME").sum() * 100
        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = df.pivot(index="year", columns="month", values="return")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

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
