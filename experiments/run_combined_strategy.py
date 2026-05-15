"""
Combine the best momentum and reversal signals into a composite strategy.

Tests four combination methods (equal weight, inverse vol, Sharpe-weighted,
min variance) and demonstrates the diversification benefit.

Usage:
    python experiments/run_combined_strategy.py
    python experiments/run_combined_strategy.py --synthetic
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._utils import (
    backtest_signals, load_data, print_summary_table, run_bt,
    save_fig, save_results,
)
from statarb.backtest.execution import ExecutionModel
from statarb.backtest.weighting import StrategyWeighting
from statarb.data.features import FeatureEngine
from statarb.evaluation.metrics import PerformanceMetrics
from statarb.signals.activity import ActivityFilter
from statarb.signals.momentum import MomentumSignals
from statarb.signals.reversal import ReversalSignals
from statarb.utils import ensure_dir, get_logger, load_config

logger = get_logger("run_combined")
_pm = PerformanceMetrics()


def main(args: argparse.Namespace) -> None:
    cfg = load_config("experiments/config.yaml")
    combo_cfg = cfg.get("combination", {})
    periods_per_year = cfg.get("evaluation", {}).get("periods_per_year", 365)
    out_dir = Path(cfg.get("output", {}).get("results_dir", "experiments/results")) / "combined"
    ensure_dir(out_dir)

    prices, volume, returns = load_data(cfg)
    fe = FeatureEngine()
    vol_ratio = fe.volume_ma_ratio(volume, window=21)
    act = ActivityFilter()
    mom = MomentumSignals()
    rev = ReversalSignals()
    em = ExecutionModel(market_order_cost=0.0020)

    # ------------------------------------------------------------------ #
    # 1. Build individual signal suite                                     #
    # ------------------------------------------------------------------ #
    logger.info("=== Building individual signals ===")
    raw_signals = {
        "mom_6_1":       mom.momentum_6_1(returns),
        "mom_3_1":       mom.momentum_3_1(returns),
        "ts_mom_63d":    mom.time_series_momentum(returns, lookback=63),
        "sharpe_mom":    mom.sharpe_momentum(returns, lookback=63),
        "reversal_5d":   rev.weekly_reversal(returns),
        "vol_adj_rev":   rev.vol_adjusted_reversal(returns),
    }

    # Activity-gate all signals
    gated = {
        name: act.activity_gate(sig, volume, min_ratio=0.5)
        for name, sig in raw_signals.items()
    }
    ranked = {name: fe.cross_sectional_rank(sig) for name, sig in gated.items()}

    # ------------------------------------------------------------------ #
    # 2. Backtest individual signals                                       #
    # ------------------------------------------------------------------ #
    logger.info("=== Individual signal backtests ===")
    individual_results = backtest_signals(ranked, returns, em, periods_per_year)
    save_results(individual_results, str(out_dir / "individual_signals.csv"))
    print_summary_table(individual_results, "Individual Signal Performance")

    # Collect return streams
    strategy_returns: dict[str, pd.Series] = {}
    for name, sig in ranked.items():
        net_rets, _ = run_bt(sig, returns, em)
        strategy_returns[name] = net_rets

    returns_df = pd.DataFrame(strategy_returns).dropna()

    # ------------------------------------------------------------------ #
    # 3. Combine using four methods                                        #
    # ------------------------------------------------------------------ #
    logger.info("=== Strategy combination ===")
    sw = StrategyWeighting()
    estimation_window = combo_cfg.get("estimation_window", 63)

    combined_returns: dict[str, pd.Series] = {}
    for method in ["equal", "inverse_vol", "sharpe", "min_variance"]:
        try:
            combined = sw.combine(returns_df, method=method,
                                  lookback=estimation_window)
            combined_returns[f"combined_{method}"] = combined
        except Exception as exc:
            logger.warning("Combination method %s failed: %s", method, exc)

    # Also add individual best for comparison
    best_individual = individual_results["sharpe"].idxmax()
    combined_returns[f"best_individual ({best_individual})"] = strategy_returns[best_individual]

    # ------------------------------------------------------------------ #
    # 4. Evaluate combined strategies                                      #
    # ------------------------------------------------------------------ #
    combo_summary = {}
    for name, rets in combined_returns.items():
        clean = rets.dropna()
        sharpe = _pm.sharpe_ratio(clean, periods_per_year=periods_per_year)
        combo_summary[name] = {
            "sharpe":            sharpe,
            "annualized_return": _pm.annualized_return(clean, periods_per_year),
            "annualized_vol":    _pm.annualized_volatility(clean, periods_per_year),
            "max_drawdown":      _pm.max_drawdown(clean),
        }
    combo_df = pd.DataFrame(combo_summary).T
    save_results(combo_df, str(out_dir / "combined_strategies.csv"))
    print_summary_table(combo_df, "Combined Strategy Performance")

    # ------------------------------------------------------------------ #
    # 5. Diversification benefit                                           #
    # ------------------------------------------------------------------ #
    indiv_sharpes = individual_results["sharpe"]
    equal_combined_sharpe = combo_summary.get("combined_equal", {}).get("sharpe", np.nan)
    avg_indiv_sharpe = float(indiv_sharpes.mean())

    logger.info(
        "Diversification benefit: avg individual Sharpe %.2f → equal-weight combined %.2f",
        avg_indiv_sharpe, equal_combined_sharpe,
    )

    # ------------------------------------------------------------------ #
    # 6. Plots                                                             #
    # ------------------------------------------------------------------ #
    # Cumulative return comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, rets in combined_returns.items():
        cum = (1 + rets.fillna(0)).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, label=name, linewidth=1.3)
    ax.set_title("Combined Strategy Cumulative Returns", fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    save_fig(fig, str(out_dir / "cumulative_returns.png"))

    # Sharpe comparison bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    all_sharpes = pd.concat([indiv_sharpes, combo_df["sharpe"]])
    colors = ["tab:blue"] * len(indiv_sharpes) + ["tab:orange"] * len(combo_df)
    ax2.bar(range(len(all_sharpes)), all_sharpes.values, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(all_sharpes)))
    ax2.set_xticklabels(list(all_sharpes.index), rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Annualized Sharpe Ratio")
    ax2.set_title("Individual vs Combined Strategy Sharpe", fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(avg_indiv_sharpe, color="tab:blue", linestyle="--",
                linewidth=1.0, label=f"Avg individual ({avg_indiv_sharpe:.2f})")
    ax2.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    save_fig(fig2, str(out_dir / "sharpe_comparison.png"))

    # Correlation matrix
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    corr = returns_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", center=0,
                cmap="coolwarm", square=True, ax=ax3,
                cbar_kws={"shrink": 0.8})
    ax3.set_title("Individual Strategy Return Correlations", fontweight="bold")
    plt.tight_layout()
    save_fig(fig3, str(out_dir / "strategy_correlations.png"))

    logger.info("=== Combined strategy research complete — results in %s ===", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined strategy research")
    parser.add_argument("--synthetic", action="store_true")
    main(parser.parse_args())
