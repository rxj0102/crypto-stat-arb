"""
Full performance evaluation of the combined strategy.

Runs the complete pipeline: signal generation, backtest, metrics,
alpha/beta vs BTC, factor exposure, drawdown analysis, all plots.

Usage:
    python experiments/run_full_evaluation.py
    python experiments/run_full_evaluation.py --synthetic
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

from experiments._utils import load_data, run_bt, save_fig, save_results
from statarb.backtest.execution import ExecutionModel
from statarb.backtest.weighting import StrategyWeighting
from statarb.data.features import FeatureEngine
from statarb.evaluation.metrics import PerformanceMetrics
from statarb.evaluation.plots import PerformancePlots
from statarb.evaluation.reporting import PerformanceReporter
from statarb.evaluation.risk import RiskAnalytics
from statarb.signals.activity import ActivityFilter
from statarb.signals.momentum import MomentumSignals
from statarb.signals.reversal import ReversalSignals
from statarb.utils import ensure_dir, get_logger, load_config

logger = get_logger("run_full_eval")
_pm = PerformanceMetrics()
_ra = RiskAnalytics()
_pp = PerformancePlots()


def main(args: argparse.Namespace) -> None:
    cfg = load_config("experiments/config.yaml")
    eval_cfg = cfg.get("evaluation", {})
    periods_per_year = eval_cfg.get("periods_per_year", 365)
    out_dir = Path(cfg.get("output", {}).get("results_dir", "experiments/results")) / "full_eval"
    ensure_dir(out_dir)

    prices, volume, returns = load_data(cfg)
    fe = FeatureEngine()
    vol_ratio = fe.volume_ma_ratio(volume, window=21)
    act = ActivityFilter()
    mom = MomentumSignals()
    rev = ReversalSignals()
    em = ExecutionModel(market_order_cost=0.0020)

    # ------------------------------------------------------------------ #
    # 1. Build the composite signal                                        #
    # ------------------------------------------------------------------ #
    logger.info("=== Building composite signal ===")
    component_signals = {
        "mom_6_1":     fe.cross_sectional_rank(
            act.activity_gate(mom.momentum_6_1(returns), volume)),
        "mom_3_1":     fe.cross_sectional_rank(
            act.activity_gate(mom.momentum_3_1(returns), volume)),
        "sharpe_mom":  fe.cross_sectional_rank(
            act.activity_gate(mom.sharpe_momentum(returns, 63), volume)),
        "reversal_5d": fe.cross_sectional_rank(
            act.activity_gate(rev.weekly_reversal(returns), volume)),
        "vol_adj_rev": fe.cross_sectional_rank(
            act.activity_gate(rev.vol_adjusted_reversal(returns), volume)),
    }

    sw = StrategyWeighting()

    # Build return series for each component first
    component_rets: dict[str, pd.Series] = {}
    for name, sig in component_signals.items():
        net_rets, _ = run_bt(sig, returns, em)
        component_rets[name] = net_rets

    comp_df = pd.DataFrame(component_rets).dropna()

    # Composite via equal weight (most robust out-of-sample)
    composite_returns = sw.equal_weight(comp_df)

    # ------------------------------------------------------------------ #
    # 2. BTC benchmark                                                     #
    # ------------------------------------------------------------------ #
    benchmark_col = eval_cfg.get("benchmark", "BTC/USDT")
    if benchmark_col in prices.columns:
        benchmark_returns = fe.log_returns(prices[[benchmark_col]])[benchmark_col]
        logger.info("Using %s as benchmark", benchmark_col)
    elif len(prices.columns) > 0:
        benchmark_col = prices.columns[0]
        benchmark_returns = fe.log_returns(prices[[benchmark_col]])[benchmark_col]
        logger.info("BTC not found — using %s as proxy benchmark", benchmark_col)
    else:
        benchmark_returns = None
        logger.warning("No benchmark available")

    # ------------------------------------------------------------------ #
    # 3. Full performance report                                           #
    # ------------------------------------------------------------------ #
    logger.info("=== Performance metrics ===")
    all_returns = dict(component_rets)
    all_returns["composite"] = composite_returns

    reporter = PerformanceReporter(
        all_returns,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
    )
    reporter.print_report(title="Full Strategy Evaluation")
    reporter.to_csv(str(out_dir / "performance_metrics.csv"))

    # ------------------------------------------------------------------ #
    # 4. Full report DataFrame (composite only)                            #
    # ------------------------------------------------------------------ #
    report = _pm.full_report(composite_returns, benchmark_returns=benchmark_returns)
    logger.info("\n=== Composite Strategy ===")
    for metric, row in report.iterrows():
        val = row["value"]
        if isinstance(val, float) and abs(val) < 100:
            logger.info("  %-30s %8.4f", metric, val)

    # ------------------------------------------------------------------ #
    # 5. Alpha / beta decomposition                                        #
    # ------------------------------------------------------------------ #
    if benchmark_returns is not None:
        logger.info("=== Alpha / Beta Decomposition ===")
        ab = _ra.alpha_beta(composite_returns, benchmark_returns)
        logger.info("  Alpha (ann.):   %.2f%%", ab["alpha"] * 100)
        logger.info("  Beta:           %.3f", ab["beta"])
        logger.info("  R²:             %.3f", ab["r_squared"])
        logger.info("  Alpha t-stat:   %.2f", ab["alpha_tstat"])

        # Rolling beta
        rolling_beta = _ra.rolling_beta(composite_returns, benchmark_returns, window=63)
        fig_rb, ax_rb = plt.subplots(figsize=(12, 3))
        ax_rb.plot(rolling_beta.index, rolling_beta.values, linewidth=1.2)
        ax_rb.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_rb.axhline(ab["beta"], color="red", linewidth=0.8, linestyle=":",
                      label=f"Full-sample β={ab['beta']:.2f}")
        ax_rb.set_title("Rolling Beta vs Benchmark (63-day)", fontweight="bold")
        ax_rb.set_ylabel("Beta")
        ax_rb.legend(frameon=False)
        plt.tight_layout()
        save_fig(fig_rb, str(out_dir / "rolling_beta.png"))

    # ------------------------------------------------------------------ #
    # 6. Factor exposure                                                   #
    # ------------------------------------------------------------------ #
    if benchmark_returns is not None and len(prices.columns) >= 3:
        logger.info("=== Factor Exposure ===")
        factors = pd.DataFrame({
            "market": benchmark_returns,
            "vol":    fe.realized_volatility(returns, window=21).mean(axis=1),
        }).dropna()
        try:
            exposure = _ra.factor_exposure(composite_returns, factors)
            logger.info("  Market beta:  %.3f (t=%.2f)",
                        exposure.get("market", 0), exposure.get("market_tstat", 0))
            logger.info("  Vol beta:     %.3f", exposure.get("vol", 0))
            logger.info("  Alpha (ann.): %.3f", exposure.get("alpha", 0))
        except Exception as exc:
            logger.warning("Factor exposure failed: %s", exc)

    # ------------------------------------------------------------------ #
    # 7. Drawdown analysis                                                 #
    # ------------------------------------------------------------------ #
    logger.info("=== Drawdown Analysis ===")
    dd_table = _ra.drawdown_analysis(composite_returns)
    if not dd_table.empty:
        logger.info("Drawdown episodes (>5%%):\n%s", dd_table.to_string())
        save_results(dd_table, str(out_dir / "drawdown_episodes.csv"))
    else:
        logger.info("No drawdown episodes exceeding 5%%")

    # ------------------------------------------------------------------ #
    # 8. Return attribution                                                #
    # ------------------------------------------------------------------ #
    logger.info("=== Return Attribution ===")
    # Build position panel from composite signal
    # (approx via equal-weight of component signals)
    comp_positions = {}
    for name, sig in component_signals.items():
        _, result = run_bt(sig, returns, em)
        comp_positions[name] = result["positions"].mean(axis=1)  # avg exposure

    # ------------------------------------------------------------------ #
    # 9. All plots                                                         #
    # ------------------------------------------------------------------ #
    logger.info("=== Generating plots ===")

    # Cumulative returns (all strategies)
    from statarb.evaluation.plots import Plotter
    plotter = Plotter()

    fig = plotter.plot_cumulative_returns(all_returns, "Cumulative Returns — All Strategies")
    save_fig(fig, str(out_dir / "cumulative_returns.png"))

    fig = plotter.plot_drawdowns(all_returns, "Drawdowns — All Strategies")
    save_fig(fig, str(out_dir / "drawdowns.png"))

    fig = plotter.plot_rolling_sharpe(all_returns, window=63)
    save_fig(fig, str(out_dir / "rolling_sharpe.png"))

    fig = plotter.plot_return_distribution(all_returns)
    save_fig(fig, str(out_dir / "return_distribution.png"))

    fig = plotter.plot_correlation_heatmap(all_returns)
    save_fig(fig, str(out_dir / "strategy_correlations.png"))

    # Composite-only detail plots
    fig = _pp.cumulative_return_plot(composite_returns, benchmark=benchmark_returns,
                                      title="Composite Strategy vs Benchmark")
    save_fig(fig, str(out_dir / "composite_vs_benchmark.png"))

    fig = _pp.drawdown_plot(composite_returns)
    save_fig(fig, str(out_dir / "composite_drawdown.png"))

    fig = _pp.return_distribution(composite_returns)
    save_fig(fig, str(out_dir / "composite_distribution.png"))

    fig = _pp.rolling_sharpe_plot(composite_returns, window=63)
    save_fig(fig, str(out_dir / "composite_rolling_sharpe.png"))

    try:
        fig = _pp.monthly_returns_heatmap(composite_returns)
        save_fig(fig, str(out_dir / "monthly_heatmap.png"))
    except Exception as exc:
        logger.warning("Monthly heatmap failed: %s", exc)

    logger.info("=== Full evaluation complete — results in %s ===", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full strategy evaluation")
    parser.add_argument("--synthetic", action="store_true")
    main(parser.parse_args())
