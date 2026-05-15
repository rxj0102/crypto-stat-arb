"""
Full performance evaluation of the combined strategy.

Runs the complete evaluation pipeline: metrics, alpha/beta,
risk decomposition, drawdown analysis, and report generation.

Usage:
    python experiments/run_full_evaluation.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from statarb.backtest.engine import BacktestConfig, BacktestEngine
from statarb.backtest.weighting import StrategyWeighter
from statarb.data.features import FeatureEngine
from statarb.data.storage import DataStore
from statarb.data.universe import TradableUniverse
from statarb.evaluation.metrics import PerformanceMetrics
from statarb.evaluation.plots import Plotter
from statarb.evaluation.reporting import PerformanceReporter
from statarb.evaluation.risk import RiskAnalytics
from statarb.signals.activity import ActivityFilter
from statarb.signals.momentum import MomentumSignals
from statarb.signals.reversal import ReversalSignals
from statarb.utils import ensure_dir, get_logger, load_config

logger = get_logger("run_full_eval")


def main() -> None:
    cfg = load_config("experiments/config.yaml")
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    exchange = data_cfg.get("exchange", "binance")

    output_dir = ensure_dir("output")
    plots_dir = ensure_dir(output_dir / "plots")
    reports_dir = ensure_dir(output_dir / "reports")

    store = DataStore(base_dir=data_cfg.get("base_dir", "data"))
    symbols = store.list_symbols(exchange, "1d")
    if not symbols:
        logger.error("No data found. Run run_data_fetch.py first.")
        return

    prices = store.build_panel(symbols, exchange, "1d", field="close")
    volume = store.build_volume_panel(symbols, exchange, "1d")
    fe = FeatureEngine()
    returns = fe.log_returns(prices)

    uni = TradableUniverse(**cfg.get("universe", {}))
    filtered_returns = uni.apply_filters(returns, volume)
    vol_ratio = fe.volume_ma_ratio(volume, window=21)

    # Build signals
    mom = MomentumSignals()
    rev = ReversalSignals()
    act = ActivityFilter()

    signals = {
        "momentum_6_1": fe.cross_sectional_rank(
            act.activity_gate(mom.momentum_6_1(filtered_returns), volume)
        ),
        "momentum_3_1": fe.cross_sectional_rank(
            act.activity_gate(mom.momentum_3_1(filtered_returns), volume)
        ),
        "reversal_5d": fe.cross_sectional_rank(
            act.activity_gate(rev.weekly_reversal(filtered_returns), volume)
        ),
    }

    weighter = StrategyWeighter(method="equal")
    composite = weighter.combine_signals(signals)
    signals["composite"] = composite

    bt_cfg = BacktestConfig(**{k: v for k, v in cfg.get("backtest", {}).items()
                                if k in BacktestConfig.__dataclass_fields__})
    engine = BacktestEngine(bt_cfg)
    results = engine.run_signals(signals, prices, volume)
    strategy_returns = {name: res.portfolio_returns for name, res in results.items()}

    # BTC benchmark
    benchmark_sym = eval_cfg.get("benchmark", "BTC/USDT")
    benchmark_returns = None
    if benchmark_sym in prices.columns:
        benchmark_returns = fe.log_returns(prices[[benchmark_sym]])[benchmark_sym]

    # Full report
    reporter = PerformanceReporter(
        strategy_returns,
        benchmark_returns=benchmark_returns,
        periods_per_year=eval_cfg.get("periods_per_year", 252),
    )
    reporter.print_report(title="Full Strategy Evaluation")
    reporter.to_csv(str(reports_dir / "performance_metrics.csv"))

    # Risk decomposition
    if benchmark_returns is not None:
        composite_rets = strategy_returns.get("composite", list(strategy_returns.values())[0])
        ra = RiskAnalytics(composite_rets)
        alpha, beta, r2, pval = ra.alpha_beta(benchmark_returns)
        logger.info("Composite Strategy Alpha: %.2f%% | Beta: %.3f | R²: %.3f",
                    alpha * 100, beta, r2)

    # Plots
    plotter = Plotter()

    fig = plotter.plot_cumulative_returns(strategy_returns, "Cumulative Returns")
    plotter.save(fig, str(plots_dir / "cumulative_returns.png"))

    fig = plotter.plot_drawdowns(strategy_returns, "Drawdowns")
    plotter.save(fig, str(plots_dir / "drawdowns.png"))

    fig = plotter.plot_rolling_sharpe(strategy_returns)
    plotter.save(fig, str(plots_dir / "rolling_sharpe.png"))

    fig = plotter.plot_return_distribution(strategy_returns)
    plotter.save(fig, str(plots_dir / "return_distribution.png"))

    fig = plotter.plot_correlation_heatmap(strategy_returns)
    plotter.save(fig, str(plots_dir / "strategy_correlations.png"))

    logger.info("Evaluation complete. Outputs in: %s", output_dir)


if __name__ == "__main__":
    main()
