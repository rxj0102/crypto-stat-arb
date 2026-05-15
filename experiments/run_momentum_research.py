"""
Momentum signal research.

Tests all momentum signals across multiple lookbacks and skip parameters.
Generates signal decay plots and a comprehensive results table.

Usage:
    python experiments/run_momentum_research.py
    python experiments/run_momentum_research.py --synthetic
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
    backtest_signals, load_data, plot_signal_decay, print_summary_table,
    run_bt, save_fig, save_results, signal_decay_ic,
)
from statarb.backtest.execution import ExecutionModel
from statarb.data.features import FeatureEngine
from statarb.signals.activity import ActivityFilter
from statarb.signals.momentum import MomentumSignals
from statarb.utils import ensure_dir, get_logger, load_config

logger = get_logger("run_momentum")


def main(args: argparse.Namespace) -> None:
    cfg = load_config("experiments/config.yaml")
    sig_cfg = cfg.get("signals", {}).get("momentum", {})
    lookbacks = sig_cfg.get("lookbacks", [7, 14, 21, 42, 63, 126])
    skip_days = sig_cfg.get("skip_days", [0, 1, 2, 3])
    periods_per_year = cfg.get("evaluation", {}).get("periods_per_year", 365)
    out_dir = Path(cfg.get("output", {}).get("results_dir", "experiments/results")) / "momentum"
    ensure_dir(out_dir)

    prices, volume, returns = load_data(cfg)
    fe = FeatureEngine()
    vol_ratio = fe.volume_ma_ratio(volume, window=21)
    act = ActivityFilter()
    mom = MomentumSignals()
    em = ExecutionModel(market_order_cost=0.0020)

    # ------------------------------------------------------------------ #
    # 1. Lookback sweep: time-series momentum at each lookback             #
    # ------------------------------------------------------------------ #
    logger.info("=== Lookback sweep ===")
    lookback_signals = {}
    for lb in lookbacks:
        sig = mom.time_series_momentum(returns, lookback=lb)
        sig_ranked = fe.cross_sectional_rank(sig)
        lookback_signals[f"ts_mom_{lb}d"] = sig_ranked

    lookback_results = backtest_signals(lookback_signals, returns, em, periods_per_year)
    save_results(lookback_results, str(out_dir / "lookback_sweep.csv"))
    print_summary_table(lookback_results, "Momentum: Lookback Sweep")

    # ------------------------------------------------------------------ #
    # 2. Skip parameter sweep (using 63d lookback)                         #
    # ------------------------------------------------------------------ #
    logger.info("=== Skip parameter sweep ===")
    skip_signals = {}
    for skip in skip_days:
        sig = fe.cross_sectional_rank(
            mom.time_series_momentum(returns, lookback=63).shift(skip)
        )
        skip_signals[f"ts_mom_63d_skip{skip}"] = sig

    skip_results = backtest_signals(skip_signals, returns, em, periods_per_year)
    save_results(skip_results, str(out_dir / "skip_sweep.csv"))
    print_summary_table(skip_results, "Momentum: Skip Parameter Sweep")

    # ------------------------------------------------------------------ #
    # 3. Signal variants                                                   #
    # ------------------------------------------------------------------ #
    logger.info("=== Signal variants ===")
    variant_signals = {
        "price_mom_12_1": fe.cross_sectional_rank(mom.momentum_12_1(returns)),
        "price_mom_6_1":  fe.cross_sectional_rank(mom.momentum_6_1(returns)),
        "price_mom_3_1":  fe.cross_sectional_rank(mom.momentum_3_1(returns)),
        "price_mom_1w":   fe.cross_sectional_rank(mom.momentum_1w(returns)),
        "ts_mom_63d":     fe.cross_sectional_rank(mom.time_series_momentum(returns, lookback=63)),
        "sharpe_mom_63d": fe.cross_sectional_rank(mom.sharpe_momentum(returns, lookback=63)),
        "ma_cross_20_60": fe.cross_sectional_rank(mom.moving_average_crossover(prices, 20, 60)),
    }

    # Volume-weighted variant
    vol_cond = mom.volume_weighted_momentum(returns, vol_ratio, lookback=63)
    variant_signals["vol_cond_mom"] = fe.cross_sectional_rank(vol_cond)

    # Activity-gated variants
    for base in ["price_mom_6_1", "ts_mom_63d"]:
        raw_sig = {
            "price_mom_6_1": mom.momentum_6_1(returns),
            "ts_mom_63d":    mom.time_series_momentum(returns, lookback=63),
        }[base]
        gated = act.activity_gate(raw_sig, volume, min_ratio=0.5)
        variant_signals[f"{base}_gated"] = fe.cross_sectional_rank(gated)

    variant_results = backtest_signals(variant_signals, returns, em, periods_per_year)
    save_results(variant_results, str(out_dir / "signal_variants.csv"))
    print_summary_table(variant_results, "Momentum: Signal Variants")

    # ------------------------------------------------------------------ #
    # 4. Signal decay plots                                                #
    # ------------------------------------------------------------------ #
    logger.info("=== Signal decay analysis ===")
    decay_signals = {
        "price_mom_6_1":  fe.cross_sectional_rank(mom.momentum_6_1(returns)),
        "ts_mom_63d":     fe.cross_sectional_rank(mom.time_series_momentum(returns, 63)),
        "sharpe_mom_63d": fe.cross_sectional_rank(mom.sharpe_momentum(returns, 63)),
        "vol_weighted_mom": fe.cross_sectional_rank(vol_cond),
    }
    decay_dict = {
        name: signal_decay_ic(sig, returns, max_horizon=20)
        for name, sig in decay_signals.items()
    }
    fig = plot_signal_decay(decay_dict, "Momentum Signal Decay (IC by Forward Horizon)")
    save_fig(fig, str(out_dir / "signal_decay.png"))

    # Also plot lookback comparison decay
    lookback_decay = {}
    for lb in [7, 21, 63, 126]:
        sig = fe.cross_sectional_rank(mom.time_series_momentum(returns, lookback=lb))
        lookback_decay[f"{lb}d"] = signal_decay_ic(sig, returns, max_horizon=20)
    fig2 = plot_signal_decay(lookback_decay, "TS Momentum IC Decay by Lookback")
    save_fig(fig2, str(out_dir / "lookback_decay.png"))

    # ------------------------------------------------------------------ #
    # 5. Cost impact analysis                                              #
    # ------------------------------------------------------------------ #
    logger.info("=== Cost impact analysis ===")
    em_market = ExecutionModel(market_order_cost=0.0020, order_type="market")
    em_limit = ExecutionModel(limit_order_cost=0.0007, order_type="limit")
    em_zero = ExecutionModel(market_order_cost=0.0, limit_order_cost=0.0, order_type="market")

    best_sig_name = variant_results["sharpe"].idxmax()
    best_sig = variant_signals[best_sig_name]

    cost_rows = {}
    for label, em_i in [("zero_cost", em_zero), ("limit", em_limit), ("market", em_market)]:
        net_rets, result = run_bt(best_sig, returns, em_i)
        from statarb.evaluation.metrics import PerformanceMetrics
        _pm = PerformanceMetrics()
        cost_rows[label] = {
            "sharpe":        _pm.sharpe_ratio(net_rets, periods_per_year=periods_per_year),
            "avg_turnover":  float(result["turnover"].mean()),
            "daily_cost_bp": float(result["execution_costs"].mean()) * 10000,
        }

    cost_df = pd.DataFrame(cost_rows).T
    save_results(cost_df, str(out_dir / "cost_impact.csv"))
    print_summary_table(cost_df, f"Cost Impact Analysis ({best_sig_name})")

    # ------------------------------------------------------------------ #
    # 6. Sharpe vs lookback chart                                          #
    # ------------------------------------------------------------------ #
    fig3, ax = plt.subplots(figsize=(9, 4))
    gross_sharpes = lookback_results["gross_sharpe"]
    net_sharpes = lookback_results["sharpe"]
    x = range(len(lookbacks))
    labels = [f"{lb}d" for lb in lookbacks]
    ax.bar([i - 0.2 for i in x], gross_sharpes.values, width=0.35,
           label="Gross Sharpe", alpha=0.7)
    ax.bar([i + 0.2 for i in x], net_sharpes.values, width=0.35,
           label="Net Sharpe", alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Lookback Period")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("TS Momentum Sharpe by Lookback", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    plt.tight_layout()
    save_fig(fig3, str(out_dir / "sharpe_by_lookback.png"))

    logger.info("=== Momentum research complete — results in %s ===", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Momentum signal research")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data generation")
    main(parser.parse_args())
