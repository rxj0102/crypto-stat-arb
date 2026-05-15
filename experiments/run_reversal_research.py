"""
Reversal signal research.

Tests all reversal signals across lookbacks [1, 2, 3, 5, 7 days].
Includes volume-filtered, beta-neutral, and activity-gated variants.

Usage:
    python experiments/run_reversal_research.py
    python experiments/run_reversal_research.py --synthetic
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
    save_fig, save_results, signal_decay_ic,
)
from statarb.backtest.execution import ExecutionModel
from statarb.data.features import FeatureEngine
from statarb.signals.activity import ActivityFilter
from statarb.signals.reversal import ReversalSignals
from statarb.utils import ensure_dir, get_logger, load_config

logger = get_logger("run_reversal")


def main(args: argparse.Namespace) -> None:
    cfg = load_config("experiments/config.yaml")
    sig_cfg = cfg.get("signals", {}).get("reversal", {})
    lookbacks = sig_cfg.get("lookbacks", [1, 2, 3, 5, 7])
    periods_per_year = cfg.get("evaluation", {}).get("periods_per_year", 365)
    out_dir = Path(cfg.get("output", {}).get("results_dir", "experiments/results")) / "reversal"
    ensure_dir(out_dir)

    prices, volume, returns = load_data(cfg)
    fe = FeatureEngine()
    vol_ratio = fe.volume_ma_ratio(volume, window=21)
    realized_vol = fe.realized_volatility(returns, window=21)
    act = ActivityFilter()
    rev = ReversalSignals()

    em_limit = ExecutionModel(limit_order_cost=0.0007, order_type="limit")
    em_market = ExecutionModel(market_order_cost=0.0020, order_type="market")

    # ------------------------------------------------------------------ #
    # 1. Lookback sweep: short-term reversal                               #
    # ------------------------------------------------------------------ #
    logger.info("=== Lookback sweep ===")
    lookback_signals = {}
    for lb in lookbacks:
        sig = fe.cross_sectional_rank(rev.short_term_reversal(returns, lookback=lb))
        lookback_signals[f"reversal_{lb}d"] = sig

    # Reversal benefits from limit orders (captures the bid-ask bounce)
    lb_limit = backtest_signals(lookback_signals, returns, em_limit, periods_per_year)
    lb_market = backtest_signals(lookback_signals, returns, em_market, periods_per_year)
    save_results(lb_limit, str(out_dir / "lookback_limit.csv"))
    save_results(lb_market, str(out_dir / "lookback_market.csv"))
    print_summary_table(lb_limit, "Reversal: Lookback Sweep (Limit Orders)")
    print_summary_table(lb_market, "Reversal: Lookback Sweep (Market Orders)")

    # ------------------------------------------------------------------ #
    # 2. Signal variants                                                   #
    # ------------------------------------------------------------------ #
    logger.info("=== Signal variants ===")
    variant_signals = {
        "reversal_1d":         fe.cross_sectional_rank(rev.short_term_reversal(returns, 1)),
        "reversal_5d":         fe.cross_sectional_rank(rev.weekly_reversal(returns)),
        "reversal_10d":        fe.cross_sectional_rank(rev.biweekly_reversal(returns)),
        "vol_adj_reversal":    fe.cross_sectional_rank(rev.vol_adjusted_reversal(returns)),
        "bollinger_20d":       fe.cross_sectional_rank(rev.bollinger_reversal(prices, 20)),
        "large_move_reversal": fe.cross_sectional_rank(rev.large_move_reversal(returns)),
    }

    # Volume-filtered reversal (low volume = uninformed = more reversal)
    vol_filtered = rev.volume_filtered_reversal(returns, volume, lookback=3)
    variant_signals["vol_filtered_reversal"] = fe.cross_sectional_rank(vol_filtered)

    # Activity-gated: suppress on low-volume days
    for base_name in ["reversal_5d", "vol_adj_reversal"]:
        raw = {
            "reversal_5d":      rev.weekly_reversal(returns),
            "vol_adj_reversal": rev.vol_adjusted_reversal(returns),
        }[base_name]
        gated = act.activity_gate(raw, volume, min_ratio=0.5)
        variant_signals[f"{base_name}_gated"] = fe.cross_sectional_rank(gated)

    variant_results_limit = backtest_signals(variant_signals, returns, em_limit, periods_per_year)
    variant_results_market = backtest_signals(variant_signals, returns, em_market, periods_per_year)
    save_results(variant_results_limit, str(out_dir / "variants_limit.csv"))
    save_results(variant_results_market, str(out_dir / "variants_market.csv"))
    print_summary_table(variant_results_limit, "Reversal: Signal Variants (Limit Orders)")

    # ------------------------------------------------------------------ #
    # 3. Beta-neutral reversal (residual after removing market beta)       #
    # ------------------------------------------------------------------ #
    logger.info("=== Beta-neutral reversal ===")
    # Compute market (equal-weighted) returns
    mkt_ret = returns.mean(axis=1)
    # Compute rolling beta for each asset vs market
    beta_panel = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    roll_window = 63
    for col in returns.columns:
        y = returns[col].dropna()
        x = mkt_ret.reindex(y.index)
        cov = y.rolling(roll_window, min_periods=21).cov(x)
        var = x.rolling(roll_window, min_periods=21).var()
        beta_panel[col] = (cov / var.replace(0, np.nan)).reindex(returns.index)

    # Residual returns = actual - beta * market
    mkt_broadcast = pd.DataFrame(
        np.outer(mkt_ret.values, np.ones(len(returns.columns))),
        index=returns.index,
        columns=returns.columns,
    )
    residual_returns = returns - beta_panel * mkt_broadcast

    beta_neutral_5d = fe.cross_sectional_rank(rev.short_term_reversal(residual_returns, 5))
    beta_neutral_sigs = {
        "raw_reversal_5d":       fe.cross_sectional_rank(rev.weekly_reversal(returns)),
        "beta_neutral_5d":       beta_neutral_5d,
        "vol_adj_beta_neutral":  fe.cross_sectional_rank(
            rev.vol_adjusted_reversal(residual_returns)
        ),
    }

    bn_results = backtest_signals(beta_neutral_sigs, returns, em_limit, periods_per_year)
    save_results(bn_results, str(out_dir / "beta_neutral.csv"))
    print_summary_table(bn_results, "Reversal: Beta-Neutral vs Raw")

    # ------------------------------------------------------------------ #
    # 4. Signal decay plots                                                #
    # ------------------------------------------------------------------ #
    logger.info("=== Signal decay analysis ===")
    decay_signals = {
        "reversal_1d":          fe.cross_sectional_rank(rev.short_term_reversal(returns, 1)),
        "reversal_5d":          fe.cross_sectional_rank(rev.weekly_reversal(returns)),
        "vol_adj_reversal":     fe.cross_sectional_rank(rev.vol_adjusted_reversal(returns)),
        "vol_filtered_reversal": fe.cross_sectional_rank(vol_filtered),
    }
    decay_dict = {
        name: signal_decay_ic(sig, returns, max_horizon=15)
        for name, sig in decay_signals.items()
    }
    fig = plot_signal_decay(decay_dict, "Reversal Signal Decay (IC by Forward Horizon)")
    save_fig(fig, str(out_dir / "signal_decay.png"))

    # ------------------------------------------------------------------ #
    # 5. Limit vs market cost comparison chart                             #
    # ------------------------------------------------------------------ #
    fig2, ax = plt.subplots(figsize=(10, 4))
    names = list(variant_signals.keys())
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], variant_results_market["sharpe"].values,
           width=0.35, label="Market orders (20bps)", alpha=0.7, color="tab:red")
    ax.bar([i + 0.2 for i in x], variant_results_limit["sharpe"].values,
           width=0.35, label="Limit orders (7bps)", alpha=0.7, color="tab:blue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Reversal: Limit vs Market Order Costs", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    plt.tight_layout()
    save_fig(fig2, str(out_dir / "limit_vs_market.png"))

    logger.info("=== Reversal research complete — results in %s ===", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reversal signal research")
    parser.add_argument("--synthetic", action="store_true")
    main(parser.parse_args())
