"""
Pairs and correlation-based signal research.

Finds cointegrated pairs, backtests spread-reversion strategies,
and tests sector-neutral reversal within thematic clusters.

Usage:
    python experiments/run_pairs_research.py
    python experiments/run_pairs_research.py --synthetic
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
    backtest_signals, load_data, print_summary_table,
    run_bt, save_fig, save_results,
)
from statarb.backtest.execution import ExecutionModel
from statarb.data.features import FeatureEngine
from statarb.signals.pairs import PairsSignals
from statarb.signals.reversal import ReversalSignals
from statarb.signals.themes import CRYPTO_THEMES
from statarb.utils import ensure_dir, get_logger, load_config

logger = get_logger("run_pairs")


def main(args: argparse.Namespace) -> None:
    cfg = load_config("experiments/config.yaml")
    pairs_cfg = cfg.get("signals", {}).get("pairs", {})
    periods_per_year = cfg.get("evaluation", {}).get("periods_per_year", 365)
    out_dir = Path(cfg.get("output", {}).get("results_dir", "experiments/results")) / "pairs"
    ensure_dir(out_dir)

    prices, volume, returns = load_data(cfg)
    fe = FeatureEngine()
    ps = PairsSignals()
    rev = ReversalSignals()
    em = ExecutionModel(limit_order_cost=0.0007, order_type="limit")

    # ------------------------------------------------------------------ #
    # 1. Find cointegrated pairs                                           #
    # ------------------------------------------------------------------ #
    logger.info("=== Scanning for cointegrated pairs ===")
    try:
        coint_pairs = ps.find_cointegrated_pairs(
            prices,
            lookback=min(pairs_cfg.get("cointegration_lookback", 252), len(prices) - 10),
            pvalue_threshold=pairs_cfg.get("min_pvalue", 0.05),
        )
        if coint_pairs:
            logger.info("Top cointegrated pairs:")
            pair_rows = []
            for a, b, pval in coint_pairs[:15]:
                logger.info("  %-12s / %-12s  p=%.4f", a, b, pval)
                pair_rows.append({"asset_a": a, "asset_b": b, "pvalue": pval})
            pairs_df = pd.DataFrame(pair_rows)
            save_results(pairs_df, str(out_dir / "cointegrated_pairs.csv"))
        else:
            logger.warning("No cointegrated pairs found (may need more data)")
            coint_pairs = []
    except Exception as exc:
        logger.warning("Cointegration scan failed (%s) — skipping", exc)
        coint_pairs = []

    # ------------------------------------------------------------------ #
    # 2. Backtest spread-reversion for top pairs                           #
    # ------------------------------------------------------------------ #
    if coint_pairs:
        logger.info("=== Backtesting top 10 pairs ===")
        top_pairs = coint_pairs[:10]
        pair_results = {}
        for a, b, pval in top_pairs:
            if a not in prices.columns or b not in prices.columns:
                continue
            pair_name = f"{a.split('/')[0]}_{b.split('/')[0]}"
            try:
                spread = ps.spread_zscore(prices[a], prices[b],
                                          window=pairs_cfg.get("zscore_window", 21))
                # Spread mean-reversion signal: negative z-score → buy A, sell B
                spread_signal = pd.DataFrame(
                    {"A": -spread, "B": spread},
                    index=spread.index,
                )
                # Build a 2-asset signal panel on returns
                pair_rets = returns[[a, b]].rename(columns={a: "A", b: "B"})
                net_rets, result = run_bt(spread_signal, pair_rets, em)
                pair_results[pair_name] = {
                    "pvalue": pval,
                    "sharpe": float(net_rets.mean() / net_rets.std() * np.sqrt(periods_per_year))
                    if net_rets.std() > 0 else 0.0,
                    "avg_turnover": float(result["turnover"].mean()),
                }
                logger.info("  %s: Sharpe %.2f", pair_name,
                            pair_results[pair_name]["sharpe"])
            except Exception as exc:
                logger.warning("  %s failed: %s", pair_name, exc)

        if pair_results:
            pair_results_df = pd.DataFrame(pair_results).T
            save_results(pair_results_df, str(out_dir / "top_pairs_backtest.csv"))
            print_summary_table(pair_results_df, "Top 10 Pairs Backtest Results")

    # ------------------------------------------------------------------ #
    # 3. Cross-sectional relative value                                    #
    # ------------------------------------------------------------------ #
    logger.info("=== Relative value signals ===")
    signals = {}

    try:
        rv_signal = ps.relative_value_signal(
            returns,
            correlation_window=63,
            return_window=21,
            min_correlation=0.5,
        )
        signals["relative_value"] = fe.cross_sectional_rank(rv_signal)
    except Exception as exc:
        logger.warning("Relative value signal failed: %s", exc)

    # ------------------------------------------------------------------ #
    # 4. Cluster / sector-neutral reversal                                 #
    # ------------------------------------------------------------------ #
    logger.info("=== Sector-neutral reversal ===")
    # Within each theme, subtract theme-level mean return (sector-neutral)
    theme_col_map = {}
    for theme, members in CRYPTO_THEMES.items():
        present = [m for m in members if m in returns.columns]
        if len(present) >= 2:
            theme_col_map[theme] = present

    if theme_col_map:
        # Sector-neutral: subtract cross-theme mean
        sector_neutral_ret = returns.copy()
        for theme, cols in theme_col_map.items():
            theme_mean = returns[cols].mean(axis=1)
            sector_neutral_ret[cols] = returns[cols].sub(theme_mean, axis=0)

        signals["sector_neutral_reversal"] = fe.cross_sectional_rank(
            rev.short_term_reversal(sector_neutral_ret, lookback=5)
        )
        signals["sector_neutral_raw_reversal"] = fe.cross_sectional_rank(
            rev.short_term_reversal(returns, lookback=5)
        )

    # Cluster momentum
    try:
        cluster_mom = ps.cluster_momentum_signal(returns, CRYPTO_THEMES, lookback=21)
        signals["cluster_momentum"] = fe.cross_sectional_rank(cluster_mom)
    except Exception as exc:
        logger.warning("Cluster momentum signal failed: %s", exc)

    if signals:
        results = backtest_signals(signals, returns, em, periods_per_year)
        save_results(results, str(out_dir / "signals_backtest.csv"))
        print_summary_table(results, "Pairs / Correlation Signals")

    # ------------------------------------------------------------------ #
    # 5. Spread visualization for best pairs                               #
    # ------------------------------------------------------------------ #
    if coint_pairs and len(coint_pairs) >= 2:
        fig, axes = plt.subplots(min(3, len(coint_pairs)), 1,
                                 figsize=(12, 3 * min(3, len(coint_pairs))))
        if not hasattr(axes, "__iter__"):
            axes = [axes]
        for ax, (a, b, pval) in zip(axes, coint_pairs[:3]):
            if a not in prices.columns or b not in prices.columns:
                continue
            try:
                spread = ps.spread_zscore(prices[a], prices[b], window=21)
                ax.plot(spread.index, spread.values, linewidth=0.8)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.axhline(1.5, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.axhline(-1.5, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.set_title(f"{a.split('/')[0]} / {b.split('/')[0]}  (p={pval:.4f})",
                             fontsize=9)
                ax.set_ylabel("Spread Z-Score")
            except Exception:
                pass
        plt.suptitle("Pair Spread Z-Scores (top cointegrated pairs)", fontweight="bold")
        plt.tight_layout()
        save_fig(fig, str(out_dir / "pair_spreads.png"))

    logger.info("=== Pairs research complete — results in %s ===", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairs and correlation signal research")
    parser.add_argument("--synthetic", action="store_true")
    main(parser.parse_args())
