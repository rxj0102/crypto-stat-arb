"""
Pairs and correlation-based signal research.

Identifies cointegrated pairs and evaluates relative-value signals.

Usage:
    python experiments/run_pairs_research.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from statarb.backtest.engine import BacktestConfig, BacktestEngine
from statarb.data.features import FeatureEngine
from statarb.data.storage import DataStore
from statarb.data.universe import TradableUniverse
from statarb.evaluation.reporting import PerformanceReporter
from statarb.signals.pairs import PairsSignals
from statarb.signals.themes import CRYPTO_THEMES
from statarb.utils import get_logger, load_config

logger = get_logger("run_pairs")


def main() -> None:
    cfg = load_config("experiments/config.yaml")
    data_cfg = cfg.get("data", {})
    exchange = data_cfg.get("exchange", "binance")

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

    pairs_cfg = cfg.get("signals", {}).get("pairs", {})
    ps = PairsSignals()

    # Scan for cointegrated pairs
    logger.info("Scanning for cointegrated pairs (Engle-Granger test)...")
    coint_pairs = ps.find_cointegrated_pairs(
        prices,
        lookback=pairs_cfg.get("cointegration_lookback", 252),
        pvalue_threshold=0.05,
    )
    logger.info("Top cointegrated pairs:")
    for a, b, pval in coint_pairs[:10]:
        logger.info("  %s / %s  (p=%.4f)", a, b, pval)

    signals = {
        "relative_value": ps.relative_value_signal(
            filtered_returns,
            correlation_window=63,
            return_window=21,
            min_correlation=0.5,
        ),
        "cluster_momentum_L1": ps.cluster_momentum_signal(
            filtered_returns, CRYPTO_THEMES, lookback=21
        ),
    }

    ranked_signals = {name: fe.cross_sectional_rank(sig) for name, sig in signals.items()}

    bt_config = BacktestConfig(**{k: v for k, v in cfg.get("backtest", {}).items()
                                  if k in BacktestConfig.__dataclass_fields__})
    engine = BacktestEngine(bt_config)
    results = engine.run_signals(ranked_signals, prices, volume)

    strategy_returns = {name: res.portfolio_returns for name, res in results.items()}
    reporter = PerformanceReporter(strategy_returns)
    reporter.print_report(title="Pairs / Correlation Signal Evaluation")


if __name__ == "__main__":
    main()
