"""
Momentum signal exploration and backtest.

Evaluates multiple momentum specifications (12-1, 6-1, 3-1, MA crossover,
time-series momentum, Sharpe-weighted momentum) across the crypto universe.

Usage:
    python experiments/run_momentum_research.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from statarb.backtest.engine import BacktestConfig, BacktestEngine
from statarb.data.features import FeatureEngine
from statarb.data.storage import DataStore
from statarb.data.universe import TradableUniverse
from statarb.evaluation.metrics import PerformanceMetrics
from statarb.evaluation.reporting import PerformanceReporter
from statarb.signals.momentum import MomentumSignals
from statarb.utils import get_logger, load_config

logger = get_logger("run_momentum")


def main() -> None:
    cfg = load_config("experiments/config.yaml")
    data_cfg = cfg.get("data", {})
    exchange = data_cfg.get("exchange", "binance")
    timeframe = "1d"

    store = DataStore(base_dir=data_cfg.get("base_dir", "data"))
    symbols = store.list_symbols(exchange, timeframe)
    if not symbols:
        logger.error("No data found. Run run_data_fetch.py first.")
        return

    logger.info("Loading data for %d symbols", len(symbols))
    prices = store.build_panel(symbols, exchange, timeframe, field="close")
    volume = store.build_volume_panel(symbols, exchange, timeframe)

    # Universe filtering
    fe = FeatureEngine()
    returns = fe.log_returns(prices)
    uni = TradableUniverse(**cfg.get("universe", {}))
    filtered_returns = uni.apply_filters(returns, volume)

    # Signal generation
    mom = MomentumSignals()
    vol_ratio = fe.volume_ma_ratio(volume, window=21)

    signals = {
        "momentum_12_1": mom.momentum_12_1(filtered_returns),
        "momentum_6_1": mom.momentum_6_1(filtered_returns),
        "momentum_3_1": mom.momentum_3_1(filtered_returns),
        "momentum_1w": mom.momentum_1w(filtered_returns),
        "ts_momentum_63d": mom.time_series_momentum(filtered_returns, lookback=63),
        "sharpe_momentum_63d": mom.sharpe_momentum(filtered_returns, lookback=63),
        "ma_crossover_20_60": mom.moving_average_crossover(prices, fast=20, slow=60),
        "vol_conditioned_mom": mom.volume_conditioned_momentum(
            filtered_returns, vol_ratio, lookback=63
        ),
    }

    # Rank signals cross-sectionally
    ranked_signals = {name: fe.cross_sectional_rank(sig) for name, sig in signals.items()}

    # Run backtest for each signal
    bt_config = BacktestConfig(**{k: v for k, v in cfg.get("backtest", {}).items()
                                  if k in BacktestConfig.__dataclass_fields__})
    engine = BacktestEngine(bt_config)
    results = engine.run_signals(ranked_signals, prices, volume)

    strategy_returns = {name: res.portfolio_returns for name, res in results.items()}

    # Reporting
    reporter = PerformanceReporter(strategy_returns, periods_per_year=252)
    reporter.print_report(title="Momentum Signal Evaluation")

    for name, rets in strategy_returns.items():
        print(PerformanceReporter.one_liner(rets, name))


if __name__ == "__main__":
    main()
