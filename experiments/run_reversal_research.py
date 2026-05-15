"""
Reversal signal exploration and backtest.

Evaluates short-term and medium-term reversal signals.

Usage:
    python experiments/run_reversal_research.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from statarb.backtest.engine import BacktestConfig, BacktestEngine
from statarb.data.features import FeatureEngine
from statarb.data.storage import DataStore
from statarb.data.universe import TradableUniverse
from statarb.evaluation.reporting import PerformanceReporter
from statarb.signals.reversal import ReversalSignals
from statarb.utils import get_logger, load_config

logger = get_logger("run_reversal")


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
    vol_ratio = fe.volume_ma_ratio(volume, window=21)
    realized_vol = fe.realized_volatility(returns, window=21)

    rev = ReversalSignals()
    signals = {
        "reversal_1d": rev.short_term_reversal(filtered_returns, lookback=1),
        "reversal_5d": rev.weekly_reversal(filtered_returns),
        "reversal_10d": rev.biweekly_reversal(filtered_returns),
        "vol_adj_reversal_5d": rev.vol_adjusted_reversal(filtered_returns),
        "bollinger_reversal_20d": rev.bollinger_reversal(prices, window=20),
        "ma_distance_reversal_20d": rev.ma_distance_reversal(prices, window=20),
        "high_vol_reversal_1d": rev.high_volume_reversal(filtered_returns, vol_ratio),
        "large_move_reversal": rev.large_move_reversal(filtered_returns),
    }

    ranked_signals = {name: fe.cross_sectional_rank(sig) for name, sig in signals.items()}

    bt_config = BacktestConfig(**{k: v for k, v in cfg.get("backtest", {}).items()
                                  if k in BacktestConfig.__dataclass_fields__})
    engine = BacktestEngine(bt_config)
    results = engine.run_signals(ranked_signals, prices, volume, realized_vol)

    strategy_returns = {name: res.portfolio_returns for name, res in results.items()}
    reporter = PerformanceReporter(strategy_returns)
    reporter.print_report(title="Reversal Signal Evaluation")


if __name__ == "__main__":
    main()
