"""
Combine the best signals into a single composite strategy.

Uses the StrategyWeighter to blend multiple signal panels.

Usage:
    python experiments/run_combined_strategy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from statarb.backtest.engine import BacktestConfig, BacktestEngine
from statarb.backtest.weighting import StrategyWeighter
from statarb.data.features import FeatureEngine
from statarb.data.storage import DataStore
from statarb.data.universe import TradableUniverse
from statarb.evaluation.reporting import PerformanceReporter
from statarb.signals.activity import ActivityFilter
from statarb.signals.momentum import MomentumSignals
from statarb.signals.reversal import ReversalSignals
from statarb.utils import get_logger, load_config

logger = get_logger("run_combined")


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

    # Signal components
    mom = MomentumSignals()
    rev = ReversalSignals()
    act = ActivityFilter()

    raw_signals = {
        "momentum_6_1": mom.momentum_6_1(filtered_returns),
        "momentum_3_1": mom.momentum_3_1(filtered_returns),
        "sharpe_momentum": mom.sharpe_momentum(filtered_returns, lookback=63),
        "reversal_5d": rev.weekly_reversal(filtered_returns),
        "vol_adj_reversal": rev.vol_adjusted_reversal(filtered_returns),
    }

    # Activity-gate: suppress signals on low-volume days
    gated_signals = {
        name: act.activity_gate(sig, volume, min_ratio=0.5)
        for name, sig in raw_signals.items()
    }

    # Rank cross-sectionally
    ranked = {name: fe.cross_sectional_rank(sig) for name, sig in gated_signals.items()}

    # Combine using configured method
    combo_cfg = cfg.get("combination", {})
    weighter = StrategyWeighter(
        method=combo_cfg.get("method", "equal"),
        estimation_window=combo_cfg.get("estimation_window", 63),
    )

    # Use forward returns for IC-weighted combination
    fwd_returns = returns.shift(-1)
    composite_signal = weighter.combine_signals(ranked, forward_returns=fwd_returns
                                                if combo_cfg.get("method") == "ic_weighted"
                                                else None)

    # Backtest individual + composite
    bt_cfg = BacktestConfig(**{k: v for k, v in cfg.get("backtest", {}).items()
                                if k in BacktestConfig.__dataclass_fields__})
    engine = BacktestEngine(bt_cfg)

    all_signals = dict(ranked)
    all_signals["composite"] = composite_signal

    results = engine.run_signals(all_signals, prices, volume)
    strategy_returns = {name: res.portfolio_returns for name, res in results.items()}

    reporter = PerformanceReporter(strategy_returns)
    reporter.print_report(title="Combined Strategy Evaluation")

    for name, rets in strategy_returns.items():
        print(PerformanceReporter.one_liner(rets, name))


if __name__ == "__main__":
    main()
