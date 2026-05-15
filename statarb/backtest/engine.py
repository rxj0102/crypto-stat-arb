"""
Unconstrained cross-sectional backtesting engine.

Runs a full backtest pipeline:
1. Receive signal panel (dates × symbols)
2. Convert to positions via PositionManager
3. Apply forward returns with transaction costs via ExecutionModel
4. Return portfolio return series and diagnostic panels
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from statarb.backtest.execution import ExecutionConfig, ExecutionModel
from statarb.backtest.positions import PositionManager
from statarb.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine."""

    # Rebalancing frequency
    rebalance_freq: Literal["daily", "weekly", "monthly"] = "daily"

    # Execution
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Position sizing
    position_method: Literal[
        "rank", "signal", "top_bottom", "equal_weight", "vol_target"
    ] = "rank"
    max_position: float = 0.10
    top_k: int = 10
    vol_target: float = 0.15
    vol_window: int = 63

    # Return computation
    return_type: Literal["log", "simple"] = "log"

    # Look-ahead guard (periods to shift signal before applying)
    signal_lag: int = 1  # default 1: use today's signal for tomorrow's return


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    portfolio_returns: pd.Series           # daily net portfolio return
    gross_returns: pd.Series               # daily gross portfolio return
    positions: pd.DataFrame                # weight panel (dates × symbols)
    signal: pd.DataFrame                   # signal panel used
    cost_returns: pd.Series                # daily costs as return drag
    turnover: pd.Series                    # daily one-way turnover
    asset_returns: pd.DataFrame            # per-asset net return panel


class BacktestEngine:
    """
    Unconstrained cross-sectional backtesting engine.

    The engine is intentionally simple and transparent:
    - No constraints other than position cap and dollar-neutrality
    - Signals are lagged by ``signal_lag`` periods to prevent look-ahead
    - Costs are applied at each rebalance date

    Usage::

        engine = BacktestEngine(config)
        result = engine.run(
            signal=signal_panel,
            prices=price_panel,
            dollar_volume=volume_panel,
        )

    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.execution = ExecutionModel(self.config.execution)
        self.position_mgr = PositionManager(
            method=self.config.position_method,
            max_position=self.config.max_position,
            top_k=self.config.top_k,
            vol_target=self.config.vol_target,
            vol_window=self.config.vol_window,
        )

    # ------------------------------------------------------------------
    # Main backtest runner
    # ------------------------------------------------------------------

    def run(
        self,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
        dollar_volume: Optional[pd.DataFrame] = None,
        realized_vol: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run the full backtest.

        Args:
            signal: cross-sectional signal panel (dates × symbols).
                    Should already be cross-sectionally ranked/normalised.
            prices: close price panel (dates × symbols)
            dollar_volume: optional ADTV panel for cost modelling
            realized_vol: optional vol panel for slippage

        Returns:
            BacktestResult with portfolio returns, positions, and diagnostics.
        """
        logger.info("Running backtest: %d dates × %d symbols",
                    len(signal), signal.shape[1])

        # Align panels
        signal, prices = self._align(signal, prices)
        if dollar_volume is not None:
            dollar_volume = dollar_volume.reindex_like(signal)
        if realized_vol is not None:
            realized_vol = realized_vol.reindex_like(signal)

        # Compute forward returns (the return realised by holding on date t+1)
        if self.config.return_type == "log":
            fwd_returns = np.log(prices / prices.shift(1)).shift(-1)
        else:
            fwd_returns = prices.pct_change().shift(-1)

        # Apply rebalancing frequency mask
        rebalance_mask = self._rebalance_dates(signal.index)

        # Position sizing: only update on rebalance dates, hold otherwise
        positions = self._compute_positions(signal, fwd_returns, rebalance_mask)

        # Transaction costs
        trades = positions.diff().fillna(positions)
        trades[~rebalance_mask] = 0.0  # only trade on rebalance dates
        costs = self.execution.compute_costs(trades, dollar_volume, realized_vol)

        # Per-asset gross returns
        lagged_pos = positions.shift(self.config.signal_lag).fillna(0)
        asset_gross = fwd_returns * lagged_pos
        asset_net = asset_gross - costs

        # Portfolio aggregation
        gross_ret = asset_gross.sum(axis=1)
        cost_drag = costs.sum(axis=1)
        net_ret = asset_net.sum(axis=1)

        turnover = self.position_mgr.turnover(positions)

        logger.info(
            "Backtest complete. Ann. Sharpe (gross): %.2f | Ann. Sharpe (net): %.2f",
            self._quick_sharpe(gross_ret),
            self._quick_sharpe(net_ret),
        )

        return BacktestResult(
            portfolio_returns=net_ret,
            gross_returns=gross_ret,
            positions=positions,
            signal=signal,
            cost_returns=cost_drag,
            turnover=turnover,
            asset_returns=asset_net,
        )

    # ------------------------------------------------------------------
    # Multi-signal runner
    # ------------------------------------------------------------------

    def run_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        dollar_volume: Optional[pd.DataFrame] = None,
        realized_vol: Optional[pd.DataFrame] = None,
    ) -> Dict[str, BacktestResult]:
        """
        Run the same backtest for multiple signals independently.

        Useful for signal evaluation before combination.

        Args:
            signals: dict of signal_name → signal panel

        Returns:
            dict of signal_name → BacktestResult
        """
        results = {}
        for name, sig in signals.items():
            logger.info("Backtesting signal: %s", name)
            results[name] = self.run(sig, prices, dollar_volume, realized_vol)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_positions(
        self,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
        rebalance_mask: pd.Series,
    ) -> pd.DataFrame:
        """Compute positions, only updating on rebalance dates."""
        all_positions = self.position_mgr.compute_positions(signal, returns)

        # On non-rebalance dates, forward-fill previous positions
        positions = all_positions.copy()
        positions[~rebalance_mask] = np.nan
        positions = positions.ffill()
        return positions.fillna(0)

    def _rebalance_dates(self, index: pd.DatetimeIndex) -> pd.Series:
        """Boolean mask for rebalance dates based on frequency."""
        freq = self.config.rebalance_freq
        if freq == "daily":
            return pd.Series(True, index=index)
        elif freq == "weekly":
            return pd.Series(index.dayofweek == 0, index=index)  # Monday
        elif freq == "monthly":
            # First trading day of each month
            return pd.Series(
                (index.month != pd.DatetimeIndex(index).shift(-1, freq="D").month),
                index=index
            )
        else:
            raise ValueError(f"Unknown rebalance_freq: {freq}")

    @staticmethod
    def _align(a: pd.DataFrame, b: pd.DataFrame):
        """Align two DataFrames to common index and columns."""
        common_idx = a.index.intersection(b.index)
        common_cols = a.columns.intersection(b.columns)
        return a.loc[common_idx, common_cols], b.loc[common_idx, common_cols]

    @staticmethod
    def _quick_sharpe(returns: pd.Series, periods: int = 252) -> float:
        """Annualised Sharpe ratio assuming 0% risk-free rate."""
        clean = returns.dropna()
        if len(clean) < 10 or clean.std() == 0:
            return 0.0
        return float(clean.mean() / clean.std() * np.sqrt(periods))
