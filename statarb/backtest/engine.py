"""
Backtesting engine for statistical arbitrage.

Two engines are provided:

UnconstrainedBacktest — the primary research engine described in Prompt 3.
    Simple, transparent, and dollar-neutral by construction.

BacktestEngine — the legacy engine from Prompt 1.  Kept for compatibility
    but not the focus of Prompt 3 tests.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from statarb.backtest.execution import ExecutionConfig, ExecutionModel
from statarb.backtest.positions import PositionManager
from statarb.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# UnconstrainedBacktest
# ---------------------------------------------------------------------------

class UnconstrainedBacktest:
    """
    Unconstrained backtest engine for statistical arbitrage.

    "Unconstrained" means:
    - Long AND short positions allowed
    - No position limits (except optional max weight per asset)
    - Signal directly determines position size (no optimization)

    Standard stat arb backtest:
    1. At each rebalancing date t:
       a. Compute signal s_{i,t} for each asset i
       b. Construct positions: w_{i,t} = s_{i,t} / Σ|s_{i,t}|
          (dollar-neutral: Σ w_long = 1, Σ w_short = −1, net = 0)
       c. Apply execution costs to the TURNOVER (|w_new − w_old|)
    2. P&L_t = Σ w_{i,t} × r_{i,t} − execution_costs_t

    Dollar-neutral construction ensures no net market exposure.
    The total gross exposure is $2 ($1 long + $1 short).

    Rebalancing frequencies to test: daily, every 2 days, weekly.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        execution_model: Optional[ExecutionModel] = None,
        rebal_frequency: int = 1,
        max_weight: float = 0.10,
        dollar_neutral: bool = True,
    ):
        """
        Args:
            returns:         forward return panel (what you earn AFTER taking the position)
            signals:         signal panel (determines the position)
            execution_model: cost model (default: 20 bps market orders)
            rebal_frequency: rebalance every N periods (1=daily)
            max_weight:      max absolute weight per asset before normalization
            dollar_neutral:  if True, construct dollar-neutral portfolio
        """
        self.returns = returns
        self.signals = signals
        self.execution_model = execution_model or ExecutionModel()
        self.rebal_frequency = rebal_frequency
        self.max_weight = max_weight
        self.dollar_neutral = dollar_neutral

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run the backtest.

        Returns:
            dict with:
            - 'returns':          pd.Series of strategy returns (after costs)
            - 'gross_returns':    pd.Series of returns before costs
            - 'positions':        pd.DataFrame of positions at each date
            - 'turnover':         pd.Series of daily one-way turnover
            - 'execution_costs':  pd.Series of daily costs
            - 'n_longs':          number of long positions per date
            - 'n_shorts':         number of short positions per date
        """
        common_idx = self.returns.index.intersection(self.signals.index)
        common_cols = self.returns.columns.intersection(self.signals.columns)
        rets = self.returns.loc[common_idx, common_cols]
        sigs = self.signals.loc[common_idx, common_cols]

        n = len(common_idx)
        all_positions: list = []
        turnover_vals: list = []
        prev_pos = pd.Series(0.0, index=common_cols)

        for i in range(n):
            if i % self.rebal_frequency == 0:
                new_pos = self._construct_positions(sigs.iloc[i], prev_pos)
            else:
                new_pos = prev_pos.copy()

            turnover_vals.append(self._compute_turnover(new_pos, prev_pos))
            all_positions.append(new_pos)
            prev_pos = new_pos

        positions = pd.DataFrame(all_positions, index=common_idx)
        turnover = pd.Series(turnover_vals, index=common_idx)
        execution_costs = self.execution_model.compute_costs(turnover)
        gross_returns = (positions * rets).sum(axis=1)
        net_returns = gross_returns - execution_costs

        return {
            "returns": net_returns,
            "gross_returns": gross_returns,
            "positions": positions,
            "turnover": turnover,
            "execution_costs": execution_costs,
            "n_longs": (positions > 0).sum(axis=1),
            "n_shorts": (positions < 0).sum(axis=1),
        }

    # ------------------------------------------------------------------
    # Position construction
    # ------------------------------------------------------------------

    def _construct_positions(
        self,
        signal: pd.Series,
        prev_positions: pd.Series,  # noqa: ARG002  (available for subclass override)
    ) -> pd.Series:
        """
        Convert signal to positions.

        Steps:
        1. Drop NaN signals
        2. Clip to max_weight: w = signal.clip(−max_weight, max_weight)
        3. If dollar_neutral: separate into long (>0) and short (<0)
           Normalize: longs sum to +1, shorts sum to −1
        4. Return position weights
        """
        result = pd.Series(0.0, index=signal.index)
        valid = signal.dropna()
        if valid.empty:
            return result

        w = valid.clip(-self.max_weight, self.max_weight)

        if self.dollar_neutral:
            longs = w[w > 0]
            shorts = w[w < 0]
            if len(longs) > 0:
                result[longs.index] = longs / longs.sum()          # sums to +1
            if len(shorts) > 0:
                result[shorts.index] = shorts / shorts.abs().sum() # sums to −1
        else:
            total = w.abs().sum()
            if total > 0:
                result[w.index] = w / total

        return result

    # ------------------------------------------------------------------
    # Turnover
    # ------------------------------------------------------------------

    def _compute_turnover(
        self,
        new_positions: pd.Series,
        old_positions: pd.Series,
    ) -> float:
        """
        Turnover = Σ |w_new − w_old| / 2
        (divided by 2 because each dollar sold is a dollar bought)
        """
        old_aligned = old_positions.reindex(new_positions.index).fillna(0.0)
        return float((new_positions - old_aligned).abs().sum() / 2)


# ---------------------------------------------------------------------------
# BacktestResult — container with analysis properties
# ---------------------------------------------------------------------------

class BacktestResult:
    """
    Container for backtest results with built-in analysis.

    Wraps the dict returned by UnconstrainedBacktest.run() and exposes
    summary statistics and a plot method.
    """

    def __init__(self, result_dict: dict):
        self._data = result_dict
        self.returns: pd.Series = result_dict["returns"]
        self.gross_returns: pd.Series = result_dict["gross_returns"]
        self.positions: pd.DataFrame = result_dict["positions"]
        self.turnover: pd.Series = result_dict["turnover"]
        self.execution_costs: pd.Series = result_dict["execution_costs"]
        self.n_longs: pd.Series = result_dict["n_longs"]
        self.n_shorts: pd.Series = result_dict["n_shorts"]

    @property
    def cumulative_return(self) -> pd.Series:
        """Cumulative compounded return."""
        return (1 + self.returns.fillna(0)).cumprod() - 1

    @property
    def annualized_return(self) -> float:
        clean = self.returns.dropna()
        return float(clean.mean() * 252) if len(clean) > 0 else 0.0

    @property
    def annualized_vol(self) -> float:
        clean = self.returns.dropna()
        return float(clean.std() * np.sqrt(252)) if len(clean) > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe = annualized_return / annualized_vol"""
        v = self.annualized_vol
        return self.annualized_return / v if v > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        cum = (1 + self.returns.fillna(0)).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())

    @property
    def average_turnover(self) -> float:
        return float(self.turnover.mean())

    def summary(self) -> pd.Series:
        """One-line summary of all key metrics."""
        return pd.Series(
            {
                "annualized_return": self.annualized_return,
                "annualized_vol": self.annualized_vol,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "average_turnover": self.average_turnover,
            }
        )

    def plot(self, title: str = None):
        """Plot cumulative return curve with drawdown subplot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        self.cumulative_return.plot(ax=axes[0], title=title or "Cumulative Return")
        axes[0].set_ylabel("Cumulative Return")
        axes[0].axhline(0, color="black", linewidth=0.8)

        cum = (1 + self.returns.fillna(0)).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        dd.plot(ax=axes[1], color="firebrick", title="Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.3)

        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Legacy BacktestEngine (Prompt 1 — kept for compatibility)
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration for BacktestEngine."""

    rebalance_freq: Literal["daily", "weekly", "monthly"] = "daily"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    position_method: Literal[
        "rank", "signal", "top_bottom", "equal_weight", "vol_target"
    ] = "rank"
    max_position: float = 0.10
    top_k: int = 10
    vol_target: float = 0.15
    vol_window: int = 63
    return_type: Literal["log", "simple"] = "log"
    signal_lag: int = 1


@dataclass
class BacktestData:
    """Container for BacktestEngine outputs."""

    portfolio_returns: pd.Series
    gross_returns: pd.Series
    positions: pd.DataFrame
    signal: pd.DataFrame
    cost_returns: pd.Series
    turnover: pd.Series
    asset_returns: pd.DataFrame


class BacktestEngine:
    """
    Legacy unconstrained cross-sectional backtesting engine (Prompt 1).

    Use UnconstrainedBacktest for new research.  This class is kept so
    that experiment scripts written against the Prompt 1 API continue to work.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        cfg = self.config.execution
        order_type = cfg.order_type
        bps = cfg.base_cost_bps / 10_000
        self.execution = ExecutionModel(
            market_order_cost=bps if order_type == "market" else 0.0020,
            limit_order_cost=bps if order_type == "limit" else 0.0007,
            order_type=order_type,
        )
        self.position_mgr = PositionManager(
            method=self.config.position_method,
            max_position=self.config.max_position,
            top_k=self.config.top_k,
            vol_target=self.config.vol_target,
            vol_window=self.config.vol_window,
        )

    def run(
        self,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
        dollar_volume: Optional[pd.DataFrame] = None,
        realized_vol: Optional[pd.DataFrame] = None,
    ) -> BacktestData:
        """Run the full backtest; returns BacktestData."""
        logger.info(
            "Running backtest: %d dates × %d symbols", len(signal), signal.shape[1]
        )
        signal, prices = self._align(signal, prices)

        if self.config.return_type == "log":
            fwd_returns = np.log(prices / prices.shift(1)).shift(-1)
        else:
            fwd_returns = prices.pct_change().shift(-1)

        rebalance_mask = self._rebalance_dates(signal.index)
        positions = self._compute_positions(signal, fwd_returns, rebalance_mask)

        turnover = self.position_mgr.turnover(positions)
        rebal_to = turnover.where(rebalance_mask, other=0.0)
        cost_series = self.execution.compute_costs(rebal_to)

        lagged_pos = positions.shift(self.config.signal_lag).fillna(0)
        asset_gross = fwd_returns * lagged_pos
        # Spread cost evenly across assets for per-asset return tracking
        n_assets = max(positions.shape[1], 1)
        asset_net = asset_gross.sub(cost_series / n_assets, axis=0)

        gross_ret = asset_gross.sum(axis=1)
        net_ret = gross_ret - cost_series

        logger.info(
            "Backtest complete. Ann. Sharpe (gross): %.2f | Ann. Sharpe (net): %.2f",
            self._quick_sharpe(gross_ret),
            self._quick_sharpe(net_ret),
        )

        return BacktestData(
            portfolio_returns=net_ret,
            gross_returns=gross_ret,
            positions=positions,
            signal=signal,
            cost_returns=cost_series,
            turnover=turnover,
            asset_returns=asset_net,
        )

    def run_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        dollar_volume: Optional[pd.DataFrame] = None,
        realized_vol: Optional[pd.DataFrame] = None,
    ) -> Dict[str, BacktestData]:
        """Run the same backtest for multiple signals independently."""
        return {
            name: self.run(sig, prices, dollar_volume, realized_vol)
            for name, sig in signals.items()
        }

    def _compute_positions(self, signal, returns, rebalance_mask):
        all_positions = self.position_mgr.compute_positions(signal, returns)
        positions = all_positions.copy()
        positions[~rebalance_mask] = np.nan
        positions = positions.ffill()
        return positions.fillna(0)

    def _rebalance_dates(self, index: pd.DatetimeIndex) -> pd.Series:
        freq = self.config.rebalance_freq
        if freq == "daily":
            return pd.Series(True, index=index)
        elif freq == "weekly":
            return pd.Series(index.dayofweek == 0, index=index)
        elif freq == "monthly":
            return pd.Series(
                index.month != pd.DatetimeIndex(index).shift(-1, freq="D").month,
                index=index,
            )
        else:
            raise ValueError(f"Unknown rebalance_freq: {freq}")

    @staticmethod
    def _align(a: pd.DataFrame, b: pd.DataFrame):
        common_idx = a.index.intersection(b.index)
        common_cols = a.columns.intersection(b.columns)
        return a.loc[common_idx, common_cols], b.loc[common_idx, common_cols]

    @staticmethod
    def _quick_sharpe(returns: pd.Series, periods: int = 252) -> float:
        clean = returns.dropna()
        if len(clean) < 10 or clean.std() == 0:
            return 0.0
        return float(clean.mean() / clean.std() * np.sqrt(periods))
