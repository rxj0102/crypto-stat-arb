"""
Execution cost modeling.

Realistic transaction cost assumptions for crypto statistical arbitrage:
- Market orders: ~7 bps commission + ~13 bps slippage = 20 bps total
- Limit orders: ~7 bps commission only

The cost is applied to TURNOVER: each dollar traded incurs the cost.

Total cost at time t:
    cost_t = cost_per_trade × turnover_t × gross_exposure / 2

For a strategy with $2 gross exposure and 50% daily turnover:
    daily cost = 20 bps × 50% × $2 = 0.20% of gross exposure per day
               = 0.10% of net capital per day ≈ 36% per year

This is SUBSTANTIAL. Low-turnover strategies are critical.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from statarb.utils import get_logger

logger = get_logger(__name__)

MARKET_ORDER_COST_BPS = 10.0
LIMIT_ORDER_COST_BPS = 3.5


@dataclass
class ExecutionConfig:
    """Legacy config object kept for BacktestEngine compatibility."""

    order_type: Literal["market", "limit"] = "market"
    one_way_cost_bps: Optional[float] = None
    market_impact_bps_per_pct_adv: float = 5.0
    slippage_vol_multiplier: float = 0.1
    min_cost_bps: float = 1.0

    @property
    def base_cost_bps(self) -> float:
        if self.one_way_cost_bps is not None:
            return self.one_way_cost_bps
        return MARKET_ORDER_COST_BPS if self.order_type == "market" else LIMIT_ORDER_COST_BPS


class ExecutionModel:
    """
    Execution cost model for cryptocurrency trading.

    Cost structure (from project spec):
    - Market orders: ~7 bps commission + ~13 bps slippage = 20 bps total
    - Limit orders: ~7 bps commission only

    The cost is applied to TURNOVER: each dollar traded incurs the cost.

    Total cost at time t:
        cost_t = cost_per_trade × turnover_t × gross_exposure / 2
    """

    def __init__(
        self,
        market_order_cost: float = 0.0020,
        limit_order_cost: float = 0.0007,
        order_type: str = "market",
    ):
        """
        Args:
            market_order_cost: one-way cost fraction for market orders (20 bps default)
            limit_order_cost:  one-way cost fraction for limit orders (7 bps default)
            order_type:        'market' or 'limit'
        """
        self.market_order_cost = market_order_cost
        self.limit_order_cost = limit_order_cost
        self.order_type = order_type

    @property
    def cost_per_trade(self) -> float:
        """One-way cost fraction for the configured order type."""
        return self.market_order_cost if self.order_type == "market" else self.limit_order_cost

    def compute_costs(
        self,
        turnover: pd.Series,
        gross_exposure: float = 2.0,
    ) -> pd.Series:
        """
        Compute execution costs per period.

        cost = cost_per_trade × turnover × gross_exposure / 2
        (divide by 2 because turnover = Σ|Δw|/2 already counts one side)

        With the default gross_exposure=2 this simplifies to:
            cost = cost_per_trade × turnover

        Args:
            turnover:       one-way turnover series (Σ|w_new − w_old| / 2)
            gross_exposure: total absolute notional (default $2: $1 long + $1 short)

        Returns:
            Series of cost fractions (same index as turnover).
        """
        return self.cost_per_trade * turnover * gross_exposure / 2

    def net_return(self, gross_return: pd.Series, turnover: pd.Series) -> pd.Series:
        """gross_return − execution_costs"""
        return gross_return - self.compute_costs(turnover)


class ExecutionOptimizer:
    """
    Techniques to reduce execution costs:

    1. Slower rebalancing (every 2-5 days instead of daily)
    2. Partial rebalancing: only trade when signal change exceeds threshold
    3. Buffer zone: don't trade into/out of positions unless signal crosses a band
    4. Limit orders: lower cost but execution risk (may not fill)
    """

    def partial_rebalance(
        self,
        target_positions: pd.DataFrame,
        current_positions: pd.DataFrame,
        threshold: float = 0.02,
    ) -> pd.DataFrame:
        """
        Only rebalance positions where |target − current| > threshold.
        Reduces turnover by avoiding small rebalancing trades.
        """
        diff = (target_positions - current_positions).abs()
        result = current_positions.copy()
        mask = diff > threshold
        result[mask] = target_positions[mask]
        return result

    def buffer_zone_positions(
        self,
        signal: pd.DataFrame,
        entry_threshold: float = 0.5,
        exit_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Enter position when |signal| > entry_threshold.
        Exit when |signal| < exit_threshold.
        In between: hold.

        This creates hysteresis, reducing unnecessary round-trips.

        Args:
            signal:          signal panel (dates × symbols)
            entry_threshold: minimum |signal| to open a position
            exit_threshold:  |signal| below which an open position is closed

        Returns:
            Position panel with same shape as signal.
        """
        positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        current = pd.Series(0.0, index=signal.columns)

        for i in range(len(signal)):
            row = signal.iloc[i]
            new_pos = current.copy()

            for col in signal.columns:
                s = row[col]
                if pd.isna(s):
                    new_pos[col] = 0.0
                    continue

                abs_s = abs(s)
                in_pos = current[col] != 0.0

                if in_pos:
                    if abs_s < exit_threshold:
                        new_pos[col] = 0.0
                    # abs_s >= exit_threshold: hold (keep current value)
                else:
                    if abs_s > entry_threshold:
                        new_pos[col] = s
                    # abs_s <= entry_threshold: stay flat

            positions.iloc[i] = new_pos
            current = new_pos

        return positions
