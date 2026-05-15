"""
Execution cost modeling.

Realistic transaction cost assumptions for crypto statistical arbitrage:
- Market orders: ~20 bps round-trip (taker fee on liquid assets)
- Limit orders: ~7 bps round-trip (maker rebate + slippage estimate)
- Market impact: additional cost proportional to trade size vs daily volume

These costs are applied at each rebalance to compute net returns.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from statarb.utils import get_logger

logger = get_logger(__name__)

# Default cost assumptions (basis points, one-way)
MARKET_ORDER_COST_BPS = 10.0   # 10 bps one-way = 20 bps round-trip
LIMIT_ORDER_COST_BPS = 3.5    # 3.5 bps one-way = 7 bps round-trip


@dataclass
class ExecutionConfig:
    """Configuration for execution cost modeling."""

    order_type: Literal["market", "limit"] = "market"
    one_way_cost_bps: Optional[float] = None   # override default if set
    market_impact_bps_per_pct_adv: float = 5.0  # 5 bps per 1% of ADTV traded
    slippage_vol_multiplier: float = 0.1        # 10% of daily vol as slippage
    min_cost_bps: float = 1.0                   # floor on transaction cost

    @property
    def base_cost_bps(self) -> float:
        if self.one_way_cost_bps is not None:
            return self.one_way_cost_bps
        return (MARKET_ORDER_COST_BPS if self.order_type == "market"
                else LIMIT_ORDER_COST_BPS)


class ExecutionModel:
    """
    Model transaction costs for a cross-sectional portfolio.

    Applies costs at each rebalance date based on:
    1. Base cost (market or limit order fee/spread)
    2. Market impact (function of trade size vs ADTV)
    3. Volatility slippage (additional cost on high-vol days)

    All costs are expressed as fractions (not percentages).
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        logger.info(
            "ExecutionModel: %s orders, base cost %.1f bps/side",
            self.config.order_type,
            self.config.base_cost_bps,
        )

    # ------------------------------------------------------------------
    # Core cost computation
    # ------------------------------------------------------------------

    def compute_costs(
        self,
        trades: pd.DataFrame,
        dollar_volume: Optional[pd.DataFrame] = None,
        returns_vol: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute transaction costs for a DataFrame of trades.

        Args:
            trades: signed position change panel (dates × symbols).
                    Values are portfolio weights (e.g., -0.05 to +0.05).
                    Positive = buy, negative = sell.
            dollar_volume: daily dollar volume panel for market impact
            returns_vol: realized volatility panel for slippage estimation

        Returns:
            DataFrame of transaction costs (as fraction of portfolio,
            same shape as ``trades``). Always non-negative.
        """
        abs_trades = trades.abs()
        base_cost = abs_trades * (self.config.base_cost_bps / 10_000)

        # Market impact: proportional to trade size / ADTV
        impact_cost = pd.DataFrame(0.0, index=trades.index, columns=trades.columns)
        if dollar_volume is not None:
            adtv = dollar_volume.rolling(window=21, min_periods=5).mean()
            adtv_aligned = adtv.reindex_like(trades)
            # Approximate portfolio $ size = 1 (normalised weights)
            pct_adv = abs_trades / adtv_aligned.replace(0, np.nan).clip(lower=1e-8)
            impact_cost = (pct_adv * self.config.market_impact_bps_per_pct_adv / 10_000)
            impact_cost = impact_cost.fillna(0).clip(lower=0)

        # Volatility slippage
        vol_cost = pd.DataFrame(0.0, index=trades.index, columns=trades.columns)
        if returns_vol is not None:
            vol_aligned = returns_vol.reindex_like(trades).fillna(0)
            vol_cost = abs_trades * vol_aligned * self.config.slippage_vol_multiplier

        total_cost = (base_cost + impact_cost + vol_cost).clip(lower=0)

        # Enforce minimum cost floor where there is any trade
        min_cost = (abs_trades > 0) * (self.config.min_cost_bps / 10_000)
        total_cost = total_cost.where(total_cost >= min_cost, other=min_cost)
        total_cost = total_cost.where(abs_trades > 0, other=0.0)

        return total_cost

    # ------------------------------------------------------------------
    # Net return computation
    # ------------------------------------------------------------------

    def apply_costs(
        self,
        gross_returns: pd.DataFrame,
        positions: pd.DataFrame,
        dollar_volume: Optional[pd.DataFrame] = None,
        returns_vol: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute net returns by subtracting transaction costs.

        Args:
            gross_returns: forward return panel (dates × symbols)
            positions: portfolio weight panel (dates × symbols)
            dollar_volume: optional ADTV for market impact
            returns_vol: optional volatility for slippage

        Returns:
            Net return panel (same shape as gross_returns).
        """
        trades = positions.diff().fillna(positions)  # first row is full trade
        costs = self.compute_costs(trades, dollar_volume, returns_vol)
        net = gross_returns * positions.shift(1).fillna(0) - costs
        return net

    def round_trip_cost_bps(self) -> float:
        """Return the round-trip base cost in basis points."""
        return self.config.base_cost_bps * 2

    def breakeven_alpha_bps(self, turnover: float) -> float:
        """
        Minimum daily alpha (bps) needed to break even given turnover.

        Args:
            turnover: daily one-way turnover as a fraction (0 to 1)

        Returns:
            Required gross alpha in basis points per day.
        """
        return self.config.base_cost_bps * turnover * 2  # two-sided
