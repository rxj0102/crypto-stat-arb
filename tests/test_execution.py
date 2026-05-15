"""Tests for execution cost modeling."""

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.execution import ExecutionConfig, ExecutionModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 100
SYMS = [f"A{i}" for i in range(5)]


@pytest.fixture()
def dates():
    return pd.date_range("2022-01-01", periods=N, freq="D")


@pytest.fixture()
def trades(dates):
    """Small random position changes."""
    np.random.seed(10)
    data = np.random.randn(N, len(SYMS)) * 0.01
    return pd.DataFrame(data, index=dates, columns=SYMS)


@pytest.fixture()
def dollar_volume(dates):
    return pd.DataFrame(
        np.ones((N, len(SYMS))) * 1e8,
        index=dates, columns=SYMS,
    )


@pytest.fixture()
def returns_vol(dates):
    return pd.DataFrame(
        np.ones((N, len(SYMS))) * 0.02,
        index=dates, columns=SYMS,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecutionModel:
    def test_compute_costs_non_negative(self, trades, dollar_volume, returns_vol):
        em = ExecutionModel()
        costs = em.compute_costs(trades, dollar_volume, returns_vol)
        assert (costs >= 0).all().all()

    def test_zero_trade_zero_cost(self, dates):
        em = ExecutionModel()
        zero_trades = pd.DataFrame(0.0, index=dates, columns=SYMS)
        costs = em.compute_costs(zero_trades)
        assert (costs == 0).all().all()

    def test_costs_shape(self, trades, dollar_volume):
        em = ExecutionModel()
        costs = em.compute_costs(trades, dollar_volume)
        assert costs.shape == trades.shape

    def test_market_order_more_expensive_than_limit(self):
        """Market base cost (10 bps) exceeds limit base cost (3.5 bps)."""
        em_market = ExecutionModel(ExecutionConfig(order_type="market"))
        em_limit = ExecutionModel(ExecutionConfig(order_type="limit"))
        assert em_market.config.base_cost_bps > em_limit.config.base_cost_bps

    def test_round_trip_cost_market(self):
        em = ExecutionModel(ExecutionConfig(order_type="market"))
        assert abs(em.round_trip_cost_bps() - 20.0) < 1e-6

    def test_round_trip_cost_limit(self):
        em = ExecutionModel(ExecutionConfig(order_type="limit"))
        assert abs(em.round_trip_cost_bps() - 7.0) < 1e-6

    def test_breakeven_alpha_positive(self):
        em = ExecutionModel()
        ba = em.breakeven_alpha_bps(turnover=0.1)
        assert ba > 0

    def test_apply_costs_reduces_returns(self, dates, dollar_volume, returns_vol):
        """Net returns should be <= gross returns."""
        np.random.seed(5)
        pos = pd.DataFrame(
            np.random.randn(N, len(SYMS)) * 0.05,
            index=dates, columns=SYMS,
        )
        gross_ret = pd.DataFrame(
            np.random.randn(N, len(SYMS)) * 0.01,
            index=dates, columns=SYMS,
        )
        em = ExecutionModel()
        net = em.apply_costs(gross_ret, pos, dollar_volume, returns_vol)
        # Net portfolio total <= gross
        assert net.sum(axis=1).sum() <= gross_ret.sum(axis=1).sum() + 1e-10

    def test_custom_cost_override(self, trades):
        em = ExecutionModel(ExecutionConfig(one_way_cost_bps=50.0))
        costs = em.compute_costs(trades)
        # Base cost should be 50 bps of |trade|
        expected_base = trades.abs() * 0.005  # 50 bps
        assert (costs >= expected_base * 0.99).all().all()
