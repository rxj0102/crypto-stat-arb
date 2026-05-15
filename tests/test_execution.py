"""Tests for ExecutionModel and ExecutionOptimizer."""

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.execution import ExecutionModel, ExecutionOptimizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 120
SYMS = ["A0", "A1", "A2", "A3"]


@pytest.fixture()
def dates():
    return pd.date_range("2022-01-01", periods=N, freq="D")


@pytest.fixture()
def turnover(dates):
    np.random.seed(0)
    return pd.Series(np.abs(np.random.randn(N)) * 0.10, index=dates)


# ---------------------------------------------------------------------------
# ExecutionModel
# ---------------------------------------------------------------------------

class TestExecutionModel:
    # --- construction / properties ---

    def test_default_order_type_is_market(self):
        em = ExecutionModel()
        assert em.order_type == "market"

    def test_market_cost_per_trade_is_20bps(self):
        em = ExecutionModel(market_order_cost=0.0020, order_type="market")
        assert em.cost_per_trade == 0.0020

    def test_limit_cost_per_trade_is_7bps(self):
        em = ExecutionModel(limit_order_cost=0.0007, order_type="limit")
        assert em.cost_per_trade == 0.0007

    def test_market_more_expensive_than_limit(self):
        assert ExecutionModel(order_type="market").cost_per_trade \
             > ExecutionModel(order_type="limit").cost_per_trade

    # --- compute_costs shape / sign ---

    def test_compute_costs_returns_series(self, turnover):
        costs = ExecutionModel().compute_costs(turnover)
        assert isinstance(costs, pd.Series)
        assert len(costs) == N

    def test_costs_are_nonneg(self, turnover):
        assert (ExecutionModel().compute_costs(turnover) >= 0).all()

    def test_zero_turnover_zero_cost(self, dates):
        em = ExecutionModel()
        zero = pd.Series(0.0, index=dates)
        assert (em.compute_costs(zero) == 0.0).all()

    # --- cost formula ---

    def test_cost_equals_rate_times_turnover_default_exposure(self, turnover):
        """With gross_exposure=2 (default): cost = cost_per_trade × turnover."""
        em = ExecutionModel(market_order_cost=0.0020, order_type="market")
        costs = em.compute_costs(turnover)  # gross_exposure=2 by default
        expected = 0.0020 * turnover
        pd.testing.assert_series_equal(costs, expected, rtol=1e-12)

    def test_gross_exposure_scales_cost_linearly(self, turnover):
        em = ExecutionModel(market_order_cost=0.0020)
        c2 = em.compute_costs(turnover, gross_exposure=2.0)
        c4 = em.compute_costs(turnover, gross_exposure=4.0)
        pd.testing.assert_series_equal(c4, 2.0 * c2, rtol=1e-12)

    def test_cost_proportional_to_cost_rate(self, turnover):
        c_low = ExecutionModel(market_order_cost=0.001).compute_costs(turnover)
        c_high = ExecutionModel(market_order_cost=0.002).compute_costs(turnover)
        pd.testing.assert_series_equal(c_high, 2.0 * c_low, rtol=1e-12)

    # --- net_return ---

    def test_net_return_reduces_gross(self, turnover, dates):
        np.random.seed(1)
        gross = pd.Series(np.random.randn(N) * 0.01, index=dates)
        net = ExecutionModel().net_return(gross, turnover)
        diff = gross - net
        assert (diff >= 0).all()

    def test_net_return_formula(self, turnover, dates):
        np.random.seed(2)
        em = ExecutionModel(market_order_cost=0.0015)
        gross = pd.Series(np.random.randn(N) * 0.01, index=dates)
        net = em.net_return(gross, turnover)
        expected = gross - em.compute_costs(turnover)
        pd.testing.assert_series_equal(net, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# ExecutionOptimizer.partial_rebalance
# ---------------------------------------------------------------------------

class TestPartialRebalance:
    def test_shape_preserved(self, dates):
        opt = ExecutionOptimizer()
        target = pd.DataFrame(np.random.randn(N, 4) * 0.1, index=dates, columns=SYMS)
        current = pd.DataFrame(np.random.randn(N, 4) * 0.1, index=dates, columns=SYMS)
        result = opt.partial_rebalance(target, current, threshold=0.02)
        assert result.shape == target.shape

    def test_small_diff_stays_at_current(self, dates):
        """Differences below threshold are not traded."""
        opt = ExecutionOptimizer()
        current = pd.DataFrame(0.10, index=dates, columns=SYMS)
        target = pd.DataFrame(0.10 + 0.005, index=dates, columns=SYMS)  # diff=0.005 < 0.02
        result = opt.partial_rebalance(target, current, threshold=0.02)
        pd.testing.assert_frame_equal(result, current)

    def test_large_diff_moves_to_target(self, dates):
        """Differences above threshold are fully traded to target."""
        opt = ExecutionOptimizer()
        current = pd.DataFrame(0.10, index=dates, columns=SYMS)
        target = pd.DataFrame(0.10 + 0.05, index=dates, columns=SYMS)  # diff=0.05 > 0.02
        result = opt.partial_rebalance(target, current, threshold=0.02)
        pd.testing.assert_frame_equal(result, target)

    def test_reduces_total_turnover(self, dates):
        """Partial rebalance generates strictly less turnover than a full trade."""
        opt = ExecutionOptimizer()
        np.random.seed(3)
        current = pd.DataFrame(np.random.randn(N, 4) * 0.1, index=dates, columns=SYMS)
        target = pd.DataFrame(np.random.randn(N, 4) * 0.1, index=dates, columns=SYMS)

        partial = opt.partial_rebalance(target, current, threshold=0.05)
        full_to = (target - current).abs().sum().sum()
        partial_to = (partial - current).abs().sum().sum()
        assert partial_to <= full_to

    def test_result_between_current_and_target(self, dates):
        """For a single cell above threshold, result equals target."""
        opt = ExecutionOptimizer()
        current = pd.DataFrame(0.0, index=dates[:5], columns=["X"])
        target = pd.DataFrame(0.10, index=dates[:5], columns=["X"])  # diff=0.10 > 0.02
        result = opt.partial_rebalance(target, current, threshold=0.02)
        pd.testing.assert_frame_equal(result, target)


# ---------------------------------------------------------------------------
# ExecutionOptimizer.buffer_zone_positions
# ---------------------------------------------------------------------------

class TestBufferZonePositions:
    def test_shape_preserved(self, dates):
        opt = ExecutionOptimizer()
        sig = pd.DataFrame(np.random.randn(N, 4), index=dates, columns=SYMS)
        pos = opt.buffer_zone_positions(sig)
        assert pos.shape == sig.shape

    def test_flat_below_entry_threshold(self, dates):
        """Signal 0.3 < entry_threshold 0.5 → no position opened."""
        opt = ExecutionOptimizer()
        sig = pd.DataFrame(0.3, index=dates, columns=["A"])
        pos = opt.buffer_zone_positions(sig, entry_threshold=0.5, exit_threshold=0.0)
        assert (pos["A"] == 0.0).all()

    def test_enters_above_entry_threshold(self, dates):
        """Signal 0.8 > entry_threshold 0.5 → position taken."""
        opt = ExecutionOptimizer()
        sig = pd.DataFrame(0.0, index=dates, columns=["A"])
        sig.iloc[10] = 0.8
        pos = opt.buffer_zone_positions(sig, entry_threshold=0.5, exit_threshold=0.3)
        assert pos.iloc[10, 0] != 0.0

    def test_exits_below_exit_threshold(self, dates):
        """Open position closes when |signal| drops below exit_threshold."""
        opt = ExecutionOptimizer()
        # First 20 rows: signal=0.8 (enter); rows 20+: signal=0.1 (exit)
        vals = [0.8] * 20 + [0.1] * (N - 20)
        sig = pd.DataFrame({"A": vals}, index=dates)
        pos = opt.buffer_zone_positions(sig, entry_threshold=0.5, exit_threshold=0.3)
        assert pos.iloc[10, 0] != 0.0   # in position
        assert pos.iloc[30, 0] == 0.0   # out after exit

    def test_holds_in_buffer_zone(self, dates):
        """
        Position is held when |signal| is in (exit_threshold, entry_threshold].
        Signal sequence: enter (0.8), hold zone (0.4), exit zone (0.1).
        """
        opt = ExecutionOptimizer()
        vals = [0.8] * 10 + [0.4] * 20 + [0.1] * (N - 30)
        sig = pd.DataFrame({"A": vals}, index=dates)
        pos = opt.buffer_zone_positions(sig, entry_threshold=0.5, exit_threshold=0.2)
        assert pos.iloc[5, 0] != 0.0    # entered at row 0
        assert pos.iloc[15, 0] != 0.0   # hold zone: still in
        assert pos.iloc[35, 0] == 0.0   # below exit_threshold: out

    def test_no_entry_when_nan(self, dates):
        """NaN signal → zero position."""
        opt = ExecutionOptimizer()
        sig = pd.DataFrame({"A": [np.nan] * N}, index=dates)
        pos = opt.buffer_zone_positions(sig, entry_threshold=0.5, exit_threshold=0.0)
        assert (pos["A"] == 0.0).all()

    def test_hysteresis_reduces_roundtrips(self, dates):
        """Buffer zone creates fewer position changes than direct signal thresholding."""
        opt = ExecutionOptimizer()
        np.random.seed(9)
        sig = pd.DataFrame(
            {"A": 0.5 + np.random.randn(N) * 0.2},
            index=dates,
        )
        pos_buffer = opt.buffer_zone_positions(sig, entry_threshold=0.6, exit_threshold=0.4)
        pos_direct = (sig.abs() > 0.5).astype(float) * np.sign(sig)
        changes_buffer = pos_buffer.diff().abs().sum().sum()
        changes_direct = pos_direct.diff().abs().sum().sum()
        assert changes_buffer <= changes_direct
