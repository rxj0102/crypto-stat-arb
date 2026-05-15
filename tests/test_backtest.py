"""Tests for UnconstrainedBacktest and BacktestResult."""

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.engine import BacktestResult, UnconstrainedBacktest
from statarb.backtest.execution import ExecutionModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DATES = 200
N_ASSETS = 6
SYMBOLS = [f"A{i}" for i in range(N_ASSETS)]


@pytest.fixture()
def dates():
    return pd.date_range("2021-01-01", periods=N_DATES, freq="D")


@pytest.fixture()
def returns(dates):
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(N_DATES, N_ASSETS) * 0.01,
        index=dates,
        columns=SYMBOLS,
    )


@pytest.fixture()
def signals(dates):
    np.random.seed(1)
    return pd.DataFrame(
        np.random.randn(N_DATES, N_ASSETS),
        index=dates,
        columns=SYMBOLS,
    )


# ---------------------------------------------------------------------------
# UnconstrainedBacktest.run — basic shape and type checks
# ---------------------------------------------------------------------------

class TestUnconstrainedBacktestRun:
    def test_run_returns_dict(self, returns, signals):
        bt = UnconstrainedBacktest(returns, signals)
        result = bt.run()
        assert isinstance(result, dict)

    def test_result_keys(self, returns, signals):
        result = UnconstrainedBacktest(returns, signals).run()
        required = {"returns", "gross_returns", "positions", "turnover",
                    "execution_costs", "n_longs", "n_shorts"}
        assert required.issubset(result.keys())

    def test_returns_length(self, returns, signals):
        result = UnconstrainedBacktest(returns, signals).run()
        assert len(result["returns"]) == N_DATES

    def test_positions_shape(self, returns, signals):
        result = UnconstrainedBacktest(returns, signals).run()
        assert result["positions"].shape == (N_DATES, N_ASSETS)

    def test_execution_costs_nonneg(self, returns, signals):
        result = UnconstrainedBacktest(returns, signals).run()
        assert (result["execution_costs"] >= 0).all()

    def test_net_equals_gross_minus_costs(self, returns, signals):
        """net return = gross return − execution costs exactly."""
        result = UnconstrainedBacktest(returns, signals).run()
        diff = result["gross_returns"] - result["returns"] - result["execution_costs"]
        np.testing.assert_allclose(diff.values, 0.0, atol=1e-12)

    def test_n_longs_shorts_leq_n_assets(self, returns, signals):
        result = UnconstrainedBacktest(returns, signals).run()
        total = result["n_longs"] + result["n_shorts"]
        assert (total <= N_ASSETS).all()


# ---------------------------------------------------------------------------
# Dollar-neutral construction
# ---------------------------------------------------------------------------

class TestDollarNeutral:
    def test_net_position_is_zero(self, returns, signals):
        """Dollar-neutral: sum of all weights = 0 on dates with both sides."""
        bt = UnconstrainedBacktest(returns, signals, dollar_neutral=True)
        positions = bt.run()["positions"]
        # Only check rows where the signal had both longs and shorts
        has_both = (positions > 0).any(axis=1) & (positions < 0).any(axis=1)
        net = positions.loc[has_both].sum(axis=1)
        np.testing.assert_allclose(net.values, 0.0, atol=1e-12)

    def test_gross_exposure_at_most_two(self, returns, signals):
        """Gross exposure (Σ|w|) ≤ 2 for a dollar-neutral portfolio."""
        bt = UnconstrainedBacktest(returns, signals, dollar_neutral=True)
        positions = bt.run()["positions"]
        gross = positions.abs().sum(axis=1)
        assert (gross <= 2.0 + 1e-9).all()

    def test_longs_sum_to_one(self, returns, dates):
        """Long book sums to exactly +1."""
        sig = pd.DataFrame(
            np.tile([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], (N_DATES, 1)),
            index=dates, columns=SYMBOLS,
        )
        bt = UnconstrainedBacktest(returns, sig, dollar_neutral=True)
        pos = bt.run()["positions"]
        longs_sum = pos[pos > 0].sum(axis=1).dropna()
        np.testing.assert_allclose(longs_sum.values, 1.0, atol=1e-12)

    def test_shorts_sum_to_minus_one(self, returns, dates):
        """Short book sums to exactly −1."""
        sig = pd.DataFrame(
            np.tile([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], (N_DATES, 1)),
            index=dates, columns=SYMBOLS,
        )
        bt = UnconstrainedBacktest(returns, sig, dollar_neutral=True)
        pos = bt.run()["positions"]
        shorts_sum = pos[pos < 0].sum(axis=1).dropna()
        np.testing.assert_allclose(shorts_sum.values, -1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Constant signal → constant positions → P&L = position · return
# ---------------------------------------------------------------------------

class TestConstantSignal:
    def test_constant_signal_gives_constant_positions(self, returns, dates):
        """Constant signal must yield identical positions on every date."""
        sig = pd.DataFrame(
            np.tile([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], (N_DATES, 1)),
            index=dates, columns=SYMBOLS,
        )
        bt = UnconstrainedBacktest(returns, sig, dollar_neutral=True)
        positions = bt.run()["positions"]
        # All rows should be identical (same signal → same normalised weights)
        ref = positions.iloc[0]
        for i in range(1, N_DATES):
            pd.testing.assert_series_equal(
                positions.iloc[i], ref, rtol=1e-12, check_names=False
            )

    def test_gross_return_equals_position_dot_return(self, returns, signals):
        """Gross return = Σ(position_i · return_i) at each date."""
        bt = UnconstrainedBacktest(returns, signals, dollar_neutral=True)
        result = bt.run()
        expected = (result["positions"] * returns).sum(axis=1)
        pd.testing.assert_series_equal(
            result["gross_returns"], expected, rtol=1e-10, check_names=False
        )


# ---------------------------------------------------------------------------
# No look-ahead bias
# ---------------------------------------------------------------------------

class TestNoLookahead:
    def test_positions_identical_up_to_split(self, returns, signals):
        """
        Truncating the inputs at row T must not change positions before T.
        """
        split = 100
        bt_full = UnconstrainedBacktest(returns, signals, dollar_neutral=True)
        bt_short = UnconstrainedBacktest(
            returns.iloc[:split], signals.iloc[:split], dollar_neutral=True
        )
        pos_full = bt_full.run()["positions"]
        pos_short = bt_short.run()["positions"]

        pd.testing.assert_frame_equal(
            pos_full.iloc[:split], pos_short.iloc[:split], rtol=1e-12
        )


# ---------------------------------------------------------------------------
# Execution cost = cost_rate × turnover (with default gross_exposure=2)
# ---------------------------------------------------------------------------

class TestExecutionCosts:
    def test_cost_equals_rate_times_turnover(self, returns, signals):
        """With gross_exposure=2 (default), cost = cost_per_trade × turnover."""
        em = ExecutionModel(market_order_cost=0.0020, order_type="market")
        bt = UnconstrainedBacktest(returns, signals, execution_model=em)
        result = bt.run()
        expected = 0.0020 * result["turnover"]
        pd.testing.assert_series_equal(
            result["execution_costs"], expected, rtol=1e-12, check_names=False
        )

    def test_limit_orders_cheaper_than_market(self, returns, signals):
        bt_mkt = UnconstrainedBacktest(
            returns, signals,
            execution_model=ExecutionModel(order_type="market")
        )
        bt_lmt = UnconstrainedBacktest(
            returns, signals,
            execution_model=ExecutionModel(order_type="limit")
        )
        costs_mkt = bt_mkt.run()["execution_costs"].sum()
        costs_lmt = bt_lmt.run()["execution_costs"].sum()
        assert costs_mkt > costs_lmt

    def test_net_leq_gross(self, returns, signals):
        result = UnconstrainedBacktest(returns, signals).run()
        diff = result["gross_returns"] - result["returns"]
        assert (diff >= -1e-12).all()


# ---------------------------------------------------------------------------
# Rebalancing frequency
# ---------------------------------------------------------------------------

class TestRebalancingFrequency:
    def test_freq2_holds_position_on_odd_days(self, returns, signals):
        """rebal_frequency=2: positions on odd rows equal the previous row."""
        bt = UnconstrainedBacktest(returns, signals, rebal_frequency=2)
        positions = bt.run()["positions"]
        for i in range(1, min(30, N_DATES), 2):
            pd.testing.assert_series_equal(
                positions.iloc[i], positions.iloc[i - 1],
                rtol=1e-12, check_names=False,
            )

    def test_freq1_lower_cost_than_freq1_with_market(self, returns, dates):
        """Slower rebalancing (freq=5) should incur less total cost than daily."""
        np.random.seed(7)
        sig = pd.DataFrame(np.random.randn(N_DATES, N_ASSETS), index=dates, columns=SYMBOLS)
        bt_daily = UnconstrainedBacktest(returns, sig, rebal_frequency=1)
        bt_slow = UnconstrainedBacktest(returns, sig, rebal_frequency=5)
        cost_daily = bt_daily.run()["execution_costs"].sum()
        cost_slow = bt_slow.run()["execution_costs"].sum()
        assert cost_slow < cost_daily


# ---------------------------------------------------------------------------
# _construct_positions unit tests
# ---------------------------------------------------------------------------

class TestConstructPositions:
    @pytest.fixture()
    def bt(self, returns, signals):
        return UnconstrainedBacktest(returns, signals, dollar_neutral=True)

    def _prev(self, *cols):
        return pd.Series(0.0, index=list(cols))

    def test_net_zero(self, bt):
        sig = pd.Series([1.0, 0.5, -0.5, -1.0], index=["A", "B", "C", "D"])
        pos = bt._construct_positions(sig, self._prev("A", "B", "C", "D"))
        assert abs(pos.sum()) < 1e-12

    def test_longs_sum_to_one(self, bt):
        sig = pd.Series([1.0, 0.5, -0.5, -1.0], index=["A", "B", "C", "D"])
        pos = bt._construct_positions(sig, self._prev("A", "B", "C", "D"))
        assert abs(pos[pos > 0].sum() - 1.0) < 1e-12

    def test_shorts_sum_to_minus_one(self, bt):
        sig = pd.Series([1.0, 0.5, -0.5, -1.0], index=["A", "B", "C", "D"])
        pos = bt._construct_positions(sig, self._prev("A", "B", "C", "D"))
        assert abs(pos[pos < 0].sum() + 1.0) < 1e-12

    def test_nan_excluded(self, bt):
        sig = pd.Series([1.0, np.nan, -1.0, np.nan], index=["A", "B", "C", "D"])
        pos = bt._construct_positions(sig, self._prev("A", "B", "C", "D"))
        assert pos["B"] == 0.0
        assert pos["D"] == 0.0

    def test_all_nan_returns_zeros(self, bt):
        sig = pd.Series([np.nan, np.nan, np.nan], index=["A", "B", "C"])
        pos = bt._construct_positions(sig, self._prev("A", "B", "C"))
        assert (pos == 0.0).all()


# ---------------------------------------------------------------------------
# _compute_turnover unit tests
# ---------------------------------------------------------------------------

class TestComputeTurnover:
    def _bt(self):
        obj = object.__new__(UnconstrainedBacktest)
        return obj

    def test_no_change_is_zero(self):
        bt = self._bt()
        pos = pd.Series([0.5, -0.5])
        assert bt._compute_turnover(pos, pos) == 0.0

    def test_from_zero_to_pos(self):
        bt = self._bt()
        new = pd.Series([0.5, -0.5])
        old = pd.Series([0.0, 0.0])
        # Σ|0.5−0| + |−0.5−0| = 1.0; /2 = 0.5
        assert abs(bt._compute_turnover(new, old) - 0.5) < 1e-12

    def test_symmetry(self):
        bt = self._bt()
        a = pd.Series([0.3, -0.3, 0.1, -0.1])
        b = pd.Series([-0.3, 0.3, -0.1, 0.1])
        assert abs(bt._compute_turnover(a, b) - bt._compute_turnover(b, a)) < 1e-12

    def test_full_flip_returns_two(self):
        bt = self._bt()
        # Full reversal of a dollar-neutral book with gross=2:
        # Σ|Δw| = 4 (each of 4 assets changes by 1.0) → /2 = 2.0
        a = pd.Series([0.5, 0.5, -0.5, -0.5])
        b = pd.Series([-0.5, -0.5, 0.5, 0.5])
        assert abs(bt._compute_turnover(a, b) - 2.0) < 1e-12


# ---------------------------------------------------------------------------
# BacktestResult container
# ---------------------------------------------------------------------------

class TestBacktestResult:
    @pytest.fixture()
    def result(self, returns, signals):
        bt = UnconstrainedBacktest(returns, signals)
        return BacktestResult(bt.run())

    def test_cumulative_return_is_series(self, result):
        assert isinstance(result.cumulative_return, pd.Series)
        assert len(result.cumulative_return) == N_DATES

    def test_annualized_return_is_float(self, result):
        assert isinstance(result.annualized_return, float)

    def test_annualized_vol_nonneg(self, result):
        assert result.annualized_vol >= 0.0

    def test_sharpe_ratio_formula(self, result):
        if result.annualized_vol > 0:
            expected = result.annualized_return / result.annualized_vol
            assert abs(result.sharpe_ratio - expected) < 1e-10

    def test_max_drawdown_nonpositive(self, result):
        assert result.max_drawdown <= 0.0

    def test_average_turnover_nonneg(self, result):
        assert result.average_turnover >= 0.0

    def test_summary_is_series(self, result):
        s = result.summary()
        assert isinstance(s, pd.Series)
        for key in ("annualized_return", "annualized_vol", "sharpe_ratio",
                    "max_drawdown", "average_turnover"):
            assert key in s.index

    def test_plot_returns_figure(self, result):
        import matplotlib
        matplotlib.use("Agg")
        fig = result.plot(title="Test")
        assert fig is not None
