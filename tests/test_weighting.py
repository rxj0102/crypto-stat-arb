"""Tests for StrategyWeighting."""

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.weighting import StrategyWeighter, StrategyWeighting

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 300
STRATS = ["alpha", "beta", "gamma"]


@pytest.fixture()
def dates():
    return pd.date_range("2020-01-01", periods=N, freq="D")


@pytest.fixture()
def strat_returns(dates):
    np.random.seed(0)
    return pd.DataFrame(
        {name: np.random.randn(N) * 0.01 for name in STRATS},
        index=dates,
    )


# ---------------------------------------------------------------------------
# equal_weight
# ---------------------------------------------------------------------------

class TestEqualWeight:
    def test_returns_series(self, strat_returns):
        result = StrategyWeighting().equal_weight(strat_returns)
        assert isinstance(result, pd.Series)

    def test_length(self, strat_returns):
        assert len(StrategyWeighting().equal_weight(strat_returns)) == N

    def test_is_mean(self, strat_returns):
        result = StrategyWeighting().equal_weight(strat_returns)
        expected = strat_returns.mean(axis=1)
        pd.testing.assert_series_equal(result, expected, rtol=1e-12)

    def test_identical_strategies_equals_that_strategy(self, dates):
        """Equal weighting of K identical series returns that series."""
        np.random.seed(1)
        r = pd.Series(np.random.randn(N) * 0.01, index=dates)
        strats = pd.DataFrame({"A": r, "B": r, "C": r})
        result = StrategyWeighting().equal_weight(strats)
        pd.testing.assert_series_equal(result, r, rtol=1e-12, check_names=False)

    def test_two_strategy_average(self, dates):
        np.random.seed(2)
        r1 = pd.Series(np.random.randn(N) * 0.01, index=dates)
        r2 = pd.Series(np.random.randn(N) * 0.01, index=dates)
        strats = pd.DataFrame({"a": r1, "b": r2})
        result = StrategyWeighting().equal_weight(strats)
        expected = (r1 + r2) / 2
        pd.testing.assert_series_equal(result, expected, rtol=1e-12, check_names=False)


# ---------------------------------------------------------------------------
# inverse_vol_weight
# ---------------------------------------------------------------------------

class TestInverseVolWeight:
    def test_returns_series(self, strat_returns):
        result = StrategyWeighting().inverse_vol_weight(strat_returns, lookback=50)
        assert isinstance(result, pd.Series)
        assert len(result) == N

    def test_nan_during_warmup(self, strat_returns):
        result = StrategyWeighting().inverse_vol_weight(strat_returns, lookback=50)
        assert result.iloc[:49].isna().all()

    def test_valid_after_warmup(self, strat_returns):
        result = StrategyWeighting().inverse_vol_weight(strat_returns, lookback=50)
        assert result.iloc[50:].notna().all()

    def test_lower_vol_gets_more_weight(self, dates):
        """
        Inverse-vol weighting of two independent strategies should produce
        lower volatility than equal weighting — because the low-vol strategy
        gets a larger weight.
        """
        np.random.seed(42)
        low = pd.Series(np.random.randn(N) * 0.01, index=dates)   # σ ≈ 0.01
        high = pd.Series(np.random.randn(N) * 0.05, index=dates)  # σ ≈ 0.05
        strats = pd.DataFrame({"low": low, "high": high})

        sw = StrategyWeighting()
        combined = sw.inverse_vol_weight(strats, lookback=100)

        equal_wt = (low + high) / 2
        valid_idx = combined.dropna().index
        # Inv-vol-weighted has lower realised vol than equal-weighted
        assert combined.loc[valid_idx].std() < equal_wt.loc[valid_idx].std()

    def test_single_strategy_returns_that_strategy(self, dates):
        """With one strategy, any weighting scheme returns that strategy."""
        np.random.seed(3)
        r = pd.Series(np.random.randn(N) * 0.01, index=dates)
        strats = pd.DataFrame({"only": r})
        result = StrategyWeighting().inverse_vol_weight(strats, lookback=20)
        valid = result.dropna().index
        pd.testing.assert_series_equal(
            result.loc[valid], r.loc[valid], rtol=1e-10, check_names=False
        )


# ---------------------------------------------------------------------------
# sharpe_weighted
# ---------------------------------------------------------------------------

class TestSharpeWeighted:
    def test_returns_series(self, strat_returns):
        result = StrategyWeighting().sharpe_weighted(strat_returns, lookback=63)
        assert isinstance(result, pd.Series)
        assert len(result) == N

    def test_nan_during_warmup(self, strat_returns):
        result = StrategyWeighting().sharpe_weighted(strat_returns, lookback=63)
        assert result.iloc[:62].isna().all()

    def test_valid_after_warmup(self, strat_returns):
        result = StrategyWeighting().sharpe_weighted(strat_returns, lookback=63)
        assert result.iloc[63:].notna().all()

    def test_favors_positive_sharpe_strategy(self, dates):
        """
        Strategy with consistently positive returns should dominate,
        pushing the combined mean above zero.
        """
        np.random.seed(5)
        good = pd.Series(np.abs(np.random.randn(N)) * 0.01, index=dates)  # all > 0
        bad = pd.Series(-np.abs(np.random.randn(N)) * 0.01, index=dates)  # all < 0
        strats = pd.DataFrame({"good": good, "bad": bad})

        result = StrategyWeighting().sharpe_weighted(strats, lookback=63)
        valid = result.dropna()
        assert valid.mean() > 0.0

    def test_zeros_negative_sharpe_strategies(self, dates):
        """
        With one clearly positive and one clearly negative Sharpe strategy,
        the combined return should outperform the equal-weighted mean.
        """
        np.random.seed(6)
        good = pd.Series(np.abs(np.random.randn(N)) * 0.01, index=dates)
        bad = pd.Series(-np.abs(np.random.randn(N)) * 0.01, index=dates)
        strats = pd.DataFrame({"good": good, "bad": bad})

        sw = StrategyWeighting()
        sw_combined = sw.sharpe_weighted(strats, lookback=63)
        eq_combined = sw.equal_weight(strats)

        valid = sw_combined.dropna().index
        assert sw_combined.loc[valid].mean() > eq_combined.loc[valid].mean()


# ---------------------------------------------------------------------------
# min_variance
# ---------------------------------------------------------------------------

class TestMinVariance:
    def test_returns_series(self, strat_returns):
        result = StrategyWeighting().min_variance(strat_returns, lookback=100)
        assert isinstance(result, pd.Series)
        assert len(result) == N

    def test_nan_during_warmup(self, strat_returns):
        result = StrategyWeighting().min_variance(strat_returns, lookback=100)
        assert result.iloc[:99].isna().all()

    def test_valid_after_warmup(self, strat_returns):
        result = StrategyWeighting().min_variance(strat_returns, lookback=100)
        assert result.iloc[100:].notna().all()

    def test_lower_vol_than_highest_vol_strategy(self, dates):
        """Min-variance combination should be less volatile than the highest-vol input."""
        np.random.seed(7)
        low = pd.Series(np.random.randn(N) * 0.01, index=dates)
        high = pd.Series(np.random.randn(N) * 0.05, index=dates)
        strats = pd.DataFrame({"low": low, "high": high})

        result = StrategyWeighting().min_variance(strats, lookback=100)
        valid = result.dropna().index
        assert result.loc[valid].std() < high.loc[valid].std()


# ---------------------------------------------------------------------------
# combine — dispatcher
# ---------------------------------------------------------------------------

class TestCombine:
    def test_equal_method(self, strat_returns):
        sw = StrategyWeighting()
        result = sw.combine(strat_returns, method="equal")
        expected = sw.equal_weight(strat_returns)
        pd.testing.assert_series_equal(result, expected, rtol=1e-12)

    def test_inverse_vol_method(self, strat_returns):
        sw = StrategyWeighting()
        result = sw.combine(strat_returns, method="inverse_vol", lookback=63)
        expected = sw.inverse_vol_weight(strat_returns, lookback=63)
        pd.testing.assert_series_equal(result, expected, rtol=1e-12)

    def test_sharpe_method(self, strat_returns):
        sw = StrategyWeighting()
        result = sw.combine(strat_returns, method="sharpe", lookback=63)
        expected = sw.sharpe_weighted(strat_returns, lookback=63)
        pd.testing.assert_series_equal(result, expected, rtol=1e-12)

    def test_min_variance_method(self, strat_returns):
        sw = StrategyWeighting()
        result = sw.combine(strat_returns, method="min_variance", lookback=100)
        expected = sw.min_variance(strat_returns, lookback=100)
        pd.testing.assert_series_equal(result, expected, rtol=1e-12)

    def test_invalid_method_raises(self, strat_returns):
        with pytest.raises(ValueError, match="unknown"):
            StrategyWeighting().combine(strat_returns, method="unknown")


# ---------------------------------------------------------------------------
# Legacy StrategyWeighter — keep passing to avoid regressions
# ---------------------------------------------------------------------------

class TestCombineSignals:
    @pytest.fixture()
    def signals(self, dates):
        np.random.seed(7)
        return {
            name: pd.DataFrame(
                np.random.randn(N, 3),
                index=dates,
                columns=["X", "Y", "Z"],
            )
            for name in STRATS
        }

    def test_equal_combine_shape(self, signals):
        w = StrategyWeighter(method="equal")
        composite = w.combine_signals(signals)
        assert composite.shape == (N, 3)

    def test_equal_combine_is_average(self, signals):
        w = StrategyWeighter(method="equal")
        composite = w.combine_signals(signals)
        stacked = np.stack([df.values for df in signals.values()], axis=2)
        expected = np.nanmean(stacked, axis=2)
        np.testing.assert_allclose(composite.values, expected, rtol=1e-10)

    def test_ic_weighted_shape(self, signals, dates):
        np.random.seed(8)
        fwd = pd.DataFrame(
            np.random.randn(N, 3) * 0.01,
            index=dates, columns=["X", "Y", "Z"],
        )
        w = StrategyWeighter(method="ic_weighted", estimation_window=30)
        composite = w.combine_signals(signals, fwd)
        assert composite.shape == (N, 3)

    def test_ic_weighted_requires_forward_returns(self, signals):
        w = StrategyWeighter(method="ic_weighted")
        with pytest.raises(ValueError, match="forward_returns"):
            w.combine_signals(signals)

    def test_custom_combine(self, signals):
        w = StrategyWeighter(
            method="custom",
            custom_weights={"alpha": 0.6, "beta": 0.3, "gamma": 0.1},
        )
        composite = w.combine_signals(signals)
        assert composite.shape == (N, 3)

    def test_empty_signals_raises(self):
        w = StrategyWeighter()
        with pytest.raises(ValueError):
            w.combine_signals({})

    def test_non_signal_method_raises(self, signals):
        w = StrategyWeighter(method="sharpe_weighted")
        with pytest.raises(ValueError):
            w.combine_signals(signals)


class TestCombineStrategyReturns:
    @pytest.fixture()
    def strategy_returns(self, dates):
        np.random.seed(9)
        return {
            name: pd.Series(np.random.randn(N) * 0.01, index=dates)
            for name in STRATS
        }

    def test_equal_combination(self, strategy_returns):
        w = StrategyWeighter(method="equal")
        combined = w.combine_strategy_returns(strategy_returns)
        assert isinstance(combined, pd.Series)
        assert len(combined) == N

    def test_equal_is_mean(self, strategy_returns):
        w = StrategyWeighter(method="equal")
        combined = w.combine_strategy_returns(strategy_returns)
        expected = pd.DataFrame(strategy_returns).mean(axis=1)
        pd.testing.assert_series_equal(
            combined.dropna(), expected.dropna(), rtol=1e-10, check_names=False
        )

    def test_sharpe_weighted_length(self, strategy_returns):
        w = StrategyWeighter(method="sharpe_weighted", estimation_window=30)
        assert len(w.combine_strategy_returns(strategy_returns)) == N

    def test_min_corr_length(self, strategy_returns):
        w = StrategyWeighter(method="min_corr", estimation_window=30)
        assert len(w.combine_strategy_returns(strategy_returns)) == N

    def test_mean_variance_length(self, strategy_returns):
        w = StrategyWeighter(method="mean_variance", estimation_window=30)
        assert len(w.combine_strategy_returns(strategy_returns)) == N

    def test_custom_weights(self, strategy_returns):
        w = StrategyWeighter(
            method="custom",
            custom_weights={"alpha": 2, "beta": 1, "gamma": 1},
        )
        combined = w.combine_strategy_returns(strategy_returns)
        assert isinstance(combined, pd.Series)
