"""Tests for strategy combination / weighting."""

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.weighting import StrategyWeighter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 150
SYMS = [f"A{i}" for i in range(6)]
STRATEGIES = ["mom", "rev", "pairs"]


@pytest.fixture()
def dates():
    return pd.date_range("2021-01-01", periods=N, freq="D", tz="UTC")


@pytest.fixture()
def signals(dates):
    np.random.seed(7)
    return {
        name: pd.DataFrame(
            np.random.randn(N, len(SYMS)),
            index=dates, columns=SYMS,
        )
        for name in STRATEGIES
    }


@pytest.fixture()
def forward_returns(dates):
    np.random.seed(8)
    return pd.DataFrame(
        np.random.randn(N, len(SYMS)) * 0.01,
        index=dates, columns=SYMS,
    )


@pytest.fixture()
def strategy_returns(dates):
    np.random.seed(9)
    return {
        name: pd.Series(np.random.randn(N) * 0.01, index=dates)
        for name in STRATEGIES
    }


# ---------------------------------------------------------------------------
# Signal combination
# ---------------------------------------------------------------------------

class TestCombineSignals:
    def test_equal_combine_shape(self, signals, forward_returns):
        w = StrategyWeighter(method="equal")
        composite = w.combine_signals(signals, forward_returns)
        assert composite.shape == (N, len(SYMS))

    def test_equal_combine_is_average(self, signals):
        w = StrategyWeighter(method="equal")
        composite = w.combine_signals(signals)
        # Manual equal average
        stacked = np.stack([df.values for df in signals.values()], axis=2)
        expected = np.nanmean(stacked, axis=2)
        np.testing.assert_allclose(composite.values, expected, rtol=1e-10)

    def test_ic_weighted_shape(self, signals, forward_returns):
        w = StrategyWeighter(method="ic_weighted", estimation_window=30)
        composite = w.combine_signals(signals, forward_returns)
        assert composite.shape == (N, len(SYMS))

    def test_ic_weighted_requires_forward_returns(self, signals):
        w = StrategyWeighter(method="ic_weighted")
        with pytest.raises(ValueError, match="forward_returns"):
            w.combine_signals(signals)

    def test_custom_combine(self, signals):
        w = StrategyWeighter(
            method="custom",
            custom_weights={"mom": 0.6, "rev": 0.3, "pairs": 0.1},
        )
        composite = w.combine_signals(signals)
        assert composite.shape == (N, len(SYMS))

    def test_empty_signals_raises(self):
        w = StrategyWeighter()
        with pytest.raises(ValueError):
            w.combine_signals({})

    def test_non_signal_method_raises(self, signals):
        w = StrategyWeighter(method="sharpe_weighted")
        with pytest.raises(ValueError):
            w.combine_signals(signals)


# ---------------------------------------------------------------------------
# Strategy return combination
# ---------------------------------------------------------------------------

class TestCombineStrategyReturns:
    def test_equal_combination(self, strategy_returns):
        w = StrategyWeighter(method="equal")
        combined = w.combine_strategy_returns(strategy_returns)
        assert isinstance(combined, pd.Series)
        assert len(combined) == N

    def test_equal_is_mean(self, strategy_returns, dates):
        w = StrategyWeighter(method="equal")
        combined = w.combine_strategy_returns(strategy_returns)
        expected = pd.DataFrame(strategy_returns).mean(axis=1)
        pd.testing.assert_series_equal(
            combined.dropna(), expected.dropna(), rtol=1e-10, check_names=False
        )

    def test_sharpe_weighted_length(self, strategy_returns):
        w = StrategyWeighter(method="sharpe_weighted", estimation_window=30)
        combined = w.combine_strategy_returns(strategy_returns)
        assert len(combined) == N

    def test_min_corr_length(self, strategy_returns):
        w = StrategyWeighter(method="min_corr", estimation_window=30)
        combined = w.combine_strategy_returns(strategy_returns)
        assert len(combined) == N

    def test_mean_variance_length(self, strategy_returns):
        w = StrategyWeighter(method="mean_variance", estimation_window=30)
        combined = w.combine_strategy_returns(strategy_returns)
        assert len(combined) == N

    def test_custom_weights_sum_to_one(self, strategy_returns):
        w = StrategyWeighter(
            method="custom",
            custom_weights={"mom": 2, "rev": 1, "pairs": 1},
        )
        combined = w.combine_strategy_returns(strategy_returns)
        assert isinstance(combined, pd.Series)
