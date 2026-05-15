"""Tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest

from statarb.evaluation.metrics import PerformanceMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flat_returns():
    """Zero-return series."""
    return pd.Series(np.zeros(252))


@pytest.fixture()
def positive_returns():
    """Consistently positive daily returns."""
    np.random.seed(42)
    return pd.Series(np.abs(np.random.randn(500)) * 0.005 + 0.001)


@pytest.fixture()
def negative_returns():
    """Consistently negative daily returns."""
    np.random.seed(43)
    return pd.Series(-np.abs(np.random.randn(500)) * 0.005 - 0.001)


@pytest.fixture()
def realistic_returns():
    """Realistic equity-like returns with Sharpe ~ 1."""
    np.random.seed(99)
    return pd.Series(np.random.randn(1000) * 0.01 + 0.0004)


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

class TestPerformanceMetrics:
    def test_total_return_zero(self, flat_returns):
        pm = PerformanceMetrics(flat_returns)
        assert abs(pm.total_return()) < 1e-9

    def test_total_return_positive(self, positive_returns):
        pm = PerformanceMetrics(positive_returns)
        assert pm.total_return() > 0

    def test_annualised_return_positive(self, positive_returns):
        pm = PerformanceMetrics(positive_returns)
        assert pm.annualised_return() > 0

    def test_annualised_volatility_positive(self, realistic_returns):
        pm = PerformanceMetrics(realistic_returns)
        assert pm.annualised_volatility() > 0

    def test_sharpe_zero_for_flat(self, flat_returns):
        pm = PerformanceMetrics(flat_returns)
        assert pm.sharpe_ratio() == 0.0

    def test_sharpe_positive_for_good_strategy(self, positive_returns):
        pm = PerformanceMetrics(positive_returns)
        assert pm.sharpe_ratio() > 0

    def test_sharpe_negative_for_bad_strategy(self, negative_returns):
        pm = PerformanceMetrics(negative_returns)
        assert pm.sharpe_ratio() < 0

    def test_max_drawdown_non_positive(self, realistic_returns):
        pm = PerformanceMetrics(realistic_returns)
        assert pm.max_drawdown() <= 0

    def test_max_drawdown_zero_for_increasing(self):
        """Monotonically increasing strategy has zero drawdown."""
        increasing = pd.Series([0.001] * 252)
        pm = PerformanceMetrics(increasing)
        assert abs(pm.max_drawdown()) < 1e-9

    def test_drawdown_series_non_positive(self, realistic_returns):
        pm = PerformanceMetrics(realistic_returns)
        dd = pm.drawdown_series()
        assert (dd <= 0 + 1e-10).all()

    def test_hit_rate_in_range(self, realistic_returns):
        pm = PerformanceMetrics(realistic_returns)
        hr = pm.hit_rate()
        assert 0 <= hr <= 1

    def test_hit_rate_near_one_for_positive(self, positive_returns):
        pm = PerformanceMetrics(positive_returns)
        assert pm.hit_rate() > 0.9

    def test_var_positive(self, realistic_returns):
        pm = PerformanceMetrics(realistic_returns)
        assert pm.var(0.95) > 0

    def test_cvar_geq_var(self, realistic_returns):
        pm = PerformanceMetrics(realistic_returns)
        assert pm.cvar(0.95) >= pm.var(0.95) - 1e-10

    def test_sortino_ratio_positive(self, realistic_returns):
        """Sortino is positive for a strategy with positive mean return."""
        pm = PerformanceMetrics(realistic_returns)
        assert pm.sortino_ratio() > 0

    def test_calmar_ratio_positive(self, realistic_returns):
        """Calmar ratio is positive for a strategy with positive return and finite drawdown."""
        pm = PerformanceMetrics(realistic_returns)
        assert pm.calmar_ratio() > 0

    def test_summary_returns_dict(self, realistic_returns):
        pm = PerformanceMetrics(realistic_returns)
        s = pm.summary()
        assert isinstance(s, dict)
        required_keys = [
            "sharpe_ratio", "annualised_return", "max_drawdown",
            "hit_rate", "total_return",
        ]
        for k in required_keys:
            assert k in s

    def test_compare_returns_dataframe(self):
        returns = {
            "a": pd.Series(np.random.randn(200) * 0.01),
            "b": pd.Series(np.random.randn(200) * 0.01 + 0.0005),
        }
        df = PerformanceMetrics.compare(returns)
        assert isinstance(df, pd.DataFrame)
        assert "a" in df.columns
        assert "b" in df.columns

    def test_information_ratio(self, realistic_returns):
        benchmark = pd.Series(np.random.randn(1000) * 0.01, name="bench")
        pm = PerformanceMetrics(realistic_returns)
        ir = pm.information_ratio(benchmark)
        assert np.isfinite(ir)
