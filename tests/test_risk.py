"""Tests for risk analytics."""

import numpy as np
import pandas as pd
import pytest

from statarb.evaluation.risk import RiskAnalytics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 500


@pytest.fixture()
def strategy_returns():
    np.random.seed(10)
    return pd.Series(np.random.randn(N) * 0.01 + 0.0003)


@pytest.fixture()
def benchmark_returns():
    np.random.seed(11)
    return pd.Series(np.random.randn(N) * 0.02)


@pytest.fixture()
def ra(strategy_returns):
    return RiskAnalytics(strategy_returns)


# ---------------------------------------------------------------------------
# Alpha / Beta
# ---------------------------------------------------------------------------

class TestAlphaBeta:
    def test_returns_four_values(self, ra, benchmark_returns):
        result = ra.alpha_beta(benchmark_returns)
        assert len(result) == 4

    def test_r_squared_in_range(self, ra, benchmark_returns):
        _, _, r2, _ = ra.alpha_beta(benchmark_returns)
        assert 0 <= r2 <= 1

    def test_beta_near_zero_for_uncorrelated(self, strategy_returns):
        """Independent strategy should have beta near zero."""
        np.random.seed(99)
        independent_benchmark = pd.Series(np.random.randn(N) * 0.02)
        ra = RiskAnalytics(strategy_returns)
        _, beta, _, _ = ra.alpha_beta(independent_benchmark)
        assert abs(beta) < 0.5  # loose check for independence

    def test_beta_near_one_for_identical(self, strategy_returns):
        """Strategy identical to benchmark should have beta = 1."""
        ra = RiskAnalytics(strategy_returns)
        _, beta, r2, _ = ra.alpha_beta(strategy_returns)
        assert abs(beta - 1.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-6

    def test_alpha_annualised(self, ra, benchmark_returns):
        """Annualised alpha should be in a reasonable range."""
        alpha, _, _, _ = ra.alpha_beta(benchmark_returns, annualise=True)
        assert -2.0 < alpha < 2.0  # within ±200% annual

    def test_insufficient_data_returns_zeros(self):
        short_rets = pd.Series(np.random.randn(5) * 0.01)
        ra = RiskAnalytics(short_rets)
        bench = pd.Series(np.random.randn(5) * 0.01)
        alpha, beta, r2, pval = ra.alpha_beta(bench)
        assert alpha == 0.0
        assert beta == 0.0


# ---------------------------------------------------------------------------
# Rolling alpha/beta
# ---------------------------------------------------------------------------

class TestRollingAlphaBeta:
    def test_shape(self, ra, benchmark_returns):
        rolling = ra.rolling_alpha_beta(benchmark_returns, window=63)
        assert rolling.shape == (N, 3)

    def test_columns(self, ra, benchmark_returns):
        rolling = ra.rolling_alpha_beta(benchmark_returns, window=63)
        assert list(rolling.columns) == ["alpha", "beta", "r_squared"]

    def test_r_squared_in_range(self, ra, benchmark_returns):
        rolling = ra.rolling_alpha_beta(benchmark_returns, window=63)
        valid = rolling["r_squared"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


# ---------------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------------

class TestTailRisk:
    def test_returns_dict(self, ra):
        result = ra.tail_risk_summary()
        assert isinstance(result, dict)

    def test_max_daily_loss_negative(self, ra):
        result = ra.tail_risk_summary()
        assert result["max_daily_loss"] <= 0

    def test_max_daily_gain_positive(self, ra):
        result = ra.tail_risk_summary()
        assert result["max_daily_gain"] >= 0

    def test_tail_ratio_positive(self, ra):
        result = ra.tail_risk_summary()
        assert result["tail_ratio"] > 0


# ---------------------------------------------------------------------------
# Conditional performance
# ---------------------------------------------------------------------------

class TestConditionalPerformance:
    def test_returns_two_regimes(self, ra):
        condition = pd.Series(np.random.rand(N) > 0.5)
        result = ra.conditional_performance(condition)
        assert "high" in result
        assert "low" in result

    def test_regime_has_metrics(self, ra):
        condition = pd.Series(np.ones(N, dtype=bool))
        condition.iloc[:N // 2] = False
        result = ra.conditional_performance(condition)
        for regime in result.values():
            if regime:
                assert "sharpe_ratio" in regime


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------

class TestRollingSharpe:
    def test_length(self, ra):
        rolling = ra.rolling_sharpe(window=63)
        assert len(rolling) == N

    def test_finite_values(self, ra):
        rolling = ra.rolling_sharpe(window=63)
        assert np.isfinite(rolling.dropna().values).all()


# ---------------------------------------------------------------------------
# Strategy correlation
# ---------------------------------------------------------------------------

class TestStrategyCorrelation:
    def test_returns_series(self, ra, benchmark_returns):
        result = ra.strategy_correlation({"benchmark": benchmark_returns})
        assert isinstance(result, pd.Series)

    def test_self_correlation_is_one(self, strategy_returns, ra):
        result = ra.strategy_correlation({"self": strategy_returns})
        assert abs(result["self"] - 1.0) < 1e-9

    def test_correlation_in_range(self, ra, benchmark_returns):
        result = ra.strategy_correlation({"bench": benchmark_returns})
        assert -1 <= result["bench"] <= 1
