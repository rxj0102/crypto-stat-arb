"""Tests for RiskAnalytics (stateless API)."""

import math

import numpy as np
import pandas as pd
import pytest

from statarb.evaluation.risk import RiskAnalytics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 400
DATES = pd.date_range("2020-01-01", periods=N, freq="D")
N_ASSETS = 4
SYMS = [f"S{i}" for i in range(N_ASSETS)]


@pytest.fixture()
def ra():
    return RiskAnalytics()


@pytest.fixture()
def benchmark():
    np.random.seed(10)
    return pd.Series(np.random.randn(N) * 0.015, index=DATES)


@pytest.fixture()
def random_returns():
    np.random.seed(42)
    return pd.Series(np.random.randn(N) * 0.01, index=DATES)


@pytest.fixture()
def returns_with_nans(random_returns):
    r = random_returns.copy()
    r.iloc[::10] = np.nan
    return r


# ---------------------------------------------------------------------------
# alpha_beta
# ---------------------------------------------------------------------------

class TestAlphaBeta:
    def test_returns_dict(self, ra, random_returns, benchmark):
        result = ra.alpha_beta(random_returns, benchmark)
        assert isinstance(result, dict)

    def test_required_keys(self, ra, random_returns, benchmark):
        result = ra.alpha_beta(random_returns, benchmark)
        assert {"alpha", "beta", "r_squared", "alpha_tstat", "beta_tstat"} == set(result)

    def test_benchmark_against_itself_alpha_zero_beta_one(self, ra, benchmark):
        """Alpha of benchmark against itself = 0, beta = 1."""
        result = ra.alpha_beta(benchmark, benchmark)
        assert abs(result["alpha"]) < 1e-8
        assert abs(result["beta"] - 1.0) < 1e-8

    def test_r_squared_self_regression_is_one(self, ra, benchmark):
        result = ra.alpha_beta(benchmark, benchmark)
        assert abs(result["r_squared"] - 1.0) < 1e-8

    def test_beta_in_reasonable_range(self, ra, random_returns, benchmark):
        result = ra.alpha_beta(random_returns, benchmark)
        assert -5.0 < result["beta"] < 5.0

    def test_r_squared_between_zero_and_one(self, ra, random_returns, benchmark):
        r2 = ra.alpha_beta(random_returns, benchmark)["r_squared"]
        assert 0.0 <= r2 <= 1.0

    def test_handles_nan_gracefully(self, ra, returns_with_nans, benchmark):
        result = ra.alpha_beta(returns_with_nans, benchmark)
        assert isinstance(result, dict)
        assert math.isfinite(result["beta"])

    def test_too_few_observations_returns_zeros(self, ra, benchmark):
        short = pd.Series([0.01, 0.02], index=DATES[:2])
        result = ra.alpha_beta(short, benchmark)
        assert result["alpha"] == 0.0 and result["beta"] == 0.0

    def test_dollar_neutral_strategy_beta_near_zero(self, ra, benchmark):
        """
        Dollar-neutral strategy with returns uncorrelated to benchmark
        should have beta close to 0.
        """
        np.random.seed(99)
        uncorrelated = pd.Series(np.random.randn(N) * 0.005, index=DATES)
        result = ra.alpha_beta(uncorrelated, benchmark)
        assert abs(result["beta"]) < 0.3


# ---------------------------------------------------------------------------
# rolling_beta
# ---------------------------------------------------------------------------

class TestRollingBeta:
    def test_returns_series(self, ra, random_returns, benchmark):
        result = ra.rolling_beta(random_returns, benchmark, window=63)
        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self, ra, random_returns, benchmark):
        result = ra.rolling_beta(random_returns, benchmark, window=63)
        assert len(result) == len(random_returns)

    def test_nan_before_window(self, ra, random_returns, benchmark):
        window = 63
        result = ra.rolling_beta(random_returns, benchmark, window=window)
        # First (window-1) entries should be NaN
        assert result.iloc[:window - 1].isna().all()

    def test_non_nan_after_window(self, ra, random_returns, benchmark):
        window = 63
        result = ra.rolling_beta(random_returns, benchmark, window=window)
        assert result.iloc[window:].notna().any()

    def test_benchmark_vs_itself_beta_is_one(self, ra, benchmark):
        result = ra.rolling_beta(benchmark, benchmark, window=30)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-8)

    def test_handles_nan_gracefully(self, ra, returns_with_nans, benchmark):
        result = ra.rolling_beta(returns_with_nans, benchmark, window=63)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# factor_exposure
# ---------------------------------------------------------------------------

class TestFactorExposure:
    @pytest.fixture()
    def factors(self):
        np.random.seed(5)
        return pd.DataFrame(
            np.random.randn(N, 3) * 0.01,
            index=DATES,
            columns=["market", "size", "momentum"],
        )

    def test_returns_dict(self, ra, random_returns, factors):
        result = ra.factor_exposure(random_returns, factors)
        assert isinstance(result, dict)

    def test_contains_alpha(self, ra, random_returns, factors):
        result = ra.factor_exposure(random_returns, factors)
        assert "alpha" in result

    def test_contains_factor_betas(self, ra, random_returns, factors):
        result = ra.factor_exposure(random_returns, factors)
        for col in factors.columns:
            assert col in result

    def test_contains_r_squared(self, ra, random_returns, factors):
        result = ra.factor_exposure(random_returns, factors)
        assert "r_squared" in result

    def test_r_squared_between_zero_and_one(self, ra, random_returns, factors):
        r2 = ra.factor_exposure(random_returns, factors)["r_squared"]
        assert 0.0 <= r2 <= 1.0


# ---------------------------------------------------------------------------
# drawdown_analysis
# ---------------------------------------------------------------------------

class TestDrawdownAnalysis:
    def test_returns_dataframe(self, ra, random_returns):
        result = ra.drawdown_analysis(random_returns)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ra, random_returns):
        result = ra.drawdown_analysis(random_returns)
        expected = {"start", "trough", "recovery", "max_depth", "duration", "recovery_time"}
        assert expected.issubset(set(result.columns))

    def test_no_drawdowns_above_threshold_for_positive_returns(self, ra):
        """Always-positive returns produce no deep drawdown episodes."""
        r = pd.Series(0.001, index=DATES)
        result = ra.drawdown_analysis(r)
        assert len(result) == 0

    def test_max_depth_is_nonpositive(self, ra, random_returns):
        result = ra.drawdown_analysis(random_returns)
        if len(result) > 0:
            assert (result["max_depth"] <= 0).all()

    def test_only_large_drawdowns_included(self, ra, random_returns):
        """Only episodes with depth ≤ -5% are included."""
        result = ra.drawdown_analysis(random_returns)
        if len(result) > 0:
            assert (result["max_depth"] <= -0.05).all()

    def test_duration_is_positive(self, ra, random_returns):
        result = ra.drawdown_analysis(random_returns)
        if len(result) > 0:
            assert (result["duration"] > 0).all()

    def test_empty_returns_empty_dataframe(self, ra):
        result = ra.drawdown_analysis(pd.Series([], dtype=float))
        assert len(result) == 0

    def test_handles_nan_gracefully(self, ra, returns_with_nans):
        result = ra.drawdown_analysis(returns_with_nans)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# rolling_sharpe
# ---------------------------------------------------------------------------

class TestRollingSharpe:
    def test_returns_series(self, ra, random_returns):
        result = ra.rolling_sharpe(random_returns, window=63)
        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self, ra, random_returns):
        result = ra.rolling_sharpe(random_returns, window=63)
        assert len(result) == len(random_returns)

    def test_nan_before_warmup(self, ra, random_returns):
        window = 63
        result = ra.rolling_sharpe(random_returns, window=window)
        # Should be NaN before min_periods = window // 2
        # At least the very first row should be NaN
        assert result.isna().any()

    def test_finite_after_warmup(self, ra, random_returns):
        window = 63
        result = ra.rolling_sharpe(random_returns, window=window)
        valid = result.dropna()
        assert valid.apply(math.isfinite).all()

    def test_handles_nan_gracefully(self, ra, returns_with_nans):
        result = ra.rolling_sharpe(returns_with_nans, window=63)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# return_attribution
# ---------------------------------------------------------------------------

class TestReturnAttribution:
    @pytest.fixture()
    def positions(self):
        np.random.seed(7)
        raw = np.random.randn(N, N_ASSETS)
        pos = raw / np.abs(raw).sum(axis=1, keepdims=True)
        return pd.DataFrame(pos, index=DATES, columns=SYMS)

    @pytest.fixture()
    def asset_returns(self):
        np.random.seed(8)
        return pd.DataFrame(
            np.random.randn(N, N_ASSETS) * 0.01,
            index=DATES,
            columns=SYMS,
        )

    def test_returns_dict(self, ra, positions, asset_returns):
        result = ra.return_attribution(positions, asset_returns)
        assert isinstance(result, dict)

    def test_required_keys(self, ra, positions, asset_returns):
        result = ra.return_attribution(positions, asset_returns)
        required = {
            "long_contribution", "short_contribution", "total_contribution",
            "selection_contribution", "timing_contribution",
        }
        assert required.issubset(set(result))

    def test_total_equals_long_plus_short(self, ra, positions, asset_returns):
        result = ra.return_attribution(positions, asset_returns)
        assert abs(
            result["total_contribution"]
            - result["long_contribution"]
            - result["short_contribution"]
        ) < 1e-8

    def test_all_values_are_floats(self, ra, positions, asset_returns):
        result = ra.return_attribution(positions, asset_returns)
        for v in result.values():
            assert isinstance(v, float)
