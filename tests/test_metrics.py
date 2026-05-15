"""Tests for PerformanceMetrics (stateless API)."""

import math

import numpy as np
import pandas as pd
import pytest

from statarb.evaluation.metrics import PerformanceMetrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 500
DATES = pd.date_range("2020-01-01", periods=N, freq="D")


@pytest.fixture()
def pm():
    return PerformanceMetrics()


@pytest.fixture()
def random_returns():
    np.random.seed(42)
    return pd.Series(np.random.randn(N) * 0.01, index=DATES)


@pytest.fixture()
def positive_returns():
    """Returns that are always strictly positive."""
    return pd.Series(0.001, index=DATES)


@pytest.fixture()
def negative_returns():
    """Returns that are always strictly negative."""
    return pd.Series(-0.001, index=DATES)


@pytest.fixture()
def returns_with_nans(random_returns):
    r = random_returns.copy()
    r.iloc[::10] = np.nan
    return r


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------

class TestAnnualizedReturn:
    def test_returns_float(self, pm, random_returns):
        assert isinstance(pm.annualized_return(random_returns), float)

    def test_positive_return_is_positive(self, pm, positive_returns):
        assert pm.annualized_return(positive_returns) > 0

    def test_negative_return_is_negative(self, pm, negative_returns):
        assert pm.annualized_return(negative_returns) < 0

    def test_empty_series_returns_zero(self, pm):
        assert pm.annualized_return(pd.Series([], dtype=float)) == 0.0

    def test_nan_only_returns_zero(self, pm):
        assert pm.annualized_return(pd.Series([np.nan, np.nan])) == 0.0

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.annualized_return(returns_with_nans)
        assert math.isfinite(result)

    def test_cagr_formula(self, pm):
        """Verify CAGR: known constant daily return over exactly 365 periods."""
        daily = 0.001
        r = pd.Series([daily] * 365)
        expected = float((1 + daily) ** 365 - 1)
        assert abs(pm.annualized_return(r) - expected) < 1e-10


# ---------------------------------------------------------------------------
# annualized_volatility
# ---------------------------------------------------------------------------

class TestAnnualizedVolatility:
    def test_returns_float(self, pm, random_returns):
        assert isinstance(pm.annualized_volatility(random_returns), float)

    def test_nonneg(self, pm, random_returns):
        assert pm.annualized_volatility(random_returns) >= 0.0

    def test_constant_returns_zero_vol(self, pm, positive_returns):
        assert pm.annualized_volatility(positive_returns) == 0.0

    def test_empty_returns_zero(self, pm):
        assert pm.annualized_volatility(pd.Series([], dtype=float)) == 0.0

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.annualized_volatility(returns_with_nans)
        assert math.isfinite(result) and result >= 0

    def test_scales_with_sqrt_periods(self, pm, random_returns):
        vol_365 = pm.annualized_volatility(random_returns, periods_per_year=365)
        vol_252 = pm.annualized_volatility(random_returns, periods_per_year=252)
        ratio = vol_365 / vol_252
        assert abs(ratio - math.sqrt(365 / 252)) < 1e-10


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_returns_float_or_inf(self, pm, random_returns):
        result = pm.sharpe_ratio(random_returns)
        assert isinstance(result, float)

    def test_constant_positive_return_is_inf(self, pm, positive_returns):
        """Sharpe of constant positive return = +inf."""
        assert pm.sharpe_ratio(positive_returns) == np.inf

    def test_constant_negative_return_is_neg_inf(self, pm, negative_returns):
        assert pm.sharpe_ratio(negative_returns) == -np.inf

    def test_empty_returns_zero(self, pm):
        assert pm.sharpe_ratio(pd.Series([], dtype=float)) == 0.0

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.sharpe_ratio(returns_with_nans)
        assert isinstance(result, float)

    def test_formula(self, pm, random_returns):
        """Sharpe = mean / std * sqrt(365) when rf=0."""
        clean = random_returns.dropna()
        expected = float(clean.mean() / clean.std() * math.sqrt(365))
        assert abs(pm.sharpe_ratio(random_returns) - expected) < 1e-10


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_returns_float_or_inf(self, pm, random_returns):
        result = pm.sortino_ratio(random_returns)
        assert isinstance(result, float)

    def test_all_positive_returns_is_inf(self, pm, positive_returns):
        """Sortino = +inf when there are no downside returns."""
        assert pm.sortino_ratio(positive_returns) == np.inf

    def test_empty_returns_zero(self, pm):
        assert pm.sortino_ratio(pd.Series([], dtype=float)) == 0.0

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.sortino_ratio(returns_with_nans)
        assert isinstance(result, float)

    def test_nonneg_for_positive_mean_series(self, pm):
        np.random.seed(0)
        r = pd.Series(np.abs(np.random.randn(N)) * 0.01, index=DATES)
        result = pm.sortino_ratio(r)
        assert result > 0 or result == np.inf


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_always_positive_returns_zero_drawdown(self, pm, positive_returns):
        """Max drawdown of always-positive returns = 0."""
        assert pm.max_drawdown(positive_returns) == 0.0

    def test_max_drawdown_is_nonpositive(self, pm, random_returns):
        assert pm.max_drawdown(random_returns) <= 0.0

    def test_empty_returns_zero(self, pm):
        assert pm.max_drawdown(pd.Series([], dtype=float)) == 0.0

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.max_drawdown(returns_with_nans)
        assert result <= 0.0

    def test_known_drawdown(self, pm):
        """50% single-period drop: MDD = -0.5."""
        r = pd.Series([0.0, -0.5, 0.0, 0.0])
        assert abs(pm.max_drawdown(r) - (-0.5)) < 1e-10

    def test_between_minus_one_and_zero(self, pm, random_returns):
        mdd = pm.max_drawdown(random_returns)
        assert -1.0 <= mdd <= 0.0


# ---------------------------------------------------------------------------
# max_drawdown_duration
# ---------------------------------------------------------------------------

class TestMaxDrawdownDuration:
    def test_no_drawdown_returns_zero(self, pm, positive_returns):
        assert pm.max_drawdown_duration(positive_returns) == 0

    def test_returns_int(self, pm, random_returns):
        assert isinstance(pm.max_drawdown_duration(random_returns), int)

    def test_nonneg(self, pm, random_returns):
        assert pm.max_drawdown_duration(random_returns) >= 0

    def test_empty_returns_zero(self, pm):
        assert pm.max_drawdown_duration(pd.Series([], dtype=float)) == 0

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.max_drawdown_duration(returns_with_nans)
        assert result >= 0


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_returns_float(self, pm, random_returns):
        assert isinstance(pm.calmar_ratio(random_returns), float)

    def test_no_drawdown_returns_zero(self, pm, positive_returns):
        assert pm.calmar_ratio(positive_returns) == 0.0

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.calmar_ratio(returns_with_nans)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_positive_is_one(self, pm, positive_returns):
        assert pm.win_rate(positive_returns) == 1.0

    def test_all_negative_is_zero(self, pm, negative_returns):
        assert pm.win_rate(negative_returns) == 0.0

    def test_between_zero_and_one(self, pm, random_returns):
        wr = pm.win_rate(random_returns)
        assert 0.0 <= wr <= 1.0

    def test_empty_returns_zero(self, pm):
        assert pm.win_rate(pd.Series([], dtype=float)) == 0.0

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.win_rate(returns_with_nans)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_all_positive_is_inf(self, pm, positive_returns):
        assert pm.profit_factor(positive_returns) == float("inf")

    def test_nonneg(self, pm, random_returns):
        assert pm.profit_factor(random_returns) >= 0.0

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.profit_factor(returns_with_nans)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# skewness and kurtosis
# ---------------------------------------------------------------------------

class TestSkewnessKurtosis:
    def test_skewness_returns_float(self, pm, random_returns):
        assert isinstance(pm.skewness(random_returns), float)

    def test_kurtosis_returns_float(self, pm, random_returns):
        assert isinstance(pm.kurtosis(random_returns), float)

    def test_symmetric_distribution_near_zero_skew(self, pm):
        np.random.seed(0)
        r = pd.Series(np.random.randn(10000))
        assert abs(pm.skewness(r)) < 0.1

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        assert isinstance(pm.skewness(returns_with_nans), float)
        assert isinstance(pm.kurtosis(returns_with_nans), float)


# ---------------------------------------------------------------------------
# tail_ratio
# ---------------------------------------------------------------------------

class TestTailRatio:
    def test_returns_float(self, pm, random_returns):
        assert isinstance(pm.tail_ratio(random_returns), float)

    def test_symmetric_near_one(self, pm):
        np.random.seed(1)
        r = pd.Series(np.random.randn(10000))
        assert abs(pm.tail_ratio(r) - 1.0) < 0.15

    def test_handles_nan_gracefully(self, pm, returns_with_nans):
        result = pm.tail_ratio(returns_with_nans)
        assert isinstance(result, float) and result > 0


# ---------------------------------------------------------------------------
# full_report
# ---------------------------------------------------------------------------

class TestFullReport:
    def test_returns_dataframe(self, pm, random_returns):
        report = pm.full_report(random_returns)
        assert isinstance(report, pd.DataFrame)

    def test_has_value_column(self, pm, random_returns):
        report = pm.full_report(random_returns)
        assert "value" in report.columns

    def test_required_metrics_present(self, pm, random_returns):
        report = pm.full_report(random_returns)
        required = {
            "annualized_return", "annualized_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "max_drawdown_duration",
            "calmar_ratio", "win_rate", "profit_factor",
            "skewness", "kurtosis", "tail_ratio",
        }
        assert required.issubset(set(report.index))

    def test_with_benchmark_adds_alpha_beta(self, pm, random_returns):
        np.random.seed(7)
        bench = pd.Series(np.random.randn(N) * 0.01, index=DATES)
        report = pm.full_report(random_returns, benchmark_returns=bench)
        assert "alpha" in report.index
        assert "beta" in report.index

    def test_handles_nan_returns(self, pm, returns_with_nans):
        report = pm.full_report(returns_with_nans)
        assert isinstance(report, pd.DataFrame)
        assert "sharpe_ratio" in report.index

    def test_all_positive_returns_zero_drawdown(self, pm, positive_returns):
        report = pm.full_report(positive_returns)
        assert report.loc["max_drawdown", "value"] == 0.0

    def test_constant_positive_sharpe_inf(self, pm, positive_returns):
        report = pm.full_report(positive_returns)
        assert report.loc["sharpe_ratio", "value"] == np.inf


# ---------------------------------------------------------------------------
# compare (static method)
# ---------------------------------------------------------------------------

class TestCompare:
    def test_returns_dataframe(self, random_returns):
        np.random.seed(1)
        results = {
            "A": random_returns,
            "B": pd.Series(np.random.randn(N) * 0.01, index=DATES),
        }
        df = PerformanceMetrics.compare(results)
        assert isinstance(df, pd.DataFrame)
        assert "A" in df.columns and "B" in df.columns

    def test_columns_are_strategy_names(self, random_returns):
        np.random.seed(2)
        results = {"X": random_returns}
        df = PerformanceMetrics.compare(results)
        assert list(df.columns) == ["X"]

    def test_index_contains_sharpe(self, random_returns):
        df = PerformanceMetrics.compare({"S": random_returns})
        assert "sharpe_ratio" in df.index
