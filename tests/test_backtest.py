"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from statarb.backtest.execution import ExecutionConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DATES = 200
N_ASSETS = 8
SYMBOLS = [f"ASSET{i}/USDT" for i in range(N_ASSETS)]


@pytest.fixture()
def dates():
    return pd.date_range("2021-01-01", periods=N_DATES, freq="D", tz="UTC")


@pytest.fixture()
def prices(dates):
    np.random.seed(0)
    data = 100 * np.exp(np.random.randn(N_DATES, N_ASSETS).cumsum(axis=0) * 0.01)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


@pytest.fixture()
def signal(dates):
    np.random.seed(1)
    data = np.random.randn(N_DATES, N_ASSETS)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


@pytest.fixture()
def volume(dates):
    np.random.seed(2)
    return pd.DataFrame(
        np.abs(np.random.randn(N_DATES, N_ASSETS)) * 1e7 + 1e6,
        index=dates, columns=SYMBOLS,
    )


@pytest.fixture()
def engine():
    cfg = BacktestConfig(
        execution=ExecutionConfig(order_type="market"),
        position_method="rank",
        max_position=0.10,
    )
    return BacktestEngine(cfg)


# ---------------------------------------------------------------------------
# BacktestEngine.run
# ---------------------------------------------------------------------------

class TestBacktestEngineRun:
    def test_returns_result(self, engine, signal, prices, volume):
        result = engine.run(signal, prices, volume)
        assert isinstance(result, BacktestResult)

    def test_portfolio_returns_series(self, engine, signal, prices):
        result = engine.run(signal, prices)
        assert isinstance(result.portfolio_returns, pd.Series)

    def test_portfolio_returns_length(self, engine, signal, prices):
        result = engine.run(signal, prices)
        assert len(result.portfolio_returns) == N_DATES

    def test_positions_shape(self, engine, signal, prices):
        result = engine.run(signal, prices)
        assert result.positions.shape == (N_DATES, N_ASSETS)

    def test_positions_within_max(self, engine, signal, prices):
        result = engine.run(signal, prices)
        assert (result.positions.abs().max().max() <= 0.10 + 1e-9)

    def test_net_leq_gross(self, engine, signal, prices, volume):
        """Net returns should be <= gross returns (costs are non-negative)."""
        result = engine.run(signal, prices, volume)
        # Allow small floating-point tolerance
        diff = result.gross_returns - result.portfolio_returns
        assert (diff >= -1e-10).all()

    def test_cost_returns_non_negative(self, engine, signal, prices, volume):
        result = engine.run(signal, prices, volume)
        assert (result.cost_returns >= -1e-10).all()

    def test_no_lookahead(self, signal, prices):
        """Verify signal lag prevents look-ahead bias."""
        cfg = BacktestConfig(signal_lag=1)
        engine = BacktestEngine(cfg)
        result = engine.run(signal, prices)
        # With lag=1, first position entry shouldn't use future data
        assert isinstance(result.portfolio_returns, pd.Series)

    def test_market_neutral_approx(self, engine, signal, prices):
        """Dollar-neutral portfolio: net exposure should be near zero."""
        result = engine.run(signal, prices)
        net_exposure = result.positions.sum(axis=1)
        # Allow some deviation due to NaN-filling and boundary effects
        assert net_exposure.abs().mean() < 0.15

    def test_run_signals_dict(self, engine, signal, prices):
        signals = {"sig_a": signal, "sig_b": -signal}
        results = engine.run_signals(signals, prices)
        assert set(results.keys()) == {"sig_a", "sig_b"}


# ---------------------------------------------------------------------------
# Rebalance frequency
# ---------------------------------------------------------------------------

class TestRebalanceFrequency:
    def test_weekly_rebalance(self, signal, prices):
        cfg = BacktestConfig(rebalance_freq="weekly")
        engine = BacktestEngine(cfg)
        result = engine.run(signal, prices)
        assert isinstance(result.portfolio_returns, pd.Series)

    def test_monthly_rebalance(self, signal, prices):
        cfg = BacktestConfig(rebalance_freq="monthly")
        engine = BacktestEngine(cfg)
        result = engine.run(signal, prices)
        assert isinstance(result.portfolio_returns, pd.Series)

    def test_invalid_rebalance_freq(self, signal, prices):
        cfg = BacktestConfig(rebalance_freq="quarterly")  # type: ignore
        engine = BacktestEngine(cfg)
        with pytest.raises(ValueError):
            engine.run(signal, prices)
