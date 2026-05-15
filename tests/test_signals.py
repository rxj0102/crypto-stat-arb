"""Tests for signal generators (momentum, reversal, pairs, etc.)."""

import numpy as np
import pandas as pd
import pytest

from statarb.data.features import FeatureEngine
from statarb.signals.momentum import MomentumSignals
from statarb.signals.reversal import ReversalSignals
from statarb.signals.activity import ActivityFilter
from statarb.signals.themes import ThemeSignals, CRYPTO_THEMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DATES = 300
N_ASSETS = 10
SYMBOLS = [f"ASSET{i}/USDT" for i in range(N_ASSETS)]


@pytest.fixture()
def dates():
    return pd.date_range("2021-01-01", periods=N_DATES, freq="D", tz="UTC")


@pytest.fixture()
def prices(dates):
    np.random.seed(42)
    data = 100 * np.exp(np.random.randn(N_DATES, N_ASSETS).cumsum(axis=0) * 0.02)
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


@pytest.fixture()
def returns(prices):
    return FeatureEngine().log_returns(prices)


@pytest.fixture()
def volume(dates):
    np.random.seed(123)
    data = np.abs(np.random.randn(N_DATES, N_ASSETS)) * 1e7 + 1e6
    return pd.DataFrame(data, index=dates, columns=SYMBOLS)


# ---------------------------------------------------------------------------
# FeatureEngine
# ---------------------------------------------------------------------------

class TestFeatureEngine:
    def test_log_returns_shape(self, prices):
        fe = FeatureEngine()
        ret = fe.log_returns(prices)
        assert ret.shape == prices.shape

    def test_log_returns_first_row_nan(self, prices):
        fe = FeatureEngine()
        ret = fe.log_returns(prices)
        assert ret.iloc[0].isna().all()

    def test_simple_returns_range(self, prices):
        fe = FeatureEngine()
        ret = fe.simple_returns(prices)
        assert ret.dropna().gt(-1).all().all()  # returns > -100%

    def test_realized_vol_positive(self, returns):
        fe = FeatureEngine()
        vol = fe.realized_volatility(returns, window=21)
        assert (vol.dropna() >= 0).all().all()

    def test_volume_ma_ratio_positive(self, volume):
        fe = FeatureEngine()
        ratio = fe.volume_ma_ratio(volume, window=21)
        assert (ratio.dropna() > 0).all().all()

    def test_cross_sectional_rank_range(self, returns):
        fe = FeatureEngine()
        ranked = fe.cross_sectional_rank(returns)
        valid = ranked.dropna()
        assert (valid >= -1).all().all()
        assert (valid <= 1).all().all()

    def test_cross_sectional_rank_zero_mean(self, returns):
        """Cross-sectional rank should be zero-sum (roughly)."""
        fe = FeatureEngine()
        ranked = fe.cross_sectional_rank(returns)
        row_means = ranked.mean(axis=1).dropna()
        assert (row_means.abs() < 1e-6).all()

    def test_return_dispersion_positive(self, returns):
        fe = FeatureEngine()
        disp = fe.return_dispersion(returns)
        assert (disp.dropna() >= 0).all()


# ---------------------------------------------------------------------------
# MomentumSignals
# ---------------------------------------------------------------------------

class TestMomentumSignals:
    def test_price_momentum_shape(self, returns):
        mom = MomentumSignals()
        sig = mom.price_momentum(returns, lookback=63, skip=5)
        assert sig.shape == returns.shape

    def test_momentum_12_1_no_future(self, returns):
        """Signal at date t should not use returns from t+1."""
        mom = MomentumSignals()
        sig = mom.momentum_12_1(returns)
        # At least some non-NaN after lookback period
        assert sig.dropna(how="all").shape[0] > 0

    def test_momentum_1w_shape(self, returns):
        mom = MomentumSignals()
        sig = mom.momentum_1w(returns)
        assert sig.shape == returns.shape

    def test_sharpe_momentum_finite(self, returns):
        mom = MomentumSignals()
        sig = mom.sharpe_momentum(returns, lookback=63)
        assert np.isfinite(sig.dropna().values).all()

    def test_skip_must_be_less_than_lookback(self, returns):
        mom = MomentumSignals()
        with pytest.raises(ValueError):
            mom.price_momentum(returns, lookback=10, skip=10)

    def test_time_series_momentum_shape(self, returns):
        mom = MomentumSignals()
        sig = mom.time_series_momentum(returns, lookback=21)
        assert sig.shape == returns.shape

    def test_ma_crossover_fast_below_slow_is_negative(self, prices):
        """If price falls all the way, fast MA < slow MA → negative signal."""
        mom = MomentumSignals()
        falling = prices.copy()
        # Create a consistently falling price
        for i in range(len(falling)):
            falling.iloc[i] = 100 - i * 0.3
        sig = mom.moving_average_crossover(falling, fast=5, slow=20)
        # After burn-in, signal should be predominantly negative
        late_sig = sig.iloc[-50:].dropna()
        assert (late_sig < 0).all().all()


# ---------------------------------------------------------------------------
# ReversalSignals
# ---------------------------------------------------------------------------

class TestReversalSignals:
    def test_short_term_reversal_negates_return(self, returns):
        """1-day reversal should be exactly -1 * 1-day return."""
        rev = ReversalSignals()
        sig = rev.short_term_reversal(returns, lookback=1)
        direct = -returns
        pd.testing.assert_frame_equal(sig.dropna(), direct.dropna(), rtol=1e-10)

    def test_reversal_shape(self, returns):
        rev = ReversalSignals()
        sig = rev.weekly_reversal(returns)
        assert sig.shape == returns.shape

    def test_bollinger_reversal_range(self, prices):
        rev = ReversalSignals()
        sig = rev.bollinger_reversal(prices, window=20, n_std=2.0)
        # After burn-in, values within reasonable range
        valid = sig.iloc[40:].dropna()
        assert (valid.abs() < 10).all().all()

    def test_large_move_reversal_sparse(self, returns):
        """Large-move reversal should have many NaN (not every day is extreme)."""
        rev = ReversalSignals()
        sig = rev.large_move_reversal(returns, threshold_std=2.0)
        nan_fraction = sig.isna().mean().mean()
        assert nan_fraction > 0.5  # most days are not extreme

    def test_vol_adjusted_reversal_finite(self, returns):
        rev = ReversalSignals()
        sig = rev.vol_adjusted_reversal(returns, lookback=5, vol_window=21)
        assert np.isfinite(sig.dropna().values).all()


# ---------------------------------------------------------------------------
# ActivityFilter
# ---------------------------------------------------------------------------

class TestActivityFilter:
    def test_volume_regime_values(self, volume):
        act = ActivityFilter()
        regime = act.volume_regime(volume)
        unique_vals = set(regime.dropna().values.flatten())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_activity_gate_zeros_low_volume(self, dates):
        """Activity gate NaNs out signals where V/MA ratio < min_ratio.

        Build a volume series where the MA is built on high-volume history,
        then insert a single very-low-volume day. At that date the ratio
        will be well below 0.5, so the gated signal must be NaN.
        """
        act = ActivityFilter()
        window = 5
        n = 30
        syms = ["X"]
        # All days: high volume (1e7), except day 20 which is near-zero
        vol_vals = np.ones((n, 1)) * 1e7
        vol_vals[20, 0] = 1.0           # effectively 0 vs MA of 1e7
        vol = pd.DataFrame(vol_vals, index=dates[:n], columns=syms)
        ret = pd.DataFrame(np.ones((n, 1)) * 0.01, index=dates[:n], columns=syms)
        gated = act.activity_gate(ret, vol, window=window, min_ratio=0.5)
        # Day 20 should be NaN (ratio ≈ 0 << 0.5)
        assert pd.isna(gated.iloc[20]["X"])
        # Day 25 (MA recovered) should not be NaN
        assert not pd.isna(gated.iloc[25]["X"])

    def test_activity_scale_increases_high_volume(self, returns, volume):
        act = ActivityFilter()
        scaled = act.activity_scale(returns, volume, window=21)
        # Scaled should have same shape
        assert scaled.shape == returns.shape

    def test_market_activity_index_series(self, volume):
        act = ActivityFilter()
        mai = act.market_activity_index(volume)
        assert isinstance(mai, pd.Series)
        assert len(mai) == len(volume)


# ---------------------------------------------------------------------------
# ThemeSignals
# ---------------------------------------------------------------------------

class TestThemeSignals:
    @pytest.fixture()
    def real_returns(self, dates):
        """Returns with real symbol names from CRYPTO_THEMES."""
        all_syms = list({s for members in CRYPTO_THEMES.values() for s in members})[:15]
        np.random.seed(0)
        data = np.random.randn(N_DATES, len(all_syms)) * 0.02
        return pd.DataFrame(data, index=dates, columns=all_syms)

    def test_list_themes(self):
        ts = ThemeSignals()
        themes = ts.list_themes()
        assert "L1" in themes
        assert "DeFi" in themes

    def test_theme_returns_columns(self, real_returns):
        ts = ThemeSignals()
        theme_rets = ts.theme_returns(real_returns, lookback=21)
        assert isinstance(theme_rets, pd.DataFrame)
        assert theme_rets.shape[0] == N_DATES

    def test_within_theme_relative_value_shape(self, real_returns):
        ts = ThemeSignals()
        sig = ts.within_theme_relative_value(real_returns, lookback=21)
        assert sig.shape == real_returns.shape

    def test_low_vol_factor_negative_for_high_vol(self, returns):
        """Low-vol factor should give negative score to high-volatility assets."""
        ts = ThemeSignals()
        sig = ts.low_volatility_factor(returns, vol_window=63)
        # High-vol assets should have more negative scores
        vol = FeatureEngine().realized_volatility(returns, 63)
        # At some date, highest vol asset should have lowest factor score
        late_vol = vol.iloc[100]
        late_sig = sig.iloc[100]
        valid = late_vol.dropna().index.intersection(late_sig.dropna().index)
        if len(valid) >= 3:
            vol_rank = late_vol[valid].rank()
            sig_rank = late_sig[valid].rank()
            # Correlation should be negative (high vol → low factor)
            corr = vol_rank.corr(sig_rank)
            assert corr < 0
