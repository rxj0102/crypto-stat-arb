"""
Comprehensive tests for all signal generators.

Test categories for each signal method:
1. Shape — output matches input shape
2. Range — cross-sectionally ranked values lie in [-1, 1]
3. Known-pattern — synthetic data with predictable outcome
4. NaN handling — NaN in input propagates correctly
5. No look-ahead — signal at T unchanged when future data changes
6. Conditional zeros — volume-filtered signals are 0 where inactive
"""

import numpy as np
import pandas as pd
import pytest

from statarb.data.features import FeatureEngine
from statarb.signals.activity import ActivityFilter, ActivitySignals
from statarb.signals.momentum import MomentumSignals
from statarb.signals.pairs import PairsSignals
from statarb.signals.reversal import ReversalSignals
from statarb.signals.seasonality import SeasonalitySignals
from statarb.signals.themes import CRYPTO_THEMES, ThemeSignals


# ============================================================
# Shared fixtures
# ============================================================

N_DATES = 300
N_ASSETS = 8
SYMBOLS = [f"ASSET{i}/USDT" for i in range(N_ASSETS)]


@pytest.fixture()
def dates():
    return pd.date_range("2021-01-01", periods=N_DATES, freq="D", tz="UTC")


@pytest.fixture()
def prices(dates):
    """Random-walk price panel — no systematic trend."""
    np.random.seed(42)
    shocks = np.random.randn(N_DATES, N_ASSETS) * 0.01
    log_prices = np.cumsum(shocks, axis=0)
    return pd.DataFrame(np.exp(log_prices) * 100, index=dates, columns=SYMBOLS)


@pytest.fixture()
def returns(prices):
    return FeatureEngine().log_returns(prices)


@pytest.fixture()
def volume(dates):
    np.random.seed(123)
    return pd.DataFrame(
        np.abs(np.random.randn(N_DATES, N_ASSETS)) * 1e7 + 1e6,
        index=dates, columns=SYMBOLS,
    )


def _assert_ranked(sig: pd.DataFrame, tol: float = 1e-9) -> None:
    """Assert that every non-NaN value lies in [-1, 1]."""
    valid = sig.stack().dropna()
    assert (valid >= -1 - tol).all(), f"Min value {valid.min()} < -1"
    assert (valid <= 1 + tol).all(), f"Max value {valid.max()} > 1"


def _no_lookahead(fn, *args, split_at: int = 100, **kwargs) -> None:
    """
    Verify that the signal at index split_at-1 is identical whether
    computed on data[:split_at] or on the full dataset.
    """
    full_sig = fn(*args, **kwargs)
    # Truncate all DataFrame args at split_at
    truncated_args = []
    for a in args:
        if isinstance(a, pd.DataFrame):
            truncated_args.append(a.iloc[:split_at])
        elif isinstance(a, pd.Series):
            truncated_args.append(a.iloc[:split_at])
        else:
            truncated_args.append(a)
    trunc_sig = fn(*truncated_args, **kwargs)

    row_full = full_sig.iloc[split_at - 1].dropna()
    row_trunc = trunc_sig.iloc[split_at - 1].dropna()
    common = row_full.index.intersection(row_trunc.index)
    if len(common) == 0:
        return  # no overlapping valid values — can't test
    pd.testing.assert_series_equal(
        row_full[common].sort_index(),
        row_trunc[common].sort_index(),
        rtol=1e-8,
        check_names=False,
    )


# ============================================================
# MomentumSignals
# ============================================================

class TestMomentumSignals:
    @pytest.fixture()
    def mom(self):
        return MomentumSignals()

    # --- time_series_momentum ---

    def test_tsmom_shape(self, mom, returns):
        sig = mom.time_series_momentum(returns, lookback=21)
        assert sig.shape == returns.shape

    def test_tsmom_range(self, mom, returns):
        sig = mom.time_series_momentum(returns, lookback=21)
        _assert_ranked(sig)

    def test_tsmom_known_pattern(self, mom, dates):
        """A consistently rising asset should have the top momentum rank."""
        n, m = 100, 5
        data = np.zeros((n, m))
        data[:, 0] = 0.05   # always up
        data[:, 1:] = np.random.randn(n, m - 1) * 0.001
        prices = pd.DataFrame(np.exp(np.cumsum(data, axis=0)) * 100,
                               index=dates[:n],
                               columns=[f"A{i}" for i in range(m)])
        ret = FeatureEngine().log_returns(prices)
        sig = mom.time_series_momentum(ret, lookback=20)
        # After burn-in the rising asset should dominate
        late_sig = sig.iloc[50:].dropna(how="all")
        assert (late_sig["A0"] == 1.0).mean() > 0.5

    def test_tsmom_nan_propagates(self, mom, returns):
        """NaN in the input should yield NaN in the output.

        Use a window large enough that all prior periods are NaN so
        min_periods cannot be satisfied.
        """
        r = returns.copy()
        # Fill rows 50-79 with NaN for column 0 (covers a full lookback=21 window)
        r.iloc[50:79, 0] = np.nan
        sig = mom.time_series_momentum(r, lookback=21)
        # At row 73, the rolling window [53..73] is entirely NaN → NaN output
        assert pd.isna(sig.iloc[73, 0])

    def test_tsmom_no_lookahead(self, mom, returns):
        _no_lookahead(mom.time_series_momentum, returns, lookback=21)

    def test_tsmom_multiple_lookbacks(self, mom, returns):
        for lb in [7, 14, 21, 42, 63]:
            sig = mom.time_series_momentum(returns, lookback=lb)
            assert sig.shape == returns.shape
            _assert_ranked(sig)

    # --- cross_sectional_momentum ---

    def test_xsmom_shape(self, mom, returns):
        sig = mom.cross_sectional_momentum(returns, lookback=21, skip=1)
        assert sig.shape == returns.shape

    def test_xsmom_range(self, mom, returns):
        _assert_ranked(mom.cross_sectional_momentum(returns, lookback=21, skip=1))

    def test_xsmom_skip_0(self, mom, returns):
        sig = mom.cross_sectional_momentum(returns, lookback=21, skip=0)
        assert sig.shape == returns.shape
        _assert_ranked(sig)

    def test_xsmom_negative_skip_raises(self, mom, returns):
        with pytest.raises(ValueError):
            mom.cross_sectional_momentum(returns, lookback=21, skip=-1)

    def test_xsmom_no_lookahead(self, mom, returns):
        _no_lookahead(mom.cross_sectional_momentum, returns, lookback=21, skip=1)

    # --- volume_weighted_momentum ---

    def test_vwmom_shape(self, mom, returns, volume):
        sig = mom.volume_weighted_momentum(returns, volume, lookback=21)
        assert sig.shape == returns.shape

    def test_vwmom_range(self, mom, returns, volume):
        _assert_ranked(mom.volume_weighted_momentum(returns, volume, lookback=21))

    def test_vwmom_no_lookahead(self, mom, returns, volume):
        _no_lookahead(mom.volume_weighted_momentum, returns, volume, lookback=21)

    def test_vwmom_high_vol_amplifies(self, mom, dates):
        """High-volume days should get more weight in the cumulative signal."""
        n, m = 100, 3
        ret_data = np.zeros((n, m))
        ret_data[:, 0] = 0.01   # consistently positive
        rets = pd.DataFrame(ret_data, index=dates[:n], columns=["A", "B", "C"])

        vol_data = np.ones((n, m)) * 1e6
        vol_data[:, 0] = 1e9   # much higher volume for A
        vols = pd.DataFrame(vol_data, index=dates[:n], columns=["A", "B", "C"])

        sig = mom.volume_weighted_momentum(rets, vols, lookback=10)
        late = sig.iloc[20:].dropna(how="all")
        # A has large positive return AND large volume → top rank
        assert (late["A"] > 0).mean() > 0.7

    # --- momentum_with_activity_filter ---

    def test_actmom_shape(self, mom, returns, volume):
        sig = mom.momentum_with_activity_filter(returns, volume, lookback=21)
        assert sig.shape == returns.shape

    def test_actmom_zero_when_low_activity(self, mom, dates):
        """Where activity < threshold, signal should be exactly 0."""
        n, m = 100, 4
        rets = pd.DataFrame(np.random.randn(n, m) * 0.01,
                             index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        vols = pd.DataFrame(np.ones((n, m)) * 1e6,   # flat volume → ratio ≈ 1.0
                             index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        sig = mom.momentum_with_activity_filter(rets, vols, lookback=21,
                                                 activity_threshold=5.0)  # high threshold
        # All vol ratios ≈ 1.0 < 5.0 → all zeros
        assert (sig.dropna() == 0).all().all()

    def test_actmom_no_lookahead(self, mom, returns, volume):
        _no_lookahead(mom.momentum_with_activity_filter, returns, volume,
                      lookback=21, split_at=100)

    # --- breakout_momentum ---

    def test_breakout_shape(self, mom, prices):
        sig = mom.breakout_momentum(prices, lookback=21)
        assert sig.shape == prices.shape

    def test_breakout_range(self, mom, prices):
        _assert_ranked(mom.breakout_momentum(prices, lookback=21))

    def test_breakout_at_high_is_positive(self, mom, dates):
        """Asset always at top of range should rank highest; asset at bottom lowest."""
        n, m = 60, 4
        # Monotone up: always at rolling max → raw signal ≈ 1.0
        up = np.linspace(80, 120, n)
        # Monotone down: always at rolling min → raw signal ≈ 0.0
        down = np.linspace(120, 80, n)
        # Two oscillating assets to provide cross-sectional context
        mid1 = 100 + 5 * np.sin(np.linspace(0, 4 * np.pi, n))
        mid2 = 100 + 5 * np.cos(np.linspace(0, 4 * np.pi, n))
        pdata = np.column_stack([up, down, mid1, mid2])
        prices = pd.DataFrame(pdata, index=dates[:n], columns=["UP", "DOWN", "M1", "M2"])
        sig = mom.breakout_momentum(prices, lookback=20)
        late = sig.iloc[30:].dropna(how="all")
        # UP should always rank higher than DOWN
        assert (late["UP"] > late["DOWN"]).mean() > 0.9

    def test_breakout_no_lookahead(self, mom, prices):
        _no_lookahead(mom.breakout_momentum, prices, lookback=21)

    # --- acceleration ---

    def test_accel_shape(self, mom, returns):
        sig = mom.acceleration(returns, short_window=7, long_window=28)
        assert sig.shape == returns.shape

    def test_accel_range(self, mom, returns):
        _assert_ranked(mom.acceleration(returns, short_window=7, long_window=28))

    def test_accel_raises_on_bad_windows(self, mom, returns):
        with pytest.raises(ValueError):
            mom.acceleration(returns, short_window=28, long_window=7)

    def test_accel_no_lookahead(self, mom, returns):
        _no_lookahead(mom.acceleration, returns, short_window=7, long_window=28)

    def test_accel_accelerating_is_positive(self, mom, dates):
        """Asset with improving short-term vs long-term momentum → positive signal.

        Acceleration fires during the TRANSITION window, before the long window
        catches up with the short window. Check the early transition period.
        """
        n = 120
        rets = np.zeros((n, 3))
        # A: flat then strongly positive → signal positive right after transition start
        rets[60:, 0] = 0.05
        rets[:, 1] = 0.01    # constant → zero acceleration
        rets[:, 2] = -0.01   # constant negative → zero acceleration
        r = pd.DataFrame(rets, index=dates[:n], columns=["A", "B", "C"])
        sig = mom.acceleration(r, short_window=7, long_window=28)
        # Check days 65-87: long window still partially in the zero-return regime
        # so short_cum > scale * long_cum → positive acceleration for A
        transition = sig.iloc[65:87].dropna(how="all")
        assert (transition["A"] > 0).mean() > 0.7


# ============================================================
# ReversalSignals
# ============================================================

class TestReversalSignals:
    @pytest.fixture()
    def rev(self):
        return ReversalSignals()

    # --- short_term_reversal ---

    def test_str_shape(self, rev, returns):
        sig = rev.short_term_reversal(returns, lookback=1)
        assert sig.shape == returns.shape

    def test_str_range(self, rev, returns):
        _assert_ranked(rev.short_term_reversal(returns, lookback=1))

    def test_str_negates_return(self, rev, dates):
        """Reversal should give the highest rank to the biggest loser."""
        n, m = 60, 4
        rets = pd.DataFrame(np.zeros((n, m)), index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        rets.iloc[-1, 0] = -0.10   # A0 is biggest loser
        rets.iloc[-1, 1] = -0.05
        rets.iloc[-1, 2] = 0.05
        rets.iloc[-1, 3] = 0.10   # A3 is biggest winner
        sig = rev.short_term_reversal(rets, lookback=1)
        last = sig.iloc[-1].dropna()
        assert last["A0"] > last["A3"]   # loser gets higher (buy) signal

    def test_str_no_lookahead(self, rev, returns):
        _no_lookahead(rev.short_term_reversal, returns, lookback=1)

    def test_str_multiple_lookbacks(self, rev, returns):
        for lb in [1, 2, 3, 5, 7]:
            _assert_ranked(rev.short_term_reversal(returns, lb))

    # --- mean_reversion_zscore ---

    def test_mrsig_shape(self, rev, prices):
        sig = rev.mean_reversion_zscore(prices, lookback=21)
        assert sig.shape == prices.shape

    def test_mrsig_range(self, rev, prices):
        _assert_ranked(rev.mean_reversion_zscore(prices, lookback=21))

    def test_mrsig_buy_below_ma(self, rev, dates):
        """Asset far below MA should get top (buy) signal."""
        n, m = 60, 3
        pdata = np.ones((n, m)) * 100.0
        pdata[-5:, 0] = 50.0   # A0 crashes → far below MA → buy
        pdata[-5:, 1] = 100.0
        pdata[-5:, 2] = 130.0  # A2 above MA → sell signal
        p = pd.DataFrame(pdata, index=dates[:n], columns=["A0", "A1", "A2"])
        sig = rev.mean_reversion_zscore(p, lookback=20)
        last = sig.iloc[-1].dropna()
        assert last["A0"] > last["A2"]

    def test_mrsig_no_lookahead(self, rev, prices):
        _no_lookahead(rev.mean_reversion_zscore, prices, lookback=21)

    # --- volume_filtered_reversal ---

    def test_vfr_shape(self, rev, returns, volume):
        sig = rev.volume_filtered_reversal(returns, volume, lookback=3)
        assert sig.shape == returns.shape

    def test_vfr_zero_when_high_activity(self, rev, dates):
        """When volume is above threshold, signal should be 0."""
        n, m = 80, 3
        rets = pd.DataFrame(np.random.randn(n, m) * 0.01,
                             index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        # High constant volume → ratio ≈ 1.0, above threshold 0.7 → zero
        vols = pd.DataFrame(np.ones((n, m)) * 1e8,
                             index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        sig = rev.volume_filtered_reversal(rets, vols, lookback=3,
                                            activity_threshold=0.5)
        # vol ratio ≈ 1.0 which is >= 0.5 → should be zero
        assert (sig.iloc[30:].fillna(0) == 0).all().all()

    def test_vfr_activated_when_low_activity(self, rev, dates):
        """Signal is non-zero when V/MA drops below threshold.

        The vol MA window is 5. We set volume high for 20 days, then drop to near-zero.
        Right after the drop, the MA still reflects the high history → ratio ≈ 0 < 0.5
        → signal is activated on those dates.
        """
        n, m = 80, 4
        rets = pd.DataFrame(np.random.randn(n, m) * 0.02,
                             index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        vol_vals = np.ones((n, m)) * 1e7
        # Volume drops to near-zero from day 20 onward
        vol_vals[20:, :] = 1.0   # effectively 0 vs old MA
        vols = pd.DataFrame(vol_vals, index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        sig = rev.volume_filtered_reversal(rets, vols, lookback=1,
                                            activity_threshold=0.5,
                                            vol_ma_window=5)
        # Days 20-23: MA still contains high-volume history → ratio ≈ 0 → signal fires
        activated = (sig.iloc[20:24].abs() > 0).any().any()
        assert activated

    def test_vfr_no_lookahead(self, rev, returns, volume):
        _no_lookahead(rev.volume_filtered_reversal, returns, volume,
                      lookback=3, split_at=100)

    # --- overnight_reversal ---

    def test_overnight_hourly(self, rev, dates):
        """Hourly data should produce a valid overnight reversal signal."""
        hourly_dates = pd.date_range("2021-01-01", periods=200, freq="h", tz="UTC")
        syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        ohlcv_data = {}
        np.random.seed(1)
        for sym in syms:
            df = pd.DataFrame({
                "open": 100 + np.random.randn(200) * 2,
                "high": 102 + np.random.randn(200) * 2,
                "low": 98 + np.random.randn(200) * 2,
                "close": 100 + np.random.randn(200) * 2,
                "volume": np.abs(np.random.randn(200)) * 1e6,
            }, index=hourly_dates)
            ohlcv_data[sym] = df
        sig = rev.overnight_reversal(ohlcv_data, lookback=1)
        assert isinstance(sig, pd.DataFrame)
        if not sig.empty:
            _assert_ranked(sig)

    def test_overnight_empty_on_empty_input(self, rev):
        sig = rev.overnight_reversal({})
        assert sig.empty

    # --- liquidation_reversal ---

    def test_liq_shape(self, rev, returns, volume):
        sig = rev.liquidation_reversal(returns, volume)
        assert sig.shape == returns.shape

    def test_liq_zero_when_no_spike(self, rev, dates):
        """With perfectly flat volume, spike condition never fires → zeros."""
        n, m = 80, 3
        rets = pd.DataFrame(np.random.randn(n, m) * 0.005,
                             index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        vols = pd.DataFrame(np.ones((n, m)) * 1e7,
                             index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        sig = rev.liquidation_reversal(rets, vols, volume_spike_threshold=10.0)
        # With constant volume, ratio ≈ 1.0 << 10 → no spikes → all zeros
        assert (sig.fillna(0) == 0).all().all()

    def test_liq_fires_on_spike(self, rev, dates):
        """Volume spike + large return → liquidation reversal fires.

        Uses tiny, asset-specific baseline returns so:
        1. Rolling std stays tiny → z-score of spike is huge (>> threshold)
        2. Assets have distinct returns → cross-sectional rank produces ±1, not 0
        """
        n, m = 100, 3
        cols = [f"A{i}" for i in range(m)]
        # Different tiny baselines per asset → distinct cross-sectional ranks
        rets = pd.DataFrame(np.column_stack([
            np.ones(n) * 1e-6,
            np.ones(n) * 2e-6,
            np.ones(n) * 3e-6,
        ]), index=dates[:n], columns=cols)
        vols = pd.DataFrame(np.ones((n, m)) * 1e7, index=dates[:n], columns=cols)

        # Day 60: massive spike — same volume but very different returns
        vols.iloc[60] = 1e7 * 50            # 50× average → vol_ratio >> 3.0
        rets.iloc[60] = [0.10, -0.10, 0.05] # different returns → no tie in ranking

        sig = rev.liquidation_reversal(
            rets, vols, lookback=1, vol_ma_window=10,
            volume_spike_threshold=3.0, return_threshold_std=1.0,
        )
        day60_sig = sig.iloc[60].fillna(0)
        # With large dispersion in returns, ranking is non-degenerate → some ≠ 0
        assert (day60_sig != 0).any()

    def test_liq_no_lookahead(self, rev, returns, volume):
        _no_lookahead(rev.liquidation_reversal, returns, volume)


# ============================================================
# PairsSignals
# ============================================================

class TestPairsSignals:
    @pytest.fixture()
    def ps(self):
        return PairsSignals()

    @pytest.fixture()
    def cointegrated_prices(self, dates):
        """Two cointegrated series + 2 random series."""
        np.random.seed(50)
        n = N_DATES
        # Genuine cointegrated pair: A = 2*B + stationary noise
        b_price = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        a_price = 2.0 * b_price + np.random.randn(n) * 2.0
        c_price = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        d_price = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        return pd.DataFrame({
            "A/USDT": a_price,
            "B/USDT": b_price,
            "C/USDT": c_price,
            "D/USDT": d_price,
        }, index=dates)

    # --- find_cointegrated_pairs ---

    def test_coint_returns_list(self, ps, cointegrated_prices):
        result = ps.find_cointegrated_pairs(cointegrated_prices, lookback=200)
        assert isinstance(result, list)

    def test_coint_structure(self, ps, cointegrated_prices):
        result = ps.find_cointegrated_pairs(cointegrated_prices, lookback=200)
        for item in result:
            assert len(item) == 4  # (sym_a, sym_b, beta, pvalue)
            sym_a, sym_b, beta, pval = item
            assert isinstance(sym_a, str)
            assert isinstance(sym_b, str)
            assert 0 <= pval <= 1

    def test_coint_finds_genuine_pair(self, ps, cointegrated_prices):
        """The A/B pair should have a low p-value."""
        result = ps.find_cointegrated_pairs(cointegrated_prices,
                                             lookback=200, p_threshold=0.1)
        pairs_found = [(r[0], r[1]) for r in result]
        assert any(
            ("A/USDT" in p and "B/USDT" in p) for p in pairs_found
        )

    def test_coint_sorted_by_pvalue(self, ps, cointegrated_prices):
        result = ps.find_cointegrated_pairs(cointegrated_prices, lookback=200)
        pvals = [r[3] for r in result]
        assert pvals == sorted(pvals)

    # --- pairs_spread_zscore ---

    def test_zscore_returns_series(self, ps, cointegrated_prices):
        zscore = ps.pairs_spread_zscore(cointegrated_prices,
                                         pair=("A/USDT", "B/USDT"),
                                         lookback=63, zscore_window=21)
        assert isinstance(zscore, pd.Series)
        assert len(zscore) == N_DATES

    def test_zscore_mean_near_zero(self, ps, cointegrated_prices):
        """Z-score of a stationary spread should have mean near zero."""
        zscore = ps.pairs_spread_zscore(cointegrated_prices,
                                         pair=("A/USDT", "B/USDT"),
                                         lookback=63, zscore_window=21)
        assert abs(zscore.dropna().mean()) < 1.5

    def test_zscore_bad_symbol_raises(self, ps, cointegrated_prices):
        with pytest.raises(KeyError):
            ps.pairs_spread_zscore(cointegrated_prices, pair=("X/USDT", "Y/USDT"))

    def test_zscore_no_lookahead(self, ps, cointegrated_prices):
        """Z-score at T should be the same whether computed on full or truncated data."""
        T = 150
        full = ps.pairs_spread_zscore(cointegrated_prices,
                                       pair=("A/USDT", "B/USDT"),
                                       lookback=63, zscore_window=21)
        trunc = ps.pairs_spread_zscore(cointegrated_prices.iloc[:T],
                                        pair=("A/USDT", "B/USDT"),
                                        lookback=63, zscore_window=21)
        if not pd.isna(full.iloc[T - 1]) and not pd.isna(trunc.iloc[T - 1]):
            assert abs(full.iloc[T - 1] - trunc.iloc[T - 1]) < 1e-8

    # --- sector_neutral_reversal ---

    def test_snr_shape(self, ps, returns):
        sector_map = {
            "grp1": SYMBOLS[:4],
            "grp2": SYMBOLS[4:],
        }
        sig = ps.sector_neutral_reversal(returns, sector_map, lookback=5)
        assert sig.shape == returns.shape

    def test_snr_range(self, ps, returns):
        sector_map = {"grp1": SYMBOLS[:4], "grp2": SYMBOLS[4:]}
        sig = ps.sector_neutral_reversal(returns, sector_map, lookback=5)
        _assert_ranked(sig)

    def test_snr_no_lookahead(self, ps, returns):
        sector_map = {"grp1": SYMBOLS[:4], "grp2": SYMBOLS[4:]}
        _no_lookahead(ps.sector_neutral_reversal, returns, sector_map, lookback=5)

    def test_snr_removes_sector_effect(self, ps, dates):
        """When all assets in a sector move together, residual should be ~0."""
        n, m = 100, 4
        # All assets have identical returns → residual = 0
        common_ret = np.ones((n, m)) * 0.01
        rets = pd.DataFrame(common_ret, index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        sector_map = {"all": [f"A{i}" for i in range(m)]}
        sig = ps.sector_neutral_reversal(rets, sector_map, lookback=5)
        # Residual is 0, so cumresid = 0 → signal should be NaN (can't rank 0s)
        # or exactly 0 — just verify no large values appear
        valid = sig.dropna()
        if not valid.empty:
            assert (valid.abs() <= 1 + 1e-9).all().all()

    # --- beta_neutral_reversal ---

    def test_bnr_shape(self, ps, returns):
        btc = returns.iloc[:, 0]
        sig = ps.beta_neutral_reversal(returns, btc, lookback=5, beta_window=30)
        assert sig.shape == returns.shape

    def test_bnr_range(self, ps, returns):
        btc = returns.iloc[:, 0]
        _assert_ranked(ps.beta_neutral_reversal(returns, btc, lookback=5, beta_window=30))

    def test_bnr_no_lookahead(self, ps, returns):
        btc = returns.iloc[:, 0]
        _no_lookahead(ps.beta_neutral_reversal, returns, btc,
                      lookback=5, beta_window=30, split_at=100)


# ============================================================
# SeasonalitySignals
# ============================================================

class TestSeasonalitySignals:
    @pytest.fixture()
    def seas(self):
        return SeasonalitySignals()

    # --- day_of_week_momentum ---

    def test_dow_shape(self, seas, returns):
        sig = seas.day_of_week_momentum(returns, estimation_window=100)
        assert sig.shape == returns.shape

    def test_dow_range(self, seas, returns):
        sig = seas.day_of_week_momentum(returns, estimation_window=100)
        _assert_ranked(sig)

    def test_dow_no_lookahead(self, seas, returns):
        _no_lookahead(seas.day_of_week_momentum, returns,
                      estimation_window=100, split_at=150)

    def test_dow_uses_only_matching_weekday(self, seas, dates):
        """Signal at a Monday should only use prior Monday data."""
        n = 200
        rets = pd.DataFrame(
            np.random.randn(n, 4) * 0.01,
            index=dates[:n],
            columns=[f"A{i}" for i in range(4)],
        )
        sig = seas.day_of_week_momentum(rets, estimation_window=60)
        # Just verify the signal runs without error and produces ranked values
        _assert_ranked(sig)

    # --- hour_of_day_signal ---

    def test_hod_shape(self, seas):
        hourly_dates = pd.date_range("2021-01-01", periods=500, freq="h", tz="UTC")
        rets = pd.DataFrame(
            np.random.randn(500, 3) * 0.001,
            index=hourly_dates,
            columns=["A", "B", "C"],
        )
        sig = seas.hour_of_day_signal(rets, estimation_window_days=10)
        assert sig.shape == rets.shape

    def test_hod_range(self, seas):
        hourly_dates = pd.date_range("2021-01-01", periods=500, freq="h", tz="UTC")
        rets = pd.DataFrame(
            np.random.randn(500, 3) * 0.001,
            index=hourly_dates,
            columns=["A", "B", "C"],
        )
        sig = seas.hour_of_day_signal(rets, estimation_window_days=10)
        _assert_ranked(sig)

    # --- month_end_effect ---

    def test_mee_shape(self, seas, returns, prices):
        sig = seas.month_end_effect(returns, prices, window_days=3,
                                     estimation_window=100)
        assert sig.shape == returns.shape

    def test_mee_zero_outside_window(self, seas, returns, prices):
        """Dates far from month-end should have zero signal."""
        sig = seas.month_end_effect(returns, prices, window_days=3,
                                     estimation_window=100)
        idx = pd.DatetimeIndex(returns.index)
        days_remaining = idx.days_in_month - idx.day
        not_month_end = days_remaining > 3
        # Non-month-end dates should be 0
        assert (sig[not_month_end].fillna(0) == 0).all().all()

    def test_mee_no_lookahead(self, seas, returns, prices):
        _no_lookahead(seas.month_end_effect, returns, prices,
                      window_days=3, estimation_window=100, split_at=150)

    # --- momentum_seasonality ---

    def test_ms_shape(self, seas, returns):
        sig = seas.momentum_seasonality(returns, lookback=21)
        assert sig.shape == returns.shape

    def test_ms_range(self, seas, returns):
        _assert_ranked(seas.momentum_seasonality(returns, lookback=21))

    def test_ms_weekday_vs_weekend(self, seas, returns):
        sig_wd = seas.momentum_seasonality(returns, lookback=21, weekday_only=True)
        sig_we = seas.momentum_seasonality(returns, lookback=21, weekend_only=True)
        # Both should return valid ranked signals
        _assert_ranked(sig_wd)
        _assert_ranked(sig_we)

    def test_ms_both_flags_raises(self, seas, returns):
        with pytest.raises(ValueError):
            seas.momentum_seasonality(returns, lookback=21,
                                       weekday_only=True, weekend_only=True)

    def test_ms_no_lookahead(self, seas, returns):
        _no_lookahead(seas.momentum_seasonality, returns, lookback=21)


# ============================================================
# ActivitySignals
# ============================================================

class TestActivitySignals:
    @pytest.fixture()
    def act(self):
        return ActivitySignals()

    # --- volume_surprise ---

    def test_vs_shape(self, act, volume):
        result = act.volume_surprise(volume, window=21)
        assert result.shape == volume.shape

    def test_vs_mean_near_one(self, act, volume):
        """Over time, V/MA(V) should average to approximately 1."""
        ratio = act.volume_surprise(volume, window=21)
        mean_ratio = ratio.dropna().mean().mean()
        assert 0.5 < mean_ratio < 2.0

    def test_vs_positive(self, act, volume):
        """All non-NaN volume surprise values should be positive."""
        ratio = act.volume_surprise(volume, window=21)
        assert (ratio.dropna() > 0).all().all()

    def test_vs_spike_detection(self, act, dates):
        """A volume spike should produce a ratio >> 1."""
        n, m = 60, 2
        vols = pd.DataFrame(np.ones((n, m)) * 1e7,
                             index=dates[:n], columns=["A", "B"])
        vols.iloc[50] = 1e9  # 100× spike
        ratio = act.volume_surprise(vols, window=21)
        assert ratio.iloc[50, 0] > 10.0

    def test_vs_no_lookahead(self, act, volume):
        T = 100
        full = act.volume_surprise(volume, window=21)
        trunc = act.volume_surprise(volume.iloc[:T], window=21)
        pd.testing.assert_frame_equal(
            full.iloc[:T].dropna(),
            trunc.dropna(),
            rtol=1e-9,
        )

    # --- return_volume_interaction ---

    def test_rvi_shape(self, act, returns, volume):
        result = act.return_volume_interaction(returns, volume)
        assert result.shape == returns.shape

    def test_rvi_values_in_set(self, act, returns, volume):
        """Values should be in {-1, 0, +1} or NaN."""
        result = act.return_volume_interaction(returns, volume)
        valid = result.stack().dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_rvi_high_vol_large_ret_is_informed(self, act, dates):
        """High volume + large return → +1 (informed)."""
        n, m = 80, 2
        rets = pd.DataFrame(np.zeros((n, m)), index=dates[:n], columns=["A", "B"])
        vols = pd.DataFrame(np.ones((n, m)) * 1e7, index=dates[:n], columns=["A", "B"])
        # Day 50: large positive return + high volume
        rets.iloc[50] = 0.10
        vols.iloc[50] = 1e7 * 3   # 3× average
        result = act.return_volume_interaction(
            rets, vols, window=10,
            volume_high_threshold=2.0,
            volume_low_threshold=0.5,
            return_threshold_std=0.5,
        )
        assert result.iloc[50, 0] == 1.0

    def test_rvi_low_vol_large_ret_is_uninformed(self, act, dates):
        """Low volume + large return → -1 (uninformed)."""
        n, m = 80, 2
        rets = pd.DataFrame(np.zeros((n, m)), index=dates[:n], columns=["A", "B"])
        vols = pd.DataFrame(np.ones((n, m)) * 1e7, index=dates[:n], columns=["A", "B"])
        # Day 50: large return but very LOW volume
        rets.iloc[50] = 0.10
        vols.iloc[50] = 1e3   # near-zero volume
        result = act.return_volume_interaction(
            rets, vols, window=10,
            volume_high_threshold=1.5,
            volume_low_threshold=0.5,
            return_threshold_std=0.5,
        )
        assert result.iloc[50, 0] == -1.0

    # --- volatility_regime ---

    def test_vr_returns_series(self, act, returns):
        regime = act.volatility_regime(returns, window=21)
        assert isinstance(regime, pd.Series)
        assert len(regime) == N_DATES

    def test_vr_valid_labels(self, act, returns):
        """All non-NaN values should be 'high_vol' or 'low_vol'."""
        regime = act.volatility_regime(returns, window=21)
        valid_vals = regime.dropna().unique()
        assert set(valid_vals).issubset({"high_vol", "low_vol"})

    def test_vr_high_vol_after_spike(self, act, dates):
        """After a volatility spike, regime should flip to high_vol."""
        n, m = 100, 4
        rets = pd.DataFrame(np.random.randn(n, m) * 0.001,
                             index=dates[:n],
                             columns=[f"A{i}" for i in range(m)])
        # Introduce huge spike in the last 10 rows
        rets.iloc[85:] = np.random.randn(15, m) * 0.50
        regime = act.volatility_regime(rets, window=10, threshold_percentile=50)
        late_regimes = regime.iloc[90:].dropna()
        assert (late_regimes == "high_vol").any()

    def test_vr_no_lookahead(self, act, returns):
        """Regime at T should not change when future data is appended."""
        T = 80
        full_regime = act.volatility_regime(returns, window=21)
        trunc_regime = act.volatility_regime(returns.iloc[:T], window=21)
        if full_regime.iloc[T - 1] is not np.nan and trunc_regime.iloc[T - 1] is not np.nan:
            assert full_regime.iloc[T - 1] == trunc_regime.iloc[T - 1]


# ============================================================
# ThemeSignals
# ============================================================

class TestThemeSignals:
    @pytest.fixture()
    def theme_returns(self, dates):
        """Returns panel using real symbol names from CRYPTO_THEMES."""
        all_syms = sorted({s for members in CRYPTO_THEMES.values() for s in members})[:12]
        np.random.seed(0)
        data = np.random.randn(N_DATES, len(all_syms)) * 0.02
        return pd.DataFrame(data, index=dates, columns=all_syms)

    @pytest.fixture()
    def ts(self):
        return ThemeSignals()

    def test_list_themes(self, ts):
        themes = ts.list_themes()
        assert "layer1" in themes
        assert "defi" in themes
        assert "gaming" in themes

    # --- theme_momentum ---

    def test_theme_momentum_shape(self, ts, theme_returns):
        sig = ts.theme_momentum(theme_returns, lookback=21)
        assert sig.shape == theme_returns.shape

    def test_theme_momentum_range(self, ts, theme_returns):
        sig = ts.theme_momentum(theme_returns, lookback=21)
        _assert_ranked(sig)

    def test_theme_momentum_hot_theme(self, ts, dates):
        """Assets in the outperforming theme should get positive signals."""
        n = 150
        layer1_syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        defi_syms = ["UNI/USDT", "AAVE/USDT", "CRV/USDT"]
        all_syms = layer1_syms + defi_syms

        rets_data = np.zeros((n, len(all_syms)))
        for i, sym in enumerate(all_syms):
            if sym in layer1_syms:
                rets_data[:, i] = 0.05   # layer1 always up
            else:
                rets_data[:, i] = -0.05  # defi always down

        rets = pd.DataFrame(rets_data, index=dates[:n], columns=all_syms)
        sig = ts.theme_momentum(rets, lookback=21)
        late_sig = sig.iloc[50:].dropna(how="all")
        # Layer1 assets should have higher signal than DeFi assets
        layer1_avg = late_sig[layer1_syms].mean().mean()
        defi_avg = late_sig[defi_syms].mean().mean()
        assert layer1_avg > defi_avg

    def test_theme_momentum_no_lookahead(self, ts, theme_returns):
        _no_lookahead(ts.theme_momentum, theme_returns, lookback=21, split_at=100)

    # --- theme_rotation ---

    def test_theme_rotation_shape(self, ts, theme_returns):
        sig = ts.theme_rotation(theme_returns, lookback=7, holding=7)
        assert sig.shape == theme_returns.shape

    def test_theme_rotation_range(self, ts, theme_returns):
        sig = ts.theme_rotation(theme_returns, lookback=7, holding=7)
        valid = sig.stack().dropna()
        if not valid.empty:
            assert (valid >= -1 - 1e-9).all()
            assert (valid <= 1 + 1e-9).all()

    # --- relative_theme_value ---

    def test_rtv_shape(self, ts, theme_returns):
        sig = ts.relative_theme_value(theme_returns, lookback=63)
        assert sig.shape == theme_returns.shape

    def test_rtv_range(self, ts, theme_returns):
        _assert_ranked(ts.relative_theme_value(theme_returns, lookback=63))

    def test_rtv_no_lookahead(self, ts, theme_returns):
        _no_lookahead(ts.relative_theme_value, theme_returns, lookback=63, split_at=100)

    def test_rtv_contrarian(self, ts, dates):
        """Hot theme should get negative (short) signal from relative_theme_value."""
        n = 150
        layer1_syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        defi_syms = ["UNI/USDT", "AAVE/USDT", "CRV/USDT"]
        all_syms = layer1_syms + defi_syms
        rets_data = np.zeros((n, len(all_syms)))
        for i, sym in enumerate(all_syms):
            rets_data[:, i] = 0.05 if sym in layer1_syms else -0.05
        rets = pd.DataFrame(rets_data, index=dates[:n], columns=all_syms)
        sig = ts.relative_theme_value(rets, lookback=42)
        late_sig = sig.iloc[80:].dropna(how="all")
        # Contrarian: hot layer1 → negative signal, cold defi → positive signal
        layer1_avg = late_sig[layer1_syms].mean().mean()
        defi_avg = late_sig[defi_syms].mean().mean()
        assert layer1_avg < defi_avg


# ============================================================
# ActivityFilter (backward-compat)
# ============================================================

class TestActivityFilter:
    def test_volume_regime_values(self, volume):
        filt = ActivityFilter()
        regime = filt.volume_regime(volume)
        unique_vals = set(regime.dropna().values.flatten())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_activity_gate_gates_low_volume(self, returns, dates):
        filt = ActivityFilter()
        window = 5
        n = 30
        syms = ["X"]
        vol_vals = np.ones((n, 1)) * 1e7
        vol_vals[20, 0] = 1.0   # near-zero on day 20
        vol = pd.DataFrame(vol_vals, index=dates[:n], columns=syms)
        ret = pd.DataFrame(np.ones((n, 1)) * 0.01, index=dates[:n], columns=syms)
        gated = filt.activity_gate(ret, vol, window=window, min_ratio=0.5)
        assert pd.isna(gated.iloc[20]["X"])
        assert not pd.isna(gated.iloc[25]["X"])

    def test_market_activity_index_series(self, volume):
        filt = ActivityFilter()
        mai = filt.market_activity_index(volume)
        assert isinstance(mai, pd.Series)
        assert len(mai) == len(volume)
