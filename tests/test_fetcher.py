"""
Tests for CryptoDataFetcher.

Uses mock data — does not call live exchange APIs.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from statarb.data.fetcher import CryptoDataFetcher, _DEFAULT_UNIVERSE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candles(n: int = 100, start_ts: int = 1_577_836_800_000) -> list:
    """Generate synthetic OHLCV candles in ccxt format."""
    ms_per_day = 86_400_000
    return [
        [start_ts + i * ms_per_day, 100 + i, 105 + i, 95 + i, 102 + i, 1000 + i * 10]
        for i in range(n)
    ]


@pytest.fixture()
def mock_exchange():
    """A ccxt-like mock exchange object."""
    ex = MagicMock()
    ex.fetch_ohlcv.return_value = _make_candles(100)
    ex.fetch_tickers.return_value = {
        "BTC/USDT": {"quoteVolume": 1e9},
        "ETH/USDT": {"quoteVolume": 5e8},
        "SOL/USDT": {"quoteVolume": 2e8},
        "USDT/USD": {"quoteVolume": 1e10},   # should be excluded by quote filter
    }
    return ex


@pytest.fixture()
def fetcher(mock_exchange):
    """CryptoDataFetcher with mocked exchange."""
    with patch("ccxt.binance", return_value=mock_exchange):
        f = CryptoDataFetcher(exchange="binance")
    f.exchange = mock_exchange
    return f


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------

class TestFetchOhlcv:
    def test_returns_dataframe(self, fetcher):
        df = fetcher.fetch_ohlcv("BTC/USDT", timeframe="1d", start_date="2020-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, fetcher):
        df = fetcher.fetch_ohlcv("BTC/USDT")
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_datetimeindex(self, fetcher):
        df = fetcher.fetch_ohlcv("BTC/USDT")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_is_utc(self, fetcher):
        df = fetcher.fetch_ohlcv("BTC/USDT")
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_no_duplicate_timestamps(self, fetcher):
        df = fetcher.fetch_ohlcv("BTC/USDT")
        assert not df.index.duplicated().any()

    def test_sorted_ascending(self, fetcher):
        df = fetcher.fetch_ohlcv("BTC/USDT")
        assert df.index.is_monotonic_increasing

    def test_empty_on_bad_symbol(self, fetcher):
        import ccxt
        fetcher.exchange.fetch_ohlcv.side_effect = ccxt.BadSymbol("bad symbol")
        df = fetcher.fetch_ohlcv("FAKE/USDT")
        assert df.empty

    def test_invalid_timeframe_raises(self, fetcher):
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            fetcher.fetch_ohlcv("BTC/USDT", timeframe="2w")

    def test_respects_start_date(self, fetcher):
        """Returned data should not predate start_date."""
        df = fetcher.fetch_ohlcv("BTC/USDT", start_date="2020-01-01")
        if not df.empty:
            assert df.index.min() >= pd.Timestamp("2020-01-01", tz="UTC")

    def test_handles_network_error_with_retry(self, fetcher, mock_exchange):
        """Should retry once on network error then succeed."""
        import ccxt
        good_candles = _make_candles(10)
        mock_exchange.fetch_ohlcv.side_effect = [
            ccxt.NetworkError("timeout"),
            good_candles,
        ]
        df = fetcher.fetch_ohlcv("BTC/USDT")
        assert not df.empty

    def test_all_numeric_values(self, fetcher):
        df = fetcher.fetch_ohlcv("BTC/USDT")
        assert df.select_dtypes(include="number").shape[1] == 5

    def test_high_gte_low(self, fetcher):
        df = fetcher.fetch_ohlcv("BTC/USDT")
        assert (df["high"] >= df["low"]).all()

    def test_volume_positive(self, fetcher):
        df = fetcher.fetch_ohlcv("BTC/USDT")
        assert (df["volume"] > 0).all()


# ---------------------------------------------------------------------------
# fetch_universe
# ---------------------------------------------------------------------------

class TestFetchUniverse:
    def test_returns_dict(self, fetcher):
        result = fetcher.fetch_universe(["BTC/USDT", "ETH/USDT"])
        assert isinstance(result, dict)

    def test_keys_are_symbols(self, fetcher):
        result = fetcher.fetch_universe(["BTC/USDT", "ETH/USDT"])
        assert "BTC/USDT" in result
        assert "ETH/USDT" in result

    def test_values_are_dataframes(self, fetcher):
        result = fetcher.fetch_universe(["BTC/USDT"])
        for df in result.values():
            assert isinstance(df, pd.DataFrame)

    def test_empty_on_all_bad_symbols(self, fetcher):
        import ccxt
        fetcher.exchange.fetch_ohlcv.side_effect = ccxt.BadSymbol("bad")
        result = fetcher.fetch_universe(["FAKE1/USDT", "FAKE2/USDT"])
        assert result == {}


# ---------------------------------------------------------------------------
# get_top_symbols
# ---------------------------------------------------------------------------

class TestGetTopSymbols:
    def test_returns_list(self, fetcher):
        result = fetcher.get_top_symbols(n=2)
        assert isinstance(result, list)

    def test_filters_by_quote_currency(self, fetcher):
        result = fetcher.get_top_symbols(n=10, quote="USDT")
        for sym in result:
            assert sym.endswith("/USDT")

    def test_respects_n_limit(self, fetcher):
        result = fetcher.get_top_symbols(n=2, quote="USDT")
        assert len(result) <= 2

    def test_sorted_by_volume_descending(self, fetcher):
        """First result should be highest-volume symbol."""
        result = fetcher.get_top_symbols(n=3, quote="USDT")
        assert result[0] == "BTC/USDT"  # highest quoteVolume in mock


# ---------------------------------------------------------------------------
# default_universe
# ---------------------------------------------------------------------------

class TestDefaultUniverse:
    def test_returns_list(self):
        u = CryptoDataFetcher.default_universe()
        assert isinstance(u, list)

    def test_non_empty(self):
        u = CryptoDataFetcher.default_universe()
        assert len(u) > 0

    def test_all_usdt_pairs(self):
        u = CryptoDataFetcher.default_universe()
        for sym in u:
            assert sym.endswith("/USDT"), f"{sym} is not a /USDT pair"

    def test_contains_major_assets(self):
        u = CryptoDataFetcher.default_universe()
        assert "BTC/USDT" in u
        assert "ETH/USDT" in u
        assert "SOL/USDT" in u

    def test_no_stablecoins(self):
        stablecoins = {"USDT/USDT", "USDC/USDT", "DAI/USDT", "BUSD/USDT"}
        u = set(CryptoDataFetcher.default_universe())
        assert u.isdisjoint(stablecoins), f"Stablecoins found: {u & stablecoins}"
