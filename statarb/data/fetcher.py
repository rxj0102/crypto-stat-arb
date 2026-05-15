"""Cryptocurrency OHLCV data fetching via ccxt."""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd

from statarb.utils import get_logger, parse_date, to_ms, today_str

logger = get_logger(__name__)

# Default research universe — liquid, long-history assets vs USDT
_DEFAULT_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "BCH/USDT",
    "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "FIL/USDT",
    "ICP/USDT", "HBAR/USDT", "VET/USDT", "ALGO/USDT", "ETC/USDT",
    "AAVE/USDT", "GRT/USDT", "MKR/USDT", "SNX/USDT", "COMP/USDT",
    "CRV/USDT", "LDO/USDT", "RUNE/USDT", "INJ/USDT", "FTM/USDT",
    "SAND/USDT", "MANA/USDT", "AXS/USDT", "ENJ/USDT", "GALA/USDT",
    "APE/USDT", "CHZ/USDT", "FLOW/USDT", "XTZ/USDT", "EOS/USDT",
    "ZEC/USDT", "DASH/USDT", "XMR/USDT", "THETA/USDT", "EGLD/USDT",
]

# Maximum candles per API request (conservative across exchanges)
_MAX_CANDLES_PER_REQUEST = 500

# Timeframe → milliseconds mapping
_TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class CryptoDataFetcher:
    """
    Fetch cryptocurrency OHLCV data from public exchange APIs via ccxt.

    Supports multiple exchanges: Binance, Coinbase, Kraken, etc.
    Handles pagination transparently — exchanges cap candles per request.

    Primary use cases for stat-arb research:
    - Daily OHLCV for medium/long-term momentum signals
    - Hourly OHLCV for short-term reversal and intraday seasonality
    - Volume data for activity-based signal filtering
    """

    def __init__(self, exchange: str = "binance", rate_limit: bool = True):
        """
        Args:
            exchange: ccxt-compatible exchange name (e.g. 'binance', 'kraken')
            rate_limit: honour the exchange's rate-limit between requests
        """
        exchange_class = getattr(ccxt, exchange, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange '{exchange}'. "
                             f"Available: {ccxt.exchanges[:10]} ...")
        self.exchange: ccxt.Exchange = exchange_class({"enableRateLimit": rate_limit})
        self.exchange_name = exchange
        logger.info("Initialized fetcher for exchange: %s", exchange)

    # ------------------------------------------------------------------
    # Single-symbol fetch
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol with automatic pagination.

        Args:
            symbol: trading pair, e.g. 'BTC/USDT'
            timeframe: one of '1m', '5m', '15m', '1h', '4h', '1d'
            start_date: ISO format start date (inclusive)
            end_date: ISO format end date (inclusive); defaults to today

        Returns:
            DataFrame indexed by UTC DatetimeIndex with columns:
            ['open', 'high', 'low', 'close', 'volume']
        """
        if timeframe not in _TF_MS:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. "
                             f"Choose from {list(_TF_MS)}")

        end_date = end_date or today_str()
        since_ms = to_ms(start_date)
        end_ms = to_ms(end_date) + _TF_MS[timeframe]  # include end candle
        tf_ms = _TF_MS[timeframe]

        all_candles: List[list] = []
        cursor = since_ms

        logger.info("Fetching %s %s from %s to %s", symbol, timeframe,
                    start_date, end_date)

        while cursor < end_ms:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe,
                    since=cursor,
                    limit=_MAX_CANDLES_PER_REQUEST,
                )
            except ccxt.BadSymbol as exc:
                logger.warning("Symbol %s not available: %s", symbol, exc)
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )
            except ccxt.NetworkError as exc:
                logger.warning("Network error fetching %s: %s. Retrying...", symbol, exc)
                time.sleep(2)
                continue
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error for %s: %s", symbol, exc)
                break

            if not candles:
                break

            all_candles.extend(candles)

            last_ts = candles[-1][0]
            if last_ts >= end_ms or len(candles) < _MAX_CANDLES_PER_REQUEST:
                break
            cursor = last_ts + tf_ms

        if not all_candles:
            logger.warning("No data returned for %s", symbol)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

        # Trim to requested window
        start_dt = pd.Timestamp(start_date, tz="UTC")
        end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        df = df.loc[start_dt:end_dt]

        # Drop duplicate timestamps (exchange artefacts)
        df = df[~df.index.duplicated(keep="last")]

        logger.info("Fetched %d candles for %s", len(df), symbol)
        return df

    # ------------------------------------------------------------------
    # Multi-symbol fetch
    # ------------------------------------------------------------------

    def fetch_universe(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple symbols.

        Returns:
            dict mapping symbol → DataFrame (same schema as fetch_ohlcv)
        """
        result: Dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols):
            logger.info("[%d/%d] Fetching %s", i + 1, len(symbols), sym)
            try:
                df = self.fetch_ohlcv(sym, timeframe, start_date, end_date)
                if not df.empty:
                    result[sym] = df
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch %s: %s", sym, exc)
        return result

    # ------------------------------------------------------------------
    # Universe helpers
    # ------------------------------------------------------------------

    def get_top_symbols(self, n: int = 50, quote: str = "USDT") -> List[str]:
        """
        Return top-n symbols by 24h trading volume on the exchange.

        Args:
            n: number of symbols to return
            quote: quote currency filter (e.g. 'USDT')

        Returns:
            List of symbol strings sorted by 24h volume descending.
        """
        try:
            tickers = self.exchange.fetch_tickers()
        except ccxt.ExchangeError as exc:
            logger.error("Could not fetch tickers: %s", exc)
            return []

        rows = []
        for sym, ticker in tickers.items():
            if not sym.endswith(f"/{quote}"):
                continue
            vol = ticker.get("quoteVolume") or 0.0
            rows.append((sym, float(vol)))

        rows.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in rows[:n]]

    @staticmethod
    def default_universe() -> List[str]:
        """
        Default research universe of ~50 liquid crypto assets vs USDT.

        Selection criteria:
        - Minimum 2-year trading history on major exchanges
        - Significant daily dollar volume (>$10M typical)
        - Diverse across L1s, L2s, DeFi, gaming, privacy, and PoW assets
        - Excludes stablecoins and highly correlated wrapped tokens
        """
        return list(_DEFAULT_UNIVERSE)
