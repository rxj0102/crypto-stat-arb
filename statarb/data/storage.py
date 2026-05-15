"""Local data storage using Parquet files."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from statarb.utils import ensure_dir, get_logger

logger = get_logger(__name__)


class DataStore:
    """
    Local data storage using Parquet files.

    Directory layout::

        {base_dir}/raw/{exchange}/{timeframe}/{symbol_safe}.parquet
        {base_dir}/processed/returns/{timeframe}/panel.parquet
        {base_dir}/processed/volume/{timeframe}/panel.parquet

    Symbol names are sanitised (``/`` → ``_``) for filesystem compatibility.
    """

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir / "raw")
        ensure_dir(self.base_dir / "processed" / "returns")
        ensure_dir(self.base_dir / "processed" / "volume")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_path(self, symbol: str, exchange: str, timeframe: str) -> Path:
        safe_sym = symbol.replace("/", "_")
        return self.base_dir / "raw" / exchange / timeframe / f"{safe_sym}.parquet"

    @staticmethod
    def _sanitise(symbol: str) -> str:
        return symbol.replace("/", "_")

    # ------------------------------------------------------------------
    # Raw OHLCV
    # ------------------------------------------------------------------

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> None:
        """Persist OHLCV DataFrame to Parquet, merging with existing data."""
        path = self._raw_path(symbol, exchange, timeframe)
        ensure_dir(path.parent)

        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df]).pipe(
                lambda d: d[~d.index.duplicated(keep="last")]
            ).sort_index()

        df.to_parquet(path)
        logger.debug("Saved %d rows → %s", len(df), path)

    def load_ohlcv(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Parquet.

        Returns empty DataFrame if file not found.
        """
        path = self._raw_path(symbol, exchange, timeframe)
        if not path.exists():
            logger.warning("No stored data for %s/%s/%s", exchange, timeframe, symbol)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.read_parquet(path)
        if start_date:
            df = df.loc[pd.Timestamp(start_date, tz="UTC"):]
        if end_date:
            df = df.loc[: pd.Timestamp(end_date, tz="UTC")]
        return df

    def list_symbols(self, exchange: str, timeframe: str) -> List[str]:
        """Return list of symbols available in storage for a given exchange/timeframe."""
        directory = self.base_dir / "raw" / exchange / timeframe
        if not directory.exists():
            return []
        return [p.stem.replace("_", "/", 1) for p in directory.glob("*.parquet")]

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    def build_panel(
        self,
        symbols: List[str],
        exchange: str,
        timeframe: str,
        field: str = "close",
    ) -> pd.DataFrame:
        """
        Build a cross-sectional panel: rows = dates, columns = symbols.

        Missing symbols are filled with NaN. Panel is aligned on a common
        DatetimeIndex (union of all available dates).

        Args:
            symbols: list of symbol strings
            exchange: exchange name
            timeframe: data frequency
            field: OHLCV field to extract (default 'close')

        Returns:
            DataFrame[dates × symbols]
        """
        series: dict = {}
        for sym in symbols:
            df = self.load_ohlcv(sym, exchange, timeframe)
            if df.empty or field not in df.columns:
                continue
            series[sym] = df[field]

        if not series:
            return pd.DataFrame()

        panel = pd.DataFrame(series)
        panel.index = pd.DatetimeIndex(panel.index)
        return panel.sort_index()

    def build_return_panel(
        self,
        symbols: List[str],
        exchange: str,
        timeframe: str,
        return_type: str = "log",
        periods: int = 1,
    ) -> pd.DataFrame:
        """
        Build a panel of returns.

        Args:
            return_type: 'log' (log returns) or 'simple' (arithmetic)
            periods: number of periods for return calculation

        Returns:
            DataFrame[dates × symbols]
        """
        prices = self.build_panel(symbols, exchange, timeframe, field="close")
        if prices.empty:
            return prices

        if return_type == "log":
            return np.log(prices / prices.shift(periods))
        elif return_type == "simple":
            return prices.pct_change(periods)
        else:
            raise ValueError(f"return_type must be 'log' or 'simple', got '{return_type}'")

    def build_volume_panel(
        self,
        symbols: List[str],
        exchange: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Build a panel of dollar-denominated volumes (close × volume).

        Returns:
            DataFrame[dates × symbols]
        """
        close_panel = self.build_panel(symbols, exchange, timeframe, field="close")
        volume_panel = self.build_panel(symbols, exchange, timeframe, field="volume")

        if close_panel.empty or volume_panel.empty:
            return pd.DataFrame()

        common_cols = close_panel.columns.intersection(volume_panel.columns)
        dollar_vol = close_panel[common_cols] * volume_panel[common_cols]
        return dollar_vol.sort_index()
