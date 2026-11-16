"""Daily prices and dollar volume for every USDT spot pair Binance has ever listed.

Source: the public Binance archive, https://data.binance.vision (free, no key). The archive
keeps delisted pairs, which is what makes a survivorship-free universe possible.

The study reads the frozen copy in data/snapshot. scripts/fetch_data.py rebuilds it.

Cleaning rules, fixed before any return was computed:
  * a day is the UTC candle; price is its close, dollar volume is its quote asset volume
  * EXCLUDED pairs are dropped: stablecoins and fiat, leveraged tokens, and tokens pegged to
    another asset
  * a gap of more than MAX_GAP days inside one symbol starts a new asset (the symbol was
    delisted and later reused or relisted); assets are named SYMBOL, SYMBOL.2, ...
  * days with zero dollar volume are treated as not traded
  * assets first listed after LAST_LISTING are dropped: from June 2026 the archive adds about a
    hundred tokenized stocks and funds, which are not crypto assets
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "snapshot"
MAX_GAP = 3
LAST_LISTING = "2026-05-31"

STABLE_OR_FIAT = {
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USDS", "USDSB", "SUSD", "UST", "USTC", "USD1",
    "EUR", "EURI", "AEUR", "GBP", "AUD", "BRL", "TRY", "RUB", "UAH", "NGN", "BIDR", "IDRT", "BKRW", "BVND",
    "VAI", "XUSD", "USDE", "PYUSD", "BFUSD", "EURC", "RLUSD", "USDX", "FRAX", "LUSD", "GUSD", "ZAR", "PLN",
    "RON", "ARS", "JPY", "MXN", "COP", "CZK", "U",
}
PEGGED = {"WBTC", "WBETH", "BETH", "BNSOL", "PAXG", "XAUT", "WETH", "STETH", "KGST"}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)$")
NOT_LEVERAGED = {"JUP", "SYRUP"}          # ordinary coins whose names end like a leveraged token


def base_of(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def excluded(symbol: str) -> bool:
    b = base_of(symbol)
    return b in STABLE_OR_FIAT or b in PEGGED or (bool(LEVERAGED.search(b)) and b not in NOT_LEVERAGED)


def parse_kline_zip(raw: bytes) -> pd.DataFrame:
    """One monthly archive file as (date, close, dollar volume). Handles ms and us timestamps."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode()
    rows = []
    for line in text.splitlines():
        p = line.split(",")
        if len(p) < 8 or not p[0].strip().isdigit():
            continue                                   # header line in some files
        t = int(p[0])
        t = t // 1000 if t > 10**14 else t             # microseconds from 2025
        rows.append((pd.Timestamp(t, unit="ms").normalize(), float(p[4]), float(p[7])))
    return pd.DataFrame(rows, columns=["date", "close", "dollar_volume"])


def split_assets(symbol: str, df: pd.DataFrame, max_gap: int = MAX_GAP) -> dict[str, pd.DataFrame]:
    """Split one symbol's history where trading stopped for more than max_gap days."""
    df = df[df["dollar_volume"] > 0].drop_duplicates("date").sort_values("date")
    if df.empty:
        return {}
    part = (df["date"].diff().dt.days > max_gap + 1).cumsum()
    out = {}
    for k, g in df.groupby(part):
        out[symbol if k == 0 else f"{symbol}.{k + 1}"] = g.set_index("date")
    return out


def build(zip_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide close and dollar volume panels (dates by asset) from a folder of archive files."""
    by_symbol: dict[str, list] = {}
    for f in sorted(Path(zip_dir).glob("*USDT-1d-*.zip")):
        sym = f.name.split("-")[0]
        if not excluded(sym):
            by_symbol.setdefault(sym, []).append(parse_kline_zip(f.read_bytes()))
    close, vol = {}, {}
    for sym, parts in sorted(by_symbol.items()):
        for name, g in split_assets(sym, pd.concat(parts, ignore_index=True)).items():
            close[name], vol[name] = g["close"], g["dollar_volume"]
    close, vol = pd.DataFrame(close).sort_index(), pd.DataFrame(vol).sort_index()
    close.index.name = vol.index.name = "date"
    return close, vol


def write_snapshot(close: pd.DataFrame, vol: pd.DataFrame, last_day: str) -> None:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    close, vol = close.loc[:last_day], vol.loc[:last_day]
    first = close.apply(lambda s: s.first_valid_index())
    keep = first.index[first.notna() & (first <= pd.Timestamp(LAST_LISTING))]
    gz = {"method": "gzip", "mtime": 0}
    close[keep].to_csv(SNAPSHOT / "close.csv.gz", float_format="%.10g", compression=gz)
    vol[keep].round(0).to_csv(SNAPSHOT / "dollar_volume.csv.gz", float_format="%.0f", compression=gz)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.read_csv(SNAPSHOT / "close.csv.gz", index_col="date", parse_dates=True)
    vol = pd.read_csv(SNAPSHOT / "dollar_volume.csv.gz", index_col="date", parse_dates=True)
    return close, vol


def symbol_of(asset: str) -> str:
    return asset.split(".")[0]
