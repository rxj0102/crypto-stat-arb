# Data

This directory stores raw and processed cryptocurrency price-volume data.

## Directory Layout

```
data/
├── raw/
│   └── {exchange}/
│       └── {timeframe}/
│           └── {SYMBOL_USDT}.parquet   # e.g. BTC_USDT.parquet
├── processed/
│   ├── returns/
│   │   └── {timeframe}/panel.parquet   # cross-sectional return panel
│   └── volume/
│       └── {timeframe}/panel.parquet   # cross-sectional volume panel
└── README.md                           # this file
```

The `raw/` and `processed/` subdirectories are **gitignored** — data files
must be fetched locally and are not committed to the repository.

## Data Source

Data is fetched from cryptocurrency exchange APIs via the
[ccxt](https://github.com/ccxt/ccxt) library. The default exchange is
**Binance** (largest spot market by volume, best historical data availability).

Alternative exchanges: `kraken`, `coinbase`, `okx`, `bybit`, `kucoin`.

## Fetching Data

### Quick start

```bash
# Fetch the default ~50-asset universe (daily OHLCV from 2020-01-01)
python experiments/run_data_fetch.py

# Fetch hourly data for specific symbols
python experiments/run_data_fetch.py --timeframe 1h --symbols BTC/USDT ETH/USDT SOL/USDT

# Fetch top 30 assets by 24h volume
python experiments/run_data_fetch.py --top-n 30

# Custom date range
python experiments/run_data_fetch.py --start-date 2022-01-01 --end-date 2023-12-31
```

### Programmatic usage

```python
from statarb.data.fetcher import CryptoDataFetcher
from statarb.data.storage import DataStore

fetcher = CryptoDataFetcher(exchange='binance')
store = DataStore(base_dir='data')

# Fetch and save BTC daily data
df = fetcher.fetch_ohlcv('BTC/USDT', timeframe='1d', start_date='2020-01-01')
store.save_ohlcv(df, 'BTC/USDT', 'binance', '1d')

# Load back
df = store.load_ohlcv('BTC/USDT', 'binance', '1d')
print(df.head())
```

## Data Schema

Each raw Parquet file contains an OHLCV DataFrame:

| Column    | Type    | Description                           |
|-----------|---------|---------------------------------------|
| timestamp | index   | UTC DatetimeIndex                     |
| open      | float64 | Open price in USDT                    |
| high      | float64 | High price in USDT                    |
| low       | float64 | Low price in USDT                     |
| close     | float64 | Close price in USDT                   |
| volume    | float64 | Base asset volume (not dollar volume) |

Dollar volume = `close × volume`.

## Data Quality Notes

- **Gaps**: Some altcoins have listing gaps or zero-volume periods.
  The `TradableUniverse` class handles these via minimum-history filters.
- **Delisted assets**: Symbols that were delisted from Binance will
  return empty DataFrames — this is expected and handled gracefully.
- **Stablecoins**: Filtered out by `TradableUniverse` by default.
- **Outliers**: Extreme returns (e.g. exchange hacks, flash crashes) are
  kept in the data. Winsorisation can be applied at the signal level.

## Rate Limits

ccxt handles exchange rate limits automatically when `rate_limit=True`
(the default). For bulk fetches of 50+ symbols, expect ~10-20 minutes
for daily data and ~1-3 hours for hourly data.

To fetch data faster, consider parallelising across multiple exchange
API keys or using a data provider like CryptoCompare / Kaiko for bulk
historical downloads.
