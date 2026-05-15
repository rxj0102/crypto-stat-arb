"""
Fetch and store crypto OHLCV data for the research universe.

Usage:
    python experiments/run_data_fetch.py
    python experiments/run_data_fetch.py --timeframe 1h --symbols BTC/USDT ETH/USDT
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from statarb.data.fetcher import CryptoDataFetcher
from statarb.data.storage import DataStore
from statarb.utils import get_logger, load_config

logger = get_logger("run_data_fetch")


def main(args: argparse.Namespace) -> None:
    cfg = load_config("experiments/config.yaml")
    data_cfg = cfg.get("data", {})

    exchange = args.exchange or data_cfg.get("exchange", "binance")
    timeframe = args.timeframe or data_cfg.get("timeframes", {}).get("daily", "1d")
    start_date = args.start_date or data_cfg.get("start_date", "2020-01-01")
    end_date = args.end_date or data_cfg.get("end_date") or None
    base_dir = data_cfg.get("base_dir", "data")

    fetcher = CryptoDataFetcher(exchange=exchange)

    if args.symbols:
        symbols = args.symbols
    elif args.top_n:
        logger.info("Fetching top %d symbols by volume", args.top_n)
        symbols = fetcher.get_top_symbols(n=args.top_n)
    else:
        symbols = CryptoDataFetcher.default_universe()

    logger.info("Universe: %d symbols | Timeframe: %s | %s → %s",
                len(symbols), timeframe, start_date, end_date or "today")

    store = DataStore(base_dir=base_dir)
    success, failures = 0, []

    for sym in symbols:
        try:
            df = fetcher.fetch_ohlcv(sym, timeframe, start_date, end_date)
            if df.empty:
                logger.warning("No data for %s — skipping", sym)
                failures.append(sym)
                continue
            store.save_ohlcv(df, sym, exchange, timeframe)
            success += 1
            logger.info("✓ Saved %s (%d rows)", sym, len(df))
        except Exception as exc:  # noqa: BLE001
            logger.error("✗ Failed %s: %s", sym, exc)
            failures.append(sym)

    logger.info("\n=== Fetch complete: %d succeeded, %d failed ===", success, len(failures))
    if failures:
        logger.warning("Failed symbols: %s", failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch crypto OHLCV data")
    parser.add_argument("--exchange", default=None, help="Exchange name (default: binance)")
    parser.add_argument("--timeframe", default=None,
                        help="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d (default: 1d)")
    parser.add_argument("--start-date", default=None, dest="start_date",
                        help="Start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end-date", default=None, dest="end_date",
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Specific symbols to fetch (e.g. BTC/USDT ETH/USDT)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Fetch top N symbols by volume")
    main(parser.parse_args())
