"""
Fetch and store crypto OHLCV data for the research universe.

If live API access is unavailable (network error, rate limit, etc.), falls
back to loading existing data from data/raw/ or generates synthetic data.

Usage:
    python experiments/run_data_fetch.py
    python experiments/run_data_fetch.py --synthetic          # force synthetic
    python experiments/run_data_fetch.py --symbols BTC/USDT ETH/USDT
    python experiments/run_data_fetch.py --timeframe 1h
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from statarb.data.fetcher import CryptoDataFetcher
from statarb.data.storage import DataStore
from statarb.utils import get_logger, load_config
from experiments.generate_synthetic_data import generate_all as generate_synthetic

logger = get_logger("run_data_fetch")


def fetch_live(args: argparse.Namespace, cfg: dict) -> tuple[int, list]:
    """Attempt to fetch live data. Returns (n_success, failures)."""
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
        except Exception as exc:
            logger.error("✗ Failed %s: %s", sym, exc)
            failures.append(sym)

    return success, failures


def main(args: argparse.Namespace) -> None:
    cfg = load_config("experiments/config.yaml")
    data_cfg = cfg.get("data", {})
    syn_cfg = cfg.get("synthetic", {})
    base_dir = data_cfg.get("base_dir", "data")
    exchange = args.exchange or data_cfg.get("exchange", "binance")

    if args.synthetic:
        logger.info("Synthetic mode: generating synthetic data")
        generate_synthetic(
            n_assets=syn_cfg.get("n_assets", 20),
            n_days=syn_cfg.get("n_days", 1000),
            start_date=syn_cfg.get("start_date", "2021-01-01"),
            seed=syn_cfg.get("seed", 42),
            exchange=exchange,
            base_dir=base_dir,
        )
        return

    # Try live fetch; fall back to synthetic if nothing succeeds
    logger.info("Attempting live data fetch from %s", exchange)
    try:
        success, failures = fetch_live(args, cfg)
        logger.info("=== Fetch complete: %d succeeded, %d failed ===",
                    success, len(failures))
        if failures:
            logger.warning("Failed symbols: %s", failures)
        if success == 0:
            raise RuntimeError("No data fetched successfully")
    except Exception as exc:
        logger.warning("Live fetch failed (%s) — checking for existing data", exc)

        store = DataStore(base_dir=base_dir)
        existing = store.list_symbols(exchange, "1d")
        if existing:
            logger.info("Found %d symbols of existing data — using that", len(existing))
        else:
            logger.warning("No existing data found — generating synthetic data")
            generate_synthetic(
                n_assets=syn_cfg.get("n_assets", 20),
                n_days=syn_cfg.get("n_days", 1000),
                start_date=syn_cfg.get("start_date", "2021-01-01"),
                seed=syn_cfg.get("seed", 42),
                exchange=exchange,
                base_dir=base_dir,
            )

    # Build processed panels
    store = DataStore(base_dir=base_dir)
    symbols = store.list_symbols(exchange, "1d")
    if symbols:
        logger.info("Building processed return and volume panels for %d symbols", len(symbols))
        prices = store.build_panel(symbols, exchange, "1d", field="close")
        volume = store.build_volume_panel(symbols, exchange, "1d")
        returns = prices.pct_change()
        logger.info("Price panel: %s × %d assets", prices.index[-1].date(), len(symbols))
        logger.info("Return panel: %.2f%% mean daily return",
                    returns.mean().mean() * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch crypto OHLCV data")
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--start-date", default=None, dest="start_date")
    parser.add_argument("--end-date", default=None, dest="end_date")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--top-n", type=int, default=None, dest="top_n")
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip API fetch; generate synthetic data instead")
    main(parser.parse_args())
