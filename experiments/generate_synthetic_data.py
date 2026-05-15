"""
Generate synthetic crypto-like OHLCV data for pipeline testing.

Produces realistic-looking price series with:
- Correlated returns (common factor + idiosyncratic)
- Time-varying volatility (GARCH-like clustering)
- Volume co-moving with volatility
- Occasional large moves (fat tails)

Usage:
    python experiments/generate_synthetic_data.py
    python experiments/generate_synthetic_data.py --n-assets 10 --n-days 500 --seed 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from statarb.data.storage import DataStore
from statarb.utils import get_logger, load_config

logger = get_logger("generate_synthetic")

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT",
    "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT", "FIL/USDT",
    "AAVE/USDT", "GRT/USDT", "CRV/USDT", "INJ/USDT", "RUNE/USDT",
]


def generate_synthetic_ohlcv(
    symbol: str,
    dates: pd.DatetimeIndex,
    market_factor: np.ndarray,
    beta: float = 1.0,
    idio_vol: float = 0.02,
    base_price: float = 100.0,
    base_volume: float = 1e6,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV for a single asset.

    Returns:
        DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(dates)

    # GARCH-like vol clustering
    vol = np.ones(n) * idio_vol
    for i in range(1, n):
        shock = abs(market_factor[i - 1]) + abs(rng.normal(0, idio_vol))
        vol[i] = 0.85 * vol[i - 1] + 0.15 * shock

    # Returns: common factor + idiosyncratic + fat tails
    idio = rng.normal(0, vol)
    jumps = rng.normal(0, 3 * idio_vol, n) * (rng.random(n) < 0.02)  # 2% jump prob
    log_returns = beta * market_factor + idio + jumps

    # Price path
    log_prices = np.log(base_price) + np.cumsum(log_returns)
    close = np.exp(log_prices)

    # OHLC construction
    daily_range = np.abs(rng.normal(0, vol * 0.5)) + 0.003
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)
    open_price = np.roll(close, 1)
    open_price[0] = base_price

    # Volume: higher on large moves, mean-reverting
    vol_factor = np.exp(np.abs(log_returns) * 5)
    volume = base_volume * vol_factor * rng.lognormal(0, 0.3, n)

    return pd.DataFrame({
        "open":   open_price,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=dates)


def generate_all(
    n_assets: int = 20,
    n_days: int = 1000,
    start_date: str = "2021-01-01",
    seed: int = 42,
    exchange: str = "binance",
    base_dir: str = "data",
) -> None:
    """Generate and save synthetic data for all assets."""
    rng = np.random.default_rng(seed)
    symbols = SYMBOLS[:n_assets]
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # Market factor with GARCH-like clustering
    mkt_vol = np.ones(n_days) * 0.03
    mkt_raw = rng.normal(0, 1, n_days)
    mkt_factor = np.zeros(n_days)
    for i in range(n_days):
        mkt_vol[i] = max(0.01, 0.9 * mkt_vol[max(0, i - 1)] + 0.1 * abs(mkt_raw[i]) * 0.03)
        mkt_factor[i] = mkt_raw[i] * mkt_vol[i]

    # Asset characteristics
    betas = rng.uniform(0.3, 1.8, n_assets)
    idio_vols = rng.uniform(0.015, 0.045, n_assets)
    base_prices = rng.uniform(10, 50000, n_assets)
    base_volumes = rng.lognormal(15, 1.5, n_assets)

    store = DataStore(base_dir=base_dir)

    for i, sym in enumerate(symbols):
        df = generate_synthetic_ohlcv(
            symbol=sym,
            dates=dates,
            market_factor=mkt_factor,
            beta=betas[i],
            idio_vol=idio_vols[i],
            base_price=base_prices[i],
            base_volume=base_volumes[i],
            rng=rng,
        )
        store.save_ohlcv(df, sym, exchange, "1d")
        logger.info("Generated %s: %d days, β=%.2f, σ=%.1f%%",
                    sym, n_days, betas[i], idio_vols[i] * 100)

    logger.info("Synthetic data saved to %s/raw/%s/1d/ (%d symbols × %d days)",
                base_dir, exchange, n_assets, n_days)


def main() -> None:
    cfg = load_config("experiments/config.yaml")
    syn_cfg = cfg.get("synthetic", {})

    parser = argparse.ArgumentParser(description="Generate synthetic crypto OHLCV data")
    parser.add_argument("--n-assets", type=int, default=syn_cfg.get("n_assets", 20))
    parser.add_argument("--n-days", type=int, default=syn_cfg.get("n_days", 1000))
    parser.add_argument("--start-date", default=syn_cfg.get("start_date", "2021-01-01"))
    parser.add_argument("--seed", type=int, default=syn_cfg.get("seed", 42))
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--base-dir", default=cfg.get("data", {}).get("base_dir", "data"))
    args = parser.parse_args()

    generate_all(
        n_assets=args.n_assets,
        n_days=args.n_days,
        start_date=args.start_date,
        seed=args.seed,
        exchange=args.exchange,
        base_dir=args.base_dir,
    )


if __name__ == "__main__":
    main()
