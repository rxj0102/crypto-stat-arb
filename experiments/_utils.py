"""
Shared helpers for experiment scripts.

Provides:
  - load_data():  load prices + volume, falling back to synthetic generation
  - run_bt():     run UnconstrainedBacktest for a signal, return (returns, result_dict)
  - backtest_signals(): batch backtest, return summary DataFrame
  - save_results():  save DataFrame to CSV in the results dir
  - save_fig():      save matplotlib figure to PNG
  - signal_decay_ic(): compute IC at each forward horizon for signal decay plot
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statarb.backtest.engine import UnconstrainedBacktest
from statarb.backtest.execution import ExecutionModel
from statarb.data.features import FeatureEngine
from statarb.data.storage import DataStore
from statarb.evaluation.metrics import PerformanceMetrics
from statarb.utils import ensure_dir, get_logger, load_config

logger = get_logger("_utils")
_pm = PerformanceMetrics()


def load_data(
    cfg: dict,
    generate_if_missing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load prices, volume, and log-returns from the local store.

    Falls back to synthetic data generation if the store is empty.

    Returns:
        (prices, volume, log_returns) — all DataFrame[dates × symbols]
    """
    data_cfg = cfg.get("data", {})
    syn_cfg = cfg.get("synthetic", {})
    exchange = data_cfg.get("exchange", "binance")
    base_dir = data_cfg.get("base_dir", "data")

    store = DataStore(base_dir=base_dir)
    symbols = store.list_symbols(exchange, "1d")

    if not symbols and generate_if_missing:
        logger.warning("No data found — generating synthetic data")
        from experiments.generate_synthetic_data import generate_all
        generate_all(
            n_assets=syn_cfg.get("n_assets", 20),
            n_days=syn_cfg.get("n_days", 1000),
            start_date=syn_cfg.get("start_date", "2021-01-01"),
            seed=syn_cfg.get("seed", 42),
            exchange=exchange,
            base_dir=base_dir,
        )
        symbols = store.list_symbols(exchange, "1d")

    if not symbols:
        raise RuntimeError(
            "No data available. Run: python experiments/run_data_fetch.py --synthetic"
        )

    prices = store.build_panel(symbols, exchange, "1d", field="close")
    volume = store.build_volume_panel(symbols, exchange, "1d")
    fe = FeatureEngine()
    returns = fe.log_returns(prices)
    logger.info("Loaded %d symbols × %d days", len(symbols), len(prices))
    return prices, volume, returns


def run_bt(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    execution_model: Optional[ExecutionModel] = None,
    rebal_frequency: int = 1,
    dollar_neutral: bool = True,
) -> Tuple[pd.Series, dict]:
    """Run UnconstrainedBacktest and return (net_returns, full_result_dict)."""
    em = execution_model or ExecutionModel()
    bt = UnconstrainedBacktest(
        returns=returns,
        signals=signal,
        execution_model=em,
        rebal_frequency=rebal_frequency,
        dollar_neutral=dollar_neutral,
    )
    result = bt.run()
    return result["returns"], result


def backtest_signals(
    signals: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    execution_model: Optional[ExecutionModel] = None,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    """
    Batch-backtest multiple signals, return summary DataFrame.

    Returns:
        DataFrame with signal names as rows and metrics as columns:
        sharpe, annualized_return, annualized_vol, max_drawdown,
        avg_turnover, gross_sharpe, cost_drag
    """
    em = execution_model or ExecutionModel()
    rows = {}
    for name, sig in signals.items():
        net_rets, result = run_bt(sig, returns, em)
        gross_rets = result["gross_returns"]
        turnover = result["turnover"]
        sharpe = _pm.sharpe_ratio(net_rets, periods_per_year=periods_per_year)
        gross_sharpe = _pm.sharpe_ratio(gross_rets, periods_per_year=periods_per_year)
        rows[name] = {
            "sharpe":            sharpe,
            "gross_sharpe":      gross_sharpe,
            "cost_drag":         gross_sharpe - sharpe,
            "annualized_return": _pm.annualized_return(net_rets, periods_per_year),
            "annualized_vol":    _pm.annualized_volatility(net_rets, periods_per_year),
            "max_drawdown":      _pm.max_drawdown(net_rets),
            "avg_turnover":      float(turnover.mean()),
        }
        logger.info("%-40s | Sharpe %5.2f | MDD %6.1f%% | TO %.1f%%",
                    name,
                    sharpe,
                    _pm.max_drawdown(net_rets) * 100,
                    float(turnover.mean()) * 100)
    return pd.DataFrame(rows).T


def save_results(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p)
    logger.info("Results saved → %s", p)


def save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save matplotlib Figure to PNG, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", p)


def signal_decay_ic(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    max_horizon: int = 20,
) -> pd.Series:
    """
    Compute rank IC (Spearman) between signal and forward returns at each horizon.

    Returns:
        pd.Series indexed by horizon (1..max_horizon) with IC values.
    """
    ics = {}
    for h in range(1, max_horizon + 1):
        fwd = returns.shift(-h)
        common = signal.index.intersection(fwd.index)
        sig_flat = signal.loc[common].stack().dropna()
        ret_flat = fwd.loc[common].stack().reindex(sig_flat.index).dropna()
        s = sig_flat.reindex(ret_flat.index).dropna()
        r = ret_flat.reindex(s.index)
        if len(s) < 50:
            ics[h] = np.nan
        else:
            ics[h] = float(s.corr(r, method="spearman"))
    return pd.Series(ics)


def plot_signal_decay(
    decay_dict: Dict[str, pd.Series],
    title: str = "Signal Decay (Rank IC by Forward Horizon)",
) -> plt.Figure:
    """Plot signal IC decay for one or multiple signals."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, ic_series in decay_dict.items():
        ax.plot(ic_series.index, ic_series.values, marker="o", markersize=4,
                linewidth=1.5, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Forward Horizon (days)")
    ax.set_ylabel("Rank IC (Spearman)")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    return fig


def print_summary_table(df: pd.DataFrame, title: str = "") -> None:
    """Pretty-print a summary results DataFrame."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print("=" * 70)
    fmt = df.copy()
    pct_cols = ["annualized_return", "annualized_vol", "max_drawdown", "avg_turnover"]
    for col in pct_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x*100:.1f}%")
    ratio_cols = ["sharpe", "gross_sharpe", "cost_drag"]
    for col in ratio_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.2f}")
    print(fmt.to_string())
    print()
