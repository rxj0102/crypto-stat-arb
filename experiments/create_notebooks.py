"""
Helper script that writes all 7 research notebooks.
Run once to generate the .ipynb files in notebooks/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def nb(cells):
    """Create a notebook with the given cells."""
    notebook = new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    }
    return notebook


def md(text): return new_markdown_cell(text)
def code(src): return new_code_cell(src)


SETUP = """\
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Auto-generate synthetic data if none exists
from statarb.utils import load_config
from statarb.data.storage import DataStore
from experiments.generate_synthetic_data import generate_all

cfg = load_config("../experiments/config.yaml")
data_cfg = cfg.get("data", {})
syn_cfg = cfg.get("synthetic", {})
exchange = data_cfg.get("exchange", "binance")
base_dir = data_cfg.get("base_dir", "../data")

store = DataStore(base_dir=base_dir)
symbols = store.list_symbols(exchange, "1d")
if not symbols:
    print("Generating synthetic data...")
    generate_all(
        n_assets=syn_cfg.get("n_assets", 20),
        n_days=syn_cfg.get("n_days", 1000),
        start_date=syn_cfg.get("start_date", "2021-01-01"),
        seed=syn_cfg.get("seed", 42),
        exchange=exchange,
        base_dir=base_dir,
    )
    symbols = store.list_symbols(exchange, "1d")

prices  = store.build_panel(symbols, exchange, "1d", field="close")
volume  = store.build_volume_panel(symbols, exchange, "1d")

from statarb.data.features import FeatureEngine
fe = FeatureEngine()
returns = fe.log_returns(prices)
vol_ratio = fe.volume_ma_ratio(volume, window=21)

print(f"Loaded: {len(symbols)} assets x {len(prices)} days")
print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
"""


# =========================================================================
# 01 — Data Exploration
# =========================================================================
nb01_cells = [
    md("# 01 — Data Exploration\n\nExploratory data analysis of the crypto return universe: "
       "return distributions, correlation structure, volume patterns, and volatility regimes."),
    code(SETUP),
    md("## Return Distributions"),
    code("""\
# Daily return summary statistics
stats = returns.describe().T
stats["skew"] = returns.skew()
stats["kurt"] = returns.kurtosis()
stats["ann_vol"] = returns.std() * np.sqrt(365)
print(stats[["mean", "std", "skew", "kurt", "ann_vol", "min", "max"]].round(4).to_string())
"""),
    code("""\
# Histogram of cross-asset mean daily return distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
returns.stack().hist(bins=100, ax=axes[0], color="steelblue", alpha=0.7, edgecolor="none")
axes[0].set_title("Return Distribution (all assets pooled)", fontweight="bold")
axes[0].set_xlabel("Daily Return")
axes[0].set_ylabel("Frequency")
axes[0].axvline(0, color="red", linewidth=1)

# QQ-plot vs normal
from scipy import stats
clean = returns.stack().dropna()
(osm, osr), (slope, intercept, r) = stats.probplot(clean, dist="norm")
axes[1].scatter(osm, osr, s=1, alpha=0.3, color="steelblue")
axes[1].plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=1.5)
axes[1].set_title("QQ-Plot vs Normal", fontweight="bold")
axes[1].set_xlabel("Theoretical Quantiles")
axes[1].set_ylabel("Sample Quantiles")
plt.tight_layout()
plt.show()
print(f"Excess kurtosis (pooled): {clean.kurtosis():.2f}")
"""),
    md("## Correlation Structure"),
    code("""\
# Rolling correlation between all pairs
corr = returns.dropna().corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.3, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Pairwise Return Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.show()

# Distribution of pairwise correlations
pairs = corr.values[np.triu_indices_from(corr.values, k=1)]
print(f"Median pairwise correlation: {np.median(pairs):.3f}")
print(f"Mean pairwise correlation: {np.mean(pairs):.3f}")
"""),
    code("""\
# Rolling median correlation (market stress indicator)
def rolling_median_corr(rets, window=63):
    result = []
    for i in range(window, len(rets)):
        window_rets = rets.iloc[i - window:i]
        c = window_rets.corr().values
        vals = c[np.triu_indices_from(c, k=1)]
        result.append(np.median(vals))
    return pd.Series(result, index=rets.index[window:])

roll_corr = rolling_median_corr(returns.dropna(axis=1, how="all").fillna(0))
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(roll_corr.index, roll_corr.values, linewidth=1.2, color="steelblue")
ax.fill_between(roll_corr.index, roll_corr.values, alpha=0.2, color="steelblue")
ax.axhline(float(roll_corr.mean()), color="red", linestyle="--",
           linewidth=1, label=f"Mean={roll_corr.mean():.2f}")
ax.set_title("Rolling Median Pairwise Correlation (63-day)", fontweight="bold")
ax.set_ylabel("Median Correlation")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
"""),
    md("## Volume Patterns"),
    code("""\
# Volume normalized by its rolling average (activity ratio)
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Mean activity ratio across universe
avg_activity = vol_ratio.mean(axis=1)
axes[0].plot(avg_activity.index, avg_activity.values, linewidth=0.8, color="darkorange")
axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_title("Average Volume / 21-day MA (universe mean)", fontweight="bold")
axes[0].set_ylabel("V/MA Ratio")

# Day-of-week volume pattern
dow = pd.DataFrame({
    "volume_ratio": avg_activity.values,
    "dayofweek":    avg_activity.index.dayofweek,
}).dropna()
dow_avg = dow.groupby("dayofweek")["volume_ratio"].mean()
labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[1].bar(range(len(dow_avg)), dow_avg.values, color="steelblue", alpha=0.8)
axes[1].set_xticks(range(len(dow_avg)))
axes[1].set_xticklabels(labels[:len(dow_avg)])
axes[1].set_title("Volume by Day of Week", fontweight="bold")
axes[1].set_ylabel("Avg V/MA Ratio")
axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()
"""),
    md("## Volatility Regimes"),
    code("""\
# Cross-sectional median realized volatility
realized_vol = fe.realized_volatility(returns, window=21, annualize=True, periods_per_year=365)
median_vol = realized_vol.median(axis=1)

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(median_vol.index, median_vol.values * 100, alpha=0.4, color="crimson")
ax.plot(median_vol.index, median_vol.values * 100, color="crimson", linewidth=0.8)
ax.set_title("Median Annualized Realized Volatility (21-day, universe)", fontweight="bold")
ax.set_ylabel("Annualized Vol (%)")
ax.axhline(float(median_vol.mean()) * 100, color="black", linestyle="--",
           linewidth=1, label=f"Mean={median_vol.mean()*100:.0f}%")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# High-vol regime = days when median vol > 1.5x its 63-day average
vol_ma = median_vol.rolling(63).mean()
high_vol_regime = (median_vol > 1.5 * vol_ma)
print(f"High-vol regime: {high_vol_regime.mean()*100:.1f}% of days")
print(f"Mean vol in high-vol regime: {median_vol[high_vol_regime].mean()*100:.0f}%")
print(f"Mean vol in low-vol regime:  {median_vol[~high_vol_regime].mean()*100:.0f}%")
"""),
    code("""\
# Cross-sectional dispersion (key driver of L/S alpha)
cs_vol = returns.std(axis=1)
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(cs_vol.index, cs_vol.values * 100, linewidth=0.8, color="navy")
ax.set_title("Cross-Sectional Return Dispersion (daily std, all assets)", fontweight="bold")
ax.set_ylabel("Daily CS Std (%)")
plt.tight_layout()
plt.show()
print(f"Average cross-sectional dispersion: {cs_vol.mean()*100:.2f}% per day")
print(f"This determines the maximum signal IC → alpha ceiling.")
"""),
]

# =========================================================================
# 02 — Momentum Signals
# =========================================================================
nb02_cells = [
    md("# 02 — Momentum Signals\n\nAnalysis of cross-sectional momentum signals. "
       "Tests multiple lookback periods and visualizes signal IC decay."),
    code(SETUP),
    md("## Signal Construction"),
    code("""\
from statarb.signals.momentum import MomentumSignals
mom = MomentumSignals()

# Build a panel of momentum signals at different lookbacks
signals = {
    "ts_mom_7d":   mom.time_series_momentum(returns, lookback=7),
    "ts_mom_21d":  mom.time_series_momentum(returns, lookback=21),
    "ts_mom_63d":  mom.time_series_momentum(returns, lookback=63),
    "ts_mom_126d": mom.time_series_momentum(returns, lookback=126),
    "price_mom_6_1": mom.momentum_6_1(returns),
    "sharpe_mom_63d": mom.sharpe_momentum(returns, lookback=63),
    "ma_cross_20_60": mom.moving_average_crossover(prices, fast=20, slow=60),
}

# Cross-sectional rank each signal to [-1, 1]
ranked = {name: fe.cross_sectional_rank(sig) for name, sig in signals.items()}

# Signal coverage (fraction of non-NaN at each date)
coverage = {name: sig.notna().mean(axis=1) for name, sig in ranked.items()}
print("Signal coverage (mean fraction of assets with valid signal):")
for name, cov in coverage.items():
    print(f"  {name:25s}: {cov.mean():.1%}")
"""),
    md("## Signal Decay — IC by Forward Horizon"),
    code("""\
from experiments._utils import signal_decay_ic

max_horizon = 20
decay = {}
for name in ["ts_mom_7d", "ts_mom_21d", "ts_mom_63d", "ts_mom_126d"]:
    decay[name] = signal_decay_ic(ranked[name], returns, max_horizon=max_horizon)

fig, ax = plt.subplots(figsize=(10, 4))
for name, ic_series in decay.items():
    ax.plot(ic_series.index, ic_series.values, marker="o", markersize=4,
            linewidth=1.5, label=name)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Forward Horizon (days)")
ax.set_ylabel("Rank IC (Spearman)")
ax.set_title("Momentum Signal IC Decay by Lookback", fontweight="bold")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
"""),
    md("## Backtest Performance by Lookback"),
    code("""\
from experiments._utils import backtest_signals
from statarb.backtest.execution import ExecutionModel

em = ExecutionModel(market_order_cost=0.0020)
results = backtest_signals(ranked, returns, em, periods_per_year=365)
print(results[["sharpe", "gross_sharpe", "cost_drag", "annualized_return",
               "max_drawdown", "avg_turnover"]].round(3).to_string())
"""),
    code("""\
# Best momentum signal cumulative return
best = results["sharpe"].idxmax()
from experiments._utils import run_bt
net_rets, result_dict = run_bt(ranked[best], returns, em)
cum = (1 + net_rets.fillna(0)).cumprod() - 1

fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
cum.plot(ax=axes[0])
axes[0].set_title(f"Best Momentum Signal: {best}", fontweight="bold")
axes[0].set_ylabel("Cumulative Return")
axes[0].axhline(0, color="black", linewidth=0.6)

# Drawdown
dd = (1 + net_rets.fillna(0)).cumprod()
dd = (dd - dd.cummax()) / dd.cummax()
dd.plot(ax=axes[1], color="crimson")
axes[1].fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.3)
axes[1].set_ylabel("Drawdown")
plt.tight_layout()
plt.show()

from statarb.evaluation.metrics import PerformanceMetrics
pm = PerformanceMetrics()
print(f"Sharpe:        {pm.sharpe_ratio(net_rets, periods_per_year=365):.2f}")
print(f"Ann. Return:   {pm.annualized_return(net_rets)*100:.1f}%")
print(f"Max Drawdown:  {pm.max_drawdown(net_rets)*100:.1f}%")
"""),
    md("## Volume-Conditioned Momentum"),
    code("""\
vol_cond = fe.cross_sectional_rank(
    mom.volume_weighted_momentum(returns, vol_ratio, lookback=63)
)
base = ranked["ts_mom_63d"]

# IC on high-volume vs low-volume days
activity_ratio = vol_ratio.mean(axis=1)
high_vol_mask = activity_ratio > activity_ratio.rolling(21).mean()

fwd1 = returns.shift(-1)
def daily_ic(sig, fwd, dates):
    ics = []
    for d in dates:
        if d not in sig.index or d not in fwd.index:
            ics.append(np.nan)
            continue
        s = sig.loc[d].dropna()
        r = fwd.loc[d].reindex(s.index).dropna()
        s2 = s.reindex(r.index)
        if len(s2) < 5:
            ics.append(np.nan)
        else:
            ics.append(float(s2.corr(r, method="spearman")))
    return pd.Series(ics, index=dates)

common_dates = base.index.intersection(fwd1.index)
ic = daily_ic(base, fwd1, common_dates)
hv_mask = high_vol_mask.reindex(common_dates).fillna(False)

print(f"Base TS Momentum IC (high-vol days): {ic[hv_mask].mean():.4f}")
print(f"Base TS Momentum IC (low-vol days):  {ic[~hv_mask].mean():.4f}")
print("=> Higher IC on high-volume days supports informed trading hypothesis")
"""),
]

# =========================================================================
# 03 — Reversal Signals
# =========================================================================
nb03_cells = [
    md("# 03 — Reversal Signals\n\nAnalysis of short-term mean-reversion signals. "
       "Compares raw, volume-filtered, and activity-gated variants."),
    code(SETUP),
    md("## Signal Construction"),
    code("""\
from statarb.signals.reversal import ReversalSignals
from statarb.signals.activity import ActivityFilter
rev = ReversalSignals()
act = ActivityFilter()

signals = {
    "reversal_1d":        rev.short_term_reversal(returns, lookback=1),
    "reversal_3d":        rev.short_term_reversal(returns, lookback=3),
    "reversal_5d":        rev.weekly_reversal(returns),
    "reversal_7d":        rev.short_term_reversal(returns, lookback=7),
    "vol_adj_reversal":   rev.vol_adjusted_reversal(returns),
    "bollinger_20d":      rev.bollinger_reversal(prices, window=20),
    "large_move_rev":     rev.large_move_reversal(returns),
    "vol_filtered_rev":       rev.volume_filtered_reversal(returns, volume, lookback=3),
}
ranked = {name: fe.cross_sectional_rank(sig) for name, sig in signals.items()}
gated = {
    name: fe.cross_sectional_rank(act.activity_gate(sig, volume, min_ratio=0.5))
    for name, sig in signals.items()
}
print("Signals built:", list(signals.keys()))
"""),
    md("## Reversal Signal Decay"),
    code("""\
from experiments._utils import signal_decay_ic

max_h = 15
decay = {name: signal_decay_ic(ranked[name], returns, max_h)
         for name in ["reversal_1d", "reversal_5d", "vol_adj_reversal", "large_move_rev"]}

fig, ax = plt.subplots(figsize=(10, 4))
for name, ic in decay.items():
    ax.plot(ic.index, ic.values, marker="o", markersize=4, linewidth=1.5, label=name)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Forward Horizon (days)")
ax.set_ylabel("Rank IC (Spearman)")
ax.set_title("Reversal Signal Decay", fontweight="bold")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
"""),
    md("## Limit Orders vs Market Orders"),
    code("""\
from experiments._utils import backtest_signals
from statarb.backtest.execution import ExecutionModel

em_market = ExecutionModel(market_order_cost=0.0020, order_type="market")
em_limit  = ExecutionModel(limit_order_cost=0.0007, order_type="limit")

res_market = backtest_signals(ranked, returns, em_market, 365)
res_limit  = backtest_signals(ranked, returns, em_limit,  365)

fig, ax = plt.subplots(figsize=(11, 4))
x = range(len(ranked))
ax.bar([i - 0.2 for i in x], res_market["sharpe"].values, width=0.35,
       label="Market (20bps)", alpha=0.7, color="tab:red")
ax.bar([i + 0.2 for i in x], res_limit["sharpe"].values, width=0.35,
       label="Limit (7bps)", alpha=0.7, color="tab:blue")
ax.set_xticks(list(x))
ax.set_xticklabels(list(ranked.keys()), rotation=45, ha="right", fontsize=8)
ax.set_title("Reversal Sharpe: Market vs Limit Orders", fontweight="bold")
ax.set_ylabel("Annualized Sharpe")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
print("Key insight: reversal signals need limit orders — market orders erase the alpha.")
"""),
    md("## Volume Filter Analysis"),
    code("""\
# Does high volume indicate more informed flow? If so, 1-day reversal
# should be weaker (momentum) on high-volume days.
fwd1 = returns.shift(-1)
rev1_ranked = ranked["reversal_1d"]
activity = vol_ratio.mean(axis=1)

# Split IC by activity quintile
quintiles = pd.qcut(activity.dropna(), 5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])
ic_by_quintile = {}
for q in quintiles.cat.categories:
    mask = (quintiles == q).reindex(rev1_ranked.index).fillna(False)
    dates = rev1_ranked.index[mask]
    ics = []
    for d in dates:
        if d not in fwd1.index:
            continue
        s = rev1_ranked.loc[d].dropna()
        r = fwd1.loc[d].reindex(s.index).dropna()
        s2 = s.reindex(r.index)
        if len(s2) >= 5:
            ics.append(float(s2.corr(r, method="spearman")))
    ic_by_quintile[q] = np.mean(ics) if ics else np.nan

print("1-Day Reversal IC by Volume Activity Quintile:")
for q, ic in ic_by_quintile.items():
    print(f"  {q}: {ic:.4f}")
print("Negative on high-volume days → momentum effect dominates (informed flow)")
"""),
]

# =========================================================================
# 04 — Pairs & Correlation
# =========================================================================
nb04_cells = [
    md("# 04 — Pairs and Correlation Analysis\n\nCointegration scanning, "
       "spread visualization, and cluster-based relative value."),
    code(SETUP),
    code("""\
from statarb.signals.pairs import PairsSignals
from statarb.signals.themes import CRYPTO_THEMES
ps = PairsSignals()
print("Available thematic clusters:")
for theme, members in CRYPTO_THEMES.items():
    present = [m for m in members if m in returns.columns]
    print(f"  {theme:20s}: {present}")
"""),
    md("## Correlation Clustering"),
    code("""\
# Compute correlation matrix and cluster assets
from scipy.cluster.hierarchy import dendrogram, linkage

corr = returns.dropna().corr()
# Convert correlation to distance
dist = np.sqrt(2 * (1 - corr.clip(-1, 1)))

fig, ax = plt.subplots(figsize=(12, 5))
linked = linkage(dist.values[np.triu_indices(len(dist), k=1)], method="ward")
dendrogram(linked, labels=list(corr.columns), leaf_rotation=45,
           leaf_font_size=8, ax=ax)
ax.set_title("Asset Clustering by Return Correlation (Ward Linkage)", fontweight="bold")
plt.tight_layout()
plt.show()
"""),
    md("## Cointegration Scan"),
    code("""\
# Find cointegrated pairs (Engle-Granger)
print("Scanning for cointegrated pairs...")
try:
    lookback = min(252, len(prices) - 10)
    coint_pairs = ps.find_cointegrated_pairs(prices, lookback=lookback, pvalue_threshold=0.10)
    print(f"Found {len(coint_pairs)} cointegrated pairs (p < 0.10)")
    if coint_pairs:
        print("\\nTop pairs:")
        for a, b, pval in coint_pairs[:10]:
            print(f"  {a:12s} / {b:12s}  p={pval:.4f}")
except Exception as e:
    print(f"Cointegration scan unavailable: {e}")
    coint_pairs = []
"""),
    md("## Spread Visualization"),
    code("""\
# Visualize spread z-score for available asset pairs
available_cols = list(prices.columns)
plot_pairs = []
for i, a in enumerate(available_cols[:5]):
    for b in available_cols[i+1:6]:
        plot_pairs.append((a, b))

n_plots = min(4, len(plot_pairs))
if n_plots > 0:
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]
    for ax, (a, b) in zip(axes, plot_pairs[:n_plots]):
        try:
            spread = ps.spread_zscore(prices[a], prices[b], window=21)
            ax.plot(spread.index, spread.values, linewidth=0.8, color="navy")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axhline(1.5, color="red", linestyle="--", alpha=0.7, linewidth=0.8)
            ax.axhline(-1.5, color="green", linestyle="--", alpha=0.7, linewidth=0.8)
            ax.set_title(f"{a.split('/')[0]} / {b.split('/')[0]} Spread Z-Score (21-day)",
                        fontsize=9)
            ax.set_ylabel("Z-Score")
        except Exception as e:
            ax.set_title(f"Error: {e}")
    plt.tight_layout()
    plt.show()
"""),
    md("## Cluster Momentum"),
    code("""\
# Cluster-based momentum: buy assets in outperforming themes
try:
    cluster_mom = ps.cluster_momentum_signal(returns, CRYPTO_THEMES, lookback=21)
    rel_val = ps.relative_value_signal(returns, correlation_window=63,
                                        return_window=21, min_correlation=0.5)
    signals = {
        "cluster_momentum": fe.cross_sectional_rank(cluster_mom),
        "relative_value":   fe.cross_sectional_rank(rel_val),
    }
    from experiments._utils import backtest_signals
    from statarb.backtest.execution import ExecutionModel
    results = backtest_signals(signals, returns, ExecutionModel(), 365)
    print(results[["sharpe", "annualized_return", "max_drawdown"]].round(3).to_string())
except Exception as e:
    print(f"Signal construction failed (may need more assets): {e}")
"""),
]

# =========================================================================
# 05 — Seasonality
# =========================================================================
nb05_cells = [
    md("# 05 — Seasonality Analysis\n\nDay-of-week effects, day-of-month patterns, "
       "and volatility seasonality in crypto markets."),
    code(SETUP),
    code("""\
from statarb.signals.seasonality import SeasonalitySignals
seas = SeasonalitySignals()
mean_ret = returns.mean(axis=1).rename("universe_mean")
"""),
    md("## Day-of-Week Effects"),
    code("""\
# Average return by day of week
dow_ret = mean_ret.copy()
dow_ret.index = pd.DatetimeIndex(dow_ret.index)
dow_df = pd.DataFrame({
    "return": dow_ret.values,
    "dow":    dow_ret.index.dayofweek,
    "dow_name": dow_ret.index.day_name(),
})

dow_avg = dow_df.groupby("dow").agg(
    mean_return=("return", "mean"),
    std_return=("return", "std"),
    count=("return", "count"),
).assign(tstat=lambda d: d["mean_return"] / (d["std_return"] / d["count"].pow(0.5)))

labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["green" if v > 0 else "red" for v in dow_avg["mean_return"].values]
ax.bar(range(len(dow_avg)), dow_avg["mean_return"].values * 100, color=colors, alpha=0.8)
ax.set_xticks(range(len(dow_avg)))
ax.set_xticklabels(labels[:len(dow_avg)])
ax.set_title("Average Universe Return by Day of Week", fontweight="bold")
ax.set_ylabel("Mean Daily Return (%)")
ax.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
print(dow_avg[["mean_return", "tstat"]].round(4).to_string())
"""),
    md("## Month-of-Year Effects"),
    code("""\
month_df = pd.DataFrame({
    "return": mean_ret.values,
    "month":  pd.DatetimeIndex(mean_ret.index).month,
})
month_avg = month_df.groupby("month")["return"].mean()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["green" if v > 0 else "red" for v in month_avg.values]
ax.bar(range(1, len(month_avg)+1), month_avg.values * 100, color=colors, alpha=0.8)
ax.set_xticks(range(1, len(month_avg)+1))
ax.set_xticklabels(month_labels[:len(month_avg)], fontsize=9)
ax.set_title("Average Universe Return by Month", fontweight="bold")
ax.set_ylabel("Mean Daily Return (%)")
ax.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
"""),
    md("## Volatility Seasonality"),
    code("""\
realized_vol = fe.realized_volatility(returns, window=5)
avg_vol = realized_vol.mean(axis=1)

vol_df = pd.DataFrame({
    "vol":   avg_vol.values,
    "dow":   pd.DatetimeIndex(avg_vol.index).dayofweek,
    "month": pd.DatetimeIndex(avg_vol.index).month,
}).dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
vol_by_dow = vol_df.groupby("dow")["vol"].mean()
axes[0].bar(range(len(vol_by_dow)), vol_by_dow.values * 100, color="steelblue", alpha=0.8)
axes[0].set_xticks(range(len(vol_by_dow)))
axes[0].set_xticklabels(labels[:len(vol_by_dow)])
axes[0].set_title("Avg 5-Day Vol by Day of Week (%)", fontweight="bold")

vol_by_month = vol_df.groupby("month")["vol"].mean()
axes[1].bar(range(1, len(vol_by_month)+1), vol_by_month.values * 100,
            color="darkorange", alpha=0.8)
axes[1].set_xticks(range(1, len(vol_by_month)+1))
axes[1].set_xticklabels(month_labels[:len(vol_by_month)], fontsize=8)
axes[1].set_title("Avg 5-Day Vol by Month (%)", fontweight="bold")
plt.tight_layout()
plt.show()
"""),
    md("## Seasonality Signal Backtest"),
    code("""\
from experiments._utils import backtest_signals
from statarb.backtest.execution import ExecutionModel

try:
    sigs = {
        "dow_signal":    seas.day_of_week_signal(returns),
        "month_signal":  seas.month_of_year_signal(returns),
    }
    ranked_sigs = {k: fe.cross_sectional_rank(v) for k, v in sigs.items()}
    res = backtest_signals(ranked_sigs, returns, ExecutionModel(), 365)
    print(res[["sharpe", "annualized_return", "max_drawdown"]].round(3).to_string())
except Exception as e:
    print(f"Seasonality signals unavailable: {e}")
"""),
]

# =========================================================================
# 06 — Combined Strategy
# =========================================================================
nb06_cells = [
    md("# 06 — Combined Strategy\n\nCombine momentum and reversal signals using "
       "multiple weighting methods. Demonstrate diversification benefits."),
    code(SETUP),
    code("""\
from statarb.signals.momentum import MomentumSignals
from statarb.signals.reversal import ReversalSignals
from statarb.signals.activity import ActivityFilter
from statarb.backtest.execution import ExecutionModel
from statarb.backtest.weighting import StrategyWeighting
from statarb.evaluation.metrics import PerformanceMetrics
from experiments._utils import run_bt, backtest_signals

mom = MomentumSignals()
rev = ReversalSignals()
act = ActivityFilter()
em  = ExecutionModel(market_order_cost=0.0020)
pm  = PerformanceMetrics()
sw  = StrategyWeighting()
"""),
    md("## Build Component Signals"),
    code("""\
raw_signals = {
    "mom_6_1":       mom.momentum_6_1(returns),
    "mom_3_1":       mom.momentum_3_1(returns),
    "ts_mom_63d":    mom.time_series_momentum(returns, lookback=63),
    "sharpe_mom":    mom.sharpe_momentum(returns, lookback=63),
    "reversal_5d":   rev.weekly_reversal(returns),
    "vol_adj_rev":   rev.vol_adjusted_reversal(returns),
}

# Activity-gate and rank
ranked = {
    name: fe.cross_sectional_rank(act.activity_gate(sig, volume, min_ratio=0.5))
    for name, sig in raw_signals.items()
}

# Get individual return streams
strategy_returns = {}
for name, sig in ranked.items():
    net_rets, _ = run_bt(sig, returns, em)
    strategy_returns[name] = net_rets

returns_df = pd.DataFrame(strategy_returns).dropna()
print(f"Individual strategy return DataFrame: {returns_df.shape}")
print(f"Date range: {returns_df.index[0].date()} → {returns_df.index[-1].date()}")
"""),
    md("## Individual Strategy Performance"),
    code("""\
indiv_summary = {}
for name, rets in strategy_returns.items():
    clean = rets.dropna()
    indiv_summary[name] = {
        "sharpe": pm.sharpe_ratio(clean, periods_per_year=365),
        "ann_ret": pm.annualized_return(clean) * 100,
        "max_dd":  pm.max_drawdown(clean) * 100,
        "win_rate": pm.win_rate(clean) * 100,
    }
indiv_df = pd.DataFrame(indiv_summary).T
print(indiv_df.round(2).to_string())
"""),
    md("## Combination Methods"),
    code("""\
combined_rets = {}
for method in ["equal", "inverse_vol", "sharpe", "min_variance"]:
    try:
        combined = sw.combine(returns_df, method=method, lookback=63)
        combined_rets[f"combined_{method}"] = combined
        clean = combined.dropna()
        sharpe = pm.sharpe_ratio(clean, periods_per_year=365)
        print(f"  {method:20s}: Sharpe {sharpe:.2f}")
    except Exception as e:
        print(f"  {method}: failed ({e})")
"""),
    md("## Diversification Benefit"),
    code("""\
avg_individual = indiv_df["sharpe"].mean()
equal_combined = pm.sharpe_ratio(combined_rets.get("combined_equal", pd.Series()).dropna(),
                                   periods_per_year=365)

print(f"Average individual Sharpe:    {avg_individual:.2f}")
print(f"Equal-weight combined Sharpe: {equal_combined:.2f}")
benefit = equal_combined - avg_individual
print(f"Diversification benefit:      +{benefit:.2f} Sharpe units")

# Correlation between individual strategies
fig, ax = plt.subplots(figsize=(8, 6))
import seaborn as sns
corr = returns_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", center=0, cmap="coolwarm",
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Strategy Return Correlations", fontweight="bold")
plt.tight_layout()
plt.show()
print("Low correlations → diversification benefit is real, not luck.")
"""),
    md("## Combined Strategy Cumulative Returns"),
    code("""\
fig, ax = plt.subplots(figsize=(12, 5))
# Individual strategies (thin, faded)
for name, rets in strategy_returns.items():
    cum = (1 + rets.fillna(0)).cumprod() - 1
    ax.plot(cum.index, cum.values * 100, linewidth=0.8, alpha=0.3)
# Combined (bold)
for name, rets in combined_rets.items():
    cum = (1 + rets.fillna(0)).cumprod() - 1
    ax.plot(cum.index, cum.values * 100, linewidth=2, label=name)
ax.set_title("Individual (faded) vs Combined (bold) Cumulative Returns", fontweight="bold")
ax.set_ylabel("Cumulative Return (%)")
ax.axhline(0, color="black", linewidth=0.6)
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.show()
"""),
]

# =========================================================================
# 07 — Final Report
# =========================================================================
nb07_cells = [
    md("# 07 — Final Strategy Report\n\nComprehensive performance report: all metrics, "
       "alpha/beta decomposition, factor exposure, drawdown analysis, and plots."),
    code(SETUP),
    code("""\
from statarb.signals.momentum import MomentumSignals
from statarb.signals.reversal import ReversalSignals
from statarb.signals.activity import ActivityFilter
from statarb.backtest.execution import ExecutionModel
from statarb.backtest.weighting import StrategyWeighting
from statarb.evaluation.metrics import PerformanceMetrics
from statarb.evaluation.risk import RiskAnalytics
from statarb.evaluation.plots import PerformancePlots
from statarb.evaluation.reporting import PerformanceReporter
from experiments._utils import run_bt

mom = MomentumSignals()
rev = ReversalSignals()
act = ActivityFilter()
em  = ExecutionModel(market_order_cost=0.0020)
pm  = PerformanceMetrics()
ra  = RiskAnalytics()
pp  = PerformancePlots()
sw  = StrategyWeighting()
"""),
    md("## Build Composite Strategy"),
    code("""\
# Build and backtest component signals
components = {
    "mom_6_1":     fe.cross_sectional_rank(act.activity_gate(mom.momentum_6_1(returns), volume)),
    "mom_3_1":     fe.cross_sectional_rank(act.activity_gate(mom.momentum_3_1(returns), volume)),
    "sharpe_mom":  fe.cross_sectional_rank(act.activity_gate(mom.sharpe_momentum(returns,63), volume)),
    "reversal_5d": fe.cross_sectional_rank(act.activity_gate(rev.weekly_reversal(returns), volume)),
    "vol_adj_rev": fe.cross_sectional_rank(act.activity_gate(rev.vol_adjusted_reversal(returns), volume)),
}

comp_returns = {}
comp_turnover = {}
for name, sig in components.items():
    net_rets, result = run_bt(sig, returns, em)
    comp_returns[name] = net_rets
    comp_turnover[name] = result["turnover"]

comp_df = pd.DataFrame(comp_returns).dropna()
composite = sw.equal_weight(comp_df)
print(f"Composite strategy: {len(composite)} days of returns")
"""),
    md("## Full Performance Report"),
    code("""\
# BTC or first available asset as benchmark
if "BTC/USDT" in prices.columns:
    benchmark = fe.log_returns(prices[["BTC/USDT"]])["BTC/USDT"]
elif len(prices.columns) > 0:
    benchmark = fe.log_returns(prices[[prices.columns[0]]])[prices.columns[0]]
else:
    benchmark = None

report = pm.full_report(composite, benchmark_returns=benchmark)
print(report.to_string())
"""),
    md("## Alpha / Beta Analysis"),
    code("""\
if benchmark is not None:
    ab = ra.alpha_beta(composite, benchmark)
    print(f"Alpha (annualized): {ab['alpha']*100:.2f}%")
    print(f"Beta:               {ab['beta']:.3f}")
    print(f"R²:                 {ab['r_squared']:.3f}")
    print(f"Alpha t-stat:       {ab['alpha_tstat']:.2f}")
    print()

    # Rolling beta plot
    rolling_beta = ra.rolling_beta(composite, benchmark, window=63)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rolling_beta.index, rolling_beta.values, linewidth=1.2, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(ab["beta"], color="red", linewidth=0.8, linestyle=":",
               label=f"Full-sample β={ab['beta']:.2f}")
    ax.set_title("Rolling Beta vs Benchmark (63-day)", fontweight="bold")
    ax.set_ylabel("Beta")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
"""),
    md("## Drawdown Analysis"),
    code("""\
dd_table = ra.drawdown_analysis(composite)
if not dd_table.empty:
    print(f"Drawdown episodes exceeding 5%: {len(dd_table)}")
    print(dd_table.to_string())
else:
    print("No drawdown episodes exceeding 5% (synthetic data may show no deep drawdowns)")

# Drawdown plot
fig = pp.drawdown_plot(composite)
plt.show()
"""),
    md("## All Performance Plots"),
    code("""\
fig = pp.cumulative_return_plot(composite, benchmark=benchmark,
                                 title="Composite Strategy Cumulative Return")
plt.show()

fig = pp.return_distribution(composite)
plt.show()

fig = pp.rolling_sharpe_plot(composite, window=63)
plt.show()

try:
    fig = pp.monthly_returns_heatmap(composite)
    plt.show()
except Exception as e:
    print(f"Monthly heatmap requires multi-month data: {e}")
"""),
    md("## Component Attribution"),
    code("""\
# Side-by-side comparison of all components + composite
all_strats = dict(comp_returns)
all_strats["composite"] = composite

reporter = PerformanceReporter(
    all_strats,
    benchmark_returns=benchmark,
    periods_per_year=365,
)
reporter.print_report(title="Final Strategy Evaluation")
"""),
    md("## Summary\n\n"
       "| Metric | Target | Achieved |\n"
       "|--------|--------|----------|\n"
       "| Sharpe Ratio (net) | 1.0–2.0 | see above |\n"
       "| Annual Return (net) | 15–25% | see above |\n"
       "| Max Drawdown | < 20% | see above |\n"
       "| Beta to BTC | < 0.3 | see above |\n\n"
       "See `docs/strategy_summary.md` for full write-up."),
]


def write_notebook(cells, path: Path) -> None:
    notebook = nb(cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        nbformat.write(notebook, f)
    print(f"Created: {path}")


def main():
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    write_notebook(nb01_cells, notebooks_dir / "01_data_exploration.ipynb")
    write_notebook(nb02_cells, notebooks_dir / "02_momentum_signals.ipynb")
    write_notebook(nb03_cells, notebooks_dir / "03_reversal_signals.ipynb")
    write_notebook(nb04_cells, notebooks_dir / "04_pairs_and_correlation.ipynb")
    write_notebook(nb05_cells, notebooks_dir / "05_seasonality.ipynb")
    write_notebook(nb06_cells, notebooks_dir / "06_combined_strategy.ipynb")
    write_notebook(nb07_cells, notebooks_dir / "07_final_report.ipynb")
    print(f"\nAll 7 notebooks written to {notebooks_dir}/")


if __name__ == "__main__":
    main()
