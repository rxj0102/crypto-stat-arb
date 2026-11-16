"""Momentum and reversal in the cross-section of crypto returns.

Design, fixed before any portfolio return was computed:

  universe    every 7 days, the 50 assets with the highest median dollar volume over the last
              30 days, among assets with a price that day and at least 30 days of history.
              Membership uses only data up to that day, and delisted coins are in the data.
  signals     the return over the past 1, 2, 4, 12 and 26 weeks
  portfolios  sort the universe on a signal; go long the top fifth and short the bottom fifth,
              equal weighted, one unit of capital on each side. Positive returns mean winners
              keep winning (momentum); negative means they give it back (reversal).
  timing      signal at the close of day t, trade at the close of day t+1, hold 7 days to the
              next trade. Positions are bought and held within the week, not rebalanced daily.
  delisting   a coin that stops trading during the week is sold at its last price
  costs       10 bps per unit traded on each side, the Binance spot taker fee
  sample      starts on the first rebalance with at least 20 universe members that have the
              longest signal, so every signal is measured on the same weeks

No parameter was searched. Shorting is assumed possible at no borrowing cost, which is not true
for every coin; the long and short legs are reported separately for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SIGNALS = {"1 week": 7, "2 weeks": 14, "4 weeks": 28, "12 weeks": 84, "26 weeks": 182}
WEEKS_PER_YEAR = 365.25 / 7


@dataclass(frozen=True)
class Config:
    top_n: int = 50
    volume_window: int = 30
    min_history: int = 30
    hold: int = 7
    lag: int = 1
    quantile: float = 0.2
    min_names: int = 20
    cost: float = 0.001
    nw_lags: int = 4


def universe_at(close: pd.DataFrame, vol: pd.DataFrame, t: int, cfg: Config, allowed=None) -> list[str]:
    """Universe on row t, from rows up to t only."""
    px = close.iloc[: t + 1]
    has_price = px.iloc[-1].notna()
    history = px.notna().sum() >= cfg.min_history
    liq = vol.iloc[max(0, t + 1 - cfg.volume_window): t + 1].median()
    ok = has_price & history & liq.notna()
    if allowed is not None:
        ok &= ok.index.isin(allowed)
    return list(liq[ok].sort_values(ascending=False, kind="mergesort").index[: cfg.top_n])


def signal_at(close: pd.DataFrame, t: int, lookback: int, names: list[str]) -> pd.Series:
    """Past return over lookback days for names with a price on both ends."""
    if t - lookback < 0:
        return pd.Series(dtype=float)
    s = close.iloc[t][names] / close.iloc[t - lookback][names] - 1.0
    return s.dropna()


def legs(signal: pd.Series, cfg: Config) -> tuple[list[str], list[str]]:
    """Top and bottom fifth by signal. Ties are broken by name so the split is deterministic."""
    order = signal.sort_index().sort_values(kind="mergesort")
    k = max(1, int(np.floor(len(order) * cfg.quantile)))
    return list(order.index[-k:]), list(order.index[:k])


def holding_returns(close: pd.DataFrame, start: int, end: int, names: list[str]) -> pd.Series:
    """Buy at the close of row start, sell at the close of row end or at the last price before."""
    window = close.iloc[start: end + 1][names].ffill()
    return window.iloc[-1] / window.iloc[0] - 1.0


def weights(long: list[str], short: list[str]) -> pd.Series:
    w = pd.Series(0.0, index=sorted(set(long) | set(short)))
    w[long] += 1.0 / len(long)
    w[short] -= 1.0 / len(short)
    return w


def first_rebalance(close: pd.DataFrame, vol: pd.DataFrame, cfg: Config) -> int:
    longest = max(SIGNALS.values())
    for t in range(longest, len(close) - cfg.lag - cfg.hold, 1):
        names = universe_at(close, vol, t, cfg)
        if len(signal_at(close, t, longest, names)) >= cfg.min_names:
            return t
    raise ValueError("sample too short")


def run(close: pd.DataFrame, vol: pd.DataFrame, cfg: Config = Config(), allowed=None, start: int | None = None) -> dict:
    """Weekly returns of every long-short portfolio and of the equal weighted universe.

    Row labelled with the trade date d = t + lag holds the return from the close of d to the
    close of d + hold, for a portfolio chosen with data up to t.
    """
    t0 = first_rebalance(close, vol, cfg) if start is None else start
    rows, prev = [], {s: pd.Series(dtype=float) for s in SIGNALS}
    for t in range(t0, len(close) - cfg.lag - cfg.hold, cfg.hold):
        names = universe_at(close, vol, t, cfg, allowed)
        a, b = t + cfg.lag, t + cfg.lag + cfg.hold
        tradable = [n for n in names if pd.notna(close.iloc[a][n])]
        hpr = holding_returns(close, a, b, tradable)
        row = {"date": close.index[a], "n_universe": len(tradable), "universe": float(hpr.mean())}
        if "BTCUSDT" in close.columns:
            row["btc"] = float(close.iloc[b]["BTCUSDT"] / close.iloc[a]["BTCUSDT"] - 1.0)
        for s, look in SIGNALS.items():
            sig = signal_at(close, t, look, tradable)
            if len(sig) < cfg.min_names:
                continue
            long, short = legs(sig, cfg)
            w = weights(long, short)
            turn = float(w.sub(prev[s], fill_value=0.0).abs().sum())
            prev[s] = w
            lr, sr = float(hpr[long].mean()), float(hpr[short].mean())
            row.update({f"{s}|gross": lr - sr, f"{s}|net": lr - sr - cfg.cost * turn, f"{s}|turnover": turn,
                        f"{s}|long": lr - row["universe"], f"{s}|short": row["universe"] - sr})
        rows.append(row)
    return {"weekly": pd.DataFrame(rows).set_index("date"), "start": t0, "cfg": cfg}


def newey_west_t(d: np.ndarray, lags: int) -> float:
    n = len(d)
    e = d - d.mean()
    v = float(e @ e) / n
    for k in range(1, lags + 1):
        v += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    return float(d.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def p_two_sided(t: float) -> float:
    from math import erfc, sqrt
    return float(erfc(abs(t) / sqrt(2.0)))


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def summarize(x: pd.Series, cfg: Config) -> dict:
    v = x.dropna().to_numpy()
    sd = v.std(ddof=1)
    cum = np.cumprod(1.0 + np.clip(v, -0.999, None))
    return {"ann_return": float(v.mean() * WEEKS_PER_YEAR), "ann_vol": float(sd * np.sqrt(WEEKS_PER_YEAR)),
            "sharpe": float(v.mean() / sd * np.sqrt(WEEKS_PER_YEAR)), "t": newey_west_t(v, cfg.nw_lags),
            "max_drawdown": float((cum / np.maximum.accumulate(cum) - 1.0).min()), "weeks": int(len(v))}
