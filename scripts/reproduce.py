"""Run the study on the frozen data and write results/.

  results/metrics.json          everything the README reports
  results/weekly_returns.csv    weekly return of every portfolio and benchmark
  results/figures/signals.png   return per year by lookback, with and without delisted coins
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from statarb import data, study  # noqa: E402

SE = 1.96


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def paired(a: pd.Series, b: pd.Series, cfg) -> dict:
    d = (a - b).dropna().to_numpy()
    return {"ann_diff": float(d.mean() * study.WEEKS_PER_YEAR), "t": study.newey_west_t(d, cfg.nw_lags)}


def figure(m: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.spines.top": False, "axes.spines.right": False})
    names = list(study.SIGNALS)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for off, key, color, label in ((-0.2, "signals", "#2a78d6", "All coins, including delisted"),
                                   (0.2, "survivors_only", "#eb6834", "Only coins still trading at the end")):
        mean = np.array([m[key][n]["gross"]["ann_return"] for n in names]) * 100
        half = np.array([abs(m[key][n]["gross"]["ann_return"] / m[key][n]["gross"]["t"]) * SE for n in names]) * 100
        ax.bar(x + off, mean, width=0.36, color=color, label=label)
        ax.errorbar(x + off, mean, yerr=half, fmt="none", ecolor=ink, elinewidth=1, capsize=3)
    ax.axhline(0, color=ink, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("Lookback of the past return used to sort coins", color=muted)
    ax.set_ylabel("Percent per year, before costs", color=muted)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Winners minus losers: return per year with 95% interval", loc="left", fontsize=10.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "signals.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = study.Config()
    close, vol = data.load()
    res = study.run(close, vol, cfg)
    w, t0 = res["weekly"], res["start"]

    last = close.apply(lambda s: s.last_valid_index())
    alive = set(last.index[last >= close.index[-1] - pd.Timedelta(days=30)])
    surv = study.run(close, vol, cfg, allowed=alive, start=t0)["weekly"]
    same_day = study.run(close, vol, replace(cfg, lag=0), start=t0)["weekly"]

    n = len(w)
    thirds = {f"{w.index[a].date()} to {w.index[b - 1].date()}": (a, b)
              for a, b in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n))}
    signals, pvals = {}, {}
    for s in study.SIGNALS:
        g = study.summarize(w[f"{s}|gross"], cfg)
        pvals[s] = study.p_two_sided(g["t"])
        signals[s] = {"gross": g, "net": study.summarize(w[f"{s}|net"], cfg),
                      "long_leg": study.summarize(w[f"{s}|long"], cfg),
                      "short_leg": study.summarize(w[f"{s}|short"], cfg),
                      "turnover_per_week": float(w[f"{s}|turnover"].mean()),
                      "thirds": {k: study.summarize(w[f"{s}|gross"].iloc[a:b], cfg) for k, (a, b) in thirds.items()},
                      "same_day_gross": study.summarize(same_day[f"{s}|gross"], cfg),
                      "lag_cost": paired(same_day[f"{s}|gross"].reset_index(drop=True),
                                         w[f"{s}|gross"].reset_index(drop=True), cfg)}
    adj = study.holm(pvals)
    for s in study.SIGNALS:
        signals[s]["p"], signals[s]["p_holm"] = pvals[s], adj[s]
    survivors = {s: {"gross": study.summarize(surv[f"{s}|gross"], cfg),
                     "minus_all": paired(surv[f"{s}|gross"], w[f"{s}|gross"], cfg)} for s in study.SIGNALS}

    first = close.apply(lambda s: s.first_valid_index())
    in_universe = set()
    for t in range(t0, len(close) - cfg.lag - cfg.hold, cfg.hold):
        in_universe |= set(study.universe_at(close, vol, t, cfg))
    payload = rounded({
        "data": {"first_day": str(close.index[0].date()), "last_day": str(close.index[-1].date()),
                 "n_assets": int(close.shape[1]), "n_stopped_trading": int(close.shape[1] - len(alive)),
                 "n_reused_symbols": int(sum("." in c for c in close.columns)),
                 "n_ever_in_universe": len(in_universe),
                 "n_ever_in_universe_stopped": len(in_universe - alive),
                 "last_listing": data.LAST_LISTING},
        "sample": {"first_trade": str(w.index[0].date()), "last_trade": str(w.index[-1].date()), "weeks": n,
                   "years": float(n / study.WEEKS_PER_YEAR), "top_n": cfg.top_n, "hold_days": cfg.hold,
                   "lag_days": cfg.lag, "cost_bps": cfg.cost * 1e4, "quantile": cfg.quantile,
                   "universe_min": int(w["n_universe"].min()), "universe_mean": float(w["n_universe"].mean()),
                   "names_per_leg": int(np.floor(cfg.top_n * cfg.quantile))},
        "benchmarks": {"universe": study.summarize(w["universe"], cfg), "btc": study.summarize(w["btc"], cfg),
                       "universe_survivors_only": study.summarize(surv["universe"], cfg),
                       "universe_survivors_minus_all": paired(surv["universe"], w["universe"], cfg)},
        "signals": signals, "survivors_only": survivors,
    })
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    w.round(8).to_csv(out / "weekly_returns.csv")
    figure(payload, out / "figures")
    print(json.dumps(payload["data"]), json.dumps(payload["sample"]))
    print("bench", {k: (round(v.get("ann_return", v.get("ann_diff")), 3), round(v["t"], 2)) for k, v in payload["benchmarks"].items()})
    for s, v in payload["signals"].items():
        print(f"{s:9s} gross {v['gross']['ann_return']:+.3f} t {v['gross']['t']:+.2f} holm {v['p_holm']:.3f} "
              f"net {v['net']['ann_return']:+.3f} SR {v['net']['sharpe']:+.2f} long {v['long_leg']['ann_return']:+.3f} "
              f"({v['long_leg']['t']:+.1f}) short {v['short_leg']['ann_return']:+.3f} ({v['short_leg']['t']:+.1f}) "
              f"thirds {[round(x['ann_return'], 2) for x in v['thirds'].values()]} "
              f"lag0 {v['same_day_gross']['ann_return']:+.3f} ({v['same_day_gross']['t']:+.1f}) cost-of-lag {v['lag_cost']['ann_diff']:+.3f} ({v['lag_cost']['t']:+.1f}) "
              f"surv {payload['survivors_only'][s]['gross']['ann_return']:+.3f} diff {payload['survivors_only'][s]['minus_all']['ann_diff']:+.3f} ({payload['survivors_only'][s]['minus_all']['t']:+.1f})")


if __name__ == "__main__":
    main()
