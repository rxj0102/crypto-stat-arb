"""Checks on the study: parsing, exclusions, look-ahead, delisting and accounting."""
import io
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from statarb import data, study
from statarb.study import Config

CFG = Config(top_n=20, min_names=10)


def panel(n_days=420, n_assets=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days)
    names = [f"C{i:02d}USDT" for i in range(n_assets)]
    close = pd.DataFrame(np.exp(rng.normal(0, 0.05, (n_days, n_assets)).cumsum(axis=0)), index=idx, columns=names)
    vol = pd.DataFrame(rng.lognormal(15, 1, (n_days, n_assets)) * np.arange(1, n_assets + 1), index=idx, columns=names)
    return close, vol


@pytest.fixture(scope="module")
def pv():
    return panel()


@pytest.fixture(scope="module")
def res(pv):
    return study.run(*pv, CFG)


# ---------- data ----------

def kline_zip(rows, header=False):
    text = ("open_time,open,high,low,close,volume,close_time,quote_volume\n" if header else "") + "\n".join(
        ",".join(str(x) for x in r) for r in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", text)
    return buf.getvalue()


def test_parser_handles_milliseconds_microseconds_and_headers():
    ms = int(pd.Timestamp("2021-01-05").timestamp() * 1000)
    a = data.parse_kline_zip(kline_zip([(ms, 1, 2, 0.5, 1.5, 10, ms + 1, 15.0)]))
    b = data.parse_kline_zip(kline_zip([(ms * 1000, 1, 2, 0.5, 1.5, 10, ms + 1, 15.0)], header=True))
    for df in (a, b):
        assert df["date"].iloc[0] == pd.Timestamp("2021-01-05") and df["close"].iloc[0] == 1.5
        assert df["dollar_volume"].iloc[0] == 15.0


@pytest.mark.parametrize("symbol,out", [("USDCUSDT", True), ("BTCUPUSDT", True), ("ETHBEARUSDT", True),
                                        ("WBTCUSDT", True), ("EURUSDT", True), ("BTCUSDT", False),
                                        ("JUPUSDT", False), ("SYRUPUSDT", False), ("SUPERUSDT", False)])
def test_exclusions(symbol, out):
    assert data.excluded(symbol) is out


def test_symbol_reuse_becomes_a_new_asset():
    days = list(pd.date_range("2022-05-01", periods=5)) + list(pd.date_range("2022-05-28", periods=4))
    df = pd.DataFrame({"date": days, "close": 1.0, "dollar_volume": 100.0})
    parts = data.split_assets("LUNAUSDT", df)
    assert list(parts) == ["LUNAUSDT", "LUNAUSDT.2"]
    assert len(parts["LUNAUSDT"]) == 5 and len(parts["LUNAUSDT.2"]) == 4


def test_short_gaps_and_zero_volume_days():
    days = pd.date_range("2022-05-01", periods=8).delete([3, 4])          # two missing days
    df = pd.DataFrame({"date": days, "close": 1.0, "dollar_volume": [5, 5, 5, 0, 5, 5]})
    parts = data.split_assets("XUSDT", df)
    assert list(parts) == ["XUSDT"] and len(parts["XUSDT"]) == 5          # zero volume day dropped


def test_snapshot_respects_the_rules():
    close, vol = data.load()
    assert list(close.columns) == list(vol.columns) and close.index.equals(vol.index)
    assert not any(data.excluded(data.symbol_of(c)) for c in close.columns)
    first = close.apply(lambda s: s.first_valid_index())
    assert first.max() <= pd.Timestamp(data.LAST_LISTING)
    assert "LUNAUSDT.2" in close.columns and "FTTUSDT" in close.columns   # dead coins are kept
    assert (close.notna() == vol.notna()).all().all()


# ---------- universe and signals ----------

def test_universe_uses_only_the_past(pv):
    close, vol = pv
    t = 300
    c2, v2 = close.copy(), vol.copy()
    c2.iloc[t + 1:] = 1.0
    v2.iloc[t + 1:] = 1e12 * np.arange(1, vol.shape[1] + 1)[::-1]
    assert study.universe_at(close, vol, t, CFG) == study.universe_at(c2, v2, t, CFG)


def test_universe_is_the_most_liquid_with_price_and_history(pv):
    close, vol = (x.copy() for x in pv)
    t = 300
    close.iloc[t, 29] = np.nan                       # most liquid asset has no price today
    close.iloc[: t - 10, 28] = np.nan                # next most liquid has only 11 days of history
    u = study.universe_at(close, vol, t, CFG)
    assert len(u) == 20 and "C29USDT" not in u and "C28USDT" not in u
    med = vol.iloc[t - 29: t + 1].median()
    assert u[0] == med.drop(["C29USDT", "C28USDT"]).idxmax()


def test_signal_is_the_past_return(pv):
    close, _ = pv
    s = study.signal_at(close, 100, 7, ["C00USDT", "C01USDT"])
    assert s["C00USDT"] == pytest.approx(close.iloc[100, 0] / close.iloc[93, 0] - 1)
    assert study.signal_at(close, 3, 7, ["C00USDT"]).empty


def test_legs_are_top_and_bottom_fifth():
    sig = pd.Series(np.arange(20.0), index=[f"n{i:02d}" for i in range(20)])
    long, short = study.legs(sig, CFG)
    assert long == ["n16", "n17", "n18", "n19"] and short == ["n00", "n01", "n02", "n03"]
    assert study.legs(sig.sample(frac=1, random_state=1), CFG) == (long, short)


# ---------- holding returns and delisting ----------

def test_delisted_coin_is_sold_at_its_last_price():
    idx = pd.date_range("2022-01-01", periods=9)
    close = pd.DataFrame({"A": np.linspace(10, 18, 9), "B": [10, 9, 5, 1, np.nan, np.nan, np.nan, np.nan, np.nan]}, index=idx)
    r = study.holding_returns(close, 1, 8, ["A", "B"])
    assert r["A"] == pytest.approx(18 / 11 - 1) and r["B"] == pytest.approx(1 / 9 - 1)


# ---------- the run ----------

def test_returns_start_one_day_after_the_signal(pv, res):
    close, vol = pv
    t0 = res["start"]
    w = res["weekly"]
    assert w.index[0] == close.index[t0 + CFG.lag]
    names = study.universe_at(close, vol, t0, CFG)
    long, short = study.legs(study.signal_at(close, t0, 7, names), CFG)
    hpr = close.iloc[t0 + 1 + 7] / close.iloc[t0 + 1] - 1
    assert w["1 week|gross"].iloc[0] == pytest.approx(hpr[long].mean() - hpr[short].mean())


def test_portfolios_ignore_the_future(pv, res):
    """Scramble prices after a cut. Every weekly return that ended before it must not move."""
    close, vol = pv
    w = res["weekly"]
    k = 10
    cut = close.index.get_loc(w.index[k]) + CFG.hold          # last close used by row k
    c2 = close.copy()
    c2.iloc[cut + 1:] *= np.random.default_rng(3).lognormal(0, 0.3, c2.iloc[cut + 1:].shape)
    again = study.run(c2, vol, CFG, start=res["start"])["weekly"]
    pd.testing.assert_frame_equal(w.iloc[: k + 1], again.iloc[: k + 1])
    assert not np.allclose(w["1 week|gross"].iloc[k + 2:], again["1 week|gross"].iloc[k + 2:])


def test_accounting_identities(res):
    w = res["weekly"]
    for s in study.SIGNALS:
        assert np.allclose(w[f"{s}|long"] + w[f"{s}|short"], w[f"{s}|gross"])
        assert np.allclose(w[f"{s}|gross"] - CFG.cost * w[f"{s}|turnover"], w[f"{s}|net"])
        assert w[f"{s}|turnover"].iloc[0] == pytest.approx(2.0)
        assert w[f"{s}|turnover"].between(0, 4 + 1e-9).all()


def test_all_signals_share_the_same_weeks(res):
    w = res["weekly"]
    assert not w[[f"{s}|gross" for s in study.SIGNALS]].isna().any().any()


def test_built_in_momentum_is_found():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=420)
    drift = np.linspace(-0.004, 0.004, 30)
    close = pd.DataFrame(np.exp((drift + rng.normal(0, 0.01, (420, 30))).cumsum(axis=0)), index=idx,
                         columns=[f"C{i:02d}USDT" for i in range(30)])
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    w = study.run(close, vol, replace(CFG, top_n=30))["weekly"]
    assert study.summarize(w["26 weeks|gross"], CFG)["t"] > 3


def test_survivor_filter_restricts_the_universe(pv):
    close, vol = pv
    allowed = set(close.columns[:22])
    assert set(study.universe_at(close, vol, 300, CFG, allowed)) <= allowed


# ---------- statistics ----------

def test_holm_adjustment():
    adj = study.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_newey_west_matches_plain_t_without_lags():
    d = np.random.default_rng(4).normal(0.01, 0.1, 300)
    assert study.newey_west_t(d, 0) == pytest.approx(d.mean() / (d.std() / np.sqrt(300)))


def test_p_value_of_196_is_five_percent():
    assert study.p_two_sided(1.96) == pytest.approx(0.05, abs=0.0005)
