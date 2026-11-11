"""Variant selection must not see the test period."""
import numpy as np
import pandas as pd

from experiments._utils import select_on_train


def _streams(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    split = int(n * 0.6)
    a = rng.normal(0.002, 0.01, n)          # good in train, flat after
    a[split:] = rng.normal(0.0, 0.01, n - split)
    b = rng.normal(0.0, 0.01, n)            # flat in train, great after
    b[split:] = rng.normal(0.004, 0.01, n - split)
    return {"A": pd.Series(a, idx), "B": pd.Series(b, idx)}


def test_selects_on_train_period_only():
    name, table = select_on_train(_streams())
    assert name == "A"  # B wins on the full sample but not before the split
    assert table.loc["B", "test_sharpe"] > table.loc["A", "test_sharpe"]
    assert table["selected"].sum() == 1


def test_future_returns_do_not_change_the_choice():
    s = _streams()
    name, table = select_on_train(s)
    split = pd.Timestamp(table.attrs["split_date"])
    shocked = {k: v.where(v.index < split, v * 0 + (0.05 if k == "B" else -0.05)) for k, v in s.items()}
    name2, _ = select_on_train(shocked)
    assert name == name2
