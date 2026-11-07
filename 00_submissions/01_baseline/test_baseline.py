"""Chris's suggest_clicks / suggest_buys (00_2_CalcSub.ipynb), verbatim apart from the
matrix globals, against baseline.py's Polars version on random sessions."""

import datetime
import itertools
from collections import Counter

import numpy as np
import polars as pl
import pytest
from baseline import suggest_buys, suggest_clicks

TYPE_MP = {1: 1, 2: 6, 3: 3}  # Chris's codes: clicks 1, carts 2, orders 3


def chris_suggest_clicks(aids, types, top_20_clicks):
    unique_aids = list(dict.fromkeys(aids[::-1]))
    if len(unique_aids) >= 20:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        for aid, w, t in zip(aids, weights, types, strict=True):
            aids_temp[aid] += w * TYPE_MP[t]
        return [k for k, v in aids_temp.most_common(20)]
    aids2 = list(
        itertools.chain(
            *[top_20_clicks[aid] for aid in unique_aids if aid in top_20_clicks]
        )
    )
    top_aids2 = [
        aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids
    ]
    return unique_aids + top_aids2[: 20 - len(unique_aids)]


def chris_suggest_buys(aids, types, top_20_buys, top_20_buy2buy):
    unique_aids = list(dict.fromkeys(aids[::-1]))
    buys = [a for a, t in zip(aids, types, strict=True) if t in (2, 3)]
    unique_buys = list(dict.fromkeys(buys[::-1]))
    if len(unique_aids) >= 20:
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        for aid, w, t in zip(aids, weights, types, strict=True):
            aids_temp[aid] += w * TYPE_MP[t]
        aids3 = list(
            itertools.chain(
                *[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]
            )
        )
        for aid in aids3:
            aids_temp[aid] += 0.1
        return [k for k, v in aids_temp.most_common(20)]
    aids2 = list(
        itertools.chain(
            *[top_20_buys[aid] for aid in unique_aids if aid in top_20_buys]
        )
    )
    aids3 = list(
        itertools.chain(
            *[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]
        )
    )
    top_aids2 = [
        aid2
        for aid2, cnt in Counter(aids2 + aids3).most_common(20)
        if aid2 not in unique_aids
    ]
    return unique_aids + top_aids2[: 20 - len(unique_aids)]


def _matrix(rng, n_aids, top):
    """{aid: [cand, ...]} in matrix order, and the same as baseline.py's frame."""
    lists = {a: [int(c) for c in rng.choice(n_aids, size=int(rng.integers(1, top + 1)), replace=False) if c != a]
             for a in range(n_aids) if rng.random() < 0.8}  # fmt: skip
    rows = [
        (a, c, float(top - n), n) for a, cs in lists.items() for n, c in enumerate(cs)
    ]
    df = pl.DataFrame(
        rows,
        schema={"aid": pl.Int32, "cand": pl.Int32, "wgt": pl.Float64, "n": pl.Int32},
        orient="row",
    )
    return lists, df


def _sessions(rng, n_sessions, n_aids):
    out = []
    for s in range(n_sessions):
        # short sessions and long ones with 20 or more unique aids
        n = int(rng.integers(1, 12)) if s % 3 else int(rng.integers(25, 60))
        out.append(
            (
                s,
                rng.integers(0, n_aids, size=n).tolist(),
                rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]).tolist(),
            )
        )
    return out


def _events(sessions):
    t0 = datetime.datetime(2022, 8, 29)
    rows = [
        (s, a, t0 + datetime.timedelta(seconds=i), t)
        for s, aids, ts in sessions
        for i, (a, t) in enumerate(zip(aids, ts, strict=True))
    ]
    schema = {
        "session": pl.Int32,
        "aid": pl.Int32,
        "ts": pl.Datetime("ms"),
        "type": pl.UInt8,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def _lists(df):
    return {s: g.sort("rank")["aid"].to_list() for (s,), g in df.group_by("session")}


@pytest.mark.parametrize("seed", [0, 1])
def test_suggest_clicks(seed):
    rng = np.random.default_rng(seed)
    n_aids = 60  # small, so the long sessions have 20+ unique aids
    lists, matrix = _matrix(rng, n_aids, 20)
    sessions = _sessions(rng, 150, n_aids)
    got = _lists(suggest_clicks(_events(sessions), matrix).collect())
    for s, aids, types in sessions:
        expected = chris_suggest_clicks(aids, [t + 1 for t in types], lists)
        assert got.get(s, []) == expected, s


@pytest.mark.parametrize("seed", [0, 1])
def test_suggest_buys(seed):
    rng = np.random.default_rng(seed)
    n_aids = 60
    buys, buys_df = _matrix(rng, n_aids, 15)
    b2b, b2b_df = _matrix(rng, n_aids, 15)
    sessions = _sessions(rng, 150, n_aids)
    got = _lists(suggest_buys(_events(sessions), buys_df, b2b_df).collect())
    for s, aids, types in sessions:
        expected = chris_suggest_buys(aids, [t + 1 for t in types], buys, b2b)
        assert got.get(s, []) == expected, s
