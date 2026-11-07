"""Benny's 01b notebooks, ported to pandas, against benny_covisits on random sessions.

Times are distinct within a session, so that the order of Benny's sorts doesn't
depend on cuDF; matrix weights are distinct per aid for the same reason.
"""

import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from benny_covisits import (
    SOURCES,
    _int16,
    basic_candidates,
    build_matrix,
    weighted_candidates,
    wt_candidates,
)
from polars.testing import assert_frame_equal

import folds
from engines import ENGINES
from engines import engine as to_engine

EVENT_SCHEMA = {
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}
T0 = datetime.datetime(2022, 8, 20)
TYPE_WEIGHT = {0: 1, 1: 6, 2: 3}


def _events(seed, n_sessions=150, n_aids=12, max_len=40):
    """Sorted by session, then ts (as the data); a `_group` per 50 sessions."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        n = int(rng.integers(1, max_len))
        gaps = rng.choice(
            [1, 30, 31, 600, 3_600, 3_601, 86_399, 86_401, 7 * 86_400], size=n
        )
        for t in np.cumsum(gaps):
            rows.append(
                (
                    s,
                    int(rng.integers(0, n_aids)),
                    T0 + datetime.timedelta(seconds=int(t)),
                    int(rng.choice(3, p=[0.6, 0.25, 0.15])),
                )
            )
    df = pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")
    return df.with_columns(_group=(pl.col("session") // 50).cast(pl.Int32))


def _pd(events: pl.DataFrame) -> pd.DataFrame:
    """Benny's frame: ts in seconds, type codes."""
    return events.with_columns(pl.col("ts").dt.epoch("s")).to_pandas()


def _tail30(df):
    df = df.sort_values(["session", "ts"], ascending=[True, False]).reset_index(
        drop=True
    )
    df["n"] = df.groupby("session").cumcount()
    return df.loc[df.n < 30].drop("n", axis=1)


# --- Benny's matrices (one data file; the per-file loop doesn't change these) ---


def ref_c2o(df):
    clicks, buys = df[df["type"] == 0], df[df["type"] != 0]
    df = clicks.merge(buys, how="left", on="session")
    df = df[~df.aid_y.isna()]
    df = df.loc[(df.ts_x - df.ts_y).abs() < 7 * 24 * 60 * 60]
    df = df[["session", "aid_x", "aid_y"]].drop_duplicates()
    df = df.groupby(["aid_x", "aid_y"]).count().reset_index()
    df.columns = ["aid_x", "aid_y", "wgt"]
    return df[df["wgt"] > 1]


def _ref_tail_pairs(df, days):
    df = _tail30(df)
    df = df.merge(df, how="left", on="session")
    df = df.loc[
        ((df.ts_x - df.ts_y).abs() < days * 24 * 60 * 60) & (df.aid_x != df.aid_y)
    ]
    return df[["session", "aid_x", "aid_y", "type_y", "ts_x"]].drop_duplicates(
        ["session", "aid_x", "aid_y"]
    )


def ref_caco(df):
    df = _ref_tail_pairs(df, 1)
    df["wgt"] = df.type_y.map(TYPE_WEIGHT).astype("float32")
    return df.groupby(["aid_x", "aid_y"]).wgt.sum().reset_index()


def ref_clicks(df):
    df = _ref_tail_pairs(df, 1)
    df["wgt"] = (1 + 3 * (df.ts_x - 1659304800) / (1662328791 - 1659304800)).astype(
        "float32"
    )
    return df.groupby(["aid_x", "aid_y"]).wgt.sum().reset_index()


def ref_b2b(df):
    df = _ref_tail_pairs(df.loc[df["type"].isin([1, 2])], 14)
    df["wgt"] = np.float32(1)
    return df.groupby(["aid_x", "aid_y"]).wgt.sum().reset_index()


def ref_weighted(df):
    """Notebook 6, with its per-file loop over `_chunk` and ts in ms for the 1 h cut."""
    out = []
    for _, d in df.groupby("_chunk"):
        d = d.sort_values(["session", "ts_ms"], ascending=[True, False])
        d["ts_2"] = d.groupby(["session"]).ts_ms.shift(1)
        d["diff_"] = (
            ((d["ts_2"] - d["ts_ms"]) / 1000 > 60 * 60).fillna(0).astype("int8")
        )
        d["new_session"] = d.groupby(["session"]).diff_.cumsum()
        d["session"] = d["session"] * 100 + d["new_session"]
        d = d[d["type"] == 0]
        d = d.merge(
            d[["session"]]
            .groupby(["session"])
            .size()
            .reset_index()
            .rename(columns={0: "session_len"}),
            on="session",
        )
        d = d.sort_values(["session", "ts"], ascending=[True, False]).reset_index(
            drop=True
        )
        d["n"] = d.groupby("session").cumcount()
        d = d.merge(d[["session", "aid", "ts", "n"]], how="left", on="session")
        d = d[
            (d["n_x"] != d["n_y"])
            & (d["ts_x"] != d["ts_y"])
            & (d["aid_x"] != d["aid_y"])
        ]
        w = (
            1
            / np.log((d["ts_x"] - d["ts_y"]).abs() + 2)
            * 1
            / np.log((d["n_x"] - d["n_y"]).abs() + 2)
        )
        d["wgt"] = (w * 1 / np.log(d["session_len"] + 2)).astype("float32")
        d = d.groupby(["aid_x", "aid_y"]).wgt.agg(["sum", "count"]).reset_index()
        out.append(d[d["count"] > 1])
    df_m = pd.concat(out).groupby(["aid_x", "aid_y"]).sum().reset_index()
    counts = df["aid"].value_counts()
    cx = df_m["aid_x"].map(counts).fillna(0) + 1
    cy = df_m["aid_y"].map(counts).fillna(0) + 1
    df_m["wgt"] = df_m["sum"] / (np.sqrt(cx) * np.sqrt(cy))
    return df_m[["aid_x", "aid_y", "wgt", "count"]]


def ref_wt_1_2(df):
    d = df.sort_values(["session", "ts"], ascending=[True, False]).reset_index(
        drop=True
    )
    d["n"] = d.groupby("session").cumcount()
    r, l = d[d["type"] == 1], d[d["type"] == 2]
    d = r.merge(l[["session", "aid", "ts", "n"]], how="left", on="session")
    d = d[~d["aid_y"].isna()]
    d["wgt"] = (
        1
        / np.log((d["ts_x"] - d["ts_y"]).abs() + 2)
        * 1
        / np.log((d["n_x"] - d["n_y"]).abs() + 2)
    )
    d = (
        d[["session", "aid_x", "aid_y", "wgt"]]
        .groupby(["session", "aid_x", "aid_y"])
        .agg(["sum", "count"])
        .reset_index()
    )
    d.columns = ["session", "aid_x", "aid_y", "wgt_sum", "wgt_count"]
    d = (
        d.groupby(["aid_x", "aid_y"])
        .agg({"wgt_sum": "sum", "wgt_count": "sum", "session": "count"})
        .reset_index()
    )
    d.columns = ["aid_x", "aid_y", "wgt_sum", "wgt_count", "count"]
    d = d[d["count"] > 1]
    cx = (
        d["aid_x"]
        .map(
            df[df["type"] == 1]
            .drop_duplicates(["session", "aid"])["aid"]
            .value_counts()
        )
        .fillna(0)
        + 1
    )
    cy = (
        d["aid_y"]
        .map(
            df[df["type"] == 2]
            .drop_duplicates(["session", "aid"])["aid"]
            .value_counts()
        )
        .fillna(0)
        + 1
    )
    d["wgt_sum_2"] = d["wgt_sum"] / (np.sqrt(cx) * np.sqrt(cy))
    d["wgt_prob"] = d["count"] / np.sqrt(cx)
    d["wgt_prob_2"] = d["count"] / (np.sqrt(cx) * np.sqrt(cy))
    return d.rename(columns={"wgt_sum_2": "wgt"})[
        ["aid_x", "aid_y", "wgt", "wgt_sum", "wgt_prob", "wgt_prob_2", "count"]
    ]


REFS = {
    "c2o": ref_c2o,
    "caco": ref_caco,
    "b2b": ref_b2b,
    "clicks": ref_clicks,
    "weighted": ref_weighted,
    "wt_1_2": ref_wt_1_2,
}


def _all_pairs(name, events, engine):
    """Our matrix with every pair (keep=0 sorts instead of taking a top)."""
    import benny_covisits

    m = benny_covisits.MATRICES[name]
    old = m.keep
    object.__setattr__(m, "keep", 0)
    try:
        return build_matrix(events.lazy(), name, engine=engine)
    finally:
        object.__setattr__(m, "keep", old)


@pytest.mark.parametrize("name", list(REFS))
@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_matrix_matches_bennys_code(name, engine):
    events = _events(seed=len(name))
    got = _all_pairs(name, events, engine)
    ref = _pd(events).assign(
        ts_ms=events["ts"].dt.epoch("ms").to_numpy(),
        _chunk=events.select(folds.benny_file_chunks(3)).to_series().to_numpy(),
    )
    expected = pl.from_pandas(REFS[name](ref)).rename({"aid_x": "aid", "aid_y": "cand"})
    expected = expected.select(got.columns).cast(got.schema)
    assert got.height > 0
    key = ["aid", "cand"]
    assert_frame_equal(got.sort(key), expected.sort(key), rel_tol=1e-5)


def test_weighted_depends_on_the_file_chunks():
    """A pair seen once in each of two chunks is dropped (count > 1 per chunk)."""
    rows = [
        (s, a, T0 + datetime.timedelta(seconds=t), 0)
        for s in (0, 1)
        for a, t in [(1, 0), (2, 100)]
    ]
    events = pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")
    same = events.with_columns(_group=pl.lit(0, pl.Int32))
    apart = events.with_columns(
        _group=pl.col("session").cast(pl.Int32)
    )  # files 0 and 1: chunks 0 and 1
    assert _all_pairs("weighted", same, "in-memory").height == 2  # (1, 2) and (2, 1)
    assert _all_pairs("weighted", apart, "in-memory").height == 0


# --- candidates ---


def _ref_rank(df):
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    df["dummy"] = 1
    df["rank"] = df.groupby(["session"]).dummy.cumsum()
    return df


def _ref_top(co, top, by="wgt"):
    co = co.sort_values(["aid", by], ascending=[True, False]).reset_index(drop=True)
    co["n"] = co.groupby("aid").cand.cumcount()
    return co[co["n"] < top]


def ref_basic(df, co, prefix, top):
    df = _ref_rank(df)
    df["score"] = 1 / df["rank"]
    df = df.merge(_ref_top(co, top)[["aid", "cand", "wgt"]], how="left", on="aid")
    df["wgt2"] = df["wgt"] * df["score"]
    for n, t in [("cl", 0), ("ca", 1), ("or", 2)]:
        df[f"wgt_{n}"] = (df["type"] == t).astype("int8") * df["wgt"]
    df = (
        df.groupby(["session", "cand"])
        .agg(
            {
                "dummy": "sum",
                "rank": "min",
                "wgt": "sum",
                "wgt2": "sum",
                "wgt_cl": "sum",
                "wgt_ca": "sum",
                "wgt_or": "sum",
            }
        )
        .reset_index()
    )
    names = ["num", "rank_min", "wgt", "wgt_2", "wgt_cl", "wgt_ca", "wgt_or"]
    df.columns = ["session", "cand"] + [f"{prefix}_{c}_{top}" for c in names]
    for c in df.columns[2:]:
        df[c] = df[c].astype("float32" if "_wgt_2_" in c else "int16")
    return df


def ref_weighted_cands(df, co, prefix="weighted", top=30):
    df = df.assign(dummy=1).merge(
        _ref_top(co, top)[["aid", "cand", "wgt", "count"]], how="left", on="aid"
    )
    df["wgt1"], df["count1"] = df["wgt"], df["count"]
    df = (
        df.groupby(["session", "cand"])
        .agg(
            {
                "dummy": "sum",
                "wgt": "sum",
                "wgt1": "max",
                "count": "sum",
                "count1": "max",
            }
        )
        .reset_index()
    )
    df.columns = ["session", "cand"] + [
        f"{prefix}_{c}_{top}" for c in ["num", "wgt", "wgt_max", "count", "count_max"]
    ]
    return df


def ref_wt_cands(df, co, prefix="weighted_1_2", top=20):
    df = _ref_rank(df[df["type"] == 1])
    df = df.merge(
        df[["session"]]
        .groupby(["session"])
        .size()
        .reset_index()
        .rename(columns={0: "session_len"}),
        on="session",
    )
    lin = 0.1 + ((1 - 0.1) / (df["session_len"] - 1)) * (df["session_len"] - df["rank"])
    df["score"] = (2**lin - 1).fillna(1.0)
    co = _ref_top(co.rename(columns={"wgt": "wgt_sum_2"}), top, by="wgt_sum_2")
    df = df.merge(co, how="left", on="aid")
    for col in ["wgt_sum_2", "wgt_prob", "count"]:
        df[col + "_2"] = df[col] * df["score"]
    agg = {"dummy": "sum"}
    for col in ["wgt_sum_2", "wgt_prob", "count"]:
        agg[col] = "sum"
        agg[col + "_2"] = "sum"
    df = df.groupby(["session", "cand"]).agg(agg).reset_index()
    df.columns = ["session", "cand"] + [f"{prefix}_{x}" for x in df.columns[2:]]
    return df


def _matrix(seed, extra=()):
    """aid -> 25 cands with distinct weights, so the top per aid is unambiguous."""
    rng = np.random.default_rng(seed)
    rows = [(a, c) for a in range(12) for c in rng.choice(40, size=25, replace=False)]
    df = pl.DataFrame(rows, schema={"aid": pl.Int32, "cand": pl.Int32}, orient="row")
    n = df.height
    cols = {"wgt": rng.permutation(n) + rng.random(n)}
    cols.update({c: rng.random(n) * 10 for c in extra})
    return df.with_columns(
        **{c: pl.Series(v, dtype=pl.Float32) for c, v in cols.items()}
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("prefix, top", [("c2o", 15), ("clicks", 20)])
def test_basic_candidates(engine, prefix, top):
    events, matrix = _events(seed=7).drop("_group"), _matrix(1)
    got = basic_candidates(events, matrix, prefix, top).collect(
        engine=to_engine(engine)
    )
    expected = pl.from_pandas(ref_basic(_pd(events), matrix.to_pandas(), prefix, top))
    expected = expected.select(got.columns).cast(got.schema).sort("session", "cand")
    # the int16 columns truncate float sums: a sum at an integer can come out one less
    # in one implementation than the other (cuDF's own float32 sum order varies too)
    ints = [c for c, t in got.schema.items() if t == pl.Int16]
    diff = (
        got.select(ints).cast(pl.Int32) - expected.select(ints).cast(pl.Int32)
    ).to_numpy()
    assert np.abs(diff).max() <= 1 and (diff != 0).sum() <= 0.001 * diff.size
    floats = [c for c in got.columns if c not in ints]  # with session and cand
    assert_frame_equal(got.select(floats), expected.select(floats), rel_tol=1e-5)


@pytest.mark.parametrize("engine", ENGINES)
def test_weighted_candidates(engine):
    events, matrix = _events(seed=8).drop("_group"), _matrix(2, extra=["count"])
    got = weighted_candidates(events, matrix).collect(engine=to_engine(engine))
    expected = pl.from_pandas(ref_weighted_cands(_pd(events), matrix.to_pandas()))
    expected = expected.select(got.columns).cast(got.schema).sort("session", "cand")
    assert_frame_equal(got, expected, rel_tol=1e-5)


@pytest.mark.parametrize("engine", ENGINES)
def test_wt_candidates(engine):
    events = _events(seed=9).drop("_group")
    matrix = _matrix(3, extra=["wgt_sum", "wgt_prob", "wgt_prob_2", "count"])
    got = wt_candidates(events, matrix).collect(engine=to_engine(engine))
    expected = pl.from_pandas(ref_wt_cands(_pd(events), matrix.to_pandas()))
    expected = expected.select(got.columns).cast(got.schema).sort("session", "cand")
    assert got.columns[2:] == [
        f"weighted_1_2_{c}"
        for c in [
            "dummy",
            "wgt_sum_2",
            "wgt_sum_2_2",
            "wgt_prob",
            "wgt_prob_2",
            "count",
            "count_2",
        ]
    ]
    assert_frame_equal(got, expected, rel_tol=1e-5)


def test_int16_wraps_like_cudf():
    x = pl.Series([1.9, -1.9, 32767, 32768, 70000, -32769], dtype=pl.Float64)
    got = pl.select(_int16(pl.lit(x))).to_series().to_list()
    assert got == [1, -1, 32767, -32768, 70000 - 65536, 32767]


def test_caco_b2b_is_empty():
    events, matrix = _events(seed=1).drop("_group"), _matrix(1)
    _, candidates, _ = SOURCES["caco_b2b_15"]
    got = candidates(events, matrix).collect()
    assert got.height == 0 and got.columns[2] == "caco_b2b_num_15"
