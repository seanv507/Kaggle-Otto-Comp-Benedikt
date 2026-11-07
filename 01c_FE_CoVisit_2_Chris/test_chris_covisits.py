import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from chris_covisits import COLUMNS, ORDER, TOP, apply_matrix, build_matrix, source_name
from polars.testing import assert_frame_equal

from engines import ENGINES
from engines import engine as to_engine
from event_types import CARTS, CLICKS

EVENT_SCHEMA = {
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}
T0 = datetime.datetime(2022, 8, 29)


def benny_apply(df: pd.DataFrame, co: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Benny's 04_CoMatrix-99-Chris-apply.ipynb on pandas instead of cuDF: the weights
    from the loop body, then get_candidates. `ts` in seconds, `type` as codes."""
    df = df.merge(
        df[["session"]]
        .groupby(["session"])
        .size()
        .reset_index()
        .rename(columns={0: "session_len"}),
        how="left",
        on="session",
    )
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    df["dummy"] = 1
    df["rank"] = df.groupby(["session"]).dummy.cumsum()
    df = df.drop(columns=["dummy"])
    min_val, max_val = (2**0.1 - 1), (2**1 - 1)
    df["wgt_1"] = min_val + (max_val - min_val) * (df["rank"] - 1) / (df["session_len"])
    min_val, max_val = (2**0.5 - 1), (2**1 - 1)
    df["wgt_2"] = min_val + (max_val - min_val) * (df["rank"] - 1) / (df["session_len"])
    # get_candidates
    df = df.sort_values(["session", "ts"])
    df["d"] = df.groupby("session").ts.diff()
    df.d = (df.d > 60 * 60 * 2).astype("int16").fillna(0)
    df.d = df.groupby("session").d.cumsum()
    df = df.merge(
        df.groupby(["session", "d"])
        .ts.max()
        .reset_index()
        .rename(columns={"ts": "ts_max_2"}),
        how="left",
        on=["session", "d"],
    )
    df = df.merge(co, how="left", on=["aid"])
    df["wgt"] = df["wgt"].fillna(0)
    df["wgt_1"] = df["wgt_1"] * df["wgt"]
    df["wgt_2"] = df["wgt_2"] * df["wgt"]
    df["dummy"] = 1
    df["t_wgt_1"] = df["wgt_1"] * df["type"]
    df["t_wgt_2"] = df["wgt_2"] * df["type"]
    df["t_dummy"] = df["dummy"] * df["type"]
    df["d_wgt_1"] = df["wgt_1"] * (df["d"] == 0)
    df["d_wgt_2"] = df["wgt_2"] * (df["d"] == 0)
    df["d_dummy"] = df["dummy"] * (df["d"] == 0)
    df["l_dummy"] = df["dummy"] * (df["ts"] == df["ts_max_2"])
    df["l_wgt_1"] = df["wgt_1"] * (df["ts"] == df["ts_max_2"])
    df["l_wgt_2"] = df["wgt_2"] * (df["ts"] == df["ts_max_2"])
    df = df.groupby(["session", "cand"]).agg({c: "sum" for c in COLUMNS})
    df.columns = [prefix + "_" + str(x) for x in df.columns]
    return df.reset_index()


def _random_events(rng, n_sessions=300):
    rows = []
    for s in range(n_sessions):
        n = rng.integers(1, 12)
        # distinct times per session (Benny's sort order among ties is cuDF's), with
        # gaps around the 2-hour cut
        gaps = rng.choice([1, 60, 3_600, 7_200, 7_201, 20_000], size=n)
        ts = np.cumsum(gaps)
        for t in ts:
            rows.append(
                (
                    s,
                    int(rng.integers(0, 30)),
                    T0 + datetime.timedelta(seconds=int(t)),
                    int(rng.integers(0, 3)),
                )
            )
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")


def _random_matrix(rng):
    pairs = {(int(a), int(c)) for a, c in rng.integers(0, 30, size=(200, 2))}
    return pl.DataFrame(
        [(a, c, float(rng.random() * 5)) for a, c in sorted(pairs)],
        schema={"aid": pl.Int32, "cand": pl.Int32, "wgt": pl.Float32},
        orient="row",
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_apply_matches_bennys_code(engine):
    rng = np.random.default_rng(0)
    events, matrix = _random_events(rng), _random_matrix(rng)
    got = apply_matrix(events, matrix, "chris__3_").collect(engine=to_engine(engine))
    ref = events.with_columns(pl.col("ts").dt.epoch("s")).to_pandas()
    expected = pl.from_pandas(benny_apply(ref, matrix.to_pandas(), "chris__3_"))
    expected = expected.select(got.columns).cast(got.schema).sort("session", "cand")
    assert got.columns[:3] == ["session", "cand", "chris__3__wgt"]
    assert_frame_equal(got, expected, rel_tol=1e-5)


def test_apply_by_hand():
    # session 1, oldest first: aid 1 click at 0 s, aid 2 cart at 3 h (a 2 h gap:
    # segment 1). Ranks: aid 2 -> 1, aid 1 -> 2; positions 0 and 1/2.
    events = pl.DataFrame(
        [(1, 1, T0, CLICKS), (1, 2, T0 + datetime.timedelta(hours=3), CARTS)],
        schema=EVENT_SCHEMA,
        orient="row",
    )
    matrix = pl.DataFrame(
        {"aid": [1, 2], "cand": [9, 9], "wgt": [2.0, 4.0]},
        schema={"aid": pl.Int32, "cand": pl.Int32, "wgt": pl.Float32},
    )
    got = apply_matrix(events, matrix, "x").collect().row(0, named=True)
    a = 2**0.1 - 1
    w_click, w_cart = 2 * (a + (1 - a) * 0.5), 4 * a  # wgt times recency
    assert got["x_wgt"] == 6 and got["x_dummy"] == 2
    assert got["x_wgt_1"] == pytest.approx(w_click + w_cart)
    assert got["x_t_dummy"] == CARTS and got["x_t_wgt_1"] == pytest.approx(
        w_cart * CARTS
    )
    # only the click is in the first segment; both are the last of their segment
    assert got["x_d_dummy"] == 1 and got["x_d_wgt_1"] == pytest.approx(w_click)
    assert got["x_l_dummy"] == 2


def test_order_and_names():
    assert sorted(ORDER, key=lambda v: f"top_40_aids_v{v}.parquet") == ORDER
    assert source_name(115) == "chris__0__15" and source_name(93) == "chris__9__15"
    assert set(TOP) == set(ORDER)


@pytest.mark.parametrize("ver", ORDER)
def test_build_matrix(ver):
    rng = np.random.default_rng(ver)
    events = _random_events(rng, 200).with_columns(
        pl.col("ts") + datetime.timedelta(days=5)
    )
    if ver == 239:
        events = events.with_columns(_group=(pl.col("session") // 50).cast(pl.Int32))
    got = build_matrix(events.lazy(), ver, engine="in-memory")
    assert got.schema == {"aid": pl.Int32, "cand": pl.Int32, "wgt": pl.Float32}
    if TOP[ver] == 0:
        assert got.height == 0  # Benny's 700
        return
    assert got.height > 0
    assert got.group_by("aid").len()["len"].max() <= TOP[ver]
    assert got.select(pl.col("aid").is_sorted()).item()


def test_build_239_uses_the_file_groups():
    """One pair per (file, aid, cand): the same pair in two files counts twice."""
    events = pl.DataFrame(
        [
            (1, 5, T0, CLICKS),
            (1, 6, T0 + datetime.timedelta(hours=1), CARTS),
            (2, 5, T0, CLICKS),
            (2, 6, T0 + datetime.timedelta(hours=1), CARTS),
        ],
        schema=EVENT_SCHEMA,
        orient="row",
    )
    one_file = events.with_columns(_group=pl.lit(0, pl.Int32))
    two_files = events.with_columns(_group=pl.col("session").cast(pl.Int32))
    w = lambda e: (
        build_matrix(e.lazy(), 239, engine="in-memory")
        .filter(
            (pl.col("aid") == 5) & (pl.col("cand") == 6)
        )["wgt"]
        .item()
    )
    assert w(one_file) == pytest.approx(0.5)  # 0.5 ** 1 hour, once per file
    assert w(two_files) == pytest.approx(1.0)
