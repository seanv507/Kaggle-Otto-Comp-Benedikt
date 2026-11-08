import datetime

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from ranking.emb_features import COLUMNS, emb_features, similarities

EVENT_SCHEMA = {
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}
T0 = datetime.datetime(2022, 8, 22)


def benny_emb_features(
    events: pd.DataFrame, rows: pd.DataFrame, emb: dict
) -> pd.DataFrame:
    """Benny's notebook on pandas: the session weights, then the per-file loop body
    (with numpy for the cosine, as his calc_cosine). `ts` in seconds."""
    df = events.copy()
    df = df.merge(
        df[["session"]]
        .groupby(["session"])
        .size()
        .reset_index()
        .rename(columns={0: "session_len"}),
        on="session",
    )
    df = df.merge(
        df[["session", "ts"]]
        .groupby(["session"])
        .max()
        .reset_index()
        .rename(columns={"ts": "ts_max"}),
        on="session",
    )
    df = df.sort_values(["session", "ts"], ascending=[True, False])
    df["dummy"] = 1
    df["rank"] = df.groupby(["session"]).dummy.cumsum()
    lo, hi = 2**0.1 - 1, 2**1 - 1
    df["wgt_1"] = lo + (hi - lo) * (df["rank"] - 1) / df["session_len"]
    lo = 2**0.5 - 1
    df["wgt_2"] = lo + (hi - lo) * (df["rank"] - 1) / df["session_len"]
    df["type_w"] = df["type"].isin([1, 2])
    df["last"] = df["ts"] == df["ts_max"]
    session_ts = df[["session", "aid", "wgt_1", "wgt_2", "type_w", "last"]]
    d = rows.merge(session_ts, how="left", on=["session"])
    pairs = d[["cand", "aid"]].drop_duplicates()
    pairs["sim"] = [
        np.dot(emb[c], emb[a]) / (np.linalg.norm(emb[c]) * np.linalg.norm(emb[a]))
        for c, a in zip(pairs["cand"], pairs["aid"], strict=True)
    ]
    pairs = pairs[pairs["sim"] > 0.5]
    d = d.merge(pairs, how="left", on=["cand", "aid"])
    d["sim"] = d["sim"].fillna(0)
    d["sim_1"] = d["sim"]
    d["sim_2"] = d["sim"]
    d["sim_3"] = d["sim"] > 0
    d["sim_wgt_1"] = d["sim"] * d["wgt_1"]
    d["sim_wgt_2"] = d["sim"] * d["wgt_2"]
    d["sim_last"] = d["sim"] * d["last"]
    d["sim_type_1"] = d["sim"] * d["type_w"]
    d = d.groupby(["session", "cand"]).agg(
        {
            "sim_1": "max",
            "sim_2": "sum",
            "sim_3": "sum",
            "sim_wgt_1": "sum",
            "sim_wgt_2": "sum",
            "sim_last": "max",
            "sim_type_1": "sum",
        }
    )
    for col in ["sim_2", "sim_wgt_1", "sim_wgt_2", "sim_type_1"]:
        d[col] = d[col] / (1 + d["sim_3"])
    return d.reset_index()


def test_emb_features_match_bennys_code():
    rng = np.random.default_rng(0)
    n_aids = 30
    # low-dimensional random vectors: some cosines are above 0.5, most aren't
    vectors = rng.standard_normal((n_aids, 4)).astype(np.float32)
    emb = pl.DataFrame(
        {
            "aid": np.arange(n_aids, dtype=np.int32),
            "emb": pl.Series(vectors, dtype=pl.Array(pl.Float32, 4)),
        }
    )
    rows, cands = [], []
    for s in range(60):
        ts = np.cumsum(rng.choice([1, 60, 3600], size=int(rng.integers(1, 8))))
        rows += [
            (
                s,
                int(rng.integers(n_aids)),
                T0 + datetime.timedelta(seconds=int(t)),
                int(rng.integers(3)),
            )
            for t in ts
        ]
        cands += [(s, int(c)) for c in rng.choice(n_aids, size=5, replace=False)]
    events = pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")
    cand_rows = pl.DataFrame(
        cands, schema={"session": pl.Int32, "cand": pl.Int32}, orient="row"
    )
    got = emb_features(cand_rows.lazy(), events, emb).sort("session", "cand")
    expected = benny_emb_features(
        events.with_columns(pl.col("ts").dt.epoch("s")).to_pandas(),
        cand_rows.to_pandas(),
        dict(enumerate(vectors)),
    )
    expected = (
        pl.from_pandas(expected)
        .select("session", "cand", *COLUMNS)
        .cast(got.schema)
        .sort("session", "cand")
    )
    assert got["sim_3"].sum() > 0 and (got["sim_3"] == 0).sum() > 0
    assert_frame_equal(got, expected, rel_tol=1e-5)


def test_similarities_no_pairs():
    emb = pl.DataFrame(
        {"aid": [1], "emb": [[1.0, 0.0]]},
        schema={"aid": pl.Int32, "emb": pl.Array(pl.Float32, 2)},
    )
    pairs = pl.DataFrame(schema={"cand": pl.Int32, "aid": pl.Int32})
    got = similarities(pairs, emb)
    assert got.height == 0 and got.columns == ["cand", "aid", "sim"]
