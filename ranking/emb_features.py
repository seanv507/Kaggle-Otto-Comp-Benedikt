"""Word2Vec similarity features (Benny's 02_Split_2_Add_Emb_Features-v2.ipynb).

For each (session, cand) row and each event of the session, sim = the cosine of the
fold's Word2Vec vectors (01a) of cand and of the event's aid, kept if above 0.5 and 0
otherwise. Per (session, cand):
    sim_1       max sim
    sim_2       sum sim / (1 + sim_3)
    sim_3       number of events with sim > 0.5
    sim_wgt_1   sum sim * the event's recency weight wgt_1 / (1 + sim_3)
    sim_wgt_2   the same with wgt_2 (session_ops.recency_weights)
    sim_last    max sim over the events in the session's last second
    sim_type_1  sum sim over carts and orders / (1 + sim_3)
Benny computed the cosines row by row with pandarallel; here once per (cand, aid)
pair, in batches.
"""

import polars as pl
from beartype import beartype

from event_types import CARTS, ORDERS
from session_ops import recency_weights

THRESHOLD = 0.5
COLUMNS = [
    "sim_1",
    "sim_2",
    "sim_3",
    "sim_wgt_1",
    "sim_wgt_2",
    "sim_last",
    "sim_type_1",
]


@beartype
def event_weights(events: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """(session, aid, wgt_1, wgt_2, type_w, last) per event."""
    ts = pl.col("ts").dt.epoch("s")
    return recency_weights(events.lazy()).select(
        "session",
        "aid",
        "wgt_1",
        "wgt_2",
        type_w=pl.col("type").is_in([CARTS, ORDERS]),
        last=ts == ts.max().over("session"),
    )


@beartype
def similarities(
    pairs: pl.DataFrame, emb: pl.DataFrame, batch: int = 1_000_000
) -> pl.DataFrame:
    """(cand, aid, sim) for the pairs whose cosine is above THRESHOLD.

    Pairs with an aid or cand that has no vector are left out (sim 0). The cosine is
    the dot product of the unit vectors, as Polars Array arithmetic, `batch` pairs at
    a time (each pair carries two 50-d vectors). Against numpy's einsum it differs by
    float32 rounding (<= 6.6e-7); on the 2% sample, 1 pair in 1.3M fell on the other
    side of THRESHOLD.
    """
    v = pl.col("emb")
    unit = emb.select("aid", emb=v / (v * v).arr.sum().sqrt())
    pairs = pairs.join(unit.rename({"aid": "cand", "emb": "e_cand"}), on="cand").join(
        unit.rename({"emb": "e_aid"}), on="aid"
    )
    cosine = (pl.col("e_cand") * pl.col("e_aid")).arr.sum()
    parts = [p.select("cand", "aid", sim=cosine) for p in pairs.iter_slices(batch)]
    out = pl.concat(parts) if parts else pairs.select("cand", "aid", sim=cosine)
    return out.filter(pl.col("sim") > THRESHOLD)


@beartype
def emb_features(
    rows: pl.LazyFrame, events: pl.LazyFrame | pl.DataFrame, emb: pl.DataFrame
) -> pl.DataFrame:
    """(session, cand, sim_*) for the (session, cand) `rows`, from the sessions' `events`."""
    joined = (
        rows.select("session", "cand")
        .join(event_weights(events), on="session")
        .collect()
    )
    sims = similarities(joined.select("cand", "aid").unique(), emb)
    sim = pl.col("sim")
    per_row = (
        joined.lazy()
        .join(sims.lazy(), on=["cand", "aid"], how="left")
        .with_columns(sim.fill_null(0.0).cast(pl.Float64))
        .group_by("session", "cand")
        .agg(
            sim_1=sim.max(),
            sim_2=sim.sum(),
            sim_3=(sim > 0).sum(),
            sim_wgt_1=(sim * pl.col("wgt_1")).sum(),
            sim_wgt_2=(sim * pl.col("wgt_2")).sum(),
            sim_last=(sim * pl.col("last").cast(pl.Float64)).max(),
            sim_type_1=(sim * pl.col("type_w").cast(pl.Float64)).sum(),
        )
    )
    per_cand = 1 + pl.col("sim_3")
    return (
        per_row.with_columns(
            pl.col("sim_2", "sim_wgt_1", "sim_wgt_2", "sim_type_1") / per_cand
        )
        .with_columns(
            pl.col(COLUMNS).exclude("sim_3").cast(pl.Float32),
            pl.col("sim_3").cast(pl.UInt32),
        )
        .select("session", "cand", *COLUMNS)
        .collect()
    )
