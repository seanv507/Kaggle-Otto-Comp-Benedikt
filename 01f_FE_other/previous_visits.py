"""Previous visits: every item in a session's history is a candidate.

Replaces Benny's 04_PreviousVisits.ipynb (source `pv`) and
04_PreviousVisits-diffWeights.ipynb (source `pv_2`), which differ only in the type
weights and in the columns kept. Neither depends on the fold, so both are shared
sources (folds.py):
    data_folds/shared/candidates/{train,sub}/{pv,pv_2}/cand.parquet

Each event of a session gets a weight
    (lo + (1 - lo) * (rank - 1) / session_len) * type_weight
with rank 1 for the most recent event, and lo = 2**0.1 - 1 (wgt_1) or 2**0.5 - 1
(wgt_2). As in Benny's code, the most recent event gets the smallest weight. The
weights are summed per (session, aid).

Run from the repo root (`--sample 0.05` for a quick run; outputs then go to
data_folds/sample_0.05):
    uv run python 01f_FE_other/previous_visits.py
"""

import argparse

import polars as pl
from beartype import beartype

import folds
import metrics
from event_types import CARTS, CLICKS, ORDERS

# Benny's type_mp per source
TYPE_WEIGHTS = {
    "pv": {CLICKS: 1.0, CARTS: 6.0, ORDERS: 3.0},
    "pv_2": {CLICKS: 0.5, CARTS: 9.0, ORDERS: 0.5},
}
LOW = {"wgt_1": 2**0.1 - 1, "wgt_2": 2**0.5 - 1}
SOURCES = tuple(TYPE_WEIGHTS)


@beartype
def reverse_rank(events: pl.LazyFrame) -> pl.LazyFrame:
    """Add `rank`: 1 for a session's most recent event, 2 for the one before, ...

    Benny ranked by a (session, ts desc) sort, whose order among events in the same
    second depends on cuDF. Here a later event in the file counts as more recent.
    """
    # the file position, taken before anything can reorder the rows
    events = events.with_row_index("i")
    # ts in seconds (Benny: ts / 1000 as int32) times 2**32, plus the file position:
    # a unique key per event that sorts by ts, then file order (< 2**63 for real ts)
    key = pl.col("ts").dt.epoch("s") * 2**32 + pl.col("i").cast(pl.Int64)
    return events.with_columns(
        rank=key.rank("ordinal", descending=True).over("session")
    ).drop("i")


@beartype
def previous_visits(events: pl.LazyFrame | pl.DataFrame, source: str) -> pl.LazyFrame:
    """(session, cand, features) for source `pv` or `pv_2`, sorted by session and cand."""
    weights = TYPE_WEIGHTS[source]
    lf = reverse_rank(events.lazy())
    ts = pl.col("ts").dt.epoch("s")
    # the GPU engine needs an explicit default; NaN makes an unknown type visible
    type_weight = pl.col("type").replace_strict(
        weights, default=float("nan"), return_dtype=pl.Float64
    )
    position = (pl.col("rank") - 1) / pl.len().over("session")
    lf = lf.with_columns(
        ts_diff=(ts.max().over("session") - ts).cast(pl.Int32),
        **{w: (lo + (1 - lo) * position) * type_weight for w, lo in LOW.items()},
    )
    wgts = [pl.col(w).sum().cast(pl.Float32).alias(f"{source}_{w}") for w in LOW]
    if source == "pv":
        aggs = [
            pl.col("ts_diff").min(),
            pl.col("rank").min().cast(pl.Int16).alias("rev_order"),
            *wgts,
            *[
                (pl.col("type") == t).sum().cast(pl.Int16).alias(name)
                for t, name in [
                    (CLICKS, "clicks"),
                    (CARTS, "carts"),
                    (ORDERS, "orders"),
                ]
            ],
        ]
    else:
        aggs = wgts
    return (
        lf.group_by("session", "aid")
        .agg(aggs)
        .rename({"aid": "cand"})
        .sort("session", "cand")
    )


def main(sample: float, engine):
    root = folds.output_root(sample)
    for dataset in folds.DATASETS:
        events = folds.candidate_events(None, dataset, sample)
        labels = folds.sample_sessions(folds.labels(None, dataset), sample)
        for source in SOURCES:
            cands = previous_visits(events, source).collect(engine=engine)
            path = folds.write_candidates(cands, None, dataset, source, root=root)
            r = metrics.recall(cands, labels, f"{source}_wgt_1")
            print(
                f"wrote {path.relative_to(root.parent)}: {cands.height:,} rows; "
                f"recall@20 by {source}_wgt_1: {metrics.format_recall(r)}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sample", type=float, default=1.0, help="fraction of sessions"
    )
    parser.add_argument(
        "--engine", default="streaming", help="streaming, in-memory or gpu"
    )
    args = parser.parse_args()
    main(
        args.sample,
        pl.GPUEngine(raise_on_fail=True) if args.engine == "gpu" else args.engine,
    )
