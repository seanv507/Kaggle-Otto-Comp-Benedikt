"""Session features: lengths, distinct items, duration and 4-hour segments per session.

Replaces Benny's 96_Session_Features.ipynb, which wrote data/train_sessions.parquet
(week-4 sessions) and data/sub_sessions.parquet (Kaggle test). They don't depend on
the fold, so they are shared feature files (folds.py):
    data_folds/shared/session_features_{train,sub}.parquet
keyed by session. Benny's 01_Split read them per fold as
data_folds/fold_<k>/train_sessions.parquet.

A session is cut into segments where two consecutive events are more than 4 hours
apart. Segment 0 is the most recent one, and `_fs` features count it only. As in
Benny's code, the per-type `_fs` features (num_aid_cl_fs, num_u_aid_cl_fs, ...)
forget that filter and equal the per-type features of the whole session.

Run from the repo root:
    uv run python 01f_FE_other/session_features.py
"""

import argparse
import datetime

import polars as pl
from beartype import beartype

import folds
from event_types import CARTS, CLICKS, ORDERS

GAP = datetime.timedelta(hours=4)
TYPES = [(CLICKS, "cl"), (CARTS, "ca"), (ORDERS, "co")]


@beartype
def segments(events: pl.LazyFrame) -> pl.LazyFrame:
    """Add `segment`: 0 for the events after a session's last 4-hour gap, 1 before it, ..."""
    later = pl.col("ts").shift(1).over("session")  # the next more recent event
    gap = ((later - pl.col("ts")) > GAP).fill_null(False)
    return (
        events.sort("session", "ts", descending=[False, True])
        # two steps: a window nested in another window is quadratic in the data size
        .with_columns(gap=gap.cast(pl.Int64))
        # Int64: the GPU engine's cum_sum of a UInt32 gives Int64 while the schema says
        # UInt32, so a later cast to UInt32 was dropped as a no-op
        .with_columns(segment=pl.col("gap").cum_sum().over("session"))
        .drop("gap")
    )


def _n_unique_where(expr: pl.Expr, cond: pl.Expr) -> pl.Expr:
    """n_unique of `expr` where `cond`, in a group_by.

    The GPU engine has no `filter` in a group_by, so the other rows become null and
    the null, counted by n_unique as one value, is taken off again.
    """
    # (~cond).sum() > 0 rather than .any(), which the GPU engine has no group_by for
    return pl.when(cond).then(expr).n_unique() - ((~cond).sum() > 0).cast(pl.UInt32)


@beartype
def session_features(events: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """One row per session, in Benny's column order, sorted by session."""
    aid, first = pl.col("aid"), pl.col("segment") == 0
    ts = pl.col("ts").dt.epoch("ms")
    is_type = {n: pl.col("type") == t for t, n in TYPES}
    by_type = [c.sum().alias(f"num_aid_{n}") for n, c in is_type.items()]
    by_type += [
        _n_unique_where(aid, c).alias(f"num_u_aid_{n}") for n, c in is_type.items()
    ]
    return (
        segments(events.lazy())
        .group_by("session")
        .agg(
            aid.len().alias("num_aid"),
            aid.n_unique().alias("num_u_aid"),
            (ts.max() - ts.min()).alias("diff"),  # ms
            pl.col("segment").max().alias("num_sessions"),  # the number of gaps
            first.sum().alias("num_aid_fs"),
            _n_unique_where(aid, first).alias("num_u_aid_fs"),
            *by_type,
        )
        # Benny's per-type _fs features: the same counts (see the module docstring)
        .with_columns(
            pl.col(e.meta.output_name()).alias(f"{e.meta.output_name()}_fs")
            for e in by_type
        )
        .with_columns(pl.selectors.starts_with("num_").cast(pl.UInt32))
        .sort("session")
    )


def main(sample: float, engine):
    root = folds.output_root(sample)
    for dataset in folds.DATASETS:
        df = session_features(folds.candidate_events(None, dataset, sample)).collect(
            engine=engine
        )
        path = folds.feature_path(None, f"session_features_{dataset}", root)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        print(f"wrote {path.relative_to(root.parent)}: {df.height:,} sessions")


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
