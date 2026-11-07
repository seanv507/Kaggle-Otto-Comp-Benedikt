"""Item activity: clicks, carts and orders per item in the current and earlier weeks.

Replaces Benny's 95_Cand_Features.ipynb. It doesn't depend on the fold (Benny copied
its files to every fold), so it writes shared feature files (folds.py):
    data_folds/shared/item_features_{train,sub}.parquet      keyed by aid
        p0w_*: the dataset's own week (truncated week 4, or the Kaggle test)
        p1w_*, p2w_*, p3w_*: the 1, 2 and 3 weeks before it, from TRAIN
    data_folds/shared/item_day_features_{train,sub}.parquet  keyed by (aid, day_)
        sday_*: the dataset's own week per day (folds.day_index)
with * in clicks, carts, orders. For `train` the earlier weeks are weeks 1-3; for
`sub` they include the whole of week 4.

As in Benny's code, the p0w counts are added twice, so they are double the counts.

Run from the repo root:
    uv run python 01f_FE_other/item_features.py
"""

import argparse
import datetime

import polars as pl
from beartype import beartype

import folds
import paths
from event_types import CARTS, CLICKS, ORDERS

TYPES = [(CLICKS, "clicks"), (CARTS, "carts"), (ORDERS, "orders")]
WEEKS = {"p1w": 1, "p2w": 2, "p3w": 3}
Frame = pl.LazyFrame | pl.DataFrame


@beartype
def type_counts(
    events: Frame, prefix: str, by: tuple[str, ...] = ("aid",)
) -> pl.LazyFrame:
    """Events per type, grouped by `by`: <prefix>_clicks, <prefix>_carts, <prefix>_orders."""
    return (
        events.lazy()
        .group_by(by)
        .agg(
            (pl.col("type") == t).sum().cast(pl.UInt32).alias(f"{prefix}_{name}")
            for t, name in TYPES
        )
    )


@beartype
def item_features(dataset: str, week: Frame, train: Frame) -> pl.LazyFrame:
    """Counts per aid in the dataset's `week` (p0w) and the 1-3 weeks before it (p1w..p3w).

    `train` holds the earlier weeks; its events from the dataset's week on are ignored.
    """
    start = datetime.datetime.combine(folds.WEEK_START[dataset], datetime.time())
    ts = pl.col("ts")
    parts = [type_counts(week, "p0w")]
    for prefix, n in WEEKS.items():
        window = ts.is_between(
            start - datetime.timedelta(weeks=n), start, closed="left"
        )
        parts.append(type_counts(train.lazy().filter(window), prefix))
    parts.append(type_counts(week, "p0w"))  # Benny appended p0w a second time
    # Benny's concat, fillna(0) and groupby(aid).sum()
    return (
        pl.concat(parts, how="diagonal")
        .group_by("aid")
        .agg(pl.exclude("aid").sum())
        .sort("aid")
    )


@beartype
def item_day_features(dataset: str, week: Frame) -> pl.LazyFrame:
    """Counts per (aid, day_) in the dataset's week: sday_clicks, sday_carts, sday_orders."""
    days = week.lazy().with_columns(day_=folds.day_index(dataset))
    return type_counts(days, "sday", by=("aid", "day_")).sort("aid", "day_")


def main(sample: float, engine):
    root = folds.output_root(sample)
    train = folds.sample_sessions(paths.TRAIN, sample)
    for dataset in folds.DATASETS:
        week = folds.candidate_events(None, dataset, sample)
        for name, lf in [
            (f"item_features_{dataset}", item_features(dataset, week, train)),
            (f"item_day_features_{dataset}", item_day_features(dataset, week)),
        ]:
            df = lf.collect(engine=engine)
            path = folds.feature_path(None, name, root)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path)
            print(f"wrote {path.relative_to(root.parent)}: {df.height:,} rows")
        days = df["day_"].unique().sort().to_list()
        assert days == list(range(7)), f"{dataset}: day_ values {days}, expected 0..6"


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
