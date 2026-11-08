"""Session sequences for the transformer: Polars port of Benny's 01_T4R_Preprocess.ipynb.

Each session becomes its last `seq_len` events in time order, with aids replaced by
dense item ids: 0 is padding, 1 an aid outside the map, and 2, 3, ... the aids by
descending event count over CV_TRAIN, XGB_TRAIN_X and TEST (Benny's `aid_`).

Benny's `split` option (cutting sessions at long gaps) was only ever run with
split=0, so it isn't ported. Run from the repo root:
    uv run python 01e_FE_Transformer/t4r_data.py
"""

import argparse

import polars as pl
from beartype import beartype

import paths

Frame = pl.LazyFrame | pl.DataFrame

PAD = 0
UNKNOWN = 1
FIRST_ITEM = 2
SEQ_LEN = 20

T4R = paths.T4R
AID_MAP = T4R / "aid_map.parquet"
# name -> events file; the sequences are written to T4R / f"seq_{name}.parquet"
SEQUENCE_SOURCES = {
    "cv_train": paths.CV_TRAIN,  # weeks 1-3
    "xgb_train_x": paths.XGB_TRAIN_X,  # week 4, truncated
    "test": paths.TEST,  # Kaggle test
}


def sequence_path(name: str):
    return T4R / f"seq_{name}.parquet"


@beartype
def aid_map(events: list[Frame], engine="streaming") -> pl.DataFrame:
    """(aid, item, count): item ids by descending count, ties broken by aid."""
    return (
        pl.concat([e.lazy().select("aid") for e in events])
        .group_by("aid")
        .agg(count=pl.len())
        .sort(["count", "aid"], descending=[True, False])
        .collect(engine=engine)
        # numbered after collecting: the GPU engine has no int_range
        .with_columns(item=(pl.int_range(pl.len()) + FIRST_ITEM).cast(pl.Int32))
        .select("aid", "item", "count")
    )


@beartype
def sequences(events: Frame, aids: Frame, seq_len: int = SEQ_LEN, engine="streaming") -> pl.DataFrame:
    """One row per session: its last `seq_len` items and types, oldest first.

    Events must be in time order within a session (all the data files are). `ts` is
    the time of the first kept event (Benny's `'ts': 'min'`).
    """
    return (
        events.lazy()
        # remember the event order: a join doesn't keep row order on the streaming engine
        .with_row_index("row")
        .join(aids.lazy().select("aid", "item"), on="aid", how="left")
        .with_columns(pl.col("item").fill_null(UNKNOWN))
        .group_by("session")
        .agg(
            pl.col("item").sort_by("row").tail(seq_len),
            pl.col("type").sort_by("row").tail(seq_len),
            pl.col("ts").sort_by("row").tail(seq_len).first(),
        )
        .sort("session")
        .collect(engine=engine)
    )


def main(seq_len: int, engine: str):
    T4R.mkdir(parents=True, exist_ok=True)
    sources = {name: pl.scan_parquet(path) for name, path in SEQUENCE_SOURCES.items()}
    aids = aid_map(list(sources.values()), engine=engine)
    aids.write_parquet(AID_MAP)
    print(f"wrote {AID_MAP.relative_to(paths.ROOT)}: {aids.height:,} items")
    for name, events in sources.items():
        seqs = sequences(events, aids, seq_len, engine=engine)
        seqs.write_parquet(sequence_path(name))
        lengths = seqs["item"].list.len()
        print(
            f"wrote {sequence_path(name).relative_to(paths.ROOT)}: {seqs.height:,} sessions, "
            f"{(lengths == 1).sum():,} of length 1"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--engine", default="streaming", help="streaming, in-memory or gpu")
    args = parser.parse_args()
    main(args.seq_len, args.engine)
