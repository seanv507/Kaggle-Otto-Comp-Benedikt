"""GRU inputs per fold (Benny's 01_Preprocess_GRU.ipynb), from folds.fe_events(k).

Items are aid + 2 (0 is padding, 1 unused), types are the type code + 2, and each
input event carries its reverse rank relative to the predicted event (2 for the event
just before it, up to 20).

- training examples: for each of a session's last MAX_TARGETS events that has an
  event before it, the target is that event and the input is the SEQ events before
  it (Benny's 40 passes, each dropping the session's latest event). The examples are
  grouped by the target's type: a clicks, carts and orders training set.
- inference inputs: the last SEQ events of fold k's held-out sessions (truncated) and
  of the Kaggle test sessions, ranked 2 to 20 as if an event followed.
"""

import polars as pl
from beartype import beartype

SEQ = 19  # input events (Benny: ranks 2..20)
MAX_TARGETS = 40
OFFSET = 2  # item id = aid + 2; type id = type + 2
LIST = ["item", "type", "rank"]


@beartype
def session_lists(events: pl.LazyFrame | pl.DataFrame, last: int) -> pl.LazyFrame:
    """(session, item, type): each session's last `last` events, in time order.

    The events' row order within a session is their time order (as folds.fe_events
    keeps it); Benny sorted by ts, with cuDF's order among ties.
    """
    return (
        events.lazy()
        .with_row_index("_i")
        .group_by("session")
        .agg(
            item=(pl.col("aid") + OFFSET).sort_by("_i").tail(last).cast(pl.Int32),
            type=(pl.col("type").cast(pl.Int8) + OFFSET).sort_by("_i").tail(last),
        )
    )


@beartype
def training_examples(events: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """(item, type, rank, target, target_type): one row per (session, target)."""
    lists = session_lists(events, MAX_TARGETS + SEQ).with_columns(
        n=pl.col("item").list.len()
    )
    # t: the target's reverse rank (1 = the session's latest event), with an event before it
    lf = lists.with_columns(
        t=pl.int_ranges(1, pl.min_horizontal(MAX_TARGETS, pl.col("n") - 1) + 1)
    ).explode("t")
    p = pl.col("n") - pl.col("t")  # the target's index in the lists
    start = pl.max_horizontal(p - SEQ, 0)
    return lf.filter(pl.col("t").is_not_null()).select(
        item=pl.col("item").list.slice(start, p - start),
        type=pl.col("type").list.slice(start, p - start),
        rank=pl.int_ranges(p - start + 1, 1, -1, dtype=pl.Int8),
        target=pl.col("item").list.get(p),
        target_type=pl.col("type").list.get(p),
    )


@beartype
def inference_inputs(events: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """(session, item, type, rank): the last SEQ events, ranked SEQ + 1 down to 2."""
    lf = session_lists(events, SEQ)
    n = pl.col("item").list.len()
    return lf.with_columns(rank=pl.int_ranges(n + 1, 1, -1, dtype=pl.Int8)).sort(
        "session"
    )
