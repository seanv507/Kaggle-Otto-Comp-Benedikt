"""Per-event recency weights within a session, shared by Benny's candidate steps.

Benny's previous visits (01f) and his apply step for Chris's matrices (01c) weight
each event by its reverse rank in the session:
    wgt_k = lo_k + (1 - lo_k) * (rank - 1) / session_len
with rank 1 for the most recent event, lo_1 = 2**0.1 - 1 and lo_2 = 2**0.5 - 1. So,
as in Benny's code, the most recent event gets the smallest weight.
"""

import polars as pl
from beartype import beartype

LOW = {"wgt_1": 2**0.1 - 1, "wgt_2": 2**0.5 - 1}


@beartype
def reverse_rank(events: pl.LazyFrame) -> pl.LazyFrame:
    """Add `rank`: 1 for a session's most recent event, 2 for the one before, ...

    Benny ranked by a (session, ts desc) sort, whose order among events in the same
    second depends on cuDF. Here a later event in the file counts as more recent.
    """
    # the file position, taken before anything can reorder the rows
    events = events.with_row_index("_i")
    # ts in seconds (Benny: ts / 1000 as int32) times 2**32, plus the file position:
    # a unique key per event that sorts by ts, then file order (< 2**63 for real ts)
    key = pl.col("ts").dt.epoch("s") * 2**32 + pl.col("_i").cast(pl.Int64)
    return events.with_columns(
        rank=key.rank("ordinal", descending=True).over("session")
    ).drop("_i")


@beartype
def recency_weights(events: pl.LazyFrame) -> pl.LazyFrame:
    """Add `rank`, `wgt_1` and `wgt_2` (Float64) to each event."""
    position = (pl.col("rank") - 1) / pl.len().over("session")
    return reverse_rank(events).with_columns(
        **{w: lo + (1 - lo) * position for w, lo in LOW.items()}
    )
