"""List columns as fixed-width arrays, for the models' batches (GRU, transformer).

The padding is Polars list operations: each list is extended with `pad` and cut to
`width`, then converted to a fixed-size Array, whose numpy form is a 2-D array without
a copy per row. It runs in row slices because the extended lists of a full fold (~126M
rows) would pass Polars' 2^32 length limit in one go.
"""

import numpy as np
import polars as pl
from beartype import beartype

ROWS_PER_SLICE = 5_000_000


@beartype
def padded(
    lists: pl.Series, width: int, pad: int = 0, rows_per_slice: int | None = None
) -> np.ndarray:
    """(n, width) array of `lists`, each left-aligned and padded with `pad` on the right.

    The dtype is the lists' inner dtype. Raises ValueError if a list is longer than
    `width` (it would be cut silently otherwise).
    """
    if (lists.list.len().max() or 0) > width:
        raise ValueError(f"a list is longer than {width}")
    rows = rows_per_slice or ROWS_PER_SLICE
    fill = pl.lit([pad] * width, dtype=pl.List(lists.dtype.inner))
    expr = pl.col(lists.name).list.concat(fill).list.head(width).list.to_array(width)
    frame = lists.to_frame()
    dtype = pl.Series([], dtype=lists.dtype.inner).to_numpy().dtype
    out = np.empty((len(lists), width), dtype=dtype)
    for i in range(0, len(lists), rows):
        out[i : i + rows] = frame.slice(i, rows).select(expr).to_series().to_numpy()
    return out
