import numpy as np
import polars as pl
import pytest

import padding


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int8, pl.Int64])
def test_padded(dtype):
    lists = pl.Series("x", [[3, 4, 5], [6], []], dtype=pl.List(dtype))
    got = padding.padded(lists, 4)
    np.testing.assert_array_equal(got, [[3, 4, 5, 0], [6, 0, 0, 0], [0, 0, 0, 0]])
    assert got.dtype == pl.Series([], dtype=dtype).to_numpy().dtype


def test_padded_value_and_too_long():
    lists = pl.Series("x", [[1, 2]], dtype=pl.List(pl.Int32))
    np.testing.assert_array_equal(padding.padded(lists, 3, pad=9), [[1, 2, 9]])
    with pytest.raises(ValueError):
        padding.padded(lists, 1)


def test_padded_in_slices():
    rng = np.random.default_rng(0)
    lists = pl.Series(
        "x",
        [list(rng.integers(1, 9, size=int(n))) for n in rng.integers(0, 6, size=23)],
        dtype=pl.List(pl.Int32),
    )
    np.testing.assert_array_equal(
        padding.padded(lists, 6, rows_per_slice=4), padding.padded(lists, 6)
    )


def test_padded_empty():
    assert padding.padded(pl.Series("x", [], dtype=pl.List(pl.Int32)), 3).shape == (
        0,
        3,
    )
