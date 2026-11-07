import polars as pl
import pytest

import metrics
from event_types import CARTS, CLICKS, ORDERS

LABELS = pl.DataFrame(
    {
        "session": [1, 1, 1, 2, 2, 3],
        "aid": [10, 20, 21, 10, 30, 40],
        "type": [CLICKS, CARTS, CARTS, CLICKS, ORDERS, CLICKS],
    },
    schema={"session": pl.Int32, "aid": pl.Int32, "type": pl.UInt8},
)


def _cands(rows):
    return pl.DataFrame(
        rows,
        schema={"session": pl.Int32, "cand": pl.Int32, "score": pl.Float64},
        orient="row",
    )


def test_recall():
    cands = _cands(
        [
            (1, 10, 0.9),  # click hit
            (1, 20, 0.5),  # cart hit (1 of 2)
            (1, 99, 0.7),
            (2, 30, 0.1),  # order hit, but only if k >= 2
            (2, 98, 0.2),
            # session 3 has no candidates: a miss
        ]
    )
    assert metrics.recall(cands, LABELS, "score") == {
        "clicks": pytest.approx(1 / 3),
        "carts": 0.5,
        "orders": 1.0,
    }
    # k = 1: only session 1's aid 10 and session 2's aid 98 are kept
    assert metrics.recall(cands, LABELS, "score", k=1) == {
        "clicks": pytest.approx(1 / 3),
        "carts": 0.0,  # 0 hits of min(2, 1) = 1
        "orders": 0.0,
    }


def test_recall_rank_column_ascending():
    cands = _cands([(1, 99, 0.0), (1, 10, 1.0)])  # rank 0 is best
    assert (
        metrics.recall(cands, LABELS, "score", k=1, descending=False)["clicks"] == 0.0
    )
    assert metrics.recall(cands, LABELS, "score", k=1)["clicks"] == pytest.approx(1 / 3)


def test_top_k_breaks_ties_by_cand():
    cands = _cands([(1, 12, 1.0), (1, 11, 1.0), (1, 13, 1.0)])
    for _ in range(3):
        got = metrics.top_k(
            cands.sample(fraction=1.0, shuffle=True), "score", 2
        ).collect()
        assert got["aid"].to_list() == [11, 12]
