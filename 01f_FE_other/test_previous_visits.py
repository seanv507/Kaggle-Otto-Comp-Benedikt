import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from previous_visits import previous_visits, reverse_rank

from engines import ENGINES
from engines import engine as to_engine
from event_types import CARTS, CLICKS, ORDERS

EVENT_SCHEMA = {
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}
T0 = datetime.datetime(2022, 8, 29)


def _events(rows):
    """rows of (session, aid, seconds after T0, type), in file order"""
    return pl.DataFrame(
        [(s, a, T0 + datetime.timedelta(seconds=t), ty) for s, a, t, ty in rows],
        schema=EVENT_SCHEMA,
        orient="row",
    )


# session 1: aid 11 (cart) and aid 10 (click) in the same second; aid 10 is later in
# the file, so it counts as more recent. Reverse ranks: 12 -> 1, 10 -> 2, 11 -> 3, 10 -> 4.
EVENTS = _events(
    [
        (1, 10, 0, CLICKS),
        (1, 11, 60, CARTS),
        (1, 10, 60, CLICKS),
        (1, 12, 120, ORDERS),
        (2, 10, 0, CLICKS),
    ]
)


@pytest.mark.parametrize("engine", ENGINES)
def test_reverse_rank(engine):
    got = reverse_rank(EVENTS.lazy()).collect(engine=to_engine(engine))
    # sorted by session, ts, aid: (0 s, 10), (60 s, 10), (60 s, 11), (120 s, 12), session 2
    assert got.sort("session", "ts", "aid")["rank"].to_list() == [4, 2, 3, 1, 1]


@pytest.mark.parametrize("engine", ENGINES)
def test_pv(engine):
    a = 2**0.1 - 1  # wgt_1's low end
    b = 2**0.5 - 1  # wgt_2's
    # session 1 has 4 events, so position (rank - 1) / 4 is 0, .25, .5, .75 by rank
    w = lambda lo, pos: lo + (1 - lo) * pos
    expected = pl.DataFrame(
        {
            "session": [1, 1, 1, 2],
            "cand": [10, 11, 12, 10],
            "ts_diff": [60, 60, 0, 0],
            "rev_order": [2, 3, 1, 1],
            # aid 10: two clicks (weight 1) at positions .25 and .75; aid 11: a cart
            # (weight 6) at .5; aid 12: an order (weight 3) at 0
            "pv_wgt_1": [w(a, 0.25) + w(a, 0.75), 6 * w(a, 0.5), 3 * a, a],
            "pv_wgt_2": [w(b, 0.25) + w(b, 0.75), 6 * w(b, 0.5), 3 * b, b],
            "clicks": [2, 0, 0, 1],
            "carts": [0, 1, 0, 0],
            "orders": [0, 0, 1, 0],
        },
        schema={
            "session": pl.Int32,
            "cand": pl.Int32,
            "ts_diff": pl.Int32,
            "rev_order": pl.Int16,
            "pv_wgt_1": pl.Float32,
            "pv_wgt_2": pl.Float32,
            "clicks": pl.Int16,
            "carts": pl.Int16,
            "orders": pl.Int16,
        },
    )
    got = previous_visits(EVENTS, "pv").collect(engine=to_engine(engine))
    assert_frame_equal(got, expected)


def test_pv_2_weights():
    a = 2**0.1 - 1
    got = previous_visits(EVENTS, "pv_2").collect()
    assert got.columns == ["session", "cand", "pv_2_wgt_1", "pv_2_wgt_2"]
    # pv_2's type weights: clicks 0.5, carts 9, orders 0.5
    expected = [0.5 * (2 * a + (1 - a)), 9 * (a + (1 - a) * 0.5), 0.5 * a, 0.5 * a]
    assert got["pv_2_wgt_1"].to_list() == pytest.approx(expected, rel=1e-6)


def test_pv_does_not_depend_on_row_order():
    shuffled = EVENTS.sample(fraction=1.0, shuffle=True, seed=3)
    # only the order within a second matters, and these rows keep it
    keep = shuffled.sort(pl.int_range(pl.len()).alias("i"))
    assert_frame_equal(
        previous_visits(keep, "pv").collect(), previous_visits(EVENTS, "pv").collect()
    )
