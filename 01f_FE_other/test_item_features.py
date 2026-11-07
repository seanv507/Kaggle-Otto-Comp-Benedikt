import datetime as dt

import polars as pl
import pytest
from item_features import item_day_features, item_features

from engines import ENGINES
from engines import engine as to_engine
from event_types import CARTS, CLICKS, ORDERS

EVENT_SCHEMA = {
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}


def _events(rows):
    """rows of (aid, datetime, type); one session each"""
    return pl.DataFrame(
        [(i, a, t, ty) for i, (a, t, ty) in enumerate(rows)],
        schema=EVENT_SCHEMA,
        orient="row",
    )


# the 4 weeks of train; the train dataset's week starts 2022-08-22, the sub's 2022-08-29
TRAIN = _events(
    [
        (1, dt.datetime(2022, 8, 21, 23), CLICKS),  # train p1w; sub p2w, p3w
        # train p1w-p3w (a window's start is included); sub p2w, p3w
        (1, dt.datetime(2022, 8, 15), CARTS),
        # train: none (it's in train's own week); sub p1w-p3w
        (2, dt.datetime(2022, 8, 22), ORDERS),
        (3, dt.datetime(2022, 8, 7, 23, 59), CLICKS),  # train p3w; sub none
    ]
)
WEEK4 = _events(
    [
        (1, dt.datetime(2022, 8, 22, 1), CLICKS),  # day_ 0
        (4, dt.datetime(2022, 8, 21, 23), CARTS),  # the 21st: day_ 0 too
        (4, dt.datetime(2022, 8, 28, 12), ORDERS),  # day_ 6
    ]
)
TEST = _events(
    [
        (2, dt.datetime(2022, 9, 4, 21), CLICKS),  # day_ 6
        (5, dt.datetime(2022, 8, 28, 22, 30), CLICKS),  # day_ 0
    ]
)

COLUMNS = ["aid"] + [
    f"{p}_{t}"
    for p in ["p0w", "p1w", "p2w", "p3w"]
    for t in ["clicks", "carts", "orders"]
]


def _rows(**by_aid):
    """{aid: {column: count}} -> rows in COLUMNS order, other counts 0"""
    return [
        tuple([int(aid[1:])] + [counts.get(c, 0) for c in COLUMNS[1:]])
        for aid, counts in by_aid.items()
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_item_features_train(engine):
    got = item_features("train", WEEK4, TRAIN).collect(engine=to_engine(engine))
    assert got.columns == COLUMNS
    # p0w is doubled, as in Benny's code; aid 2 only has events in its own week
    assert got.rows() == _rows(
        a1={
            "p0w_clicks": 2,
            **{f"p{n}w_{t}": 1 for n in (1, 2, 3) for t in ("clicks", "carts")},
        },
        a3={"p3w_clicks": 1},
        a4={"p0w_carts": 2, "p0w_orders": 2},
    )
    assert all(got.schema[c] == pl.UInt32 for c in COLUMNS[1:])


@pytest.mark.parametrize("engine", ENGINES)
def test_item_features_sub(engine):
    got = item_features("sub", TEST, TRAIN).collect(engine=to_engine(engine))
    assert got.rows() == _rows(
        a1={f"p{n}w_{t}": 1 for n in (2, 3) for t in ("clicks", "carts")},
        a2={"p0w_clicks": 2, "p1w_orders": 1, "p2w_orders": 1, "p3w_orders": 1},
        a5={"p0w_clicks": 2},
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_item_day_features(engine):
    got = item_day_features("train", WEEK4).collect(engine=to_engine(engine))
    assert got.columns == ["aid", "day_", "sday_clicks", "sday_carts", "sday_orders"]
    assert got.rows() == [(1, 0, 1, 0, 0), (4, 0, 0, 1, 0), (4, 6, 0, 0, 1)]
    got = item_day_features("sub", TEST).collect(engine=to_engine(engine))
    assert got.rows() == [(2, 6, 1, 0, 0), (5, 0, 1, 0, 0)]
