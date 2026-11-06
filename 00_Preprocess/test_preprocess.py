import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from engines import ENGINES
from engines import engine as to_engine
from event_types import CARTS, CLICKS, ORDERS
from preprocess import (
    check_truncation_is_prefix,
    labels_after_cut,
    labels_long,
    partition_sessions,
)

EVENT_SCHEMA = {  # same dtypes as data/train.parquet
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}
LABEL_SCHEMA = {  # same dtypes as valid_test_labels.parquet
    "session": pl.Int32,
    "type": pl.UInt8,
    "ground_truth": pl.List(pl.Int32),
}


@pytest.mark.parametrize("engine", ENGINES)
def test_labels_long(engine):
    labels = pl.DataFrame(
        [
            (5, ORDERS, [30, 31]),
            (5, CLICKS, [10]),
            (5, CARTS, [21, 20]),
            (2, CLICKS, [11]),
            (2, CARTS, []),
        ],
        schema=LABEL_SCHEMA,
        orient="row",
    )
    expected = pl.DataFrame(
        [
            (2, 11, CLICKS),
            (5, 10, CLICKS),
            (5, 20, CARTS),
            (5, 21, CARTS),
            (5, 30, ORDERS),
            (5, 31, ORDERS),
        ],
        schema={"session": pl.Int32, "aid": pl.Int32, "type": pl.UInt8},
        orient="row",
    )
    assert_frame_equal(labels_long(labels, engine=to_engine(engine)), expected)


SESSIONS = pl.Series("session", list(range(100, 203)) * 2, dtype=pl.Int32)  # 103 unique, duplicated


def test_partition_covers_and_balanced():
    parts = partition_sessions(SESSIONS, n_parts=10, seed=0)
    assert parts.schema == {"session": pl.Int32, "part": pl.UInt8}
    assert parts["session"].to_list() == list(range(100, 203))  # each session once
    sizes = parts.group_by("part").len()["len"]
    assert sizes.len() == 10
    assert sizes.min() == 10 and sizes.max() == 11


def test_partition_seeded():
    a = partition_sessions(SESSIONS, n_parts=10, seed=0)
    assert_frame_equal(a, partition_sessions(SESSIONS.shuffle(seed=1), n_parts=10, seed=0))
    assert not a.equals(partition_sessions(SESSIONS, n_parts=10, seed=1))


def _events(rows):
    t0 = datetime.datetime(2022, 8, 22)
    return pl.DataFrame(
        [(s, a, t0 + datetime.timedelta(minutes=m), t) for s, a, m, t in rows],
        schema=EVENT_SCHEMA,
        orient="row",
    )


FULL = _events(
    [
        (1, 10, 0, CLICKS),
        (1, 11, 1, CARTS),
        (1, 12, 2, ORDERS),
        (2, 20, 0, CLICKS),
        (2, 21, 5, CLICKS),
        (3, 30, 0, CLICKS),  # not in truncated: ignored
        (3, 31, 1, CLICKS),
    ]
)
CPU_ENGINES = ["streaming", "in-memory"]  # int_range().over() is not on the GPU engine


@pytest.mark.parametrize("engine", CPU_ENGINES)
def test_prefix_passes(engine):
    truncated = _events([(1, 10, 0, CLICKS), (1, 11, 1, CARTS), (2, 20, 0, CLICKS)])
    assert check_truncation_is_prefix(FULL, truncated, engine=engine).is_empty()


@pytest.mark.parametrize("engine", CPU_ENGINES)
def test_prefix_changed_aid(engine):
    truncated = _events([(1, 10, 0, CLICKS), (1, 99, 1, CARTS), (2, 20, 0, CLICKS)])
    bad = check_truncation_is_prefix(FULL, truncated, engine=engine)
    assert bad.select("session", "aid", "aid_full").rows() == [(1, 99, 11)]


@pytest.mark.parametrize("engine", CPU_ENGINES)
def test_prefix_missing_session(engine):
    truncated = _events([(1, 10, 0, CLICKS), (4, 40, 0, CLICKS)])
    bad = check_truncation_is_prefix(FULL, truncated, engine=engine)
    assert bad.select("session", "aid_full").rows() == [(4, None)]


LABELS_FULL = _events(
    [
        (1, 10, 0, CLICKS),  # session 1 cut after 1 event
        (1, 11, 1, CARTS),
        (1, 12, 2, CLICKS),  # first click after the cut
        (1, 13, 3, CLICKS),
        (1, 11, 4, CARTS),  # carted twice -> one label
        (1, 14, 5, ORDERS),
        (2, 20, 0, CLICKS),  # session 2 cut after 2 events
        (2, 21, 1, CARTS),  # before the cut: not a label
        (2, 22, 2, ORDERS),
        (3, 30, 0, CLICKS),  # not truncated: no labels
        (3, 31, 1, CLICKS),
    ]
)


@pytest.mark.parametrize("engine", CPU_ENGINES)
def test_labels_after_cut(engine):
    truncated = _events([(1, 10, 0, CLICKS), (2, 20, 0, CLICKS), (2, 21, 1, CARTS)])
    expected = pl.DataFrame(
        [(1, 12, CLICKS), (1, 11, CARTS), (1, 14, ORDERS), (2, 22, ORDERS)],
        schema={"session": pl.Int32, "aid": pl.Int32, "type": pl.UInt8},
        orient="row",
    )
    assert_frame_equal(labels_after_cut(LABELS_FULL, truncated, engine=engine), expected)
