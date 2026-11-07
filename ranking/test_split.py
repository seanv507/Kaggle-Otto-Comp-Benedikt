import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from engines import ENGINES
from engines import engine as to_engine
from event_types import CARTS, CLICKS, ORDERS
from ranking.split import (
    SPLITS,
    add_features,
    add_sources,
    add_target,
    candidates,
    generator_columns,
)

I32 = pl.Int32


def _source(rows, **cols):
    df = pl.DataFrame(
        {"session": [r[0] for r in rows], "cand": [r[1] for r in rows]},
        schema={"session": I32, "cand": I32},
    )
    return df.with_columns(**{c: pl.Series(v) for c, v in cols.items()}).lazy()


@pytest.mark.parametrize("engine", ENGINES)
def test_candidates_top_n_per_source(engine):
    sources = {
        # session 1: 10 > 11 > 12; the top 2 are 10 and 11
        "c2o_15": _source(
            [(1, 12), (1, 10), (1, 11), (2, 20)], c2o_wgt_15=[1, 3, 2, 5]
        ),
        # ties at the cut go to the smaller cand: 13 rather than 14
        "pv": _source([(1, 14), (1, 13), (1, 10)], pv_wgt_1=[1.0, 1.0, 2.0]),
        # not a generating source
        "t4r": _source([(1, 99)], t4r_score=[1.0]),
    }
    got = candidates(sources, {"c2o_15": "c2o_wgt_15", "pv": "pv_wgt_1"}, n=2)
    assert sorted(got.collect(engine=to_engine(engine)).rows()) == [
        (1, 10),
        (1, 11),
        (1, 13),
        (2, 20),
    ]


def test_add_sources_fills_zero_and_drops_ignored():
    cands = _source([(1, 10), (1, 11)])
    sources = {
        # c2o_num_15 isn't in all 3 of Benny's ignore lists, so it's kept
        "c2o_15": _source([(1, 10)], c2o_wgt_15=[3], c2o_num_15=[1]),
        # caco_wgt_15 is in all 3, so it's dropped
        "caco_15": _source([(1, 11)], caco_wgt_15=[2.5], caco_num_15=[4]),
    }
    got = add_sources(cands, sources).collect()
    assert got.columns == ["session", "cand", "c2o_wgt_15", "c2o_num_15", "caco_num_15"]
    assert got.rows() == [(1, 10, 3, 1, 0), (1, 11, 0, 0, 4)]


def test_add_sources_rejects_repeated_columns():
    sources = {"a": _source([(1, 10)], x=[1]), "b": _source([(1, 10)], x=[2])}
    with pytest.raises(ValueError, match="repeats"):
        add_sources(_source([(1, 10)]), sources).collect()


def test_add_features():
    t0 = datetime.datetime(2022, 8, 22, 12)
    events = pl.LazyFrame(
        {
            "session": [1, 1],
            "aid": [10, 11],
            "ts": [t0, t0 + datetime.timedelta(days=2)],
            "type": [CLICKS, CARTS],
        },
        schema={"session": I32, "aid": I32, "ts": pl.Datetime("ms"), "type": pl.UInt8},
    )
    item = pl.LazyFrame(
        {"aid": [10], "p0w_clicks": [4]}, schema={"aid": I32, "p0w_clicks": pl.UInt32}
    )
    session = pl.LazyFrame(
        {"session": [1], "num_aid": [2]}, schema={"session": I32, "num_aid": pl.UInt32}
    )
    item_day = pl.LazyFrame(
        {"aid": [10, 11], "day_": [2, 0], "sday_clicks": [7, 9]},
        schema={"aid": I32, "day_": pl.Int8, "sday_clicks": pl.UInt32},
    )
    got = add_features(
        _source([(1, 10), (1, 11)]), "train", events, item, session, item_day
    ).collect()
    # the session's last event is on the 24th: day_ 2
    assert got.select(
        "cand", "p0w_clicks", "num_aid", "day_", "sday_clicks"
    ).rows() == [
        (10, 4, 2, 2, 7),
        (11, 0, 2, 2, 0),
    ]


def test_add_target():
    labels = pl.LazyFrame(
        {"session": [1, 1, 1], "aid": [10, 11, 10], "type": [CLICKS, ORDERS, ORDERS]},
        schema={"session": I32, "aid": I32, "type": pl.UInt8},
    )
    cands = _source([(1, 10), (1, 11), (1, 12)])
    clicks = add_target(cands, labels, "clicks").collect()
    assert_frame_equal(
        clicks.select("target"),
        pl.DataFrame({"target": [1, 0, 0]}, schema={"target": pl.Int8}),
    )
    assert add_target(cands, labels, "orders").collect()["target"].to_list() == [
        1,
        1,
        0,
    ]


def test_candidates_min_sources():
    sources = {
        "a": _source([(1, 10), (1, 11)], a_wgt=[2.0, 1.0]),
        "b": _source([(1, 10), (1, 12)], b_wgt=[1.0, 2.0]),
    }
    got = candidates(sources, {"a": "a_wgt", "b": "b_wgt"}, n=5, min_sources=2)
    assert got.collect().rows() == [(1, 10)]  # the only cand both propose


def test_generator_columns():
    sources = {
        "c2o_15": _source([(1, 10)], c2o_wgt_15=[1], c2o_num_15=[1]),
        "t4r": _source([(1, 10)], t4r_rank=[0], t4r_score=[0.5]),
        "gru_scores_20": _source([(1, 10)], gru_score=[0.1]),
    }
    spec = SPLITS["carts"]
    assert generator_columns(sources, spec) == {
        "c2o_15": "c2o_wgt_15",
        "t4r": "t4r_score",
    }
    bad = {"x": _source([(1, 10)], something=[1])}
    with pytest.raises(ValueError, match="ranking columns"):
        generator_columns(bad, spec)
    assert generator_columns(sources, SPLITS["clicks"]) == {"c2o_15": "c2o_wgt_15"}
