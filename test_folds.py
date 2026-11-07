import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import folds
import paths
from engines import ENGINES, engine
from event_types import CARTS, CLICKS, ORDERS

EVENT_SCHEMA = {
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}
T0 = datetime.datetime(2022, 8, 22)


def _events(rows):
    """rows of (session, aid, minutes after T0, type)"""
    return pl.DataFrame(
        [(s, a, T0 + datetime.timedelta(minutes=m), t) for s, a, m, t in rows],
        schema=EVENT_SCHEMA,
        orient="row",
    )


# sessions 1-4 are in week 4 (1, 2 in fold 0; 3, 4 in fold 1), 5 only in weeks 1-3, 9 in the Kaggle test
TRAIN = _events(
    [(5, 50, 0, CLICKS), (5, 51, 1, CARTS)]
    + [(s, 10 * s + i, 100 + i, CLICKS) for s in (1, 2, 3, 4) for i in range(3)]
)
TRUNCATED = _events(
    [(s, 10 * s, 100, CLICKS) for s in (1, 2, 3, 4)]
)  # the first event of each
TEST = _events([(9, 90, 200, ORDERS)])
SESSIONS_EVAL = pl.DataFrame(
    {"session": [1, 2, 3, 4], "part": [0, 1, 2, 3], "fold": [0, 0, 1, 1]},
    schema={"session": pl.Int32, "part": pl.UInt8, "fold": pl.UInt8},
)
INPUTS = {
    "train": TRAIN,
    "truncated": TRUNCATED,
    "test": TEST,
    "sessions_eval": SESSIONS_EVAL,
}


def _sorted(df):
    return df.sort("session", "ts", "aid")


@pytest.mark.parametrize("engine_name", ENGINES)
def test_fe_events_fold_0(engine_name):
    got = folds.fe_events(0, **INPUTS).collect(engine=engine(engine_name))
    expected = pl.concat(
        [
            TRAIN.filter(pl.col("session").is_in([3, 4, 5])),
            TRUNCATED.filter(pl.col("session").is_in([1, 2])),
            TEST,
        ]
    )
    assert_frame_equal(_sorted(got), _sorted(expected))


def test_each_week4_session_is_truncated_in_exactly_one_fold():
    n_events = {s: TRAIN.filter(pl.col("session") == s).height for s in (1, 2, 3, 4)}
    truncated_in = {s: [] for s in n_events}
    for k in range(folds.N_FOLDS):
        counts = dict(
            folds.fe_events(k, **INPUTS).group_by("session").len().collect().iter_rows()
        )
        assert counts[5] == 2 and counts[9] == 1
        for s, n in n_events.items():
            assert counts[s] in (1, n)
            if counts[s] == 1:
                truncated_in[s].append(k)
    assert truncated_in == {1: [0], 2: [0], 3: [1], 4: [1]}


def test_heldout_sessions_rejects_bad_fold():
    with pytest.raises(ValueError, match="fold 5"):
        folds.heldout_sessions(5, SESSIONS_EVAL)


@pytest.mark.parametrize(
    ("k", "dataset", "sessions"),
    [
        (0, "train", [1, 2]),
        (1, "train", [3, 4]),
        (None, "train", [1, 2, 3, 4]),
        (0, "sub", [9]),
        (None, "sub", [9]),
    ],
)
def test_candidate_events(k, dataset, sessions):
    got = folds.candidate_events(
        k, dataset, truncated=TRUNCATED, test=TEST, sessions_eval=SESSIONS_EVAL
    ).collect()
    assert sorted(got["session"].unique()) == sessions


def test_candidate_events_rejects_bad_dataset():
    with pytest.raises(ValueError, match="valid"):
        folds.candidate_events(
            0, "valid", truncated=TRUNCATED, test=TEST, sessions_eval=SESSIONS_EVAL
        )


def test_labels_are_filtered_to_the_fold():
    week4 = pl.DataFrame(
        {"session": [1, 3], "aid": [11, 31], "type": [CLICKS, CLICKS]},
        schema={"session": pl.Int32, "aid": pl.Int32, "type": pl.UInt8},
    )
    got = folds.labels(1, week4_labels=week4, sessions_eval=SESSIONS_EVAL).collect()
    assert got["session"].to_list() == [3]


@pytest.mark.parametrize("engine_name", ENGINES)
def test_sample_sessions_keeps_whole_sessions(engine_name):
    sessions = pl.Series(
        "session", [s for s in range(10_000) for _ in range(3)], dtype=pl.Int32
    )
    events = sessions.to_frame()
    kept = folds.sample_sessions(events, 0.1, seed=1).collect(
        engine=engine(engine_name)
    )
    per_session = kept.group_by("session").len()
    assert (per_session["len"] == 3).all()
    assert 900 < per_session.height < 1100
    # the same sessions on another engine and in another file
    again = folds.sample_sessions(events.unique(), 0.1, seed=1).collect()
    assert sorted(again["session"]) == sorted(per_session["session"])
    # another seed, other sessions
    other = folds.sample_sessions(events.unique(), 0.1, seed=2).collect()
    assert set(other["session"]) != set(again["session"])
    assert folds.sample_sessions(events, 1.0).collect().height == events.height


def _cands(rows, **extra):
    return pl.DataFrame(
        {"session": [r[0] for r in rows], "cand": [r[1] for r in rows], **extra},
        schema_overrides={"session": pl.Int32, "cand": pl.Int32},
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_write_candidates_round_trip(tmp_path, lazy):
    df = _cands([(1, 10), (1, 11), (2, 10)], x_wgt=[0.5, 0.25, 1.0])
    path = folds.write_candidates(
        df.lazy() if lazy else df, 0, "train", "x", root=tmp_path
    )
    assert path == tmp_path / "fold_0" / "candidates" / "train" / "x" / "cand.parquet"
    assert_frame_equal(pl.read_parquet(path), df)


@pytest.mark.parametrize(
    ("df", "match"),
    [
        (_cands([(1, 10), (1, 10)]), "duplicate"),
        (_cands([(1, 10)]).select("cand", "session"), "start with"),
        (_cands([(1, 10)]).with_columns(pl.col("cand").cast(pl.Int64)), "Int32"),
        (_cands([(1, None)]), "null"),
    ],
)
def test_write_candidates_rejects_bad_files(tmp_path, df, match):
    with pytest.raises(ValueError, match=match):
        folds.write_candidates(df, 0, "train", "x", root=tmp_path)
    assert not folds.candidate_path(0, "train", "x", tmp_path).exists()


def test_candidate_paths_and_shared_sources(tmp_path):
    folds.write_candidates(
        _cands([(1, 10), (3, 30)]), None, "train", "t4r", root=tmp_path
    )
    folds.write_candidates(_cands([(1, 12)]), 0, "train", "c2o_15", root=tmp_path)
    folds.write_candidates(_cands([(3, 31)]), 1, "train", "c2o_15", root=tmp_path)
    assert list(folds.candidate_paths(0, "train", tmp_path)) == ["c2o_15", "t4r"]
    assert folds.candidate_paths(0, "sub", tmp_path) == {}
    # a shared train file is filtered to the fold's sessions
    t4r = folds.read_candidates(
        0, "train", "t4r", tmp_path, sessions_eval=SESSIONS_EVAL
    ).collect()
    assert t4r["session"].to_list() == [1]
    c2o = folds.read_candidates(
        1, "train", "c2o_15", tmp_path, sessions_eval=SESSIONS_EVAL
    ).collect()
    assert c2o.rows() == [(3, 31)]


@pytest.mark.parametrize("engine_name", ENGINES)
def test_day_index_matches_bennys_tables(engine_name):
    # Benny's day-of-month -> day_ tables
    tables = {
        "train": {21: 0, 22: 0, 23: 1, 24: 2, 25: 3, 26: 4, 27: 5, 28: 6},
        "sub": {28: 0, 29: 0, 30: 1, 31: 2, 1: 3, 2: 4, 3: 5, 4: 6},
    }
    for dataset, table in tables.items():
        start = datetime.datetime.combine(folds.WEEK_START[dataset], datetime.time())
        # each hour of the week, from 22:00 the day before it starts
        ts = [start + datetime.timedelta(hours=h - 2) for h in range(7 * 24)]
        df = pl.LazyFrame({"ts": ts}, schema={"ts": pl.Datetime("ms")})
        got = df.select(folds.day_index(dataset)).collect(engine=engine(engine_name))
        assert got.to_series().to_list() == [table[t.day] for t in ts]


def test_output_root_keeps_samples_apart():
    assert folds.output_root() == paths.DATA_FOLDS
    assert folds.output_root(0.05) == paths.DATA_FOLDS / "sample_0.05"
    assert folds.feature_path(None, "x", folds.output_root(0.05)) == (
        paths.DATA_FOLDS / "sample_0.05" / "shared" / "x.parquet"
    )


def test_candidate_paths_rejects_a_source_in_both_places(tmp_path):
    folds.write_candidates(_cands([(1, 10)]), None, "train", "pv", root=tmp_path)
    folds.write_candidates(_cands([(1, 10)]), 0, "train", "pv", root=tmp_path)
    with pytest.raises(ValueError, match="both"):
        folds.candidate_paths(0, "train", tmp_path)
