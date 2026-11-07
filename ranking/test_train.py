from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from ranking.combine import combined_scores, top20
from ranking.train import (
    XGB_PARAMS,
    _matrix,
    _with_positive,
    feature_columns,
    score_path,
    train_chunks,
)

I32 = pl.Int32


def test_train_chunks():
    files = [Path(f"fold_{k}/train_{i}") for k in range(3) for i in (0, 1)]
    # clicks: the first 2 other chunks in (fold, chunk) order
    assert train_chunks(files, files[0], "clicks") == files[1:3]
    assert train_chunks(files, files[4], "clicks") == files[0:2]
    # carts and orders: all other chunks
    assert train_chunks(files, files[4], "carts") == files[:4] + files[5:]


def test_with_positive():
    df = pl.DataFrame(
        {"session": [1, 1, 2, 2], "cand": [1, 2, 1, 2], "target": [0, 1, 0, 0]}
    )
    assert _with_positive(df)["session"].to_list() == [1, 1]


def test_feature_columns_and_a_fit():
    rng = np.random.default_rng(0)
    n = 2_000
    x = rng.random(n)
    df = pl.DataFrame(
        {
            "session": np.arange(n) // 10,
            "cand": np.arange(n),
            "x": x,
            "noise": rng.random(n),
            "target": (x > 0.7).astype(np.int8),
        }
    )
    cols = feature_columns(df)
    assert cols == ["x", "noise"]
    model = xgb.train(
        {**XGB_PARAMS, "device": "cpu", "seed": 0},
        _matrix(df, cols),
        num_boost_round=20,
    )
    p = model.predict(_matrix(df, cols, label=False))
    assert p[x > 0.8].min() > p[x < 0.6].max()


def _scores(root, fold, model, rows, seeds):
    path = score_path(root, "clicks", fold, model, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {"session": [r[0] for r in rows], "cand": [r[1] for r in rows]},
        schema={"session": I32, "cand": I32},
    )
    df.with_columns(
        **{f"score_{s}": pl.Series(v, dtype=pl.Float32) for s, v in enumerate(seeds)}
    ).write_parquet(path)


def test_combined_scores_follow_the_first_pair(tmp_path):
    # fold 0, model 0 sets the rows; the other pair's (1, 12) isn't one of them
    _scores(tmp_path, 0, 0, [(1, 10), (1, 11)], [[1, 2], [1, 2], [1, 2]])
    _scores(tmp_path, 1, 0, [(1, 10), (1, 12)], [[5, 9], [5, 9], [5, 9]])
    got = combined_scores("clicks", tmp_path, [0, 1], [0]).collect().sort("cand")
    assert got.rows() == [(1, 10, 3 + 15), (1, 11, 6)]


def test_top20():
    scores = pl.DataFrame(
        {
            "session": [1] * 25,
            "cand": list(range(25)),
            "score": [float(c % 5) for c in range(25)],
        }
    )
    got = top20(scores).collect()
    assert got.height == 20 and got["rank"].to_list() == list(range(20))
    # the 5 with score 4 first, smaller cands first
    assert got["cand"].to_list()[:5] == [4, 9, 14, 19, 24]
