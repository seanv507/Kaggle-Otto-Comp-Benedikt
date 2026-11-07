import datetime

import numpy as np
import polars as pl
from word2vec import PARAMS, train, write_corpus

EVENT_SCHEMA = {
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}
T0 = datetime.datetime(2022, 8, 1)


def _events(sessions):
    rows = [(s, a, T0, 0) for s, aids in sessions for a in aids]
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")


def test_write_corpus_keeps_the_event_order(tmp_path):
    events = _events([(7, [3, 1, 2]), (2, [10]), (5, [4, 4])])
    path = tmp_path / "c.txt"
    assert write_corpus(events, path) == 3
    assert path.read_text().splitlines() == ["10", "4 4", "3 1 2"]  # sorted by session


def _corpus(tmp_path, seed=0, n=600):
    """Items 0 and 1 appear in the same context (10 _ 11), items 2 and 3 in another
    (20 _ 21); items 30-59 at random around them."""
    rng = np.random.default_rng(seed)
    sessions = []
    for s in range(n):
        left, right = (10, 11) if s % 2 else (20, 21)
        middle = int(rng.choice([0, 1])) if s % 2 else int(rng.choice([2, 3]))
        noise = rng.choice(np.arange(30, 60), size=2).tolist()
        sessions.append((s, [noise[0], left, middle, right, noise[1]]))
    path = tmp_path / "c.txt"
    write_corpus(_events(sessions), path)
    return path


def _cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def test_train(tmp_path):
    corpus = _corpus(tmp_path)
    # no downsampling: in a corpus this small every item counts as frequent
    emb = train(corpus, workers=1, epochs=20, sample=0)
    assert emb.schema == {
        "aid": pl.Int32,
        "emb": pl.Array(pl.Float32, PARAMS["vector_size"]),
    }
    # min_count 1: every aid
    assert emb["aid"].to_list() == [0, 1, 2, 3, 10, 11, 20, 21, *range(30, 60)]
    v = dict(zip(emb["aid"].to_list(), emb["emb"].to_numpy(), strict=True))
    # items with the same contexts are closer than items with different ones
    assert _cos(v[0], v[1]) > _cos(v[0], v[2]) + 0.1
    assert _cos(v[2], v[3]) > _cos(v[1], v[3]) + 0.1


def test_train_is_deterministic_with_one_worker(tmp_path):
    corpus = _corpus(tmp_path)
    a, b = train(corpus, workers=1), train(corpus, workers=1)
    assert np.array_equal(a["emb"].to_numpy(), b["emb"].to_numpy())
