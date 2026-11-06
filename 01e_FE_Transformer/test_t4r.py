import dataclasses
import datetime

import numpy as np
import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal
from t4r_data import PAD, UNKNOWN, aid_map, sequences, to_padded
from t4r_model import Params, SessionTransformer, false_negative_mask, mlm_targets
from t4r_train import predict, recall_at_20

from event_types import CARTS, CLICKS, ORDERS

EVENT_SCHEMA = {  # same dtypes as data/train.parquet
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}
T0 = datetime.datetime(2022, 8, 1)


def _events(rows):
    return pl.DataFrame(
        [(s, a, T0 + datetime.timedelta(minutes=m), t) for s, a, m, t in rows],
        schema=EVENT_SCHEMA,
        orient="row",
    )


def _gpu_available():
    try:
        pl.LazyFrame({"a": [1]}).collect(engine=pl.GPUEngine(raise_on_fail=True))
        return True
    except Exception:
        return False


GPU = pytest.param("gpu", marks=pytest.mark.skipif(not _gpu_available(), reason="GPU engine unavailable"))


def _engine(name):
    return pl.GPUEngine(raise_on_fail=True) if name == "gpu" else name


@pytest.mark.parametrize("engine", ["streaming", "in-memory", GPU])
def test_aid_map(engine):
    a = _events([(1, 7, 0, CLICKS), (1, 5, 1, CLICKS), (2, 5, 0, CLICKS)])
    b = _events([(3, 9, 0, CLICKS), (3, 7, 1, CARTS)])
    expected = pl.DataFrame(
        {"aid": [5, 7, 9], "item": [2, 3, 4], "count": [2, 2, 1]},  # count tie: smaller aid first
        schema={"aid": pl.Int32, "item": pl.Int32, "count": pl.UInt32},
    )
    assert_frame_equal(aid_map([a, b], engine=_engine(engine)), expected)


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])  # sort_by in group_by isn't on GPU
def test_sequences(engine):
    events = _events(
        [
            (1, 10, 0, CLICKS),  # dropped: only the last 3 are kept
            (1, 11, 1, CARTS),
            (1, 12, 2, CLICKS),
            (1, 13, 2, ORDERS),  # same ts as the previous event: file order decides
            (2, 99, 5, CLICKS),  # not in the map -> UNKNOWN
        ]
    )
    aids = pl.DataFrame({"aid": [10, 11, 12, 13], "item": [2, 3, 4, 5]}, schema={"aid": pl.Int32, "item": pl.Int32})
    expected = pl.DataFrame(
        [
            (1, [3, 4, 5], [CARTS, CLICKS, ORDERS], T0 + datetime.timedelta(minutes=1)),
            (2, [UNKNOWN], [CLICKS], T0 + datetime.timedelta(minutes=5)),
        ],
        schema={"session": pl.Int32, "item": pl.List(pl.Int32), "type": pl.List(pl.UInt8), "ts": pl.Datetime("ms")},
        orient="row",
    )
    assert_frame_equal(sequences(events, aids, seq_len=3, engine=engine), expected)


def test_to_padded():
    items = pl.Series([[3, 4, 5], [6], []], dtype=pl.List(pl.Int32))
    np.testing.assert_array_equal(to_padded(items, 4), [[3, 4, 5, 0], [6, 0, 0, 0], [0, 0, 0, 0]])
    with pytest.raises(ValueError):
        to_padded(items, 2)


def test_mlm_targets():
    torch.manual_seed(0)
    items = torch.tensor([[2, 3, 4, 5, 6, 0, 0], [7, 8, 0, 0, 0, 0, 0], [9, 0, 0, 0, 0, 0, 0]])
    for p in [0.0, 0.35, 1.0]:
        for _ in range(50):
            labels = mlm_targets(items, p)
            masked = labels != PAD
            assert (labels[masked] == items[masked]).all()  # targets are the items themselves
            assert not masked[items == PAD].any()  # padding is never a target
            n_masked, n_real = masked.sum(1), (items != PAD).sum(1)
            assert n_masked[:2].min() >= 1  # at least one target ...
            assert (n_masked < n_real).all()  # ... but never all items
            assert n_masked[2] == 0  # so a length-1 sequence has none


def test_false_negative_mask():
    candidates = torch.tensor([2, 4, 6, 8])
    sequences = torch.tensor([[4, 6, 9, 0], [2, 2, 0, 0]])
    labels = torch.tensor([6, 8])
    expected = torch.tensor(
        [
            [False, True, False, False],  # 4 is in the sequence; 6 is the label; 9 isn't a candidate
            [True, False, False, False],  # 2 (twice) is in the sequence
        ]
    )
    assert torch.equal(false_negative_mask(sequences, labels, candidates), expected)


TINY = Params(d_model=16, n_head=2, n_layer=1, dropout=0.0, neg_factor=2, batch_size=8)


def test_model_learns_and_predicts():
    """A tiny model learns that 2 and 3 alternate, and predicts the next one."""
    torch.manual_seed(0)
    # with remove_false_neg, 2 is never a negative for 3 (both are in the session),
    # so the model would never learn to tell them apart
    params = dataclasses.replace(TINY, remove_false_neg=False)
    model = SessionTransformer(n_items=20, params=params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    # 2,3,2,... and 3,2,3,... of lengths 2 to 8, so position alone doesn't give the answer
    rows = []
    for first, second in [(2, 3), (3, 2)]:
        for n in range(2, 9):
            rows.append([first if i % 2 == 0 else second for i in range(n)] + [PAD] * (8 - n))
    pattern = torch.tensor(rows)
    losses = []
    for _ in range(300):
        loss = model.loss(pattern)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20])

    model.eval()
    sessions = torch.tensor(
        [
            [2, 3, 2, 0, 0, 0, 0, 0],
            [2, 3, 2, 3, 2, 3, 2, 3],  # no padding left: predict still has room for the mask
            [3, 2, 0, 0, 0, 0, 0, 0],
        ]
    )
    scores, items = model.predict(sessions, k=5)
    assert items.shape == (3, 5)
    assert (items >= 2).all()  # never PAD or UNKNOWN
    assert items[:, 0].tolist() == [3, 2, 3]
    assert (scores[:, :-1] >= scores[:, 1:]).all()


def test_predict_frame():
    torch.manual_seed(0)
    model = SessionTransformer(n_items=6, params=TINY)
    item_to_aid = np.array([-1, -1, 100, 101, 102, 103])
    items = np.array([[2, 3, 0], [4, 0, 0], [5, 2, 3]], dtype=np.int32)
    cands = predict(model, np.array([7, 8, 9]), items, item_to_aid, k=3, batch_size=2, device="cpu")
    assert cands.schema == {"session": pl.Int32, "cand": pl.Int32, "t4r_rank": pl.UInt8, "t4r_score": pl.Float32}
    assert cands["session"].to_list() == [7, 7, 7, 8, 8, 8, 9, 9, 9]
    assert cands["t4r_rank"].to_list() == [0, 1, 2] * 3
    assert cands["cand"].is_in([100, 101, 102, 103]).all()
    assert cands.group_by("session").agg(pl.col("t4r_score").sum())["t4r_score"].to_numpy() == pytest.approx(1.0)


def test_recall_at_20():
    cands = pl.DataFrame(
        {"session": [1, 1, 2], "cand": [10, 11, 20], "t4r_rank": [0, 25, 0]},  # rank 25 is outside the top 20
        schema={"session": pl.Int32, "cand": pl.Int32, "t4r_rank": pl.UInt8},
    )
    labels = pl.DataFrame(
        [
            (1, 10, CLICKS),  # hit
            (2, 21, CLICKS),  # miss
            (1, 11, CARTS),  # in the candidates but ranked 25: miss
            (1, 12, CARTS),
            (3, 30, ORDERS),  # session without candidates: miss
        ],
        schema={"session": pl.Int32, "aid": pl.Int32, "type": pl.UInt8},
        orient="row",
    )
    assert recall_at_20(cands, labels) == {"clicks": 0.5, "carts": 0.0, "orders": 0.0}
