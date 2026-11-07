import datetime

import numpy as np
import polars as pl
import torch
from gru_data import inference_inputs, training_examples, training_examples_chunked
from gru_model import CONFIGS, GRU, Config, false_negative_mask
from gru_train import combine, padded, predict, tensors

from event_types import CARTS, CLICKS, ORDERS

EVENT_SCHEMA = {
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}
T0 = datetime.datetime(2022, 8, 1)


def _events(sessions):
    rows = [
        (s, a, T0 + datetime.timedelta(seconds=i), t)
        for s, evs in sessions
        for i, (a, t) in enumerate(evs)
    ]
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")


def test_training_examples():
    # session 1: aids 10, 11, 12 (click, cart, order); targets 12 (inputs 10, 11) and 11 (input 10)
    events = _events(
        [(1, [(10, CLICKS), (11, CARTS), (12, ORDERS)]), (2, [(20, CLICKS)])]
    )
    got = training_examples(events).collect().sort("target")
    assert got["target"].to_list() == [13, 14]  # aid + 2
    assert got["item"].to_list() == [[12], [12, 13]]
    assert got["type"].to_list() == [[CLICKS + 2], [CLICKS + 2, CARTS + 2]]
    # reverse rank relative to the target: 2 for the event just before it
    assert got["rank"].to_list() == [[2], [3, 2]]
    assert got["target_type"].to_list() == [CARTS + 2, ORDERS + 2]


def test_training_examples_limits():
    """At most 40 targets per session, at most 19 inputs each."""
    events = _events([(1, [(a, CLICKS) for a in range(100)])])
    got = training_examples(events).collect()
    assert got.height == 40
    assert got["item"].list.len().max() == 19
    latest = got.filter(pl.col("target") == 99 + 2)
    assert latest["item"].to_list() == [[a + 2 for a in range(80, 99)]]
    assert latest["rank"].to_list() == [list(range(20, 1, -1))]


def test_inference_inputs():
    events = _events([(5, [(a, CLICKS) for a in range(25)]), (3, [(7, CARTS)])])
    got = inference_inputs(events).collect()
    assert got["session"].to_list() == [3, 5]
    assert got["item"].to_list()[0] == [9] and got["rank"].to_list()[0] == [2]
    assert got["item"].to_list()[1] == [a + 2 for a in range(6, 25)]
    assert got["rank"].to_list()[1] == list(range(20, 1, -1))


def test_padded():
    df = pl.DataFrame({"x": [[1, 2], [3]]}, schema={"x": pl.List(pl.Int32)})
    a = padded(df, "x")
    assert (
        a.shape == (2, 19)
        and a[0, :3].tolist() == [1, 2, 0]
        and a[1, :2].tolist() == [3, 0]
    )


def test_false_negative_mask():
    inputs = torch.tensor([[5, 7, 0], [9, 0, 0]])
    labels = torch.tensor([7, 3])
    candidates = torch.tensor([3, 5, 7, 9])
    got = false_negative_mask(inputs, labels, candidates)
    # row 0: 5 is an input and not the label; 7 is the label. row 1: 9 is an input
    assert got.tolist() == [[False, True, False, False], [False, False, False, True]]


def test_configs_are_bennys():
    assert [c.name for c in CONFIGS] == [
        "gru_F_cl",
        "gru_T_cl",
        "gru_F_ca",
        "gru_T_ca",
        "gru_F_or",
        "gru_T_or",
    ]
    assert all(c.width == 50 for c in CONFIGS if c.use_pretrained)


def test_gru_learns_a_next_item_pattern():
    """Sessions cycle 2 -> 3 -> 4 -> 2 ...; the model must predict the next one."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    sessions = []
    for s in range(400):
        start, n = int(rng.integers(3)), int(rng.integers(3, 12))
        sessions.append((s, [((start + i) % 3 + 10, CLICKS) for i in range(n)]))
    examples = training_examples(_events(sessions)).collect()
    c = Config("clicks", False, 64, 0.01, 2, 0.0, 1.0, 16, 1, 0.0)
    model = GRU(30, c)
    opt = torch.optim.Adam(model.parameters(), lr=c.lr)
    for step in range(200):
        b = tensors(examples.sample(64, seed=step), "cpu")
        loss = model.loss(b["item"], b["type"], b["rank"], b["length"], b["target"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    inputs = inference_inputs(
        _events([(1, [(10, CLICKS), (11, CLICKS)]), (2, [(12, CLICKS)])])
    ).collect()
    got = predict(model, inputs, "cpu")
    top = (
        got.group_by("session")
        .agg(pl.col("cand").first())
        .sort("session")["cand"]
        .to_list()
    )
    assert top == [12, 10]  # after 10, 11 comes 12; after 12 comes 10
    assert got["cand"].min() >= 0  # padding and id 1 are never predicted


def test_combine():
    a = pl.DataFrame({"session": [1, 1], "cand": [10, 11], "score": [0.6, 0.4]})
    b = pl.DataFrame({"session": [1], "cand": [12], "score": [1.0]})
    got = combine({"gru_F_cl": a, "gru_T_cl": b})
    assert got.rows() == [(1, 10, 0.6, 0.0), (1, 11, 0.4, 0.0), (1, 12, 0.0, 1.0)]


def test_training_examples_chunked():
    """Chunks over session ids give the same examples as one pass."""
    rng = np.random.default_rng(1)
    sessions = [
        (
            s,
            [
                (int(a), int(t))
                for a, t in zip(
                    rng.integers(50, size=n), rng.integers(3, size=n), strict=True
                )
            ],
        )
        for s, n in enumerate(rng.integers(1, 70, size=60))
    ]
    events = _events(sessions)
    keys = ["target", "item", "type", "rank", "target_type"]
    want = training_examples(events).collect().sort(keys)
    for chunk in (1, 7, 1000):
        got = training_examples_chunked(events, chunk).sort(keys)
        assert got.equals(want)


def test_padded_in_slices(monkeypatch):
    """Padding in row slices gives the same array as one pass."""
    import gru_train

    df = pl.DataFrame(
        {"x": [[1, 2], [3], [4, 5, 6], [], [7]]}, schema={"x": pl.List(pl.Int32)}
    )
    want = padded(df, "x")
    monkeypatch.setattr(gru_train, "ROWS_PER_SLICE", 2)
    got = gru_train.padded(df, "x")
    assert got.shape == (5, 19) and (got == want).all()
