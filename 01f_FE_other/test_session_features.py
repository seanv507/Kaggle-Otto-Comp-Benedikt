import datetime

import polars as pl
import pytest
from session_features import session_features

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
H = datetime.timedelta(hours=1)


def _events(rows):
    """rows of (session, aid, time after T0, type)"""
    return pl.DataFrame(
        [(s, a, T0 + t, ty) for s, a, t, ty in rows], schema=EVENT_SCHEMA, orient="row"
    )


# not in time order, to check that the segments don't depend on the file order
EVENTS = _events(
    [
        # session 1, newest first: 10h (aid 3 order, aid 4 click) | 5h gap | 5h (3 click)
        # | 4.5h gap | 0.5h (2 cart), 0h (1 click). Segments 0, 0, 1, 2, 2.
        (1, 3, 5 * H, CLICKS),
        (1, 1, 0 * H, CLICKS),
        (1, 2, 0.5 * H, CARTS),
        (1, 3, 10 * H, ORDERS),
        (1, 4, 10 * H, CLICKS),
        # session 2: one cart
        (2, 7, 0 * H, CARTS),
        # session 3: exactly 4h is not a gap, 4h + 1 ms is
        (3, 1, 0 * H, CLICKS),
        (3, 2, 4 * H, CLICKS),
        (3, 3, 8 * H + datetime.timedelta(milliseconds=1), CLICKS),
    ]
)

EXPECTED = {
    "session": [1, 2, 3],
    "num_aid": [5, 1, 3],
    "num_u_aid": [4, 1, 3],
    "diff": [10 * 3_600_000, 0, 8 * 3_600_000 + 1],
    "num_sessions": [2, 0, 1],
    "num_aid_fs": [2, 1, 1],
    "num_u_aid_fs": [2, 1, 1],
    "num_aid_cl": [3, 0, 3],
    "num_aid_ca": [1, 1, 0],
    "num_aid_co": [1, 0, 0],
    "num_u_aid_cl": [3, 0, 3],
    "num_u_aid_ca": [1, 1, 0],
    "num_u_aid_co": [1, 0, 0],
}
# Benny's per-type _fs features equal the whole-session ones
for _t in ["cl", "ca", "co"]:
    EXPECTED[f"num_aid_{_t}_fs"] = EXPECTED[f"num_aid_{_t}"]
for _t in ["cl", "ca", "co"]:
    EXPECTED[f"num_u_aid_{_t}_fs"] = EXPECTED[f"num_u_aid_{_t}"]


@pytest.mark.parametrize("engine", ENGINES)
def test_session_features(engine):
    got = session_features(EVENTS).collect(engine=to_engine(engine))
    assert got.columns == list(EXPECTED)  # Benny's column order
    assert got.to_dict(as_series=False) == EXPECTED
    assert got.schema["session"] == pl.Int32 and got.schema["diff"] == pl.Int64
    assert all(got.schema[c] == pl.UInt32 for c in EXPECTED if c.startswith("num_"))
