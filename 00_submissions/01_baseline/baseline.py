"""Chris's public LB 0.575 baseline, which Benny used to fill sessions with fewer than
20 recommendations (00_1_CoMatrix-Basic.ipynb and 00_2_CalcSub.ipynb).

Three covisitation matrices from all of TRAIN and TEST (not per fold): carts-orders
(top 15), buy2buy (top 15) and time-weighted clicks (top 20). Their pairs are the
same as Benny's caco, b2b and clicks matrices (01b_FE_CoVisit_1/benny_covisits.py).
Then Chris's rules, per Kaggle test session:
    clicks          suggest_clicks
    carts, orders   suggest_buys
written to data_tmp/baseline/<target>.parquet (folds.baseline_dir) as (session, aid, rank), rank 0 first,
with their recall@20 against TEST_LABELS.

Chris's rules use Counter.most_common, which breaks ties by first insertion; the
Polars version here keeps that order (`first`). His candidates can leave a session
with fewer than 20.

Run from the repo root:
    uv run python 00_submissions/01_baseline/baseline.py --sample 0.05
"""

import argparse
import sys
import time

import covisits
import polars as pl
from beartype import beartype

import folds
import metrics
import paths
from event_types import CARTS, CLICKS, ORDERS

sys.path.insert(0, str(paths.ROOT / "01b_FE_CoVisit_1"))
import benny_covisits as bc

TOP = {
    "caco": 15,
    "b2b": 15,
    "clicks": 20,
}  # Chris's top_15_carts_orders, top_15_buy2buy, top_20_clicks
TYPE_MP = {CLICKS: 1.0, CARTS: 6.0, ORDERS: 3.0}  # type_weight_multipliers
N = 20


@beartype
def build_matrix(
    events: pl.LazyFrame,
    name: str,
    engine="streaming",
    session_chunk: int | None = None,
) -> pl.DataFrame:
    """(aid, cand, wgt, n): a baseline matrix, n = the cand's position (wgt descending)."""
    m = bc.MATRICES[name]
    df = covisits.build_covisits(
        events, m.pairs, session_chunk, engine, TOP[name], "streaming"
    )
    return df.select(
        pl.col("aid").cast(pl.Int32),
        pl.col("aid_y").cast(pl.Int32).alias("cand"),
        pl.col("value").alias("wgt"),
        pl.int_range(pl.len(), dtype=pl.Int32).over("aid").alias("n"),
    )


def _history(events: pl.LazyFrame) -> pl.LazyFrame:
    """Per event: i (time order), r (the aid's recency rank among unique aids, 0 =
    most recent), n_events and n_unique of the session. Events in time order."""
    i = pl.int_range(pl.len()).over("session")
    lf = events.select("session", "aid", "type").with_columns(
        i=i, n_events=pl.len().over("session")
    )
    last = lf.group_by("session", "aid").agg(last=pl.col("i").max())
    last = last.with_columns(
        r=pl.col("last").rank("ordinal", descending=True).over("session").cast(pl.Int64)
        - 1,
        n_unique=pl.len().over("session"),
    )
    return lf.join(last, on=["session", "aid"])


def _logspace_weight(lo: float) -> pl.Expr:
    """np.logspace(lo, 1, n, base=2) - 1 at event i (endpoint included)."""
    n = pl.col("n_events")
    t = pl.when(n > 1).then(pl.col("i") / (n - 1)).otherwise(0.0)
    return 2 ** (lo + (1 - lo) * t) - 1


def _most_common(items: pl.LazyFrame, k: int = N) -> pl.LazyFrame:
    """Counter(...).most_common(k) per session: items are (session, aid, score, first);
    sort by score descending, then first (insertion) ascending."""
    return (
        items.group_by("session", "aid")
        .agg(pl.col("score").sum(), pl.col("first").min())
        .sort("session", "score", "first", descending=[False, True, False])
        .with_columns(pos=pl.int_range(pl.len()).over("session"))
        .filter(pl.col("pos") < k)
    )


def _covisit_items(
    hist: pl.LazyFrame, matrix: pl.DataFrame, of: pl.Expr, offset: int = 0
) -> pl.LazyFrame:
    """The concatenated matrix lists of the unique aids selected by `of`, in recency
    order: (session, aid=cand, score=1, first=(offset, recency, n))."""
    unique = hist.filter(of).group_by("session", "aid").agg(pl.col("i").max())
    unique = unique.with_columns(
        rr=pl.col("i").rank("ordinal", descending=True).over("session").cast(pl.Int64)
    )
    m = matrix.lazy().select("aid", "cand", "n")
    return unique.join(m, on="aid").select(
        "session",
        aid="cand",
        score=pl.lit(1.0),
        first=offset * 10**9 + pl.col("rr") * 1000 + pl.col("n").cast(pl.Int64),
    )


def _history_items(hist: pl.LazyFrame, lo: float) -> pl.LazyFrame:
    return hist.select(
        "session",
        "aid",
        score=_logspace_weight(lo)
        * pl.col("type").replace_strict(TYPE_MP, default=float("nan")),
        first=pl.col("i").cast(pl.Int64),
    )


def _fill(hist: pl.LazyFrame, top: pl.LazyFrame) -> pl.LazyFrame:
    """unique_aids (recency order) + the top items not among them, up to N."""
    own = (
        hist.group_by("session", "aid")
        .agg(pl.col("r").first())
        .select("session", "aid", pos=pl.col("r"))
    )
    extra = top.join(own, on=["session", "aid"], how="anti").select(
        "session", "aid", pos=10**6 + pl.col("pos")
    )
    return (
        pl.concat([own, extra])
        .sort("session", "pos")
        .with_columns(rank=pl.int_range(pl.len()).over("session"))
        .filter(pl.col("rank") < N)
        .select("session", "aid", "rank")
    )


@beartype
def suggest_clicks(
    events: pl.LazyFrame | pl.DataFrame, clicks: pl.DataFrame
) -> pl.LazyFrame:
    """Chris's suggest_clicks. `events` in time order within a session."""
    hist = _history(events.lazy())
    many = pl.col("n_unique") >= N
    weighted = _most_common(_history_items(hist.filter(many), 0.1)).select(
        "session", "aid", rank="pos"
    )
    few = hist.filter(~many)
    top = _most_common(_covisit_items(few, clicks, pl.lit(True)))
    return pl.concat([weighted, _fill(few, top)]).sort("session", "rank")


@beartype
def suggest_buys(
    events: pl.LazyFrame | pl.DataFrame, buys: pl.DataFrame, buy2buy: pl.DataFrame
) -> pl.LazyFrame:
    """Chris's suggest_buys (carts and orders). `events` in time order within a session."""
    hist = _history(events.lazy())
    many = pl.col("n_unique") >= N
    is_buy = pl.col("type").is_in([CARTS, ORDERS])
    h = hist.filter(many)
    # history weights, then +0.1 for each buy2buy item of the unique buys (inserted after)
    b2b = _covisit_items(h, buy2buy, is_buy, offset=1).with_columns(score=pl.lit(0.1))
    weighted = _most_common(pl.concat([_history_items(h, 0.5), b2b])).select(
        "session", "aid", rank="pos"
    )
    few = hist.filter(~many)
    items = pl.concat(
        [
            _covisit_items(few, buys, pl.lit(True)),
            _covisit_items(few, buy2buy, is_buy, offset=1),
        ]
    )
    return pl.concat([weighted, _fill(few, _most_common(items))]).sort(
        "session", "rank"
    )


def main(sample: float, engine, session_chunk: int | None):
    out = folds.baseline_dir(sample)
    out.mkdir(parents=True, exist_ok=True)
    # all of train and test, as Chris's LB run
    events = folds.sample_sessions(
        pl.concat([pl.scan_parquet(paths.TRAIN), pl.scan_parquet(paths.TEST)]), sample
    )
    matrices = {}
    for name, top in TOP.items():
        start = time.time()
        path = out / f"top_{top}_{name}.parquet"
        if not path.exists():
            build_matrix(events, name, engine, session_chunk).write_parquet(path)
        matrices[name] = pl.read_parquet(path)
        print(
            f"{path.relative_to(paths.ROOT)}: {matrices[name].height:,} pairs ({time.time() - start:.0f} s)"
        )
    test = folds.sample_sessions(paths.TEST, sample)
    labels = folds.sample_sessions(paths.TEST_LABELS, sample)
    clicks = suggest_clicks(test, matrices["clicks"]).collect()
    buys = suggest_buys(test, matrices["caco"], matrices["b2b"]).collect()
    weights = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}
    score = 0.0
    for target, df in [("clicks", clicks), ("carts", buys), ("orders", buys)]:
        df.write_parquet(out / f"{target}.parquet")
        code = {"clicks": CLICKS, "carts": CARTS, "orders": ORDERS}[target]
        r = metrics.recall(
            df.select("session", cand="aid", score=-pl.col("rank")),
            labels.filter(pl.col("type") == code),
            "score",
        )[target]
        score += weights[target] * r
        print(f"{target}: recall@20 {r:.4f}")
    print(f"Kaggle test score: {score:.4f} (Chris's public LB: 0.575)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sample", type=float, default=1.0, help="fraction of sessions"
    )
    parser.add_argument(
        "--engine", default="streaming", help="streaming, in-memory or gpu"
    )
    parser.add_argument(
        "--session-chunk",
        type=int,
        default=None,
        help="sessions per chunk for the matrices",
    )
    args = parser.parse_args()
    main(
        args.sample,
        pl.GPUEngine(raise_on_fail=True) if args.engine == "gpu" else args.engine,
        args.session_chunk,
    )
