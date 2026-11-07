"""Benny's own covisitation matrices and their candidates, per fold.

Replaces 01b_FE_CoVisit_1/03_CoMatrix-{1-Click2Order, 2-CartsOrders, 3-buy2buy,
4-clicks, 6-weighted-clicks-dev, 7-weighted-types}.ipynb. Each notebook builds a
matrix from folds.fe_events(k) and turns it into candidates for fold k's held-out
sessions (`train`) and the Kaggle test sessions (`sub`). The matrices are built with
the helpers of Chris's part (covisits.py in ../Kaggle-OTTO-Comp): `build_covisits`
runs the pairs per session chunk, and its conventions apply (row order is time order
within a session; after a self-join the y event's columns end in `_y`).

Per fold k it writes
    data_folds/fold_<k>/covisits/benny_<matrix>.parquet   (aid, cand, wgt[, ...])
    data_folds/fold_<k>/candidates/{train,sub}/<source>/cand.parquet
for the sources in SOURCES. See README.md for what each one is.

Run from the repo root (a sample first: outputs then go to data_folds/sample_<f>):
    uv run python 01b_FE_CoVisit_1/benny_covisits.py --sample 0.02 --folds 0 --engine streaming
"""

import argparse
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import covisits as cv
import polars as pl
from beartype import beartype

import folds
import metrics
from event_types import CARTS, CLICKS, ORDERS
from session_ops import reverse_rank

ROW = cv.ROW
TS = pl.col("ts").dt.epoch("s")  # Benny: (ts / 1000).astype('int32')
TS_Y = pl.col("ts_y").dt.epoch("s")
DT_S = (TS - TS_Y).abs()  # seconds apart
TYPE_WEIGHT = {CLICKS: 1.0, CARTS: 6.0, ORDERS: 3.0}  # Benny's type_weight
DAY = 24 * 60 * 60


def _last_n(events: pl.LazyFrame, n: int = 30) -> pl.LazyFrame:
    """Each session's last n events (Benny: sort ts descending, cumcount < n), with ROW."""
    k = pl.col(ROW)
    return cv.with_row(events).filter(k + n > k.max().over("session"))


def _log_weight(x: pl.Expr) -> pl.Expr:
    return 1 / (x.cast(pl.Float64) + 2).log()


# --- 1. click to order (c2o) ---


def c2o_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    """A click and a cart/order of the same session within 7 days, once per session.

    No aid != aid_y filter. value: the number of sessions.
    """
    x = events.filter(pl.col("type") == CLICKS)
    y = events.filter(pl.col("type") != CLICKS)
    return (
        cv.session_self_join(x, y)
        .filter(DT_S < 7 * DAY)
        .unique(subset=["session", "aid", "aid_y"])
        .group_by("aid", "aid_y")
        .agg(pl.len().alias("value"))
    )


def c2o_reduce(pairs: pl.LazyFrame) -> pl.LazyFrame:
    return cv.sum_value(pairs).filter(pl.col("value") > 1)


# --- 2. carts-orders (caco) and 4. clicks: the last 30 events, within 24 hours ---


def _tail_pairs(events: pl.LazyFrame, value: pl.Expr) -> pl.LazyFrame:
    """Pairs of the last 30 events less than 24 h apart, aid != aid_y, once per session
    (the latest copy, as Benny's descending sort + drop_duplicates); `value` summed."""
    e = _last_n(events)
    return (
        cv.session_self_join(e)
        .filter((DT_S < DAY) & cv.aid_diff_filter)
        .pipe(cv.dedup_pairs, last=True)
        .group_by("aid", "aid_y")
        .agg(value.cast(pl.Float32).sum().alias("value"))
    )


def caco_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    """value: the type weight of the y event (clicks 1, carts 6, orders 3)."""
    weight = pl.col("type_y").replace_strict(TYPE_WEIGHT, default=float("nan"))
    return _tail_pairs(events, weight)


def clicks_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    """value: 1 + 3 * the x event's position in time from train start to test end."""
    span = cv.END_TEST_S - cv.START_TRAIN_S
    return _tail_pairs(events, 1 + 3 * (TS - cv.START_TRAIN_S) / span)


# --- 3. buy2buy (b2b) ---


def b2b_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    """Carts/orders: the last 30 of them, pairs within 14 days, aid != aid_y.
    value: the number of sessions."""
    e = _last_n(events.filter(cv.carts_orders_filter))
    return (
        cv.session_self_join(e)
        .filter((DT_S < 14 * DAY) & cv.aid_diff_filter)
        .unique(subset=["session", "aid", "aid_y"])
        .group_by("aid", "aid_y")
        .agg(pl.len().cast(pl.Float32).alias("value"))
    )


# --- 6. weighted clicks ---


def weighted_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    """Clicks within 1-hour sub-sessions; partial (sum, count) per Benny file chunk.

    Sub-sessions are cut where two events (of any type) are more than an hour apart;
    Benny numbered them session * 100 + k with k = 0 for the latest, and so does this
    (with the file chunk in the key, as Benny's per-file loop had it). Pairs: two
    different clicks, at different seconds, of different aids. Weight
    1 / ln(seconds apart + 2) * 1 / ln(positions apart + 2) * 1 / ln(clicks + 2),
    with positions counted from the latest click and `clicks` in the sub-session.
    Needs `_group` from fe_events(benny_files=True).
    """
    later = pl.col("ts").shift(-1).over("session")  # the next event in time
    gap = ((later - pl.col("ts")).dt.total_milliseconds() / 1000 > 3600).fill_null(
        False
    )
    key = ["_chunk", "sub"]
    e = (
        cv.with_row(events)
        .with_columns(_chunk=folds.benny_file_chunks(3), gap=gap.cast(pl.Int64))
        # separate steps: a window inside another window is quadratic
        .with_columns(new=pl.col("gap").cum_sum(reverse=True).over("session"))
        .with_columns(
            sub=(pl.col("session").cast(pl.Int64) * 100 + pl.col("new")).cast(pl.Int32)
        )
        .filter(pl.col("type") == CLICKS)
        .with_columns(
            n=pl.col(ROW).rank("ordinal", descending=True).over(key).cast(pl.Int64)
            - 1,  # signed: n - n_y
            clicks=pl.len().over(key),
        )
        .select(*key, "aid", "ts", "n", "clicks")
    )
    pairs = e.join(e.select(*key, "aid", "ts", "n"), on=key, suffix="_y")
    w = (
        _log_weight(DT_S)
        * _log_weight((pl.col("n") - pl.col("n_y")).abs())
        * _log_weight(pl.col("clicks"))
    )
    return (
        pairs.filter((pl.col("n") != pl.col("n_y")) & (TS != TS_Y) & cv.aid_diff_filter)
        .group_by("_chunk", "aid", "aid_y")
        .agg(w.cast(pl.Float32).sum().alias("sum"), pl.len().alias("count"))
    )


def weighted_reduce(pairs: pl.LazyFrame, aid_counts: pl.LazyFrame) -> pl.LazyFrame:
    """Keep pairs seen more than once per file chunk, sum over chunks, and normalise:
    value = sum / (sqrt(count_x + 1) * sqrt(count_y + 1)), with count_* the number
    of events of the aid. Keeps `count`, the number of pairs."""
    per_chunk = (
        pairs.group_by("_chunk", "aid", "aid_y")
        .agg(pl.col("sum").sum(), pl.col("count").sum())
        .filter(pl.col("count") > 1)
    )
    total = per_chunk.group_by("aid", "aid_y").agg(
        pl.col("sum").sum(), pl.col("count").sum()
    )
    return _normalise(total, aid_counts, aid_counts).select(
        "aid",
        "aid_y",
        value=pl.col("sum") / pl.col("norm_x").sqrt() / pl.col("norm_y").sqrt(),
        count=pl.col("count").cast(pl.Float32),
    )


def _normalise(
    pairs: pl.LazyFrame, x_counts: pl.LazyFrame, y_counts: pl.LazyFrame
) -> pl.LazyFrame:
    """Add norm_x, norm_y = count + 1 (count 0 for aids without one)."""
    norm = (pl.col("n").fill_null(0) + 1).cast(pl.Float64)
    return (
        pairs.join(x_counts, on="aid", how="left")
        .with_columns(norm_x=norm)
        .drop("n")
        .join(y_counts.rename({"aid": "aid_y"}), on="aid_y", how="left")
        .with_columns(norm_y=norm)
        .drop("n")
    )


# --- 7. weighted types: carts to orders ---


def wt_1_2_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    """A cart and an order of the same session (no aid filter), weight
    1 / ln(seconds apart + 2) * 1 / ln(positions apart + 2), with positions among all
    the session's events from the latest. Per (aid, aid_y): wgt_sum, and `count`, the
    number of sessions."""
    e = cv.with_row(events).with_columns(
        n=pl.col(ROW).rank("ordinal", descending=True).over("session").cast(pl.Int64)
        - 1
    )
    x = e.filter(pl.col("type") == CARTS).select("session", "aid", "ts", "n")
    y = e.filter(pl.col("type") == ORDERS).select("session", "aid", "ts", "n")
    w = _log_weight(DT_S) * _log_weight((pl.col("n") - pl.col("n_y")).abs())
    return (
        cv.session_self_join(x, y)
        .group_by("session", "aid", "aid_y")
        .agg(w.sum().alias("wgt_sum"))
        .group_by("aid", "aid_y")
        .agg(pl.col("wgt_sum").sum(), pl.len().alias("count"))
    )


def wt_1_2_reduce(
    pairs: pl.LazyFrame, cart_sessions: pl.LazyFrame, order_sessions: pl.LazyFrame
) -> pl.LazyFrame:
    """Keep pairs in more than one session; value = wgt_sum_2, normalised by the number
    of sessions with a cart of aid (count_x) and with an order of aid_y (count_y)."""
    total = (
        pairs.group_by("aid", "aid_y")
        .agg(pl.col("wgt_sum").sum(), pl.col("count").sum())
        .filter(pl.col("count") > 1)
    )
    sx, sy = pl.col("norm_x").sqrt(), pl.col("norm_y").sqrt()
    return _normalise(total, cart_sessions, order_sessions).select(
        "aid",
        "aid_y",
        value=pl.col("wgt_sum") / (sx * sy),  # Benny's wgt_sum_2
        wgt_sum=pl.col("wgt_sum"),
        wgt_prob=pl.col("count") / sx,
        wgt_prob_2=pl.col("count") / (sx * sy),
        count=pl.col("count").cast(pl.Float64),
    )


def _aid_counts(events: pl.LazyFrame) -> pl.LazyFrame:
    return events.group_by("aid").agg(n=pl.len())


def _sessions_with(events: pl.LazyFrame, type_: int) -> pl.LazyFrame:
    return (
        events.filter(pl.col("type") == type_)
        .unique(subset=["session", "aid"])
        .group_by("aid")
        .agg(n=pl.len())
    )


# --- candidates ---


def _int16(x: pl.Expr) -> pl.Expr:
    """cuDF's astype('int16'): truncate, then wrap around on overflow."""
    v = x.cast(pl.Int64)
    return ((v + 2**15) % 2**16 - 2**15).cast(pl.Int16)


def _matrix_top(matrix: pl.LazyFrame | pl.DataFrame, top: int) -> pl.LazyFrame:
    """Each aid's `top` cands by wgt (ties: smaller cand; Benny's were cuDF's order).

    Taken on the CPU, whatever engine runs the rest: the GPU engine has no
    group_by(...).head, and the matrices are small.
    """
    m = matrix.lazy().rename({"cand": "aid_y", "wgt": "value"})
    top_m = cv.top_covisit(m, top).rename({"aid_y": "cand", "value": "wgt"})
    return top_m.collect(engine="streaming").lazy()


@beartype
def basic_candidates(
    events: pl.LazyFrame | pl.DataFrame,
    matrix: pl.LazyFrame | pl.DataFrame,
    prefix: str,
    top: int,
) -> pl.LazyFrame:
    """Benny's get_candidates of c2o, caco, b2b and clicks.

    Each event's top `top` matrix rows, summed per (session, cand):
    <prefix>_num_<top> (events), _rank_min_<top> (the most recent event's rank, 1 for
    the latest), _wgt_<top>, _wgt_2_<top> (wgt / rank), _wgt_cl/_ca/_or_<top> (wgt of
    clicks, carts, orders). As in Benny's code, every column but _wgt_2 is cast to
    int16, so _wgt is truncated (and wraps around above 32767).
    """
    m = _matrix_top(matrix, top).select("aid", "cand", "wgt")
    wgt = pl.col("wgt").cast(pl.Float64)
    by_type = {"cl": CLICKS, "ca": CARTS, "or": ORDERS}
    lf = reverse_rank(events.lazy()).join(m, on="aid")
    s = f"_{top}"
    return (
        lf.group_by("session", "cand")
        .agg(
            _int16(pl.len()).alias(f"{prefix}_num{s}"),
            _int16(pl.col("rank").min()).alias(f"{prefix}_rank_min{s}"),
            _int16(wgt.sum()).alias(f"{prefix}_wgt{s}"),
            (wgt / pl.col("rank")).sum().cast(pl.Float32).alias(f"{prefix}_wgt_2{s}"),
            *[
                _int16((wgt * (pl.col("type") == t).cast(pl.Float64)).sum()).alias(
                    f"{prefix}_wgt_{n}{s}"
                )
                for n, t in by_type.items()
            ],
        )
        .sort("session", "cand")
    )


@beartype
def weighted_candidates(
    events: pl.LazyFrame | pl.DataFrame,
    matrix: pl.LazyFrame | pl.DataFrame,
    prefix: str = "weighted",
    top: int = 30,
) -> pl.LazyFrame:
    """Benny's get_candidates of weighted clicks: <prefix>_num_, _wgt_, _wgt_max_,
    _count_, _count_max_ (suffix _<top>) per (session, cand)."""
    m = _matrix_top(matrix, top).select("aid", "cand", "wgt", "count")
    s = f"_{top}"
    f32 = pl.Float32
    return (
        events.lazy()
        .join(m, on="aid")
        .group_by("session", "cand")
        .agg(
            pl.len().cast(pl.Int16).alias(f"{prefix}_num{s}"),
            pl.col("wgt").cast(pl.Float64).sum().cast(f32).alias(f"{prefix}_wgt{s}"),
            pl.col("wgt").max().cast(f32).alias(f"{prefix}_wgt_max{s}"),
            pl.col("count")
            .cast(pl.Float64)
            .sum()
            .cast(f32)
            .alias(f"{prefix}_count{s}"),
            pl.col("count").max().cast(f32).alias(f"{prefix}_count_max{s}"),
        )
        .sort("session", "cand")
    )


@beartype
def wt_candidates(
    events: pl.LazyFrame | pl.DataFrame,
    matrix: pl.LazyFrame | pl.DataFrame,
    prefix: str = "weighted_1_2",
    top: int = 20,
) -> pl.LazyFrame:
    """Benny's get_candidates of weighted types, from the session's carts only.

    Each cart gets score = 2 ** (0.1 + 0.9 * (carts - rank) / (carts - 1)) - 1 (1 for
    a single cart). Per (session, cand): <prefix>_dummy, _wgt_sum_2, _wgt_sum_2_2
    (times score), _wgt_prob, _wgt_prob_2, _count, _count_2. As in Benny's code,
    _wgt_prob_2 is wgt_prob times score: it overwrote the matrix's wgt_prob_2.
    """
    m = _matrix_top(matrix, top).select(
        "aid", "cand", wgt_sum_2="wgt", wgt_prob="wgt_prob", count="count"
    )
    n = pl.len().over("session")
    score = (2 ** (0.1 + 0.9 / (n - 1) * (n - pl.col("rank"))) - 1).fill_nan(1.0)
    lf = (
        reverse_rank(events.lazy().filter(pl.col("type") == CARTS))
        .with_columns(score=pl.when(n > 1).then(score).otherwise(1.0))
        .join(m, on="aid")
    )
    f = pl.Float64
    cols = {
        "dummy": pl.len().cast(pl.UInt32),
        "wgt_sum_2": pl.col("wgt_sum_2").cast(f).sum(),
        "wgt_sum_2_2": (pl.col("wgt_sum_2").cast(f) * pl.col("score")).sum(),
        "wgt_prob": pl.col("wgt_prob").cast(f).sum(),
        "wgt_prob_2": (pl.col("wgt_prob").cast(f) * pl.col("score")).sum(),
        "count": pl.col("count").cast(f).sum(),
        "count_2": (pl.col("count").cast(f) * pl.col("score")).sum(),
    }
    return (
        lf.group_by("session", "cand")
        .agg(
            e.cast(pl.Float32 if name != "dummy" else pl.UInt32).alias(
                f"{prefix}_{name}"
            )
            for name, e in cols.items()
        )
        .sort("session", "cand")
    )


# --- the matrices and sources ---


@dataclass(frozen=True)
class Matrix:
    pairs: Callable[[pl.LazyFrame], pl.LazyFrame]
    keep: int  # top per aid, by value (Benny's saved files kept 50 or all; only these are used)
    reduce: Callable[[pl.LazyFrame, pl.LazyFrame], pl.LazyFrame] | None = (
        None  # (pairs, events)
    )
    benny_files: bool = False  # needs Benny's file groups


MATRICES = {
    "c2o": Matrix(c2o_pairs, 15, lambda p, e: c2o_reduce(p)),
    "caco": Matrix(caco_pairs, 50),
    "b2b": Matrix(b2b_pairs, 50),
    "clicks": Matrix(clicks_pairs, 50),
    "weighted": Matrix(
        weighted_pairs,
        30,
        lambda p, e: weighted_reduce(p, _aid_counts(e)),
        benny_files=True,
    ),
    "wt_1_2": Matrix(
        wt_1_2_pairs,
        20,
        lambda p, e: wt_1_2_reduce(
            p, _sessions_with(e, CARTS), _sessions_with(e, ORDERS)
        ),
    ),
}

# source -> (matrix, candidate function, the column Benny's 01_Split ranks it by)
SOURCES = {
    "c2o_15": (
        "c2o",
        functools.partial(basic_candidates, prefix="c2o", top=15),
        "c2o_wgt_15",
    ),
    "caco_15": (
        "caco",
        functools.partial(basic_candidates, prefix="caco", top=15),
        "caco_wgt_15",
    ),
    "b2b_15": (
        "b2b",
        functools.partial(basic_candidates, prefix="b2b", top=15),
        "b2b_wgt_15",
    ),
    # Benny's `df['type'].isin(type_filter)` passed [['carts', 'orders'], 'caco'] and
    # matched nothing, so these candidates were empty; kept as he had them
    "caco_b2b_15": (
        "b2b",
        lambda events, matrix: basic_candidates(
            events.lazy().head(0), matrix, prefix="caco_b2b", top=15
        ),
        "caco_b2b_wgt_15",
    ),
    "clicks_20": (
        "clicks",
        functools.partial(basic_candidates, prefix="clicks", top=20),
        "clicks_wgt_20",
    ),
    "weighted_30": ("weighted", weighted_candidates, "weighted_wgt_30"),
    "weighted_1_2_20": ("wt_1_2", wt_candidates, "weighted_1_2_wgt_sum_2"),
}


@beartype
def build_matrix(
    events: pl.LazyFrame,
    name: str,
    engine="streaming",
    session_chunk: int | None = None,
    agg_engine="streaming",
) -> pl.DataFrame:
    """(aid, cand, wgt, extra columns): matrix `name`'s top `keep` per aid.

    `events` are folds.fe_events(k), with benny_files=True for `weighted`.
    """
    m = MATRICES[name]
    reduce = cv.sum_value if m.reduce is None else functools.partial(m.reduce, e=events)
    df = cv.build_covisits(
        events, m.pairs, session_chunk, engine, m.keep, agg_engine, reduce=reduce
    )
    extra = [c for c in df.columns if c not in ("aid", "aid_y", "value")]
    return df.select(
        pl.col("aid").cast(pl.Int32),
        pl.col("aid_y").cast(pl.Int32).alias("cand"),
        pl.col("value").cast(pl.Float32).alias("wgt"),
        *[pl.col(c).cast(pl.Float32) for c in extra],
    )


def matrix_path(k: int, name: str, root: Path) -> Path:
    return folds.fold_dir(k, root) / "covisits" / f"benny_{name}.parquet"


def main(args):
    root = folds.output_root(args.sample)
    engine = pl.GPUEngine(raise_on_fail=True) if args.engine == "gpu" else args.engine
    for k in args.folds:
        for name in args.matrices:
            path = matrix_path(k, name, root)
            if path.exists() and not args.force:
                print(f"fold {k}: {path.relative_to(root.parent)} exists")
            else:
                start = time.time()
                events = folds.fe_events(
                    k, args.sample, benny_files=MATRICES[name].benny_files
                )
                df = build_matrix(
                    events, name, engine, args.session_chunk, args.agg_engine
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(path)
                print(
                    f"fold {k}: wrote {path.relative_to(root.parent)}: {df.height:,} pairs ({time.time() - start:.0f} s)"
                )
            matrix = pl.read_parquet(path)
            for source, (matrix_name, candidates, score) in SOURCES.items():
                if matrix_name != name:
                    continue
                for dataset in folds.DATASETS:
                    start = time.time()
                    events = folds.candidate_events(k, dataset, args.sample)
                    cands = candidates(events, matrix).collect(engine=engine)
                    out = folds.write_candidates(cands, k, dataset, source, root=root)
                    labels = folds.sample_sessions(
                        folds.labels(k, dataset), args.sample
                    )
                    r = metrics.recall(cands, labels, score) if cands.height else {}
                    # the weights are positive, so a negative int16 sum wrapped around
                    wrapped = ""
                    if cands.schema[score] == pl.Int16:
                        wrapped = f"; {cands.filter(pl.col(score) < 0).height:,} wrapped int16 sums"
                    print(
                        f"  {dataset}: wrote {out.relative_to(root.parent)}: {cands.height:,} rows "
                        f"({time.time() - start:.0f} s); recall@20 by {score}: {metrics.format_recall(r)}"
                        f"{wrapped}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(folds.N_FOLDS))
    )
    parser.add_argument(
        "--matrices", nargs="+", default=list(MATRICES), choices=list(MATRICES)
    )
    parser.add_argument(
        "--sample", type=float, default=1.0, help="fraction of sessions"
    )
    parser.add_argument(
        "--engine", default="streaming", help="streaming, in-memory or gpu"
    )
    parser.add_argument(
        "--agg-engine", default="streaming", help="engine to combine chunks"
    )
    parser.add_argument(
        "--session-chunk",
        type=int,
        default=None,
        help="sessions per chunk (default: one)",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild existing matrices"
    )
    main(parser.parse_args())
