"""Chris's covisitation matrices, built per fold and applied the way Benny did.

Replaces Benny's 03_CoMatrix-99-Chris-gpu-<VER>.ipynb (one per matrix) and
04_CoMatrix-99-Chris-apply.ipynb. The pairs come from Chris's part
(../Kaggle-OTTO-Comp/train/covisit_matrices/covisits.py, a path dependency): Benny's
notebooks copy Chris's pair logic line for line. Only the data differs: each fold's
matrices are built from folds.fe_events(k), and the top kept per aid is Benny's.

Per fold k, it writes
    data_folds/fold_<k>/covisits/chris_<VER>.parquet      (aid, cand, wgt)
    data_folds/fold_<k>/candidates/{train,sub}/chris__<i>__15/cand.parquet
where i is the matrix's position in Benny's sorted file names (115 is 0, ..., 93 is
9), and the candidate columns are chris__<i>__{wgt, dummy, wgt_1, ...} (see
`apply_matrix`). These are features for the rankers' merge; Benny's rankers don't
take candidates from them.

Run from the repo root (a sample first: outputs then go to data_folds/sample_<f>):
    uv run python 01c_FE_CoVisit_2_Chris/chris_covisits.py --sample 0.02 --folds 0
"""

import argparse
import time
from pathlib import Path

import covisits
import polars as pl
import save_covisits
from beartype import beartype

import folds
import metrics
from session_ops import recency_weights

# Benny's matrices in the order of his sorted file names top_40_aids_v<VER>.parquet,
# which is how the apply step numbered them (chris__0 = 115, ..., chris__9 = 93)
ORDER = [115, 217, 220, 226, 232, 235, 239, 700, 701, 93]
# Sessions per chunk on the GPU: the sibling's A100 defaults. Its all-pairs self-joins
# (115, 217-235) exceed cudf's 2^31-row column limit unchunked on the full data, and 93
# runs out of GPU memory. fe_events(k) has about as many sessions as its train + test.
SESSION_CHUNK = {int(m.ver): m.session_chunk for m in save_covisits.MATRICES}
# the top kept per aid. Benny's 700 notebook ends with `n < 0`, so his 700 matrix is
# empty and so are its features; kept as he had it.
TOP = {
    93: 80,
    115: 40,
    217: 40,
    220: 80,
    226: 80,
    232: 80,
    235: 80,
    239: 20,
    700: 0,
    701: 40,
}
GAP_S = 2 * 60 * 60  # the apply step's segments: cut at gaps of more than 2 hours
MATRIX_SCHEMA = {"aid": pl.Int32, "cand": pl.Int32, "wgt": pl.Float32}
COLUMNS = [
    "wgt", "dummy", "wgt_1", "wgt_2", "t_dummy", "t_wgt_1", "t_wgt_2",
    "d_wgt_1", "d_wgt_2", "d_dummy", "l_dummy", "l_wgt_1", "l_wgt_2",
]  # fmt: skip


def source_name(ver: int) -> str:
    """Benny's candidate directory for a matrix, e.g. chris__9__15 for 93."""
    return f"chris__{ORDER.index(ver)}__15"


def matrix_path(k: int, ver: int, root: Path) -> Path:
    return folds.fold_dir(k, root) / "covisits" / f"chris_{ver}.parquet"


@beartype
def build_matrix(
    events: pl.LazyFrame,
    ver: int,
    engine="gpu",
    session_chunk: int | None = None,
    agg_engine="streaming",
) -> pl.DataFrame:
    """(aid, cand, wgt): Chris's matrix `ver` with Benny's top per aid.

    `events` are folds.fe_events(k); for 239 with benny_files=True. Benny summed the
    per-file results and kept the top 100-150 per aid after every 50 files (every 5
    for 115), in `glob` order; that order can't be reproduced, so here the sums are
    exact and only the final top is kept.
    """
    if TOP[ver] == 0:
        return pl.DataFrame(schema=MATRIX_SCHEMA)
    pairs = getattr(covisits, f"covisits_{ver}_pairs")
    # 239 keeps one pair per Benny file, then sums over files
    reduce = covisits._reduce_239 if ver == 239 else covisits.sum_value
    df = covisits.build_covisits(
        events, pairs, session_chunk, engine, TOP[ver], agg_engine, reduce=reduce
    )
    return df.select(
        pl.col("aid").cast(pl.Int32),
        cand=pl.col("aid_y").cast(pl.Int32),
        wgt=pl.col("value").cast(pl.Float32),
    )


@beartype
def apply_matrix(
    events: pl.LazyFrame | pl.DataFrame,
    matrix: pl.LazyFrame | pl.DataFrame,
    prefix: str,
) -> pl.LazyFrame:
    """Benny's get_candidates: the matrix's rows of each session's items, summed per
    (session, cand), with columns <prefix>_<name> for name in COLUMNS.

    Per (event, matrix row) the values are the matrix weight `wgt`, `dummy` = 1,
    `wgt_1`/`wgt_2` = wgt times the event's recency weight (session_ops), then:
        t_*  times the event's type code (clicks 0, carts 1, orders 2)
        d_*  only events in the session's first segment (segments are cut at gaps
             of more than 2 hours, counted from the session's start)
        l_*  only events at the last second of their segment
    """
    ts = pl.col("ts").dt.epoch("s")
    lf = (
        recency_weights(events.lazy())
        .with_columns(ts=ts)
        .sort("session", "ts", maintain_order=True)
        # separate steps: a window inside another window is quadratic
        .with_columns(
            gap=(pl.col("ts").diff().over("session") > GAP_S).fill_null(False)
        )
        .with_columns(d=pl.col("gap").cast(pl.Int64).cum_sum().over("session"))
        .with_columns(last=pl.col("ts") == pl.col("ts").max().over("session", "d"))
        .join(matrix.lazy().select("aid", "cand", "wgt"), on="aid")
    )
    wgt = pl.col("wgt").cast(pl.Float64)
    w1, w2 = pl.col("wgt_1") * wgt, pl.col("wgt_2") * wgt
    t = pl.col("type").cast(pl.Float64)
    first, last = pl.col("d") == 0, pl.col("last")
    values = {
        "wgt": wgt,
        # one per row (a literal 1 would be summed once per group, not per row)
        "dummy": pl.col("cand").is_not_null().cast(pl.UInt32),
        "wgt_1": w1,
        "wgt_2": w2,
        "t_dummy": pl.col("type").cast(pl.UInt32),
        "t_wgt_1": w1 * t,
        "t_wgt_2": w2 * t,
        "d_wgt_1": w1 * first.cast(pl.Float64),
        "d_wgt_2": w2 * first.cast(pl.Float64),
        "d_dummy": first.cast(pl.UInt32),
        "l_dummy": last.cast(pl.UInt32),
        "l_wgt_1": w1 * last.cast(pl.Float64),
        "l_wgt_2": w2 * last.cast(pl.Float64),
    }
    assert list(values) == COLUMNS
    aggs = [
        e.sum()
        .cast(pl.UInt32 if name.endswith("dummy") else pl.Float32)
        .alias(f"{prefix}_{name}")
        for name, e in values.items()
    ]
    return lf.group_by("session", "cand").agg(aggs).sort("session", "cand")


def main(args):
    root = folds.output_root(args.sample)
    engine = pl.GPUEngine(raise_on_fail=True) if args.engine == "gpu" else args.engine
    for k in args.folds:
        for ver in args.matrices:
            path = matrix_path(k, ver, root)
            if path.exists() and not args.force:
                print(f"fold {k}: {path.relative_to(root.parent)} exists")
            else:
                start = time.time()
                events = folds.fe_events(k, args.sample, benny_files=ver == 239)
                chunk = args.session_chunk or (
                    SESSION_CHUNK.get(ver) if args.sample >= 1.0 else None
                )
                df = build_matrix(events, ver, engine, chunk, args.agg_engine)
                path.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(path)
                print(
                    f"fold {k}: wrote {path.relative_to(root.parent)}: {df.height:,} pairs "
                    f"({time.time() - start:.0f} s)"
                )
            matrix = pl.read_parquet(path)
            source = source_name(ver)
            prefix = source.removesuffix("_15")  # chris__<i>_ + _<name>
            for dataset in folds.DATASETS:
                start = time.time()
                events = folds.candidate_events(k, dataset, args.sample)
                cands = apply_matrix(events, matrix, prefix).collect(engine=engine)
                out = folds.write_candidates(cands, k, dataset, source, root=root)
                labels = folds.sample_sessions(folds.labels(k, dataset), args.sample)
                r = (
                    metrics.recall(cands, labels, f"{prefix}_wgt")
                    if cands.height
                    else {}
                )
                print(
                    f"  {dataset}: wrote {out.relative_to(root.parent)}: {cands.height:,} rows "
                    f"({time.time() - start:.0f} s); recall@20 by {prefix}_wgt: "
                    f"{metrics.format_recall(r)}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(folds.N_FOLDS))
    )
    parser.add_argument("--matrices", type=int, nargs="+", default=sorted(TOP))
    parser.add_argument(
        "--sample", type=float, default=1.0, help="fraction of sessions"
    )
    parser.add_argument("--engine", default="gpu", help="gpu, streaming or in-memory")
    parser.add_argument(
        "--agg-engine", default="streaming", help="engine to combine chunks"
    )
    parser.add_argument(
        "--session-chunk",
        type=int,
        default=None,
        help="sessions per chunk (default: SESSION_CHUNK on the full data, one on a sample)",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild existing matrices"
    )
    main(parser.parse_args())
