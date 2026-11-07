"""Fold inputs and the candidate-file layout, shared by every per-fold step.

Benny's week-4 scheme (see 00_Preprocess/README.md): fold k holds out the week-4
sessions with `fold == k` in SESSIONS_EVAL. For those sessions the features only see
their truncated events (XGB_TRAIN_X); every other session is seen in full (TRAIN),
together with the Kaggle test sessions (TEST). Candidates are generated for two
datasets:
    train  fold k's held-out sessions, truncated (labels in XGB_TRAIN_Y)
    sub    the Kaggle test sessions (labels in TEST_LABELS, which Benny didn't have)

Candidate files live at
    DATA_FOLDS/fold_<k>/candidates/<dataset>/<source>/cand.parquet
with the columns (session Int32, cand Int32, <source features>...), one row per
(session, cand). Sources that don't depend on the fold (t4r, previous visits) are
written once to DATA_FOLDS/shared/candidates/<dataset>/<source>/cand.parquet instead
of being copied to every fold as Benny did; `candidate_paths` lists both. A shared
`train` file covers all week-4 sessions, so readers filter it to the fold.

Run from the repo root to check the folds on the real data:
    uv run python folds.py --check
"""

import argparse
import datetime
from pathlib import Path

import polars as pl
from beartype import beartype

import paths

N_FOLDS = 5
DATASETS = ("train", "sub")
SHARED = "shared"
KEY = ("session", "cand")

Frame = Path | pl.LazyFrame | pl.DataFrame


def _scan(frame: Frame) -> pl.LazyFrame:
    return pl.scan_parquet(frame) if isinstance(frame, Path) else frame.lazy()


@beartype
def sample_sessions(frame: Frame, fraction: float = 1.0, seed: int = 0) -> pl.LazyFrame:
    """Keep about `fraction` of the sessions, the same ones in every file for a given seed.

    For developing on the local GPU. A hash of the session id decides, so a session is
    either kept with all its events or dropped, without collecting the session list.
    The hash is integer arithmetic (Knuth's multiplicative hash) rather than
    `Expr.hash`, which gives different values on the GPU engine.
    """
    lf = _scan(frame)
    if fraction >= 1.0:
        return lf
    h = (pl.col("session").cast(pl.UInt64) + seed) * 2654435761 % 2**32
    return lf.filter(h < int(fraction * 2**32))


@beartype
def heldout_sessions(
    k: int, sessions_eval: Frame = paths.SESSIONS_EVAL
) -> pl.LazyFrame:
    """(session, part): fold k's held-out week-4 sessions, in Benny's 2 parts per fold."""
    if not 0 <= k < N_FOLDS:
        raise ValueError(f"fold {k} is not in 0..{N_FOLDS - 1}")
    return _scan(sessions_eval).filter(pl.col("fold") == k).select("session", "part")


@beartype
def fe_events(
    k: int,
    sample: float = 1.0,
    *,
    train: Frame = paths.TRAIN,
    truncated: Frame = paths.XGB_TRAIN_X,
    test: Frame = paths.TEST,
    sessions_eval: Frame = paths.SESSIONS_EVAL,
) -> pl.LazyFrame:
    """Events for fold k's features: (session, aid, ts, type).

    TRAIN without fold k's held-out sessions, those sessions truncated, and TEST. This
    is the `files_split` / `isin(sess_eval)` block at the top of Benny's 01* notebooks.
    Within a session the events keep their file order.
    """
    held = heldout_sessions(k, sessions_eval).select("session")
    parts = [
        _scan(train).join(held, on="session", how="anti"),
        _scan(truncated).join(held, on="session", how="semi"),
        _scan(test),
    ]
    return sample_sessions(pl.concat(parts), sample)


@beartype
def candidate_events(
    k: int | None,
    dataset: str,
    sample: float = 1.0,
    *,
    truncated: Frame = paths.XGB_TRAIN_X,
    test: Frame = paths.TEST,
    sessions_eval: Frame = paths.SESSIONS_EVAL,
) -> pl.LazyFrame:
    """Events of the sessions to generate candidates for.

    train: fold k's held-out sessions, truncated; with k None, all week-4 sessions
    (for shared sources). sub: the Kaggle test sessions, whatever k is.
    """
    _check_dataset(dataset)
    if dataset == "sub":
        events = _scan(test)
    elif k is None:
        events = _scan(truncated)
    else:
        events = _scan(truncated).join(
            heldout_sessions(k, sessions_eval).select("session"),
            on="session",
            how="semi",
        )
    return sample_sessions(events, sample)


@beartype
def labels(
    k: int | None,
    dataset: str = "train",
    *,
    week4_labels: Frame = paths.XGB_TRAIN_Y,
    test_labels: Frame = paths.TEST_LABELS,
    sessions_eval: Frame = paths.SESSIONS_EVAL,
) -> pl.LazyFrame:
    """(session, aid, type) labels of `candidate_events(k, dataset)`'s sessions."""
    _check_dataset(dataset)
    if dataset == "sub":
        return _scan(test_labels)
    lf = _scan(week4_labels)
    if k is None:
        return lf
    return lf.join(
        heldout_sessions(k, sessions_eval).select("session"), on="session", how="semi"
    )


# the day each dataset's week of candidates starts on (UTC; the weeks start at 22:00
# the day before, which Benny counted as day 0 too)
WEEK_START = {"train": datetime.date(2022, 8, 22), "sub": datetime.date(2022, 8, 29)}


@beartype
def day_index(dataset: str, ts: pl.Expr | None = None) -> pl.Expr:
    """Benny's `day_` of a timestamp in the dataset's week: 0 for the first two calendar
    days (the week starts at 22:00 UTC), then 1 to 6.

    Benny mapped day of month to day_ with a table (train 21, 22 -> 0, 23 -> 1, ...,
    28 -> 6; sub 28, 29 -> 0, ..., Sep 4 -> 6). Here it's integer arithmetic, which
    the GPU engine runs; clipped below at 0.
    """
    _check_dataset(dataset)
    ts = pl.col("ts") if ts is None else ts
    first = (WEEK_START[dataset] - datetime.date(1970, 1, 1)).days
    return (ts.dt.epoch("s") // 86_400 - first).clip(lower_bound=0).cast(pl.Int8)


def _check_dataset(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(f"dataset {dataset!r} is not one of {DATASETS}")


@beartype
def output_root(sample: float = 1.0) -> Path:
    """Where a step's outputs go: DATA_FOLDS, or DATA_FOLDS/sample_<fraction> on a sample.

    So a run on a sample never overwrites the full outputs, and the steps after it
    read the outputs of the same sample.
    """
    return (
        paths.DATA_FOLDS if sample >= 1.0 else paths.DATA_FOLDS / f"sample_{sample:g}"
    )


@beartype
def fold_dir(k: int | None, root: Path = paths.DATA_FOLDS) -> Path:
    """DATA_FOLDS/fold_<k>, or DATA_FOLDS/shared for k None."""
    return root / (SHARED if k is None else f"fold_{k}")


@beartype
def feature_path(k: int | None, name: str, root: Path = paths.DATA_FOLDS) -> Path:
    """A feature file (not candidates), e.g. Benny's train_sessions.parquet.

    k None: shared by all folds.
    """
    return fold_dir(k, root) / f"{name}.parquet"


@beartype
def candidate_path(
    k: int | None, dataset: str, source: str, root: Path = paths.DATA_FOLDS
) -> Path:
    _check_dataset(dataset)
    return fold_dir(k, root) / "candidates" / dataset / source / "cand.parquet"


@beartype
def candidate_paths(
    k: int, dataset: str, root: Path = paths.DATA_FOLDS
) -> dict[str, Path]:
    """{source: path} of fold k's candidate files plus the shared ones, sorted by source.

    A source in both places is an error rather than silently picking one.
    """
    found = {}
    for kk in (k, None):
        for p in sorted(
            candidate_path(kk, dataset, "*", root).parent.parent.glob("*/cand.parquet")
        ):
            source = p.parent.name
            if source in found:
                raise ValueError(
                    f"source {source!r} is in both {found[source]} and {p}"
                )
            found[source] = p
    return dict(sorted(found.items()))


@beartype
def check_candidates(frame: Frame) -> None:
    """Raise if `frame` breaks the candidate-file contract."""
    lf = _scan(frame)
    schema = lf.collect_schema()
    names = schema.names()
    if names[:2] != list(KEY):
        raise ValueError(f"candidate columns must start with {KEY}, got {names[:2]}")
    for c in KEY:
        if schema[c] != pl.Int32:
            raise ValueError(f"{c} must be Int32, got {schema[c]}")
    dupes = lf.group_by(KEY).len().filter(pl.col("len") > 1).head(3).collect()
    if dupes.height:
        raise ValueError(f"duplicate (session, cand) rows, e.g. {dupes.rows()}")
    nulls = lf.select(pl.col(KEY).null_count()).collect().row(0)
    if any(nulls):
        raise ValueError(f"null session or cand: {nulls}")


@beartype
def write_candidates(
    frame: pl.LazyFrame | pl.DataFrame,
    k: int | None,
    dataset: str,
    source: str,
    root: Path = paths.DATA_FOLDS,
) -> Path:
    """Write a candidate file (k None: shared by all folds), check it, and return its path.

    A LazyFrame is streamed to disk. If the check fails the file is removed.
    """
    path = candidate_path(k, dataset, source, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, pl.LazyFrame):
        frame.sink_parquet(path)
    else:
        frame.write_parquet(path)
    try:
        check_candidates(path)
    except ValueError:
        path.unlink()
        raise
    return path


@beartype
def read_candidates(
    k: int,
    dataset: str,
    source: str,
    root: Path = paths.DATA_FOLDS,
    *,
    sessions_eval: Frame = paths.SESSIONS_EVAL,
) -> pl.LazyFrame:
    """A source's candidates for fold k: its fold file, or its shared file filtered to the fold."""
    path = candidate_paths(k, dataset, root)[source]
    lf = pl.scan_parquet(path)
    if dataset == "train" and path.is_relative_to(fold_dir(None, root)):
        lf = lf.join(
            heldout_sessions(k, sessions_eval).select("session"),
            on="session",
            how="semi",
        )
    return lf


def _check(engine):
    """Every week-4 session is truncated in exactly one fold and complete in the others."""
    count = pl.len()
    n = {
        name: pl.scan_parquet(p).select(count).collect().item()
        for name, p in [
            ("train", paths.TRAIN),
            ("truncated", paths.XGB_TRAIN_X),
            ("test", paths.TEST),
        ]
    }
    week4 = (
        pl.scan_parquet(paths.XGB_TRAIN_X)
        .select(pl.col("session").n_unique())
        .collect()
        .item()
    )
    held = (
        pl.scan_parquet(paths.SESSIONS_EVAL)
        .select(pl.len(), pl.col("session").n_unique())
        .collect()
        .row(0)
    )
    assert held == (week4, week4), (
        f"SESSIONS_EVAL has {held} rows/sessions, week 4 has {week4} sessions"
    )
    total_held = 0
    for k in range(N_FOLDS):
        held_k = heldout_sessions(k).select("session")
        full_k = (
            pl.scan_parquet(paths.TRAIN)
            .join(held_k, on="session", how="semi")
            .select(count)
            .collect()
            .item()
        )
        trunc_k = (
            pl.scan_parquet(paths.XGB_TRAIN_X)
            .join(held_k, on="session", how="semi")
            .select(count)
            .collect()
            .item()
        )
        got = fe_events(k).select(count).collect(engine=engine).item()
        want = n["train"] - full_k + trunc_k + n["test"]
        assert got == want, f"fold {k}: {got:,} events, expected {want:,}"
        total_held += trunc_k
        print(
            f"fold {k}: {got:,} events ({full_k:,} held-out events replaced by {trunc_k:,} truncated ones)"
        )
    assert total_held == n["truncated"], "the folds don't cover the truncated week 4"
    print("folds check: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="check the folds on the real data"
    )
    parser.add_argument(
        "--engine", default="streaming", help="streaming, in-memory or gpu"
    )
    args = parser.parse_args()
    if args.check:
        _check(
            pl.GPUEngine(raise_on_fail=True) if args.engine == "gpu" else args.engine
        )
