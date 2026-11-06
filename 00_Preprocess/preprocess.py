"""Prepare Benny's inputs from the Polars data of ../Kaggle-OTTO-Comp.

Replaces 01_Preprocess.ipynb (jsonl -> parquet) and 02_Preprocess.ipynb (session splits).
See README.md in this directory for the file mapping and the week-4 scheme.

Run from the repo root:
    uv run python 00_Preprocess/preprocess.py --check
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from beartype import beartype

import paths
from event_types import CARTS, CLICKS, ORDERS

Frame = pl.LazyFrame | pl.DataFrame

DEFAULT_SRC = paths.ROOT.parent / "Kaggle-OTTO-Comp" / "data"

# link in this repo -> file in the source directory
INPUT_LINKS = {
    paths.TRAIN: "train.parquet",  # all 4 weeks
    paths.TEST: "test.parquet",  # Kaggle test week
    paths.CV_TRAIN: "valid_train.parquet",  # weeks 1-3
    paths.CV_TEST: "valid_test.parquet",  # week 4, truncated
    paths.TEST_FULL: "test_full.parquet",  # Kaggle test week, untruncated
}
LABELS_SOURCE = "valid_test_labels.parquet"


@beartype
def link_inputs(src: Path, links: dict[Path, str] = INPUT_LINKS) -> None:
    """Symlink the event files from `src`, pointing at their resolved targets."""
    for link, name in links.items():
        target = (src / name).resolve(strict=True)
        if link.is_symlink():
            if link.resolve() == target:
                print(f"ok      {link.relative_to(paths.ROOT)} -> {target}")
                continue
            link.unlink()
        elif link.exists():
            raise FileExistsError(f"{link} is a regular file; not replacing it")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        print(f"linked  {link.relative_to(paths.ROOT)} -> {target}")


@beartype
def labels_long(labels: Frame, engine="streaming") -> pl.DataFrame:
    """(session, type, ground_truth: List) -> one row per (session, aid, type).

    This is Benny's test_labels / xgb_train_y layout, with `type` as the
    UInt8 codes of event_types instead of strings.
    """
    return (
        labels.lazy()
        # empty_as_null=False would skip empty lists directly, but the GPU engine rejects it
        .explode("ground_truth", empty_as_null=True)
        .select("session", pl.col("ground_truth").alias("aid"), "type")
        .drop_nulls("aid")
        .sort("session", "type", "aid")
        .collect(engine=engine)
    )


@beartype
def labels_after_cut(full: Frame, truncated: Frame, engine="streaming") -> pl.DataFrame:
    """Labels of each `truncated` session from the events of `full` after the cut.

    Same rules as the host script (and so valid_test_labels): clicks is the first
    click after the cut, carts and orders are the sets of aids carted / ordered after
    it. Truncated sessions must be prefixes of the full ones (see
    check_truncation_is_prefix), both in event order within a session. Returns
    Benny's long format, like labels_long.
    """
    cut = truncated.lazy().group_by("session").agg(n=pl.len())
    # number events before joining: a join doesn't keep row order on the streaming engine
    after = (
        full.lazy()
        .with_columns(i=pl.int_range(pl.len()).over("session"))
        .join(cut, on="session")  # only the truncated sessions
        .filter(pl.col("i") >= pl.col("n"))
    )
    clicks = (
        after.filter(pl.col("type") == CLICKS)
        .group_by("session")
        .agg(pl.col("aid").sort_by("i").first(), pl.lit(CLICKS, pl.UInt8).alias("type"))
    )
    buys = after.filter(pl.col("type").is_in([CARTS, ORDERS])).select("session", "aid", "type").unique()
    return (
        pl.concat([clicks.select("session", "aid", "type"), buys])
        .sort("session", "type", "aid")
        .collect(engine=engine)
    )


def scan_test_full(path: Path = paths.TEST_FULL) -> pl.LazyFrame:
    """The Kaggle test_full in the dtypes of the other files.

    It is stored with Int64 session/aid and millisecond ts, while test.parquet has
    whole seconds; ts is truncated to seconds so the two match event for event.
    """
    return pl.scan_parquet(path).with_columns(
        pl.col("session", "aid").cast(pl.Int32), pl.col("ts").dt.truncate("1s")
    )


@beartype
def partition_sessions(sessions: pl.Series, n_parts: int, seed: int) -> pl.DataFrame:
    """Randomly split unique sessions into `n_parts` parts of (almost) equal size.

    Benny's `partition`: shuffle, then take every n-th session. Sessions are sorted
    before shuffling, so the result depends only on the set of sessions and the seed.
    """
    unique = sessions.unique().sort()
    shuffled = unique.gather(np.random.default_rng(seed).permutation(len(unique)))
    return (
        pl.DataFrame({"session": shuffled})
        .with_columns(part=(pl.int_range(pl.len()) % n_parts).cast(pl.UInt8))
        .sort("session")
    )


@beartype
def check_truncation_is_prefix(full: Frame, truncated: Frame, engine="streaming") -> pl.DataFrame:
    """Rows of `truncated` that don't match the same position in the same session of `full`.

    Benny's folds replace a session of the full train by its truncated version, which
    only makes sense if each truncated session is a prefix of the full one. Both inputs
    must be in event order within each session. Returns the mismatching rows (empty
    when the check passes).
    """
    position = pl.int_range(pl.len()).over("session").alias("i")
    truncated = truncated.lazy().with_columns(position)
    # number events before joining: a join doesn't keep row order on the streaming engine
    full = (
        full.lazy()
        .with_columns(position)
        .join(truncated.select("session").unique(), on="session", how="semi")
    )
    same = [pl.col(c).eq_missing(pl.col(f"{c}_full")) for c in ("aid", "ts", "type")]
    return (
        truncated.join(full, on=["session", "i"], how="left", suffix="_full")
        .filter(~pl.all_horizontal(same))
        .collect(engine=engine)
    )


def _write_split(df: pl.DataFrame, path: Path, force: bool) -> None:
    # downstream fold files depend on the split, so don't silently replace it
    if path.exists() and not force:
        print(f"kept    {path.relative_to(paths.ROOT)} (exists; --force to overwrite)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    sizes = df.group_by("part").len()["len"]
    print(
        f"wrote   {path.relative_to(paths.ROOT)}: {df.height:,} sessions in "
        f"{sizes.len()} parts of {sizes.min():,}-{sizes.max():,}"
    )


def main(src: Path, seed: int, n_eval_parts: int, n_sub_parts: int, check: bool, force: bool, engine: str):
    if n_eval_parts % 2:
        raise ValueError("a fold is two eval parts, so --n-eval-parts must be even")

    link_inputs(src)

    raw_labels = pl.scan_parquet(src / LABELS_SOURCE)
    labels = labels_long(raw_labels, engine=engine)
    expected = raw_labels.select(pl.col("ground_truth").list.len().sum()).collect().item()
    assert labels.height == expected, (labels.height, expected)
    labels.write_parquet(paths.CV_TEST_LABELS)
    print(f"wrote   {paths.CV_TEST_LABELS.relative_to(paths.ROOT)}: {labels.height:,} rows")
    print(labels.group_by("type").len().sort("type"))

    # Benny holds out only sessions with labels
    sessions_eval = partition_sessions(labels["session"], n_eval_parts, seed).with_columns(
        fold=pl.col("part") // 2
    )
    _write_split(sessions_eval, paths.SESSIONS_EVAL, force)

    test_labels = labels_after_cut(scan_test_full(), pl.scan_parquet(paths.TEST), engine=engine)
    test_labels.write_parquet(paths.TEST_LABELS)
    print(f"wrote   {paths.TEST_LABELS.relative_to(paths.ROOT)}: {test_labels.height:,} rows")
    print(test_labels.group_by("type").len().sort("type"))

    test_sessions = pl.scan_parquet(paths.TEST).select("session").unique().collect(engine=engine)
    _write_split(partition_sessions(test_sessions["session"], n_sub_parts, seed), paths.SUB_SESSIONS, force)

    if check:
        _check(engine)


def _check(engine):
    failed = False
    for name, full, truncated in [
        ("week 4", pl.scan_parquet(paths.TRAIN), pl.scan_parquet(paths.CV_TEST)),
        ("Kaggle test", scan_test_full(), pl.scan_parquet(paths.TEST)),
    ]:
        bad = check_truncation_is_prefix(full, truncated, engine=engine)
        print(f"prefix check, {name}: {bad.height} mismatches")
        if bad.height:
            print(bad.head())
            failed = True

    # labels_after_cut must reproduce the host script's week-4 labels
    rebuilt = labels_after_cut(pl.scan_parquet(paths.TRAIN), pl.scan_parquet(paths.CV_TEST), engine=engine)
    host = pl.read_parquet(paths.CV_TEST_LABELS)
    diff = rebuilt.join(host, on=["session", "aid", "type"], how="anti").height
    diff += host.join(rebuilt, on=["session", "aid", "type"], how="anti").height
    print(f"labels check, week 4 rebuilt vs valid_test_labels: {diff} differing rows")
    failed |= diff > 0
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="directory with Chris's converted parquets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval-parts", type=int, default=10, help="2 parts per fold")
    parser.add_argument("--n-sub-parts", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="check the truncations and the label rules")
    parser.add_argument("--force", action="store_true", help="overwrite existing session splits")
    parser.add_argument("--engine", default="streaming", help="streaming, in-memory or gpu")
    args = parser.parse_args()
    main(args.src, args.seed, args.n_eval_parts, args.n_sub_parts, args.check, args.force, args.engine)
