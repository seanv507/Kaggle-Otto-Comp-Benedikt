"""Candidates and features per fold, for one target (Benny's 01_Split.ipynb).

For each fold k and dataset (train: fold k's held-out sessions; sub: the Kaggle test
sessions) it builds one row per (session, cand):
- candidates: the union of each generating source's top NO_CANDS per session, ranked
  by the source's column in benny_columns.CLICKS_GENERATORS;
- the columns of every candidate source of the fold (folds.candidate_paths), except
  benny_columns.TOTAL_IGNORE; 0 where a source has no row;
- the item activity (item_features_*, 0 where missing), the session features
  (session_features_*) and the item's activity on the session's day
  (item_day_features_* on folds.day_index of the session's last event, 0 where
  missing);
- for train, `target`: 1 if the cand is one of the session's labels of the target.
- the fold's Word2Vec similarity features (ranking.emb_features), unless with_emb
  is False.
Floats are Float32. The rows are written in chunks, as Benny did:
    data_folds/fold_<k>/ranker/<target>/train_<i>.parquet   i = Benny's 2 parts per fold
    data_folds/fold_<k>/ranker/<target>/sub_<j>.parquet     j = SUB_SESSIONS part // 10
"""

import time
from pathlib import Path

import polars as pl
from beartype import beartype

import folds
import metrics
import paths
from event_types import TYPE_NAMES

from .benny_columns import CLICKS_GENERATORS, NO_CANDS, TOTAL_IGNORE
from .emb_features import emb_features

KEY = ["session", "cand"]
SUB_CHUNK_PARTS = 10  # SUB_SESSIONS parts per sub chunk (Benny's sub_sessoins_10)
TARGET_CODES = {name: code for code, name in TYPE_NAMES.items()}


def chunk_path(
    k: int, target: str, dataset: str, i: int, root: Path = paths.DATA_FOLDS
) -> Path:
    return folds.fold_dir(k, root) / "ranker" / target / f"{dataset}_{i}.parquet"


@beartype
def candidates(
    sources: dict[str, pl.LazyFrame],
    generators: dict[str, str] = CLICKS_GENERATORS,
    n: int = NO_CANDS,
) -> pl.LazyFrame:
    """(session, cand): each generating source's top n per session, deduplicated.

    Ties at the cut go to the smaller cand (Benny's were cuDF's sort order).
    """
    tops = [
        metrics.top_k(sources[s], column, n).rename({"aid": "cand"})
        for s, column in generators.items()
        if s in sources
    ]
    return pl.concat(tops).unique(KEY)


@beartype
def add_sources(cands: pl.LazyFrame, sources: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Left-join every source's columns, except TOTAL_IGNORE; 0 where it has no row."""
    seen = set(KEY)
    for name, lf in sources.items():
        cols = [
            c
            for c in lf.collect_schema().names()
            if c not in KEY and c not in TOTAL_IGNORE
        ]
        clash = seen.intersection(cols)
        if clash:
            raise ValueError(f"source {name} repeats columns {sorted(clash)}")
        seen.update(cols)
        cands = cands.join(lf.select(*KEY, *cols), on=KEY, how="left").with_columns(
            pl.col(cols).fill_null(0)
        )
    return cands


@beartype
def add_features(
    cands: pl.LazyFrame,
    dataset: str,
    events: pl.LazyFrame,
    item: pl.LazyFrame,
    session: pl.LazyFrame,
    item_day: pl.LazyFrame,
) -> pl.LazyFrame:
    """The item, session and same-day item features (01f)."""
    item_cols = [c for c in item.collect_schema().names() if c != "aid"]
    day_cols = [
        c for c in item_day.collect_schema().names() if c not in ("aid", "day_")
    ]
    session_day = events.group_by("session").agg(
        day_=folds.day_index(dataset, pl.col("ts").max())
    )
    return (
        cands.join(
            item.rename({"aid": "cand"}).with_columns(pl.col(item_cols).cast(pl.Int32)),
            on="cand",
            how="left",
        )
        .with_columns(pl.col(item_cols).fill_null(0))
        .join(session, on="session", how="left")
        .join(session_day, on="session", how="left")
        .join(item_day.rename({"aid": "cand"}), on=["cand", "day_"], how="left")
        .with_columns(pl.col(day_cols).fill_null(0))
    )


@beartype
def add_target(cands: pl.LazyFrame, labels: pl.LazyFrame, target: str) -> pl.LazyFrame:
    hits = labels.filter(pl.col("type") == TARGET_CODES[target]).select(
        "session", cand="aid", target=pl.lit(1, pl.Int8)
    )
    return cands.join(hits.unique(KEY), on=KEY, how="left").with_columns(
        pl.col("target").fill_null(0)
    )


def _chunks(k: int, dataset: str) -> dict[int, list[int]]:
    """{chunk: sessions}: Benny's two parts of fold k, or the Kaggle test in 10 chunks."""
    if dataset == "train":
        parts = folds.heldout_sessions(k).collect()
        parts = parts.with_columns(chunk=pl.col("part") - pl.col("part").min())
    else:
        parts = pl.read_parquet(paths.SUB_SESSIONS).with_columns(
            chunk=pl.col("part") // SUB_CHUNK_PARTS
        )
    return {c: g["session"].to_list() for (c,), g in parts.group_by("chunk")}


def build(
    k: int, dataset: str, target: str, sample: float, engine, with_emb: bool = True
) -> None:
    """with_emb: add the Word2Vec similarity features (ranking.emb_features), from
    the fold's data_folds/fold_<k>/word2vec.parquet (01a)."""
    root = folds.output_root(sample)
    emb = pl.read_parquet(folds.feature_path(k, "word2vec", root)) if with_emb else None
    for i, sessions in sorted(_chunks(k, dataset).items()):
        start = time.time()
        in_chunk = pl.col("session").is_in(sessions)
        sources = {
            s: folds.read_candidates(k, dataset, s, root).filter(in_chunk)
            for s in folds.candidate_paths(k, dataset, root)
        }
        events = folds.candidate_events(k, dataset, sample).filter(in_chunk)
        shared = {
            name: pl.scan_parquet(folds.feature_path(None, f"{name}_{dataset}", root))
            for name in ("item_features", "session_features", "item_day_features")
        }
        lf = add_features(
            add_sources(candidates(sources), sources),
            dataset,
            events,
            shared["item_features"],
            shared["session_features"],
            shared["item_day_features"],
        )
        if dataset == "train":
            lf = add_target(lf, folds.labels(k, dataset).filter(in_chunk), target)
        df = (
            lf.with_columns(pl.selectors.float().cast(pl.Float32))
            .sort(KEY)
            .collect(engine=engine)
        )
        if (
            emb is not None
        ):  # Benny's 02_Split_2_Add_Emb_Features, which rewrote the chunks
            df = df.join(emb_features(df.lazy(), events, emb), on=KEY, how="left")
        path = chunk_path(k, target, dataset, i, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        msg = f"fold {k} {dataset} {i}: {df.height:,} rows x {df.width} columns, {df['session'].n_unique():,} sessions"
        if dataset == "train":
            msg += f", {df['target'].sum():,} positives"
        print(f"{msg} ({time.time() - start:.0f} s)")
