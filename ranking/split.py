"""Candidates and features per fold (Benny's 01_Split notebooks of 02_Clicks,
03_Carts and 04_Orders), one SplitSpec per notebook (SPLITS).

For each fold k and dataset (train: fold k's held-out sessions; sub: the Kaggle test
sessions) it builds one row per (session, cand):
- candidates: each generating source's top n per session (ranked by its generator
  column), kept if at least min_sources sources propose it. For train, a spec can
  restrict the candidates to sessions with labels of some types (Benny's labels_filter);
- the columns of every candidate source of the fold (folds.candidate_paths), except
  the spec's ignored sources and benny_columns.TOTAL_IGNORE; 0 where a source has no row;
- the item activity (item_features_*, 0 where missing), the session features
  (session_features_*) and the item's activity on the session's day
  (item_day_features_* on folds.day_index of the session's last event, 0 where
  missing);
- the fold's Word2Vec similarity features (ranking.emb_features), unless with_emb
  is False;
- for train, `target`: 1 if the cand is one of the session's labels of the output's
  target; with row_filter, only the sessions that have such a label.
Floats are Float32. The rows are written in chunks, as Benny did:
    data_folds/fold_<k>/ranker/<output>/train_<i>.parquet   i = Benny's 2 parts per fold
    data_folds/fold_<k>/ranker/<sub_dir>/sub_<j>.parquet    j = SUB_SESSIONS part // 10
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from beartype import beartype

import folds
import metrics
import paths
from event_types import TYPE_NAMES

from .benny_columns import CLICKS_GENERATORS, LIST_SORT_COLUMNS, NO_CANDS, TOTAL_IGNORE
from .emb_features import emb_features

KEY = ["session", "cand"]
SUB_CHUNK_PARTS = 10  # SUB_SESSIONS parts per sub chunk (Benny's sub_sessoins_10)
TARGET_CODES = {name: code for code, name in TYPE_NAMES.items()}


def chunk_path(
    k: int, target: str, dataset: str, i: int, root: Path = paths.DATA_FOLDS
) -> Path:
    return folds.fold_dir(k, root) / "ranker" / target / f"{dataset}_{i}.parquet"


@dataclass(frozen=True)
class SplitSpec:
    """One of Benny's 01_Split notebooks."""

    outputs: dict[str, str]  # train chunk directory -> its target
    sub_dir: str | None  # directory of the sub chunks (None: no sub)
    generators: dict[str, str] | None = (
        None  # source -> ranking column; None: every source
    )
    n: int = NO_CANDS  # top n per generating source
    min_sources: int = 1  # keep a candidate proposed by at least this many sources
    candidate_types: tuple[
        str, ...
    ] = ()  # train: candidates only for sessions with these labels
    row_filter: bool = (
        False  # train: each output only has sessions with its target's labels
    )
    ignore_sources: tuple[str, ...] = ()  # sources left out of the features
    not_generators: tuple[str, ...] = field(default=("gru_scores_20",))


SPLITS = {
    # 02_Clicks/01_Split.ipynb
    "clicks": SplitSpec({"clicks": "clicks"}, "clicks", CLICKS_GENERATORS),
    # 03_Carts/01_Split.ipynb: every source, top 100, at least 2 sources
    "carts": SplitSpec(
        {"carts": "carts"},
        "carts",
        None,
        n=100,
        min_sources=2,
        candidate_types=("carts", "orders"),
        row_filter=True,
    ),
    # 04_Orders/01a_Split-clicks.ipynb: as clicks, train only, without chris__1__15
    "orders/clicks": SplitSpec(
        {"orders/clicks": "clicks"},
        None,
        CLICKS_GENERATORS,
        ignore_sources=("chris__1__15",),
    ),
    # 04_Orders/01b_Split-carts-orders-sub.ipynb: top 80, at least 2 sources
    "orders/carts_orders": SplitSpec(
        {"orders/carts": "carts", "orders/orders": "orders"},
        "orders/sub",
        None,
        n=80,
        min_sources=2,
        candidate_types=("carts", "orders"),
        row_filter=True,
        ignore_sources=("chris__1__15",),
    ),
}


def generator_columns(
    sources: dict[str, pl.LazyFrame], spec: SplitSpec
) -> dict[str, str]:
    """{source: ranking column}: the spec's, or each source's one column in Benny's
    LIST_SORT_COLUMNS (his get_cands asserted there is exactly one)."""
    if spec.generators is not None:
        return {s: c for s, c in spec.generators.items() if s in sources}
    out = {}
    for name, lf in sources.items():
        if name in spec.not_generators:
            continue
        cols = [c for c in lf.collect_schema().names() if c in LIST_SORT_COLUMNS]
        if len(cols) != 1:
            raise ValueError(f"source {name} has ranking columns {cols}, expected one")
        out[name] = cols[0]
    return out


@beartype
def candidates(
    sources: dict[str, pl.LazyFrame],
    generators: dict[str, str] = CLICKS_GENERATORS,
    n: int = NO_CANDS,
    min_sources: int = 1,
) -> pl.LazyFrame:
    """(session, cand): each generating source's top n per session, kept if at least
    `min_sources` sources propose it.

    Ties at the cut go to the smaller cand (Benny's were cuDF's sort order).
    """
    tops = [
        metrics.top_k(sources[s], column, n).rename({"aid": "cand"})
        for s, column in generators.items()
        if s in sources
    ]
    counts = pl.concat(tops).group_by(KEY).len()
    return counts.filter(pl.col("len") >= min_sources).select(KEY)


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


def _with_labels(labels: pl.LazyFrame, targets: tuple[str, ...]) -> list[int]:
    codes = [TARGET_CODES[t] for t in targets]
    return (
        labels.filter(pl.col("type").is_in(codes))
        .select("session")
        .unique()
        .collect()["session"]
        .to_list()
    )


def build(
    k: int, dataset: str, spec_name: str, sample: float, engine, with_emb: bool = True
) -> None:
    """Write fold k's chunks of SPLITS[spec_name] for `dataset`.

    with_emb: add the Word2Vec similarity features (ranking.emb_features), from the
    fold's data_folds/fold_<k>/word2vec.parquet (01a).
    """
    spec = SPLITS[spec_name]
    if dataset == "sub" and spec.sub_dir is None:
        return
    root = folds.output_root(sample)
    emb = pl.read_parquet(folds.feature_path(k, "word2vec", root)) if with_emb else None
    labels = (
        folds.sample_sessions(folds.labels(k, dataset), sample)
        if dataset == "train"
        else None
    )
    for i, sessions in sorted(_chunks(k, dataset).items()):
        start = time.time()
        in_chunk = pl.col("session").is_in(sessions)
        sources = {
            s: folds.read_candidates(k, dataset, s, root).filter(in_chunk)
            for s in folds.candidate_paths(k, dataset, root)
        }
        generating = sources
        if dataset == "train" and spec.candidate_types:  # Benny's labels_filter
            with_labels = pl.col("session").is_in(
                _with_labels(labels.filter(in_chunk), spec.candidate_types)
            )
            generating = {s: lf.filter(with_labels) for s, lf in sources.items()}
        features = {s: lf for s, lf in sources.items() if s not in spec.ignore_sources}
        cands = candidates(
            generating, generator_columns(sources, spec), spec.n, spec.min_sources
        )
        events = folds.candidate_events(k, dataset, sample).filter(in_chunk)
        shared = {
            name: pl.scan_parquet(folds.feature_path(None, f"{name}_{dataset}", root))
            for name in ("item_features", "session_features", "item_day_features")
        }
        lf = add_features(
            add_sources(cands, features),
            dataset,
            events,
            shared["item_features"],
            shared["session_features"],
            shared["item_day_features"],
        )
        df = (
            lf.with_columns(pl.selectors.float().cast(pl.Float32))
            .sort(KEY)
            .collect(engine=engine)
        )
        if (
            emb is not None
        ):  # Benny's 02_Split_2_Add_Emb_Features, which rewrote the chunks
            df = df.join(emb_features(df.lazy(), events, emb), on=KEY, how="left")
        if dataset == "sub":
            _write(
                df,
                chunk_path(k, spec.sub_dir, "sub", i, root),
                f"fold {k} {spec.sub_dir} sub {i}",
                start,
            )
            continue
        chunk_labels = labels.filter(in_chunk)
        for out, target in spec.outputs.items():
            rows = df.lazy()
            if spec.row_filter:
                rows = rows.filter(
                    pl.col("session").is_in(_with_labels(chunk_labels, (target,)))
                )
            rows = add_target(rows, chunk_labels, target).collect()
            _write(
                rows,
                chunk_path(k, out, "train", i, root),
                f"fold {k} {out} train {i}",
                start,
            )


def _write(df: pl.DataFrame, path: Path, name: str, start: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    msg = f"{name}: {df.height:,} rows x {df.width} columns, {df['session'].n_unique():,} sessions"
    if "target" in df.columns:
        msg += f", {df['target'].sum():,} positives"
    print(f"{msg} ({time.time() - start:.0f} s)")
