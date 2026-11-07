"""Combine the submission models' scores into each session's top 20 (Benny's
04_Combine-Bags.ipynb).

Each (fold, model) pair scored the Kaggle test sessions with the features of that
fold; the score of a pair is the sum of its seeds. As in Benny's get_df, the first
pair's (session, cand) rows are the candidates, and every other pair's score is added
where it has the row. Each session's top 20 by the summed score are written to
    <root>/ranker/<target>/sub_top20.parquet   (session, cand, rank)
Sessions with fewer than 20 are then filled from Chris's public 0.575 baseline
(00_submissions/01_baseline, folds.baseline_dir), as Benny did, if it exists.
"""

from pathlib import Path

import polars as pl
from beartype import beartype

import folds
import metrics

from .split import KEY, TARGET_CODES
from .train import N_SEEDS, score_path


def _pair_scores(root, target: str, fold: int, model: int) -> pl.LazyFrame:
    files = sorted(
        score_path(root, target, fold, model, 0).parent.glob(
            f"fold_{fold}_model_{model}_*.parquet"
        )
    )
    if not files:
        raise FileNotFoundError(f"no scores for fold {fold}, model {model}")
    seeds = pl.sum_horizontal(f"score_{s}" for s in range(N_SEEDS))
    return pl.concat([pl.scan_parquet(f) for f in files]).select(
        *KEY, score=seeds.cast(pl.Float64)
    )


@beartype
def combined_scores(
    target: str, root: Path, fold_ids: list[int], model_ids: list[int]
) -> pl.LazyFrame:
    pairs = [(f, m) for f in fold_ids for m in model_ids]
    lf = _pair_scores(root, target, *pairs[0])
    for f, m in pairs[1:]:
        other = _pair_scores(root, target, f, m).rename({"score": "other"})
        lf = lf.join(other, on=KEY, how="left").select(
            *KEY, score=pl.col("score") + pl.col("other").fill_null(0)
        )
    return lf


@beartype
def fill(
    top: pl.LazyFrame | pl.DataFrame, baseline: pl.LazyFrame | pl.DataFrame
) -> pl.LazyFrame:
    """Top up each session of `top` (session, cand, rank) to 20 with the baseline's
    (session, aid, rank) items it doesn't have, in the baseline's order (Benny's
    prep_xgb). Sessions only in the baseline are left out, as in Benny's merge."""
    ours = top.lazy().select(
        "session", "cand", src=pl.lit(0), pos=pl.col("rank").cast(pl.Int64)
    )
    sessions = ours.select("session").unique()
    base = (
        baseline.lazy()
        .join(sessions, on="session", how="semi")
        .select("session", cand="aid", src=pl.lit(1), pos=pl.col("rank").cast(pl.Int64))
    )
    return (
        pl.concat([ours, base])
        .sort("session", "src", "pos")
        .unique(KEY, keep="first", maintain_order=True)
        .with_columns(rank=pl.int_range(pl.len(), dtype=pl.UInt8).over("session"))
        .filter(pl.col("rank") < 20)
        .select("session", "cand", "rank")
    )


@beartype
def top20(scores: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """(session, cand, rank): each session's 20 best by score (rank 0 is best)."""
    return (
        metrics.top_k(scores, "score", 20)
        .rename({"aid": "cand"})
        .with_columns(rank=pl.int_range(pl.len(), dtype=pl.UInt8).over("session"))
    )


def main(target: str, sample: float, fold_ids: list[int], model_ids: list[int]) -> None:
    root = folds.output_root(sample)
    top = top20(combined_scores(target, root, fold_ids, model_ids)).collect(
        engine="in-memory"
    )
    baseline = folds.baseline_dir(sample) / f"{target}.parquet"
    if baseline.exists():
        short = top.group_by("session").len().filter(pl.col("len") < 20).height
        top = fill(top, pl.scan_parquet(baseline)).collect(engine="in-memory")
        print(
            f"filled {short:,} sessions with fewer than 20 from {baseline.relative_to(baseline.parents[2])}"
        )
    else:
        print(f"no baseline at {baseline}: sessions with fewer than 20 aren't filled")
    path = root / "ranker" / target / "sub_top20.parquet"
    top.write_parquet(path)
    labels = folds.sample_sessions(folds.labels(None, "sub"), sample).filter(
        pl.col("type") == TARGET_CODES[target]
    )
    r = metrics.recall(
        top.with_columns(score=-pl.col("rank").cast(pl.Int32)), labels, "score"
    )[target]
    short = top.group_by("session").len().filter(pl.col("len") < 20).height
    print(
        f"wrote {path.relative_to(root.parent)}: {top['session'].n_unique():,} sessions "
        f"({short:,} with fewer than 20); Kaggle test {target} recall@20 {r:.4f}"
    )
