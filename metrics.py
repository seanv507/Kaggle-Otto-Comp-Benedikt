"""Recall@k of candidate files, as in the competition metric.

Every candidate source can be scored on its own against XGB_TRAIN_Y (week 4) or
TEST_LABELS (Kaggle test), without a ranker: a cheap check that it is sensible.
"""

import polars as pl
from beartype import beartype

from event_types import TYPE_NAMES

Frame = pl.LazyFrame | pl.DataFrame


@beartype
def top_k(cands: Frame, score: str, k: int, descending: bool = True) -> pl.LazyFrame:
    """(session, aid): each session's k best candidates by `score`.

    Ties are broken by the smaller cand, so the result doesn't depend on row order.
    """
    return (
        cands.lazy()
        .select("session", "cand", score)
        .sort("session", score, "cand", descending=[False, descending, False])
        .with_columns(n=pl.int_range(pl.len()).over("session"))
        .filter(pl.col("n") < k)
        .select("session", aid="cand")
    )


@beartype
def recall(
    cands: Frame, labels: Frame, score: str, k: int = 20, descending: bool = True
) -> dict[str, float]:
    """Recall@k per type: hits / sum over sessions of min(number of labels, k).

    `labels` are (session, aid, type). Sessions with labels but no candidates count as
    misses, so filter `labels` to the sessions that were scored (e.g. on a sample).
    """
    labels = labels.lazy()
    top = top_k(cands, score, k, descending)
    hits = labels.join(top, on=["session", "aid"], how="semi").group_by("type").len()
    totals = (
        labels.group_by("session", "type")
        .len()
        .group_by("type")
        .agg(pl.col("len").clip(upper_bound=k).sum().alias("total"))
    )
    rows = (
        totals.join(hits, on="type", how="left")
        .fill_null(0)
        .collect(engine="in-memory")
        .rows(named=True)
    )
    return {
        TYPE_NAMES[r["type"]]: r["len"] / r["total"]
        for r in sorted(rows, key=lambda r: r["type"])
    }


def format_recall(recalls: dict[str, float]) -> str:
    return "  ".join(f"{t} {r:.4f}" for t, r in recalls.items())
