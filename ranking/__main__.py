"""Benny's rankers from the command line. Run from the repo root:

    uv run python -m ranking split <split> [--folds ...] [--sample f]
    uv run python -m ranking cv <model> ...       local CV on week 4
    uv run python -m ranking sub <model> ...      submission models, scores, top 20
    uv run python -m ranking stack <model> ...    04_Orders' click and cart scores
    uv run python -m ranking combine <model> ...  only the top 20 of existing scores
    uv run python -m ranking submission ...       the CSV and its Kaggle-test score

<split> is one of ranking.split.SPLITS, <model> one of ranking.train.MODELS. The
order for each of Benny's pipelines:
    clicks (02_Clicks)  split clicks; cv clicks; sub clicks
    carts (03_Carts)    split carts; cv carts; sub carts
    orders (04_Orders)  split orders/clicks; split orders/carts_orders;
                        stack orders/clicks; stack orders/carts; cv orders; sub orders
    submission          after the three
"""

import argparse

import polars as pl

import folds
import metrics
from event_types import TYPE_NAMES

from . import combine, split, train

WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
FINAL = {
    "clicks": "clicks",
    "carts": "carts",
    "orders": "orders",
}  # target -> MODELS name


def submission(sample: float) -> None:
    """Benny's final CSV: clicks from 02_Clicks, carts from 03_Carts, orders from
    04_Orders (06_Combine-Bags), and its weighted recall@20 on the Kaggle test."""
    root = folds.output_root(sample)
    labels = folds.sample_sessions(folds.labels(None, "sub"), sample).collect()
    codes = {name: code for code, name in TYPE_NAMES.items()}
    parts, score = [], 0.0
    for target, name in FINAL.items():
        top = pl.read_parquet(
            root / "ranker" / train.MODELS[name].chunks / "sub_top20.parquet"
        )
        r = metrics.recall(
            top.with_columns(score=-pl.col("rank").cast(pl.Int32)),
            labels.filter(pl.col("type") == codes[target]),
            "score",
        )[target]
        score += WEIGHTS[target] * r
        print(f"{target}: recall@20 {r:.4f}")
        parts.append(
            top.sort("session", "rank")
            .group_by("session", maintain_order=True)
            .agg(labels=pl.col("cand").cast(pl.String).str.join(" "))
            .select(
                session_type=pl.col("session").cast(pl.String) + f"_{target}",
                labels="labels",
            )
        )
    path = root / "submission.csv"
    pl.concat(parts).write_csv(path)
    print(f"wrote {path}; Kaggle test score {score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=["split", "cv", "sub", "stack", "combine", "submission"]
    )
    parser.add_argument("name", nargs="?", help="the split or model")
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(folds.N_FOLDS))
    )
    parser.add_argument(
        "--sample", type=float, default=1.0, help="fraction of sessions"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(folds.DATASETS), choices=folds.DATASETS
    )
    parser.add_argument(
        "--engine", default="streaming", help="split: streaming, in-memory or gpu"
    )
    parser.add_argument(
        "--no-emb", action="store_true", help="split: without the Word2Vec features"
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=None,
        help="sub, stack: trees per model (default Benny's)",
    )
    parser.add_argument("--device", default="cuda", help="XGBoost device: cuda or cpu")
    args = parser.parse_args()
    params = {"device": args.device}
    if args.command == "submission":
        submission(args.sample)
    elif args.command == "split":
        engine = (
            pl.GPUEngine(raise_on_fail=True) if args.engine == "gpu" else args.engine
        )
        for k in args.folds:
            for dataset in args.datasets:
                split.build(
                    k, dataset, args.name, args.sample, engine, with_emb=not args.no_emb
                )
    elif args.command == "cv":
        train.cv(args.name, args.sample, args.folds, params=params)
    elif args.command == "stack":
        train.stack(args.name, args.sample, args.folds, args.trees, params=params)
    else:
        if args.command == "sub":
            train.sub(args.name, args.sample, args.folds, args.trees, params=params)
        combine.main(
            args.name, args.sample, args.folds, train.n_models(args.name, args.folds)
        )


if __name__ == "__main__":
    main()
