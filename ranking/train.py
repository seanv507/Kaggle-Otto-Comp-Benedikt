"""XGBoost rankers on the split chunks (Benny's 03_XGB-v19-clicks.ipynb and the
carts/orders versions).

Two modes, as in Benny's notebook:
- cv: each train chunk (2 per fold) is predicted by models trained on other chunks:
  for clicks the first 2 others in (fold, chunk) order, for carts and orders all
  others. Each of N_SEEDS models stops early on the predicted chunk (so the recall
  is optimistic, as Benny's was); their best iterations set the submission's number
  of trees. Prints recall@20 per chunk and in total.
- sub: one model per fold, trained on that fold's 2 chunks, restricted to sessions
  with a positive (Benny's bl_pos), N_SEEDS seeds x n_trees; each predicts the sub
  chunks of every fold. Writes
      <root>/ranker/<target>/scores/fold_<f>_model_<m>_<j>.parquet
  with (session, cand, score_<seed>...), for ranking.combine.
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from beartype import beartype

import folds
import metrics

from .split import KEY, TARGET_CODES, chunk_path

XGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "eval_metric": "logloss",
    "objective": "binary:logistic",
    "tree_method": "hist",  # Benny: gpu_hist, which XGBoost 2 replaced with device=cuda
    "device": "cuda",
}
N_SEEDS = 3
NOT_FEATURES = [
    "session",
    "cand",
    "target",
    "target_clicks",
    "target_carts",
    "target_orders",
]
SUB_FILL = -999  # Benny filled missing sub values with -999 (train had none)


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NOT_FEATURES]


def _matrix(df: pl.DataFrame, columns: list[str], label: bool = True) -> xgb.DMatrix:
    x = df.select(columns).cast(pl.Float32).to_numpy()
    return xgb.DMatrix(
        x, label=df["target"].to_numpy() if label else None, feature_names=columns
    )


def _with_positive(df: pl.DataFrame) -> pl.DataFrame:
    """Only sessions with at least one positive candidate (Benny's bl_pos)."""
    return df.filter((pl.col("target").max() > 0).over("session"))


@beartype
def train_chunks(files: list[Path], test: Path, target: str) -> list[Path]:
    """Benny's training chunks for predicting `test` in cv mode."""
    others = [f for f in files if f != test]
    return others[:2] if target == "clicks" else others


def _read(paths_: list[Path]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in paths_])


def cv(
    target: str, sample: float, fold_ids: list[int], max_trees: int = 1000, params=None
) -> dict:
    params = {**XGB_PARAMS, **(params or {})}
    root = folds.output_root(sample)
    chunks = {
        chunk_path(k, target, "train", i, root): k for k in fold_ids for i in (0, 1)
    }
    files = list(chunks)
    scored, labels, best = [], [], []
    for test, k in chunks.items():
        start = time.time()
        train_df = _read(train_chunks(files, test, target))
        if target != "clicks":  # Benny's cv filtered carts and orders only
            train_df = _with_positive(train_df)
        test_df = pl.read_parquet(test)
        cols = feature_columns(train_df)
        dtrain, dtest = _matrix(train_df, cols), _matrix(test_df, cols)
        score = np.zeros(test_df.height)
        for seed in range(N_SEEDS):
            model = xgb.train(
                {**params, "seed": seed},
                dtrain,
                num_boost_round=max_trees,
                evals=[(dtest, "test")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            best.append(model.best_iteration)
            score += model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        # Benny's calc_recall: the target's labels of the chunk's sessions
        sessions = test_df["session"].unique().to_list()
        chunk_labels = (
            folds.sample_sessions(folds.labels(k), sample)
            .filter(
                pl.col("type") == TARGET_CODES[target],
                pl.col("session").is_in(sessions),
            )
            .collect()
        )
        chunk_scores = test_df.select(KEY).with_columns(score=score)
        r = metrics.recall(chunk_scores, chunk_labels, "score")[target]
        scored.append(chunk_scores)
        labels.append(chunk_labels)
        print(
            f"{test.relative_to(root)}: recall@20 {r:.4f} (trained on {train_df.height:,} rows; "
            f"best iterations {best[-N_SEEDS:]}; {time.time() - start:.0f} s)"
        )
    recall = metrics.recall(pl.concat(scored), pl.concat(labels), "score")[target]
    print(
        f"{target} cv recall@20: {recall:.4f}; mean best iteration {np.mean(best):.0f}"
    )
    return {"recall": recall, "best_iterations": best}


def score_path(root: Path, target: str, fold: int, model: int, chunk: int) -> Path:
    return (
        root
        / "ranker"
        / target
        / "scores"
        / f"fold_{fold}_model_{model}_{chunk}.parquet"
    )


def sub(
    target: str, sample: float, fold_ids: list[int], n_trees: int, params=None
) -> None:
    params = {**XGB_PARAMS, **(params or {})}
    root = folds.output_root(sample)
    for m in fold_ids:
        start = time.time()
        train_df = _with_positive(
            _read([chunk_path(m, target, "train", i, root) for i in (0, 1)])
        )
        cols = feature_columns(train_df)
        dtrain = _matrix(train_df, cols)
        models = [
            xgb.train({**params, "seed": s}, dtrain, num_boost_round=n_trees)
            for s in range(N_SEEDS)
        ]
        print(f"model {m}: {train_df.height:,} rows ({time.time() - start:.0f} s)")
        for f in fold_ids:
            for path in sorted(
                chunk_path(f, target, "sub", 0, root).parent.glob("sub_*.parquet")
            ):
                df = pl.read_parquet(path).fill_null(SUB_FILL)
                dsub = _matrix(df, cols, label=False)
                out = df.select(KEY).with_columns(
                    **{
                        f"score_{s}": pl.Series(model.predict(dsub), dtype=pl.Float32)
                        for s, model in enumerate(models)
                    }
                )
                j = int(path.stem.removeprefix("sub_"))
                dest = score_path(root, target, f, m, j)
                dest.parent.mkdir(parents=True, exist_ok=True)
                out.write_parquet(dest)
        print(
            f"model {m}: predicted {len(fold_ids)} folds ({time.time() - start:.0f} s)"
        )
