"""XGBoost rankers on the split chunks, one ModelSpec per model of Benny's (MODELS):
02_Clicks/03_XGB-v19-clicks, 03_Carts/03_XGB-v20-carts and 04_Orders/03-05.

Three modes:
- cv: each train chunk (2 per fold) is predicted by models trained on other chunks
  (the first cv_train_chunks others in (fold, chunk) order, or all others). Each of
  cv_seeds models stops early on the predicted chunk, so the recall is optimistic,
  as Benny's was. Their best iterations set the submission's number of trees. Prints
  recall@20 per chunk and in total.
- sub: the train chunks in (fold, chunk) order are cut into groups of
  chunks_per_model (or one group of all), and each group trains n_seeds models of
  n_trees, on the sessions with a positive (Benny's bl_pos). Each model predicts the
  sub chunks of every fold and writes
      <root>/ranker/<chunks>/scores/fold_<f>_model_<m>_<j>.parquet
  with (session, cand, score_<seed>...), for ranking.combine.
- stack (04_Orders' click and cart models): the same models, but their mean score
  over all models and seeds becomes a feature column (stack_column) of the train
  chunks of stack_into and of the sub chunks, written into those files. They train
  without the sessions that have labels of `exclude`, which are the sessions the
  later models train on.
"""

import time
from dataclasses import dataclass
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
NOT_FEATURES = [
    "session",
    "cand",
    "target",
    "target_clicks",
    "target_carts",
    "target_orders",
]
SUB_FILL = -999  # Benny filled missing sub values with -999 (train had none)


@dataclass(frozen=True)
class ModelSpec:
    chunks: str  # the train chunk directory (ranking.split outputs)
    target: str
    sub_dir: str  # the sub chunk directory the models predict
    n_seeds: int
    n_trees: int
    cv_seeds: int
    cv_train_chunks: int | None  # cv: the first n other chunks; None: all others
    chunks_per_model: int | None  # sub: chunks per model; None: one model on all
    cv_positive_only: bool = True  # Benny's cv filtered carts and orders, not clicks
    exclude: tuple[str, ...] = ()  # leave out sessions with labels of these types
    stack_column: str | None = None
    stack_into: tuple[str, ...] = ()


MODELS = {
    "clicks": ModelSpec(
        "clicks", "clicks", "clicks", 3, 394, 3, 2, 2, cv_positive_only=False
    ),
    "carts": ModelSpec("carts", "carts", "carts", 10, 377, 10, 5, 5),
    # Benny's 03_XGB_Clicks used no_seeds before defining it (a value from an earlier
    # kernel state); 3 as in his clicks ranker
    "orders/clicks": ModelSpec(
        "orders/clicks",
        "clicks",
        "orders/sub",
        3,
        394,
        3,
        2,
        2,
        cv_positive_only=False,
        exclude=("carts", "orders"),
        stack_column="xgb_c_score",
        stack_into=("orders/carts", "orders/orders"),
    ),
    "orders/carts": ModelSpec(
        "orders/carts",
        "carts",
        "orders/sub",
        10,
        308,
        10,
        5,
        5,
        exclude=("orders",),
        stack_column="xgb_ca_score",
        stack_into=("orders/orders",),
    ),
    "orders": ModelSpec(
        "orders/orders", "orders", "orders/sub", 10, 345, 3, None, None
    ),
}


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NOT_FEATURES]


def _matrix(df: pl.DataFrame, columns: list[str], label: bool = True) -> xgb.DMatrix:
    """The features as an XGBoost matrix, from the Polars frame itself (through Arrow):
    no dense float32 copy first. Predictions are identical to the numpy route's."""
    return xgb.DMatrix(
        df.select(columns),
        label=df.select("target") if label else None,
        feature_names=columns,
    )


def _with_positive(df: pl.DataFrame) -> pl.DataFrame:
    """Only sessions with at least one positive candidate (Benny's bl_pos)."""
    return df.filter((pl.col("target").max() > 0).over("session"))


@beartype
def train_chunks(files: list[Path], test: Path, n: int | None) -> list[Path]:
    """Benny's training chunks for predicting `test` in cv mode: the first n others."""
    others = [f for f in files if f != test]
    return others if n is None else others[:n]


@beartype
def model_groups(files: list[Path], chunks_per_model: int | None) -> list[list[Path]]:
    """The train chunks of each submission model (Benny's step over sorted files)."""
    if chunks_per_model is None:
        return [files]
    return [
        files[i : i + chunks_per_model] for i in range(0, len(files), chunks_per_model)
    ]


def _files(spec: ModelSpec, root: Path, fold_ids: list[int]) -> dict[Path, int]:
    return {
        chunk_path(k, spec.chunks, "train", i, root): k
        for k in fold_ids
        for i in (0, 1)
    }


def _read(
    files: dict[Path, int], sample: float, exclude: tuple[str, ...] = ()
) -> pl.DataFrame:
    frames = []
    for path, k in files.items():
        df = pl.read_parquet(path)
        if exclude:
            codes = [TARGET_CODES[t] for t in exclude]
            labelled = folds.sample_sessions(folds.labels(k), sample).filter(
                pl.col("type").is_in(codes)
            )
            drop = labelled.select("session").unique().collect()["session"].to_list()
            df = df.filter(~pl.col("session").is_in(drop))
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed")


def cv(
    name: str, sample: float, fold_ids: list[int], max_trees: int = 1000, params=None
) -> dict:
    spec = MODELS[name]
    params = {**XGB_PARAMS, **(params or {})}
    root = folds.output_root(sample)
    chunks = _files(spec, root, fold_ids)
    files = list(chunks)
    scored, labels, best = [], [], []
    for test, k in chunks.items():
        start = time.time()
        train_df = _read(
            {f: chunks[f] for f in train_chunks(files, test, spec.cv_train_chunks)},
            sample,
            spec.exclude,
        )
        if spec.cv_positive_only:
            train_df = _with_positive(train_df)
        test_df = pl.read_parquet(test)
        cols = feature_columns(train_df)
        dtrain, dtest = _matrix(train_df, cols), _matrix(test_df, cols)
        score = np.zeros(test_df.height)
        for seed in range(spec.cv_seeds):
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
                pl.col("type") == TARGET_CODES[spec.target],
                pl.col("session").is_in(sessions),
            )
            .collect()
        )
        chunk_scores = test_df.select(KEY).with_columns(score=score)
        r = metrics.recall(chunk_scores, chunk_labels, "score")[spec.target]
        scored.append(chunk_scores)
        labels.append(chunk_labels)
        print(
            f"{test.relative_to(root)}: recall@20 {r:.4f} (trained on {train_df.height:,} rows; "
            f"best iterations {best[-spec.cv_seeds :]}; {time.time() - start:.0f} s)"
        )
    recall = metrics.recall(pl.concat(scored), pl.concat(labels), "score")[spec.target]
    print(f"{name} cv recall@20: {recall:.4f}; mean best iteration {np.mean(best):.0f}")
    return {"recall": recall, "best_iterations": best}


def score_path(root: Path, chunks: str, fold: int, model: int, chunk: int) -> Path:
    return (
        root
        / "ranker"
        / chunks
        / "scores"
        / f"fold_{fold}_model_{model}_{chunk}.parquet"
    )


def _sub_files(root: Path, sub_dir: str, fold_ids: list[int]) -> list[tuple[int, Path]]:
    return [
        (f, p)
        for f in fold_ids
        for p in sorted(
            chunk_path(f, sub_dir, "sub", 0, root).parent.glob("sub_*.parquet")
        )
    ]


def _train_models(
    spec: ModelSpec, group: dict[Path, int], sample: float, params: dict, n_trees: int
):
    df = _with_positive(_read(group, sample, spec.exclude))
    cols = feature_columns(df)
    dtrain = _matrix(df, cols)
    models = [
        xgb.train({**params, "seed": s}, dtrain, num_boost_round=n_trees)
        for s in range(spec.n_seeds)
    ]
    return models, cols, df.height


def sub(
    name: str,
    sample: float,
    fold_ids: list[int],
    n_trees: int | None = None,
    params=None,
) -> int:
    """Train the submission models and write their scores; returns the number of models."""
    spec = MODELS[name]
    params = {**XGB_PARAMS, **(params or {})}
    root = folds.output_root(sample)
    chunks = _files(spec, root, fold_ids)
    groups = model_groups(list(chunks), spec.chunks_per_model)
    for m, group in enumerate(groups):
        start = time.time()
        models, cols, n_rows = _train_models(
            spec, {f: chunks[f] for f in group}, sample, params, n_trees or spec.n_trees
        )
        for f, path in _sub_files(root, spec.sub_dir, fold_ids):
            df = pl.read_parquet(path).fill_null(SUB_FILL)
            dsub = _matrix(df, cols, label=False)
            out = df.select(KEY).with_columns(
                **{
                    f"score_{s}": pl.Series(model.predict(dsub), dtype=pl.Float32)
                    for s, model in enumerate(models)
                }
            )
            dest = score_path(
                root, spec.chunks, f, m, int(path.stem.removeprefix("sub_"))
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            out.write_parquet(dest)
        print(
            f"{name} model {m}: {n_rows:,} rows, {len(group)} chunks; scored {len(fold_ids)} folds' sub ({time.time() - start:.0f} s)"
        )
    return len(groups)


def stack(
    name: str,
    sample: float,
    fold_ids: list[int],
    n_trees: int | None = None,
    params=None,
) -> None:
    """Add the models' mean score as `stack_column` to the stack_into and sub chunks."""
    spec = MODELS[name]
    params = {**XGB_PARAMS, **(params or {})}
    root = folds.output_root(sample)
    chunks = _files(spec, root, fold_ids)
    models, cols = [], None
    for group in model_groups(list(chunks), spec.chunks_per_model):
        start = time.time()
        group_models, cols, n_rows = _train_models(
            spec, {f: chunks[f] for f in group}, sample, params, n_trees or spec.n_trees
        )
        models += group_models
        print(
            f"{name}: {len(group_models)} models on {n_rows:,} rows ({time.time() - start:.0f} s)"
        )
    targets = [
        p
        for d in spec.stack_into
        for k in fold_ids
        for p in sorted(
            chunk_path(k, d, "train", 0, root).parent.glob("train_*.parquet")
        )
    ]
    targets += [p for _, p in _sub_files(root, spec.sub_dir, fold_ids)]
    for path in targets:
        df = pl.read_parquet(path)
        dmat = _matrix(df.fill_null(SUB_FILL), cols, label=False)
        score = np.mean([m.predict(dmat) for m in models], axis=0)
        df.with_columns(
            pl.Series(spec.stack_column, score, dtype=pl.Float32)
        ).write_parquet(path)
    print(f"{name}: wrote {spec.stack_column} into {len(targets)} chunk files")


def n_models(name: str, fold_ids: list[int]) -> int:
    """The number of submission models of MODELS[name] for these folds."""
    per_model = MODELS[name].chunks_per_model
    n_chunks = 2 * len(fold_ids)
    return 1 if per_model is None else -(-n_chunks // per_model)
