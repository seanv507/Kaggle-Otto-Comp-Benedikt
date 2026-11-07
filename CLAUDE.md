# CLAUDE.md

## What this repo is

The user's fork of **Benny Schifferer's part** of the 3rd place [OTTO](https://www.kaggle.com/competitions/otto-recommender-system) solution (`seanv507/Kaggle-Otto-Comp-Benedikt`). It is an XGBoost reranker with Transformers4Rec, GRU and Word2Vec features and a 5-fold week-4 split. See README.md for the pipeline.

The sibling repo `../Kaggle-OTTO-Comp` is Chris Deotte's part, and it is being converted to Polars too. **Read `../Kaggle-OTTO-Comp/CLAUDE.md`.** Its rules apply here as well:
- the Polars pin (1.44.x, for `cudf-polars-cu13`);
- the GPU-engine limits;
- the lazy API with an `engine` parameter;
- `type` codes from `event_types.py` rather than strings;
- `ts` as Datetime(ms), not epoch seconds;
- the testing conventions;
- the hardware notes (local RTX 2050 vs cloud A100).

`../Kaggle-OTTO-Comp/docs/solutions.md` summarises both write-ups.

The main work is **replacing Benny's cuDF notebooks with Polars Python scripts**, one pipeline step at a time, next to the notebooks they replace. Done so far:
- `00_Preprocess/preprocess.py`
- `01e_FE_Transformer/` (`t4r_data.py`, `t4r_model.py`, `t4r_train.py`): Benny's Transformers4Rec model, reimplemented in plain PyTorch with HF `XLNetModel`. See its README for the differences from Benny's code and the A100 recall, which beats Benny's.
- `01f_FE_other/` (`previous_visits.py`, `session_features.py`, `item_features.py`): the `pv` and `pv_2` candidates, the session features and the item activity features. All of them are shared by the folds.
- `folds.py` (root): the fold inputs (`fe_events(k)`, `candidate_events(k, dataset)`, `labels`), `sample_sessions` for local development, and the candidate-file layout (`write_candidates`, `candidate_paths`, `read_candidates`, `feature_path`). Every per-fold step uses it instead of rebuilding the inputs. `metrics.py` scores any candidate file with recall@k.

`docs/conversion_plan.md` has the remaining steps. Each step goes on its own branch and is developed locally on a sample. The A100 runs come at the end.

## Data

This repo uses the same data as `../Kaggle-OTTO-Comp`. `00_Preprocess/preprocess.py` symlinks it in and builds the labels and session splits. `00_Preprocess/README.md` maps each of Benny's files to its source.

- Use the constants in `paths.py` (`TRAIN`, `TEST`, `TEST_FULL`, `TEST_LABELS`, `CV_TRAIN`, `XGB_TRAIN_X`, `XGB_TRAIN_Y`, `SESSIONS_EVAL`, `SUB_SESSIONS`, ...). Don't use the per-directory `./data` symlinks that the notebooks rely on.
- Benny's notebooks read `data/train/interim/*.parquet` (all 4 weeks), which is now `paths.TRAIN`. They read `preprocess/train/interim/*.parquet` (weeks 1–3), which is now `paths.CV_TRAIN`.
- Week-4 folds: fold *k* holds out the sessions with `fold == k` in `SESSIONS_EVAL`. For those sessions the truncated events come from `XGB_TRAIN_X`. All other sessions come untruncated from `TRAIN`.
- `TEST_LABELS` are the Kaggle test labels, built from the untruncated `TEST_FULL`. Benny never had them. Use them to score the LB pipeline locally.
- Number events within a session (`int_range().over("session")`) **before** any join. On the streaming engine a join doesn't keep row order, and numbering after it gave wrong labels with no error.
- Session splits are pickles of Python lists in the notebooks. Here they are parquet `(session, part[, fold])`, so they can be used in a join instead of `isin`.

## Tests

- Parametrize Polars code over `engines.ENGINES` (streaming, in-memory, and GPU when available) and collect with `engines.engine(name)`.
- Every step's CLI takes `--sample`, and its outputs then go to `folds.output_root(sample)`. Check a new step on the real data with a sample, plus invariants (counts that must add up to the number of events) and its candidate recall (`metrics.recall`).

## Polars pitfalls found here

These add to the GPU-engine limits in `../Kaggle-OTTO-Comp/CLAUDE.md`.
- **The GPU engine gives different answers:**
  - `Expr.hash` gives different values from the CPU engines. Use integer arithmetic (as `folds.sample_sessions` does) where the result must not depend on the engine.
  - `cum_sum` of a UInt32 returns Int64 while the plan's schema says UInt32, so a later cast to UInt32 is dropped as a no-op. Cast to Int64 before `cum_sum`.
- **The GPU engine rejects:**
  - `filter` inside a `group_by` aggregation. Use a sum of a boolean for counts, and `pl.when(cond).then(x).n_unique()` minus the null (see `01f_FE_other/session_features.py`).
  - `.any()` in a `group_by`. Use `.sum() > 0`.
  - `replace_strict` without an explicit `default`.
- **Nested windows are quadratic.** An expression with one `.over()` inside another (e.g. `gap(shift().over(s)).cum_sum().over(s)`) took minutes on 7M rows. Compute the inner window in its own `with_columns` first.

## Environment

- Managed with `uv`. Run from the repo root: `uv run python <dir>/<script>.py` and `uv run pytest`.
- Editable install with hatch `dev-mode-dirs = ["."]`, so scripts in numbered directories can import `paths` and `event_types`. Tests import modules from their own directory.
- torch comes from an extra: `uv sync --extra cu132` on a GPU machine (CUDA 13.2 wheels), `--extra cpu` otherwise. A plain `uv sync` removes torch.
- Only the dependencies needed so far are installed. Add xgboost, gensim and so on when converting the step that needs them.
