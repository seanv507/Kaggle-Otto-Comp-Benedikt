# Data Preprocessing

`preprocess.py` replaces Benny's `01_Preprocess.ipynb` (jsonl → parquet) and `02_Preprocess.ipynb` (session splits). It reuses the Polars data already converted in `../Kaggle-OTTO-Comp/data/`, so there are no jsonl files and no host script to run.

```bash
uv run python 00_Preprocess/preprocess.py --check
```

Options: `--src` (default `../Kaggle-OTTO-Comp/data`), `--seed 42`, `--n-eval-parts 10`, `--n-sub-parts 100`, `--engine streaming`, `--force`. The session splits are never overwritten without `--force`, because the per-fold files downstream depend on them.

## Inputs

The local CV data is radek's [local validation dataset](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991), the host script applied to the 4th week. Chris uses the same dataset. The script symlinks the files in, and the paths are constants in `paths.py`:

| here (`paths.py`) | Benny's original | source in `../Kaggle-OTTO-Comp/data/` |
|---|---|---|
| `data/train.parquet` (`TRAIN`) | `data/train/interim/*.parquet`, all 4 weeks | `train.parquet` |
| `data/test.parquet` (`TEST`) | `data/test.parquet` | `test.parquet` |
| `preprocess/train.parquet` (`CV_TRAIN`) | `preprocess/train/interim/*.parquet`, weeks 1–3 | `valid_train.parquet` |
| `preprocess/test.parquet` (`CV_TEST` = `XGB_TRAIN_X`) | `preprocess/test.parquet`, `data_tmp/xgb_train_x.parquet` | `valid_test.parquet` |
| `data/test_full.parquet` (`TEST_FULL`) | none; Benny didn't have it | `test_full.parquet` |

Event schema: `session` Int32, `aid` Int32, `ts` Datetime(ms), `type` UInt8 (codes in `event_types.py`: 0 clicks, 1 carts, 2 orders). Benny's data had `ts` in epoch milliseconds and `type` as strings.

## Outputs

- `preprocess/test_labels.parquet` (`CV_TEST_LABELS` = `XGB_TRAIN_Y`): `valid_test_labels.parquet` exploded to Benny's long format, one row per (`session` Int32, `aid` Int32, `type` UInt8). It has 2,650,372 rows.
- `data/test_labels.parquet` (`TEST_LABELS`): labels for the Kaggle test sessions, built from `TEST_FULL` by `labels_after_cut` and stored in the same long format. It has 2,414,212 rows. These were never available during the competition. They let the LB pipeline be scored locally.
- `data_tmp/sessions_eval.parquet` (`SESSIONS_EVAL`): (`session`, `part` 0–9, `fold` = `part // 2`). This replaces `sessions_eval.pickle`, where fold *k* was `sessions[2k] + sessions[2k+1]`. It covers all 1,801,251 week-4 sessions, because every session has at least one label.
- `data_tmp/split/sub_sessions.parquet` (`SUB_SESSIONS`): (`session`, `part` 0–99) for the Kaggle test sessions. This replaces `split/sub_sessoins_10.pickle`.

The splits are seeded, while Benny's were not, so the fold assignment differs from his.

## How week 4 is used

For fold *k*, Benny's feature notebooks read the **full 4-week train**, drop the fold's held-out sessions, and add those sessions back in truncated form from `XGB_TRAIN_X`. So 20% of week 4 is truncated and 80% is untruncated. That only works if each truncated session is a prefix of the same session in the full train. `--check` verifies this, and on the current data it finds 0 mismatches.

## The `test_full` files

A `test_full` file is the labelled version of a test file: the same sessions, untruncated. The test file is the start of each session, the part a model sees, and the events after the cut are what it has to predict. There are two:

- **Local CV, week 4**: Benny's `preprocess/test_full.parquet` (the host script's `test_sessions_full`) is the untruncated `CV_TEST`. Nothing downstream reads it, and `CV_TEST_LABELS` already holds its labels, so it isn't produced. If it's needed, it is `TRAIN` restricted to the sessions in `CV_TEST`.
- **Kaggle test, week 5**: `../Kaggle-OTTO-Comp/data/test_full.parquet` is the untruncated `TEST`. It has 13.85M events against 6.93M, and it was published after the competition. Each `TEST` session is a prefix of it: there are 0 mismatches once its timestamps are truncated to whole seconds, and every session has events after the cut. Benny didn't have this file. It is linked as `TEST_FULL`, and `TEST_LABELS` holds its labels. It is stored with Int64 `session`/`aid` and millisecond `ts`. Read it with `preprocess.scan_test_full()`, which casts it to the usual dtypes.

`labels_after_cut` follows the host script's rules: clicks is the first click after the cut, and carts and orders are the sets of aids carted and ordered after it. `--check` rebuilds the week-4 labels from `TRAIN` and `CV_TEST` with it and compares them with `valid_test_labels`. They match exactly, with 0 differing rows.
