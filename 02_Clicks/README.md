# Clicks ranker

This is Benny's clicks reranker. The steps are thin scripts over the shared `ranking/` package, which the carts and orders rankers will reuse.

```bash
uv run python 02_Clicks/split.py --sample 0.02 --folds 0 1                     # candidates + features
uv run python 02_Clicks/ranker.py cv --sample 0.02 --folds 0 1                 # local CV on week 4
uv run python 02_Clicks/ranker.py sub --sample 0.02 --folds 0 1 --trees 50     # submission models + top 20
uv run pytest ranking
```

They need every candidate source and feature file of the same `--sample`:
- `01b`, `01c` and `01a` per fold;
- `01f` shared;
- the shared `t4r` source, from `01e_FE_Transformer/t4r_fetch.py`, which copies the A100 candidates (use `--sample` for a sample's copy).

| file | replaces | does |
|---|---|---|
| `split.py` → `ranking/split.py` | `01_Split.ipynb` | Writes `data_folds/fold_<k>/ranker/clicks/{train_<i>,sub_<j>}.parquet`. |
| `ranker.py cv` → `ranking/train.py` | `03_XGB-v19-clicks.ipynb`, `bl_sub = False` | Local CV: recall@20 and the best number of trees. |
| `ranker.py sub` → `ranking/train.py`, `ranking/combine.py` | `03_XGB-v19-clicks.ipynb`, `bl_sub = True`; `04_Combine-Bags.ipynb` | One model per fold, the scores for every fold's `sub`, and `ranker/clicks/sub_top20.parquet` with its recall on the Kaggle test. |
| `02_Split_2_Add_Emb_Features-v2.ipynb` | | Word2Vec similarity features (not converted yet, conversion plan step 9). |

## What each step does

- **Split:**
  - **Candidates:** the union of the top 80 per session from `c2o_15`, `clicks_20`, `b2b_15`, `caco_15`, `pv`, `weighted_30` and `caco_b2b_15`.
  - **Features:** the columns of every candidate source of the fold, except those in Benny's ignore lists (`ranking/benny_columns.py`), with 0 where a source has no row. Then the item, session and same-day item features.
  - **Target:** the clicks label. `train` comes in Benny's 2 chunks per fold and `sub` in 10.
- **CV:** each chunk is predicted by models trained on the first 2 other chunks. 3 seeds, early stopping on the predicted chunk, at most 1,000 trees. Recall@20 is computed over the chunk's sessions.
- **Submission:**
  - **Models:** one per fold, trained on that fold's 2 chunks, sessions with a positive only, 3 seeds × 394 trees (Benny's number).
  - **Scoring:** every model scores every fold's `sub` features.
  - **Combine:** the seeds and pairs are summed onto the first pair's candidates (fold 0, model 0), and each session's top 20 is kept.
- **XGBoost:** `max_depth 8, learning_rate 0.1, subsample 0.8, colsample_bytree 0.3, binary:logistic`, on CUDA.

## Checks on a 2% sample of folds 0 and 1

| step | result | time |
|---|---|---|
| split | 77 candidates per session, 162 features; the candidates hold 50.4% of the click labels | 49 s |
| CV | recall@20 0.475; best iteration 44 on average | 41 s |
| submission, 50 trees | Kaggle test recall@20 0.463; 5,896 of 33,436 sessions have fewer than 20 | 36 s |

These numbers come from covisit matrices built on 2% of the sessions, so the candidates are weak. The A100 `t4r` candidates alone score 0.50 here, but Benny's clicks ranker uses `t4r` only as features, not as candidates. The real comparison has to wait for the full sources.

## Differences from Benny's code

- **Kept as Benny had them:**
  - **CV:** early stopping on the predicted chunk, which makes the recall optimistic. Clicks trains on only 2 other chunks.
  - **Submission models** train only on sessions with a positive. Benny's CV doesn't filter clicks.
  - **Combine:** the first (fold, model) pair's candidates are the rows.
  - **`t4r_rank` is 0 for candidates without a `t4r` row**, which looks like t4r's best rank. That comes from the fill with 0, as with Benny's rank column, which he called `order`.
- **Ties:** at the top-80 cut, and in the final top 20, ties go to the smaller cand. Benny's went by cuDF's sort order.
- **Not done yet:**
  - **Short sessions aren't filled.** Benny topped them up from Chris's public 0.575 submission (step 11).
  - **No Word2Vec features yet** (step 9).
- **Outputs:** the scores and top 20 are under `data_folds/.../ranker/clicks/`, not `../00_submissions/<run name>/`. There's no CSV yet; the final submission comes with the orders ranker.
- **XGBoost 3:** `tree_method=hist, device=cuda`, which replaces `gpu_hist`.
