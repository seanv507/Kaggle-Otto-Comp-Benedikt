# Rankers

This package is Benny's three rankers (`02_Clicks`, `03_Carts`, `04_Orders`) as one implementation. Each of his notebooks is a configuration: a `SplitSpec` in `split.py` (`SPLITS`) or a `ModelSpec` in `train.py` (`MODELS`).

```bash
uv run python -m ranking split clicks --sample 0.02 --folds 0 1
uv run python -m ranking cv clicks --sample 0.02 --folds 0 1
uv run python -m ranking sub clicks --sample 0.02 --folds 0 1 --trees 50
uv run python -m ranking submission --sample 0.02
uv run pytest ranking
```

| pipeline | commands, in order |
|---|---|
| clicks (`02_Clicks`) | `split clicks`; `cv clicks`; `sub clicks` |
| carts (`03_Carts`) | `split carts`; `cv carts`; `sub carts` |
| orders (`04_Orders`) | `split orders/clicks`; `split orders/carts_orders`; `stack orders/clicks`; `stack orders/carts`; `cv orders`; `sub orders` |
| submission | `submission`, after the three |

They need every input of the same `--sample`:
- the candidate sources: `01b`, `01c`, `01e` (the shared `t4r` source, from `01e_FE_Transformer/t4r_fetch.py`) and `01f`;
- the shared `01f` feature files;
- the fold's Word2Vec embeddings (`01a`);
- for the fill, the baseline (`00_submissions/01_baseline`).

| module | replaces | does |
|---|---|---|
| `split.py`, `emb_features.py` | `01_Split*.ipynb`, `02_Split_2_Add_Emb_Features*.ipynb` | candidates and features: `data_folds/fold_<k>/ranker/<dir>/{train_<i>,sub_<j>}.parquet` |
| `train.py` | `03_XGB*`, `04_Orders/03-05` | local CV, submission models and their scores, and the stacked click and cart scores |
| `combine.py` | `04_Combine-Bags`, `06_Combine-Bags` | the summed scores' top 20, filled from the baseline: `ranker/<dir>/sub_top20.parquet` |
| `__main__.py` | `06_Combine-Bags` (CSV) | the CLI, and `submission.csv` with its weighted recall@20 on the Kaggle test |
| `benny_columns.py` | | Benny's ignore lists and ranking columns, copied as they are |

## The splits

| split | generating sources | top n | min. sources | train candidates from sessions with | train rows | ignored source |
|---|---|---|---|---|---|---|
| `clicks` | 7 covisit and `pv` sources | 80 | 1 | any labels | all sessions | |
| `carts` | every source but the GRU | 100 | 2 | carts or orders labels | sessions with carts labels | |
| `orders/clicks` | as `clicks` | 80 | 1 | any labels | all sessions (train only) | `chris__1__15` |
| `orders/carts_orders` | every source but the GRU | 80 | 2 | carts or orders labels | `orders/carts`: sessions with carts labels; `orders/orders`: with orders labels | `chris__1__15` |

Every split merges in:
- the columns of every candidate source except Benny's ignore lists, with 0 where a source has no row;
- the item, session and same-day item features;
- the Word2Vec similarities;
- for train, the target.

## The models

All use XGBoost with `max_depth 8, learning_rate 0.1, subsample 0.8, colsample_bytree 0.3, binary:logistic` on CUDA.

| model | seeds | trees | CV: trained on | submission models | left out | stacked as |
|---|---|---|---|---|---|---|
| `clicks` | 3 | 394 | the first 2 other chunks | 5 (one per fold) | | |
| `carts` | 10 | 377 | the first 5 other chunks | 2 (5 chunks each) | | |
| `orders/clicks` | 3 | 394 | | 5 | sessions with carts or orders labels | `xgb_c_score` into `orders/carts`, `orders/orders` and `orders/sub` |
| `orders/carts` | 10 | 308 | | 2 | sessions with orders labels | `xgb_ca_score` into `orders/orders` and `orders/sub` |
| `orders` | 10 (CV: 3) | 345 | all other chunks | 1 (every chunk) | | |

- **Training rows:** the submission and stacked models train on sessions with a positive (Benny's `bl_pos`). His CV did too for carts and orders, but not for clicks.
- **Stacked models:** they leave out the sessions the next model trains on. Their mean score over every model and seed becomes a feature in those files.
- **Combine:** each (fold features, model) pair's seeds are summed, then added onto the first pair's candidates. The top 20 is filled from the baseline.

## Checks on a 2% sample of folds 0 and 1

Trees: 50 instead of Benny's numbers, which suit the full data.

| target | candidates | CV recall@20 | Kaggle test recall@20 | baseline, same sample |
|---|---|---|---|---|
| clicks | 77 per session, 169 features | 0.474 | 0.463 | 0.431 |
| carts | 104 per session | 0.402 | 0.395 | 0.360 |
| orders | 96 per session, 2 stacked scores | 0.657 | 0.644 | 0.624 |
| **score** (0.1, 0.3, 0.6) | | | **0.551** | 0.526 |

The covisit matrices of the sample come from 2% of the sessions. The full sources should score much higher (Benny: CV 0.592, LB 0.598 before Word2Vec).

## Differences from Benny's code

- **Kept as Benny had them:**
  - **CV:** early stopping on the predicted chunk.
  - **Combine:** the first (fold, model) pair's candidates are the rows.
  - **`t4r_rank` is 0** where `t4r` has no row.
  - **Benny's ignore lists**, including names no source writes.
- **Seeds of `orders/clicks`:** his notebook used `no_seeds` before defining it, so its value came from an earlier kernel state. 3 is assumed, as in his clicks ranker.
- **Orders combine:** his reads a `score` column that his orders notebook never wrote (only `score_0`–`score_9`). Here every combine sums the seeds.
- **Ties:** at the top-n cut and in the top 20, ties go to the smaller cand. His went by cuDF's sort order.
- **Files:** the stacked scores are written into the chunk files, rather than into copies (`chunks_c`, `chunks_c_2`, `sub_c`, `sub_c_2`). The Word2Vec features are added while splitting. Outputs are under `data_folds/` instead of `../00_submissions/<run>/`.
- **GRU features:** `gru_scores_20` (conversion plan step 7) isn't built yet. The splits merge it as features once it exists, and never use it to generate candidates, as in Benny's code.
- **XGBoost 3:** `tree_method=hist, device=cuda`, which replaces `gpu_hist`.
