# Plan: converting the rest of Benny's pipeline to Polars

Status on 2026-11-07:
- **Done, on master:** `00_Preprocess`, and `01e_FE_Transformer`. The full A100 run of `01e` beat Benny's recall@20 (week 4: 0.503/0.392/0.524 against his 0.455/0.331/0.491).
- **Step 1 (`folds.py`):** done on branch `polars-folds`.
- **Step 2 (`01f` previous visits and session features):** done on branch `polars-simple-features`, which is based on `polars-folds`.
- **Step 3 (`01f` item activity):** done on branch `polars-item-features`, which is based on `polars-simple-features`.
- **Step 4 (`01b` Benny's covisits):** done on branch `polars-benny-covisits`, which is based on `polars-covisits`. It's checked on a 2% sample; the full run is for the A100.
- **Step 6 (`01a` Word2Vec):** done on branch `polars-word2vec`, which is based on `polars-benny-covisits`. That branch also has `t4r_fetch.py`, which installed the A100 `t4r` candidates locally.
- **Step 8 (clicks ranker):** done on branch `polars-clicks-ranker`, which is based on `polars-word2vec`. The baseline fill came with step 11. It's checked on a 2% sample of folds 0 and 1.
- **Step 9 (Word2Vec features):** done for clicks on branch `polars-emb-features`, which is based on `polars-clicks-ranker`.
- **Step 11 (baseline):** done on branch `polars-baseline`, which is based on `polars-emb-features`. The clicks combine now fills short sessions from it.
- **Step 10 (carts and orders rankers):** done on branch `polars-carts-orders`, which is based on `polars-baseline`. The whole pipeline now runs end to end on the 2% sample, ending in `submission.csv` (Kaggle-test score 0.551, against 0.526 for the baseline).
- **Step 5 (`01c` Chris's covisits):** done on branch `polars-covisits`, which is based on `polars-item-features`. It's checked on a 2% sample; the full run is for the A100.

How the work is done:
- Each step is developed and tested locally on its own branch, on a sample of the sessions.
- The A100 runs are done only at the end, once every step is ready. Everything that can be done before that is done first (see "A100 runs at the end").

## What's left, and how it fits together

```text
00_Preprocess ─┬─> 01a Word2Vec (per fold) ──> 01d GRU (per fold)
               ├─> 01b Benny's covisits (per fold)        ─┐
               ├─> 01c Chris's covisits (per fold)        ─┤
               ├─> 01e Transformer (shared)               ─┤  candidate files
               ├─> 01f previous visits (shared)           ─┤  data_folds/fold_k/candidates/{train,sub}/<source>/cand.parquet
               ├─> 01f item activity features (shared)    ─┤  + feature files
               └─> 01f session features (shared)          ─┘
                                                            │
00_submissions baseline (fills sessions with < 20 recs)     │
                                                            v
   02_Clicks / 03_Carts:  01 merge -> 02 add word2vec features -> 03 XGBoost -> 04 combine folds and bags
   04_Orders: the same, plus click and cart model scores as order features
```

Most steps are independent because they only meet through files with a fixed layout:
- **Candidate sources** (`01b`, `01c`, `01d`, `01e`, `01f` previous visits) each write `(session, cand, <prefix>_*)` for the truncated week-4 sessions (`train`) and the Kaggle test sessions (`sub`).
- **Feature files** (`01f` 95/96, `01a` embeddings) are keyed by `cand`, `session` or `aid`. The `01f` ones are shared by the folds: `data_folds/shared/{session,item,item_day}_features_{train,sub}.parquet`.
- **The rankers** read whatever candidate files exist.

So each source can be built and tested alone. The rankers can then be built against a few sources and pick up the others as they're added.

## How each step is tested

1. **Unit tests** on small hand-checked frames with the real dtypes. These are parametrized over the Polars engines, as in `00_Preprocess` and `01e`.
2. **Comparison with a reference port.** A pandas port of the notebook's logic runs on random sessions, and the output must be equal. This is the pattern of the sibling repo's `covisits_reference.py`. It's needed wherever the logic is hard to check by hand (covisits, merges).
3. **Candidate recall.** Every candidate source can be scored on its own: recall@20 and @N of its candidates against `XGB_TRAIN_Y` (week 4) and `TEST_LABELS` (Kaggle test). This needs no ranker. It's a cheap check that a source is sensible, and it catches order or join bugs that unit tests miss.
4. **End-to-end score** against Benny's ablation table: CV 0.592, LB 0.598 for 5 folds before Word2Vec. With `TEST_LABELS`, the "LB" can now be scored locally.

Shared code goes in one place and isn't copied per notebook:
- the fold inputs;
- the "matrix → candidates" aggregation;
- the per-target ranker pipeline. `02_Clicks`, `03_Carts` and `04_Orders` are near-copies of each other.

## Steps

Each step is a branch. It ends with tests passing and a short README in its directory that records what differs from Benny's code.

### 1. Shared fold helpers (foundation) — done
A `folds.py` module at the repo root, so no notebook has to rebuild the inputs by hand:
- `fe_events(k)`: the events for fold *k*'s feature engineering. That's `TRAIN` without fold *k*'s held-out sessions, plus those sessions truncated from `XGB_TRAIN_X`, plus `TEST`. This replaces the `files_split` / `isin(sess_eval)` code at the top of nearly every `01*` notebook.
- `heldout_sessions(k)`, and the `train` / `sub` session sets that candidates are generated for.
- `write_candidates(df, k, dataset, source)` and `read_candidates(...)`: they write and read the candidate-file layout and check its schema.

**Test:** synthetic events with fold *k* checked by hand. There's also a data check: across all *k*, every week-4 session appears truncated exactly once and untruncated four times.

As built:
- Sources that don't depend on the fold (`t4r`, previous visits) are written once to `data_folds/shared/candidates/...` rather than copied to each fold. `candidate_paths(k, dataset)` lists the fold's and the shared files, and `read_candidates` filters a shared `train` file to the fold.
- `sample_sessions(frame, fraction, seed)` keeps the same sessions in every file, for local development. Every step's CLI should take `--sample`.
- `uv run python folds.py --check` checks the folds on the real data (4 s).
- `01e` now writes its candidates through `write_candidates` as the shared source `t4r`.

### 2. `01f` simple features (no fold dependence) — done
- `04_PreviousVisits*.ipynb` (`pv`, `pv_2`): history items as candidates, with recency and type weights.
- `96_Session_Features.ipynb`: session length, day, time, and so on.

**Test:** hand-checked frames. For `pv`, its candidate recall on week 4.

As built: see `01f_FE_other/README.md`.
- Recall@20 of `pv` on week 4 is 0.322/0.308/0.589.
- Both scripts run on the full data locally in under a minute, since they only read the week-4 and test events, so their outputs exist locally already.

### 3. `01f` item activity features: `95_Cand_Features.ipynb` — done
Clicks, carts and orders per item on the same day and over the previous 1, 2 and 3 weeks. There are two files, `*_cand_meta_comb` and `*_cand_day_meta_comb`. Neither depends on the fold: Benny copied `*_cand_meta_comb` to every fold.
**Test:** hand-checked counts over a few days. The week windows need care, because `ts` is now a Datetime rather than epoch milliseconds.

As built: `01f_FE_other/item_features.py`, with shared files `item_features_*` and `item_day_features_*`. On the real data:
- the window totals equal direct counts of `TRAIN`;
- `day_` (`folds.day_index`) matches Benny's tables;
- Benny's doubled `p0w` counts are kept.

### 4. `01b` Benny's covisitation matrices (6 notebooks) — done
These are `c2o`, carts-orders, buy2buy, clicks, weighted clicks and weighted types. Each one builds a matrix from `fe_events(k)`, then candidates from a shared `get_candidates`. That function sums the matrix weights over the session's items into `<prefix>_num/rank_min/wgt/wgt_2/wgt_cl/wgt_ca/wgt_or`. Reuse `build_covisits` / `top_covisit` from the sibling repo's `covisits.py` (see step 5).

**Test:**
- each matrix against a pandas reference port on random sessions;
- `get_candidates` on a hand-checked example;
- candidate recall per source.

As built: `01b_FE_CoVisit_1/benny_covisits.py` (see its README).
- **Not Chris's matrices:** they're Benny's own variants. For example, his buy2buy takes the last 30 events with a 14-day window, while Chris's 90 has no last-30 limit. So they're implemented here with the sibling's helpers rather than its `make_covisits_*`.
- **Reference ports:** every matrix and candidate function is checked against a pandas port of Benny's code.
- **Kept bugs:** `caco_b2b` is empty, and the int16 casts wrap around.

### 5. `01c` Chris's covisitation matrices, Benny's way — done
All 10 matrices that Benny uses (93, 115, 217, 220, 226, 232, 235, 239, 700, 701) are already ported in `../Kaggle-OTTO-Comp/train/covisit_matrices/covisits.py`, with parity tests. The work here:
- **Share the code.** Decide how this repo uses the sibling's `covisits.py`. A uv path dependency is better than a copy.
- **Check the definitions.** Diff Benny's `03_CoMatrix-99-Chris-gpu-*.ipynb` against Chris's notebooks to confirm they are the same, and note any changes.
- **Port the apply step.** Port `04_CoMatrix-99-Chris-apply.ipynb`. It splits sessions at 2-hour gaps and produces `chris__<i>__{wgt,dummy,t_*,d_*,l_*}` features. It reads hard-coded `/otto/data_radek/...` paths and `sessions[0]+sessions[1]`, so check that it really loops over folds.

**Test:** the apply step against a reference port, then candidate recall.

As built: `01c_FE_CoVisit_2_Chris/chris_covisits.py` (see its README).
- **Definitions:** Benny's pair logic is Chris's, line for line. Benny differs only in the top kept per aid (700: none, so it's empty) and in intermediate pruning, which isn't reproducible.
- **Apply step:** it loops over all 5 folds. Its sorted file order numbers the matrices (`chris__0` is 115).
- **Reference test:** passes against Benny's code run on pandas.
- **For the A100:** on the 2% sample, `sub` has up to 4M rows per matrix per fold, so about 200M on the full data. That's 10 matrices × 5 folds × 2 datasets of such files, so size the pod's disk and RAM for it.

### 6. `01a` Word2Vec, per fold — done
gensim on `fe_events(k)` sessions; the output is the item embeddings. gensim is CPU-only and pip-installable, so this step only needs its input changed.
**Test:** a small corpus with `workers=1` and a fixed seed for determinism, plus a sanity check that frequently co-visited items are close.

As built: `01a_FE_Word2Vec/word2vec.py`, which writes parquet (`aid`, `emb`) per fold. On 5% of fold 0, the mean cosine to an aid's top `caco` partner is 0.88, against 0.47 to a random aid.

### 7. `01d` GRU, per fold (needs step 6) — done
`train_gru.py` and `to_sparse.py` use the Merlin dataloader with torch. Port them the way `01e` was ported: a plain PyTorch model, a dataloader built from Polars, and the Word2Vec weights as initialisation. Its candidates are `gru_scores_20`.
**Test:** the same pattern as `01e`, with a tiny model that must learn a pattern and recall@20 tracked against any numbers logged in Benny's notebooks.

As built: `01d_FE_GRU/` (`gru_data.py`, `gru_model.py`, `gru_train.py`; see its README).
- **Plain PyTorch:** Benny's 6 configurations, with the examples built in memory with Polars list slices. Benny's notebooks log no recall to track against.
- **Local check:** 2% of fold 0, 300 steps, width 32: about 5 minutes for all 6 models, and it writes `gru_scores_20` for `train` and `sub`. Recall is low at that size, so it only checks the pipeline. The real numbers come from the A100.
- **For the A100:** about 150M training examples per fold, held in RAM.

### 8. Thin end-to-end slice for clicks (start this early) — done
Build `02_Clicks` against only the sources that exist at that point: `pv`, `t4r`, one covisit, and the session and item features. The parts are:
- **`01_Split`:** merge the candidate files, keep the top `no_cands` per source, add features and labels, and write chunks.
- **`03_XGB`:** XGBoost rank or binary training on folds ≠ *k* and prediction for fold *k*, plus prediction for `sub`. `xgboost` has a CUDA build on PyPI.
- **`04_Combine-Bags`:** average over folds and seeds and fill sessions with fewer than 20 recs from the baseline.

Write these as one module parametrized by target, not three copies.
**Test:**
- the merge on hand-made candidate files;
- leakage checks: training rows never include fold *k*'s sessions;
- clicks recall@20 locally on week 4 and on the Kaggle test.

This gives a working pipeline and a number, and each later source can be judged by how much it adds.

As built: the `ranking/` package (`split`, `train`, `combine`) with `02_Clicks/split.py` and `02_Clicks/ranker.py` (see `02_Clicks/README.md`).
- It uses every source built so far.
- On a 2% sample: CV recall@20 is 0.475, and the Kaggle-test recall@20 is 0.463. Both are limited by the 2% covisit matrices: the candidates hold only 50% of the click labels.
- The 25 score files (5 folds × 5 models) and the combine follow Benny. There's no CSV yet.

### 9. `02_Split_2_Add_Emb_Features` (needs step 6) — done for clicks
Cosine similarity between the candidate and the session's items, aggregated. Benny used `pandarallel` row loops. In Polars it can be an `explode`, a join and a group-by, or a numpy batch matmul.
**Test:** a hand-checked cosine on small embeddings.

As built: `ranking/emb_features.py`, added in `ranking/split.py`, and checked against a pandas port of Benny's code. On the 2% sample, clicks CV recall@20 is 0.474 with these features and 0.475 without. The carts and orders versions of the notebook come with step 10.

### 10. Carts and orders rankers — done
- **Carts:** the step 8 module with `target="carts"`, plus its own candidate rules (more candidates, kept only if 2 or more sources agree).
- **Orders (`04_Orders`):** the hierarchical setup. The click and cart models are trained on order sessions (`03_XGB_Clicks`, `04_XGB_Carts`), and their scores become features for the order model (`05a`/`05b`). Then `06_Combine-Bags` writes the final submission.

**Test:** the same as step 8, plus a check that the hierarchical features exist for every row.

As built: every ranker is a configuration of the `ranking` package (`SPLITS`, `MODELS`; see `ranking/README.md`), run as `python -m ranking`. The clicks scripts of step 8 went into the same CLI.
- **Orders chain:** the click and cart models' mean scores are written into the later chunks as `xgb_c_score` and `xgb_ca_score`.
- **Results on 2%:** carts CV 0.402, orders CV 0.657.
- **Two of Benny's bugs are worked around and documented:** his `03_XGB_Clicks` seed count was undefined, and his orders combine read a column that was never written.

### 11. `00_submissions/01_baseline` — done
Chris's public LB 0.575 solution fills sessions with fewer than 20 recommendations. It's small: three covisits and a rule. Either port it, or replace it with the sibling repo's popular-item padding lists.

**Test:** its recall on the Kaggle test against Chris's published 0.575.

As built: `00_submissions/01_baseline/baseline.py`. Its matrices come from `benny_covisits` (the same pair rules), and Chris's two rules are checked against his own code.
- **Score:** 0.545 on a 5% sample, where the matrices are weaker. The full run, which should approach 0.575, needs the A100's RAM.
- **Fill:** `ranking/combine.fill` tops up the rankers' short sessions with it, as Benny's `prep_xgb` did.

## Suggested order

| order | step | why then |
|---|---|---|
| 1 | 1 folds | everything per fold depends on it |
| 2 | 2, 3 simple features | quick, and they make step 8 possible |
| 3 | 8 thin clicks slice | gives an end-to-end score early |
| 4 | 5 Chris's covisits | biggest candidate source; mostly reuse |
| 5 | 4 Benny's covisits | the same pattern |
| 6 | 6 Word2Vec, then 9 embedding features | Benny's ablation: +0.001 |
| 7 | 10 carts and orders | needs the full set of sources to be meaningful |
| 8 | 7 GRU | most work for one feature |
| 9 | 11 baseline | only needed for the final submission |

## Decisions (2026-11-07)
- **Sharing code with the sibling repo:** an editable uv path dependency on `../Kaggle-OTTO-Comp` (package `otto`). Done on branch `polars-covisits`, together with the sibling's branch `importable-covisits`, which adds `train/covisit_matrices` to its hatch `dev-mode-dirs`, so `import covisits` works here. `run_a100.sh` clones the sibling next to this repo.
- **How faithful to be:** match Benny bit for bit where it's cheap, and document deliberate differences in each step's README, as in `01e`. One example is the length-1 sessions that Benny gave no T4R candidates.
- **Compute:** five folds of covisits, Word2Vec and GRU need the A100. Every step takes `--sample` (and chunk parameters where memory needs them) so that it can be developed on the local GPU.

## A100 runs at the end
Done locally first, for every step:
- the unit tests and reference-port tests;
- a run on a sample (`--sample 0.02` or so) of fold 0, with candidate recall reported;
- a check of the step's peak memory on the sample, to scale up to the full data.

Left for the A100, in one script like `01e/run_a100.sh`:
- every per-fold step on the full data for all five folds;
- the rankers on the full candidate sets;
- the final scores (CV on week 4, "LB" against `TEST_LABELS`).

`01e` was the exception: its full run is already done, and its candidates are on HF in `benny_t4r/`. Local work on the rankers (step 8) can download them instead of using the small local model's candidates.
