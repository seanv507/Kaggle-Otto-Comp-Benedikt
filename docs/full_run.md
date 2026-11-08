# Full run on the A100: steps, timings, costs and metrics

The first full run of the Polars pipeline (`run_a100.sh`): all five week-4 folds, then the three rankers and the submission.

**Status (2026-11-08 11:30 UTC):**
- Done: all candidate sources, features and the baseline, plus the clicks ranker.
- Running: the carts ranker.
- Not started: the orders ranker and `submission.csv`.

Rows marked *pending* get filled in when the run ends.

## The run

| | |
|---|---|
| Pod | Runpod `benny-full` (`4unmtt3ry1a6ib`), Secure Cloud, US-MD-1 |
| Hardware | 1× A100-SXM4-80GB, 250 GB RAM, a quota of 14 CPUs, an 800 GB volume |
| Price | **$1.79/hr** for the GPU. The volume bills separately (its rate wasn't in the catalog read). |
| Code | branch `polars-gru`. It started at `d3ef945`. The fixes below brought it to `7c48e22` from the GRU step on. |
| Started | 2026-11-07 18:54 UTC |
| Inputs | `seanv507/kaggle-otto-multi-objective` on HF: train, test, `test_full`, `local_valid/*` |
| Outputs | HF `benny_full/`: the logs, `submission.csv`, and each target's `sub_top20.parquet` |

The run is resumable. Each step leaves a marker in `/workspace/done`, so a restart after a failure skips the finished steps (see the script's header).

## Timeline and cost per step

The minutes are the steps' own times, from their ntfy messages. The cost is the minutes at $1.79/hr.

| step | what it does | per fold | 5 folds / total | cost |
|---|---|---|---|---|
| `code`, `inputs`, `preprocess` | clone and `uv sync` (~2 min), HF download (1 min), links, labels, splits and checks (~1.5 min) | | ~5 min | $0.15 |
| `t4r` | install the earlier A100 run's transformer candidates as the shared `t4r` source | | 2 min | $0.06 |
| `other` (`01f`) | `pv` and `pv_2` candidates, session and item features (shared by all folds) | | 1 min | $0.03 |
| `benny_covisits_k` (`01b`) | Benny's 6 matrices, his 7 candidate sources | 23–24 min | 117 min | $3.49 |
| `chris_covisits_k` (`01c`) | Chris's 10 matrices on the GPU, in session chunks, and the `chris__<i>` features | 17–18 min | 86 min | $2.57 |
| `word2vec_k` (`01a`) | 50-d gensim embeddings, 5 epochs | 15–16 min | 78 min | $2.33 |
| `gru_k` (`01d`) | 6 GRU models (up to 30,000 steps each), scoring all 1.86M items for each session | 54–55 min | 272 min | $8.11 |
| **one fold, all four steps** | | **~1 h 51 min** | **~9 h 13 min** | **$16.50** |
| `baseline` | Chris's covisit baseline (fills short sessions) | | 8 min | $0.24 |
| `clicks` ranker | split (1.6 h), CV with 10 models (~1 h), 5 submission models (~2.2 h) | | 289 min | $8.62 |
| `carts` ranker | split, CV, 2 submission models with 10 seeds | | *pending* | |
| `orders` ranker | 2 splits, 2 stacked models, CV, 1 submission model | | *pending* | |
| `submission` | `submission.csv` and its score on the Kaggle test labels | | *pending* | |

**Elapsed:** the sources finished at 04:48 UTC (9 h 54 min in, including the restarts below), the baseline at 04:56, and clicks at 09:46. That's about $29 by 11:30 UTC. The estimate for the whole run is **$48–53**, finishing around 21:00–24:00 UTC on 2026-11-08.

**Not included:** the transformer itself (`01e`). It was trained in an earlier, separate A100 run (2026-11-06): 38.5k steps at 9.3 it/s, about 70 minutes of training. Its candidates are on HF in `benny_t4r/`.

## Failures on the way, and their fixes

The 2% sample hadn't exposed three problems. Each cost a restart, about 25 minutes (~$0.75) in total. The resume markers meant no finished work was redone.

| step | symptom | cause | fix |
|---|---|---|---|
| `chris_covisits_0` | cuDF `OverflowError: … exceeds the column size limit` | unchunked pair self-joins exceed cuDF's 2³¹-row column limit on the full data | `7cd5241`: each matrix uses the sibling's A100 session-chunk sizes (115: 4M, 217/220/232/235: 250k, 239: 2M, 93: 2M) |
| `gru_0` | Polars panic: "maximum length reached" | before slicing, each target's row carries the session's 59-event lists: ~141M rows × 59 values, past Polars' 2³² limit | `3d2aad3`: training examples built in chunks of 2M session ids |
| `gru_0` | GPU idle, under 3 steps/s | each batch was a random gather from the 126M-row Polars frame of list columns | `7c48e22`: padded NumPy arrays built once per model; batches are NumPy indexing. **138 steps/s.** |

A fourth fix, `af59b17`, caps Word2Vec's threads at the container's CPU quota (14 rather than the host's 128). It made no measurable difference: Word2Vec took 16 min before it and 15–16 min after.

## Metrics

Unless noted, the figures are recall@20 per target on week 4's held-out sessions ("train"). They're from fold 3; folds 0–4 agree to about ±0.005. For the shared sources, "sub" is the Kaggle test, scored against `TEST_LABELS`.

### Candidate sources

Each source is scored on its own, ranked by its main weight.

| source | step | rows (train, one fold) | clicks | carts | orders |
|---|---|---|---|---|---|
| `t4r` (Kaggle test) | 01e | 83.6M | 0.500 | 0.400 | 0.543 |
| `t4r` (week 4) | 01e | 90.1M | 0.503 | 0.392 | 0.524 |
| `pv`, by `pv_wgt_1` | 01f | 5.5M | 0.322 | 0.308 | 0.589 |
| `pv_2`, by `pv_2_wgt_1` | 01f | 5.5M | 0.322 | 0.308 | 0.591 |
| `caco_15` | 01b | 12.8M | 0.270 | 0.194 | 0.258 |
| `b2b_15` | 01b | 12.9M | 0.167 | 0.166 | 0.223 |
| `caco_b2b_15` | 01b | 0 | — | — | — |
| `clicks_20` | 01b | 16.3M | 0.282 | 0.181 | 0.201 |
| `weighted_30` | 01b | 27.2M | 0.273 | 0.185 | 0.207 |
| `weighted_1_2_20` | 01b | 0.94M | 0.040 | 0.088 | 0.402 |
| `chris__3` (226) | 01c | 47.9M | 0.149 | 0.150 | 0.202 |
| `chris__4` (232) | 01c | 61.9M | 0.283 | 0.181 | 0.196 |
| `chris__5` (235) | 01c | 61.9M | 0.279 | 0.177 | 0.188 |
| `chris__6` (239) | 01c | 16.8M | 0.436 | 0.379 | 0.554 |
| `chris__7` (700) | 01c | 0 | — | — | — |
| `chris__8` (701) | 01c | 27.4M | 0.228 | 0.166 | 0.203 |

**Notes:**
- **Empty sources:** `caco_b2b_15` and `chris__7` (matrix 700) are empty in Benny's code too; both bugs are kept on purpose (see the `01b` and `01c` READMEs).
- **`int16` wraparound:** the Benny covisit sums also wrap in Benny's code, which uses `int16`. For example, `caco_15` has ~3,500 wrapped sums per fold on train.
- **Missing rows:** the ntfy messages carry only each step's last 12 lines, so `c2o_15` and `chris__0`, `chris__1`, `chris__2` and `chris__9` (115, 217, 220, 93) aren't captured here. They're in the step logs uploaded to HF (`benny_full/logs/`).

### GRU (`gru_scores_20`)

Each model's recall@20 is for its own target, on each fold's held-out sessions.

| model | target | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | time per fold |
|---|---|---|---|---|---|---|---|
| `gru_F_cl` | clicks | 0.439 | 0.439 | 0.440 | 0.440 | 0.438 | 19 min |
| `gru_T_cl` | clicks | 0.375 | 0.375 | 0.371 | 0.380 | 0.373 | 10 min |
| `gru_F_ca` | carts | 0.272 | 0.272 | 0.269 | 0.273 | 0.270 | 7 min |
| `gru_T_ca` | carts | 0.262 | 0.261 | 0.260 | 0.265 | 0.260 | 3 min |
| `gru_F_or` | orders | 0.479 | 0.472 | 0.480 | 0.479 | 0.478 | 5.5 min |
| `gru_T_or` | orders | 0.327 | 0.320 | 0.328 | 0.330 | 0.331 | 2.3 min |

**Notes:**
- **The two variants:** `F` uses random item embeddings and `T` is initialised from the fold's Word2Vec vectors, frozen. The random ones score higher in every case.
- **Training size:** fold 0 has 141M training examples, 126M of them clicks targets. The clicks models stop at 30,000 steps (61M examples); the carts and orders models see all of theirs.

### Word2Vec

No recall at full size. On 5% of fold 0, an aid's mean cosine to its top `caco` partner was 0.88, against 0.47 for a random aid (`01a` README).

### Baseline (Chris's public notebook)

| clicks | carts | orders | Kaggle test score |
|---|---|---|---|
| 0.530 | 0.421 | 0.660 | **0.5754** (Chris's public LB: 0.575) |

These are on the Kaggle test (`TEST_LABELS`). The score is 0.1 × clicks + 0.3 × carts + 0.6 × orders.

### Rankers

| target | CV recall@20 (week 4, 5 folds) | Kaggle test recall@20 | notes |
|---|---|---|---|
| clicks | **0.5660** (chunks 0.5649–0.5683) | **0.5600** | 3 seeds; mean best iteration 451; 31.6M training rows per CV model |
| carts | *pending* | *pending* | |
| orders | *pending* | *pending* | stacked click and cart scores |
| **score** (0.1 / 0.3 / 0.6) | *pending* | *pending* | |

For comparison:

| | clicks | carts | orders | score |
|---|---|---|---|---|
| Baseline, Kaggle test (above) | 0.530 | 0.421 | 0.660 | 0.575 |
| This pipeline, 2% sample, 50 trees, Kaggle test | 0.463 | 0.396 | 0.643 | 0.551 |
| Chris's part, the sibling's `otto-valid-full` run (week 4) | 0.558 | 0.440 | 0.669 | 0.589 |
| Benny's write-up: CV / LB | | | | 0.593 / 0.601 |

The sibling run's clicks rankers took 3.7 h, carts 1.1 h and orders 0.5 h.

## Lessons for the next run

- **Test at full size before the pod run.** All three failures were size limits that a 2% sample can't reach: cuDF's 2³¹ rows, Polars' 2³² values, and the cost of gathering from a frame of 126M rows. A one-fold smoke test on the full data, or a check of intermediate sizes, would have caught them for a few dollars.
- **The rankers are the long pole.** The sources take ~9 h for 5 folds; the three rankers take longer, mostly scoring the 10 sub chunks of each fold with every model and seed.
- **Sizing:** 250 GB of RAM and 80 GB of GPU memory were enough everywhere once the code chunked. The 14-CPU quota limits the Polars steps more than the GPU does.
- **Notifications:** watch the pod's ntfy topic directly in the ntfy app. A watcher on a laptop lapses whenever its session is idle.
