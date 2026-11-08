# 3rd Place Solution - Kaggle-OTTO-Comp - Benny's Part

[Otto's Kaggle competition](https://www.kaggle.com/competitions/otto-recommender-system/) was a session-based, multi-task recommender system problem. Otto provided 12M sessions over 4 weeks with clicks, add to carts and orders events for training. Our team ([Chris](https://www.kaggle.com/cdeotte), [Giba](https://www.kaggle.com/titericz) and [Theo](https://www.kaggle.com/theoviel) won the 3rd place and achieved a score of 0.60437 on public LB (private LB 0.60382). Our solution is an ensemble of our individual models. 

My model achieved public LB 0.60102 and private LB 0.6003. This repository contains the code to run my model end-to-end.

I shared [my approach on kaggle write-up](https://www.kaggle.com/competitions/otto-recommender-system/discussion/386497). You can find my teammate's solution: [Chris](https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013) and [Theo](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382975).

## This fork

This is a fork of Benny's code. The cuDF notebooks are replaced with Polars Python scripts: one per step, next to the notebook it replaced, with tests and a README of its own. The transformer and the GRU are reimplemented in plain PyTorch. Each step's README lists where it differs from Benny's code. Benny's own text below is kept as he wrote it, apart from the run instructions.

Chris's part of the solution is being converted the same way in the sibling repo [`Kaggle-OTTO-Comp`](https://github.com/seanv507/Kaggle-OTTO-Comp). This repo uses its covisitation code as a uv path dependency, so it needs that repo checked out next to this one.

## Model Summary

I provide a detailed description [here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/386497).

![model](./img/model.png)

My final model is a similar pipeline as proposed by Chris and described by many top solution. I will briefly summarize my solution:

Pipeline for all targets:<br>
1. Feature Engineering<br>

Pipelines per target:<br>
2. Generating Candidates<br>
3. Add Features/Scores for session x candidate pairs<br>
4. Training a XGBoost model<br>

![model](./img/datasplit.png)

I splitted my dataset as proposed by Radek and truncated the 4th week. A difference is, that I splitted the 4th week into 5 chunks and comobine only 1 chunk of the truncated sessions of week 4 with the remaining untruncated sessions of week 4. Therefore, some of my pipeline has to be executed 5 times!

## Files Inventory

```
├── 00_Preprocess          # input links, labels, the week-4 folds and the Kaggle-test session parts
├── 00_submissions         # 01_baseline: Chris's public baseline (fills short sessions)
├── 01a_FE_Word2Vec        # Word2Vec item embeddings, per fold
├── 01b_FE_CoVisit_1       # Benny's covisitation matrices and their candidates, per fold
├── 01c_FE_CoVisit_2_Chris # Chris's 10 matrices (from the sibling repo) and their features, per fold
├── 01d_FE_GRU             # 6 GRU models' next-item scores, per fold
├── 01e_FE_Transformer     # the transformer's candidates (trained once, shared by the folds)
├── 01f_FE_other           # previous-visit candidates, session and item features (shared)
├── 02_Clicks, 03_Carts, 04_Orders  # READMEs only: the rankers are in ranking/
├── ranking                # the three rankers as configurations of one package
├── folds.py               # fold inputs and the candidate-file layout, used by every step
├── paths.py, metrics.py, session_ops.py, event_types.py
├── run_a100.sh            # the whole pipeline on a cloud GPU pod
├── docs                   # the conversion plan and the full run's timings and metrics
├── data, preprocess, data_tmp, data_folds  # inputs, intermediate files and outputs (git-ignored)
└── img
```

## How to run the code

The environment is managed with [uv](https://docs.astral.sh/uv/). Run everything from the repo root:

```bash
uv sync --extra cu132          # CUDA 13.2 torch on a GPU machine; --extra cpu otherwise
uv run pytest                  # tests (the GPU engine's are skipped without a GPU)
```

**The whole pipeline** runs from `run_a100.sh`: preprocessing, every candidate source and feature for all 5 folds, the three rankers, and `submission.csv` with its score on the Kaggle test labels. It's written for a Runpod pod (A100 80 GB); its header has the start command and the pod's requirements. It resumes after a failure from the last step that finished. [`docs/full_run.md`](docs/full_run.md) has the first full run's timings, costs and metrics: about 9 h and $17 for the sources of 5 folds.

**Locally, on a sample:** every step takes `--sample <fraction>` and writes to `data_folds/sample_<fraction>/`, so it never overwrites the full outputs:

```bash
LOCAL=1 SAMPLE=0.02 FOLDS="0 1" GRU_ARGS="--max-steps 300 --width 32 --eval-batch-size 64" \
    RANKER_ARGS="--trees 50" WORK=/tmp/otto bash run_a100.sh
```

Or one step at a time, in this order. Each step's README has the details.

1. [`00_Preprocess/preprocess.py`](00_Preprocess/README.md): links the inputs from the sibling repo's data, and builds the labels and session splits.
2. Per-fold and shared sources and features. These are independent, except that the GRU needs Word2Vec:
   - [`01e_FE_Transformer/t4r_fetch.py`](01e_FE_Transformer/README.md): the transformer's candidates, from the HF dataset (or train it with `t4r_train.py`);
   - [`01f_FE_other/`](01f_FE_other/README.md): `previous_visits.py`, `session_features.py`, `item_features.py`;
   - [`01b_FE_CoVisit_1/benny_covisits.py`](01b_FE_CoVisit_1/README.md) and [`01c_FE_CoVisit_2_Chris/chris_covisits.py`](01c_FE_CoVisit_2_Chris/README.md);
   - [`01a_FE_Word2Vec/word2vec.py`](01a_FE_Word2Vec/README.md), then [`01d_FE_GRU/gru_train.py`](01d_FE_GRU/README.md).
3. [`00_submissions/01_baseline/baseline.py`](00_submissions/01_baseline/README.md): Chris's baseline.
4. [`ranking/`](ranking/README.md): `uv run python -m ranking split|cv|sub|stack ...` for clicks, carts and orders, then `submission`.

The original notebooks are in the git history (before this fork's conversion commits).
