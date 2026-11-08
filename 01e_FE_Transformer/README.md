# Transformer next-item scores

A transformer trained on session sequences scores the next item. Its top 50 per session become candidates with a `t4r_score` feature for the rankers. This directory replaces Benny's Transformers4Rec notebooks with Polars and plain PyTorch. There's no Merlin, NVTabular or cuDF.

```bash
uv sync --extra cu132                                  # torch for CUDA 13.2; --extra cpu otherwise
uv run python 01e_FE_Transformer/t4r_data.py           # sequences (about 30 s, ~9 GB RAM)
uv run python 01e_FE_Transformer/t4r_train.py          # train, then write the candidates
uv run pytest 01e_FE_Transformer
```

For the full run on a cloud GPU, `T4R_TRAIN=1 bash run_a100.sh` (the root script) does all of the steps above, from the inputs on HF to uploading the candidates. Without `T4R_TRAIN=1`, the script installs the 2026-11-06 run's candidates with `t4r_fetch.py` instead. The script's header explains how to launch it as a Runpod start command. The full model needs a large GPU. The item table is 1.85M × 256, and with its gradients and Adam state that's about 8 GB before activations. On the local 4 GB GPU, run a smaller version, e.g. `--d-model 32 --n-head 4 --sample 0.02`.

To use the A100 run's candidates locally, run `uv run python 01e_FE_Transformer/t4r_fetch.py`. It downloads `benny_t4r/cand_{train,sub}.parquet` from the HF dataset (1.4 GB), checks them, installs them as the shared `t4r` source and prints their recall. It needs about 12 GB of RAM.

| file | replaces | does |
|---|---|---|
| `t4r_data.py` | `01_T4R_Preprocess.ipynb` | item ids, and the last 20 items of every session |
| `t4r_model.py` | T4R + `custom_t4r.py` | the model and its loss |
| `t4r_train.py` | `02_T4R_Train-v2.ipynb`, `03_Convert.ipynb` | training, recall@20, candidate files |
| `t4r_fetch.py` | | installs the A100 run's candidates from HF |

Benny's `custom_t4r.py` was deleted in commit `1a76a2e`. Read it with `git show 83e570e:01e_FE_Transformer/custom_t4r.py`.

## Data (`data_tmp/t4r/`)

- `aid_map.parquet`: (`aid`, `item`, `count`). Items are numbered 2, 3, … by descending event count over `CV_TRAIN`, `XGB_TRAIN_X` and `TEST`. 0 is padding and 1 is an unknown aid.
- `seq_{cv_train,xgb_train_x,test}.parquet`: (`session`, `item`, `type`, `ts`). These are the last 20 events of each session, oldest first, and `ts` is the time of the first kept event.
- `checkpoints/epoch_{e}.pt`: the model's `state_dict` after each epoch.

The candidates are the shared source `t4r` (see `folds.py`): `data_folds/shared/candidates/train/t4r/cand.parquet` (all week-4 sessions) and `.../sub/t4r/cand.parquet` (Kaggle test sessions). Their columns are (`session`, `cand`, `t4r_rank`, `t4r_score`). These are the top 50 per session, and `t4r_score` is a softmax over those 50. The model doesn't depend on the fold. Benny copied the same files to every `data_folds/fold_*/candidates/{train,sub}/t4r/cand.parquet`; `folds.read_candidates` filters the shared `train` file to the fold instead. `run_a100.sh` with `T4R_TRAIN=1` uploads them to HF as `benny_t4r/cand_train.parquet` and `benny_t4r/cand_sub.parquet`.

## Model

The model is the one Benny's final parameters build in T4R. His other options were switched off: `using_type`, `multi_task_emb`, `item_correction`, and type-weighted losses.

- **Layers:** item embedding (256), then `Linear(256, 256)`, then XLNet (3 layers, 32 heads, bidirectional, layer-norm eps 0.03). Scores come from the dot product with the item embeddings.
- **Training:** masked-item prediction. 35% of the items are masked, each session keeps at least one item masked and at least one unmasked, and masked items are replaced by a learned mask embedding.
- **Loss:** cross-entropy over the batch's masked items plus 4 random items per masked item, at temperature 1.57. Items in the session other than the target get logit 0 (Benny's `remove_false_neg`).
- **Prediction:** a mask is placed after the last item, every padding position also gets the mask embedding, and the hidden state at that position is scored against all items.
- **Schedule:** AdamW (lr 5e-4, no weight decay), batch 1024, and a cosine schedule with 1.5 cycles per epoch. There are 3 epochs on weeks 1–3, week 4 (truncated) and the Kaggle test, then 1 epoch on week 4 and the Kaggle test only.

Benny's logged recall@20 on week 4 is the target:

| epoch | clicks | carts | orders |
|---|---|---|---|
| 0 | 0.447 | 0.321 | 0.471 |
| 1 | 0.455 | 0.331 | 0.491 |

The full A100 run (2026-11-06, commit `22c9bbd`, fp32 without TF32) beat it:

| epoch | week 4 clicks | carts | orders | Kaggle test clicks | carts | orders |
|---|---|---|---|---|---|---|
| 0 | 0.484 | 0.376 | 0.498 | 0.479 | 0.385 | 0.520 |
| 1 | 0.498 | 0.390 | 0.517 | 0.494 | 0.398 | 0.539 |
| 2 | 0.503 | 0.394 | 0.528 | 0.500 | 0.402 | 0.550 |
| 3 | 0.502 | 0.391 | 0.523 | 0.499 | 0.403 | 0.548 |
| 3, all sessions | 0.503 | 0.392 | 0.524 | 0.500 | 0.400 | 0.543 |

The per-epoch rows are measured on a 10% sample of the sessions. The last epoch (week 4 and test only) slightly lowers recall. Training ran at 9.3 it/s, so the 38.5k steps took about 70 minutes.

## Differences from Benny's code

- **Length-1 sessions:** they get predictions. Benny dropped them from `xgb_train_x`, which is 44% of the truncated week-4 sessions, so they had no T4R candidates. They're still not used for training, because they have no target.
- **Room for the mask:** prediction adds a column, so sessions with 20 items still have a position for the mask.
- **Learning-rate schedule:** it restarts every epoch. Benny's T4R Trainer kept one schedule across his `train()` calls.
- **Negative sampling:** random negatives are drawn from all items. Benny's were drawn from `1..1825323`, a vocabulary size hard-coded from an earlier run.
- **Data:** the item map differs slightly (1,848,472 items against Benny's 1,840,500), because the data comes from radek's split rather than Benny's own host-script run.
- **Recall on the Kaggle test:** `t4r_train.py` also reports it, against `TEST_LABELS`.
- **Column name:** the rank column is `t4r_rank`. Benny's `rename` wasn't assigned, so his column stayed `order`.
