# GRU next-item scores

Benny trained 6 GRU models per fold: for each target, one with random item embeddings and one with the fold's Word2Vec vectors, frozen. Their top 50 per session become the candidate source `gru_scores_20`, which the rankers merge as features (`gru_<T|F>_<cl|ca|or>`). This directory replaces his Merlin-dataloader scripts with Polars and plain PyTorch.

```bash
uv run python 01d_FE_GRU/gru_train.py --sample 0.02 --folds 0 --max-steps 300 --width 32 --eval-batch-size 64
uv run python 01d_FE_GRU/gru_train.py                 # all folds and configurations (A100)
uv run pytest 01d_FE_GRU
```

| file | replaces | does |
|---|---|---|
| `gru_data.py` | `01_Preprocess_GRU.ipynb` | training examples and inference inputs |
| `gru_model.py` | `train_gru.py` (model, loss), `start.sh` (configurations) | the model, its sampled loss, and Benny's 6 configurations |
| `gru_train.py` | `train_gru.py` (training, prediction), `02_Combine_GRU_Output.ipynb` | training, recall@20, and `data_folds/fold_<k>/candidates/{train,sub}/gru_scores_20/cand.parquet` |

## Data

Items are aid + 2 and types are the type code + 2. Each input event carries its reverse rank relative to the predicted event: 2 for the event just before it, up to 20.
- **Training examples:** for each of a session's last 40 events that has an event before it, the target is that event and the input is the 19 events before it. The examples come from `folds.fe_events(k)` and are split by the target's type.
- **Inference inputs:** the last 19 events of fold *k*'s held-out sessions (truncated) and of the Kaggle test sessions.

## Model

- **Layers:** the item embedding (width *w*), a type embedding (8) and a rank embedding (8) are concatenated, then go through `Linear(w + 16, w)` and a GRU. The hidden state at the last input is scored against every item.
- **Loss:** cross-entropy over the batch's targets plus `neg_factor` × batch random items, at a temperature, with label smoothing. Candidates that are items of the row's input, other than the target, get logit 0.
- **Training:** Adam for one epoch of at most 30,000 batches.
- **Scores:** the top 50 per session, with a softmax over those 50 as the score.

| configuration | width | layers | batch | lr | negatives | temperature | label smoothing |
|---|---|---|---|---|---|---|---|
| `gru_F_cl` | 128 | 3 | 2048 | 0.001 | 2 | 1 | 0 |
| `gru_T_cl` | 50 (Word2Vec) | 2 | 2048 | 0.001 | 4 | 0.531 | 0 |
| `gru_F_ca` | 256 | 1 | 2048 | 0.001 | 4 | 1 | 0 |
| `gru_T_ca` | 50 (Word2Vec) | 1 | 2048 | 0.00131 | 4 | 1 | 0 (dropout 0.237) |
| `gru_F_or` | 256 | 2 | 1024 | 0.000569 | none | 1 | 0.165 |
| `gru_T_or` | 50 (Word2Vec) | 2 | 2048 | 0.00188 | 4 | 1 | 0 |

## Differences from Benny's code

- **Shuffled training:** the examples are shuffled before the epoch. Benny's Merlin loader read them in file order.
- **Predictions:** padding (id 0) and the unused id 1 are never predicted. Benny's top 50 could include them, which became cand −2 and −1; his carts and orders embedding notebooks filtered −1 out.
- **Ties in time:** within a session, events in the same second keep their file order. Benny's sort used cuDF's order.
- **Sequence width:** the inputs are padded to 19, not 20. Benny's 20th column was always padding.
- **Data in memory:** the examples are built in memory with Polars list slices instead of 40 passes over the files written to disk. On the full data (about 150M examples per fold) that needs the A100 pod's RAM.
- **Local runs:** `--width` (the random models only), `--batch-size`, `--max-steps` and `--eval-batch-size` are for local runs. Prediction scores every session against all 1.86M items, so a batch of 256 sessions makes 1.9 GB of logits.

## Local check

2% of fold 0, `--max-steps 300 --width 32 --eval-batch-size 64`, on the RTX 2050: 2.8M training examples, about 5 minutes for all 6 models. The recall@20 on fold 0's held-out sessions only shows the pipeline works:

| model | target | recall@20 |
|---|---|---|
| `gru_F_cl` | clicks | 0.123 |
| `gru_T_cl` | clicks | 0.081 |
| `gru_F_ca` | carts | 0.177 |
| `gru_T_ca` | carts | 0.053 |
| `gru_F_or` | orders | 0.011 |
| `gru_T_or` | orders | 0.023 |
