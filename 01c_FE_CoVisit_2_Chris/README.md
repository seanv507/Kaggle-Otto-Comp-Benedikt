# Chris's covisitation matrices, Benny's way

Benny built 10 of Chris's covisitation matrices for each fold and turned each one into session × candidate features (`chris__<i>__*`) for his rankers. Here, `chris_covisits.py` does both. The pair logic is imported from Chris's part: `covisits.py` in `../Kaggle-OTTO-Comp`, an editable uv path dependency, on branch `importable-covisits`.

```bash
uv run python 01c_FE_CoVisit_2_Chris/chris_covisits.py --sample 0.02 --folds 0 --engine streaming
uv run python 01c_FE_CoVisit_2_Chris/chris_covisits.py          # all folds and matrices (A100)
uv run pytest 01c_FE_CoVisit_2_Chris
```

The options are `--folds`, `--matrices`, `--sample`, `--engine` (gpu by default), `--session-chunk` and `--force`. Existing matrices are reused unless `--force` is given.

| file | replaces |
|---|---|
| `chris_covisits.py` | `03_CoMatrix-99-Chris-gpu-{93,115,217,220,226,232,235,239,700,701}.ipynb`, `04_CoMatrix-99-Chris-apply.ipynb` |

## Outputs, per fold *k*

- `data_folds/fold_<k>/covisits/chris_<VER>.parquet`: (`aid`, `cand`, `wgt`), the top `cand` per `aid`. The matrix is built from `folds.fe_events(k)`: all of `TRAIN` and `TEST`, with fold *k*'s held-out sessions truncated.
- `data_folds/fold_<k>/candidates/{train,sub}/chris__<i>__15/cand.parquet`: the matrix applied to fold *k*'s held-out sessions (`train`) and to the Kaggle test sessions (`sub`). The rankers merge these as features; Benny didn't take candidates from them.

`i` is the matrix's position in Benny's sorted file names:

| i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| matrix | 115 | 217 | 220 | 226 | 232 | 235 | 239 | 700 | 701 | 93 |
| top per aid | 40 | 40 | 80 | 80 | 80 | 80 | 20 | 0 | 40 | 80 |

## The features (`apply_matrix`)

Each event of a session is joined to its item's matrix row. Each (event, cand) pair has these values, which are summed per (session, cand):

| column | value |
|---|---|
| `wgt` | the matrix weight |
| `dummy` | 1 |
| `wgt_1`, `wgt_2` | `wgt` times the event's recency weight (`session_ops.recency_weights`, as in `pv`) |
| `t_*` | the above times the event's type code (clicks 0, carts 1, orders 2) |
| `d_*` | only events in the session's first segment. Segments are cut at gaps of more than 2 hours, counted from the session's start |
| `l_*` | only events at the last second of their segment |

`test_chris_covisits.py` checks `apply_matrix` against Benny's own code, run on pandas, on random sessions.

## Checks on a 2% sample of fold 0

The matrices were built from 2% of the sessions, so they're weaker than the full ones. Recall@20 of the `train` rows, ranked by `chris__<i>__wgt`:

| matrix | clicks | carts | orders |
|---|---|---|---|
| 115 | 0.426 | 0.351 | 0.590 |
| 239 | 0.232 | 0.251 | 0.556 |
| 93 | 0.239 | 0.207 | 0.408 |
| 235 | 0.236 | 0.190 | 0.333 |

- On the full fold 0 (222M events), `fe_events` keeps every session's events in time order.
- The 93 and 115 matrices built on the GPU engine equal the streaming ones exactly.
- The whole sample run takes 94 s on the streaming engine with 4 GB of RAM.
- On the local 4 GB GPU, 217 runs out of memory even in session chunks, because its windows can't be partitioned. Use `--engine streaming` locally.

## Differences from Benny's code

- **Kept as Benny had them:**
  - **Matrix 700 is empty.** His 700 notebook ends with `n < 0`, so it keeps no pairs, and the `chris__7__15` files are empty.
  - **The whole matrix is applied.** `treshhold_topn=15` was never used; the `_15` only appears in the names.
  - **`t_*` weights clicks by 0**, because it multiplies by the type code.
- **Exact sums:** Benny summed the per-file results in `glob` order and kept only the top 100–150 per aid after every 50 files (every 5 for 115). That order can't be reproduced, so here the sums are exact and only the final top is kept.
- **239's files:** 239 keeps one pair per data file. Benny's files were the Kaggle test, the fold's truncated sessions, and train files of 100,000 sessions each. `folds.fe_events(k, benny_files=True)` rebuilds those groups.
- **Ties in time:** as in `pv`, the event later in the file counts as more recent.
- **Dtypes:** the sums are Float32 and the counts UInt32. Benny's were float64 and int64.
- **`sub` file layout:** Benny wrote the `sub` candidates directly in 10 session parts (`split/.../cand_<i>.parquet`). Here they're one `cand.parquet`, like `train`.
