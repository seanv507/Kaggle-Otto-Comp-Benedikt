# Benny's covisitation matrices

These are Benny's own six matrices, each built per fold and turned into candidates. `benny_covisits.py` does it with the session-chunked builder and helpers of Chris's part: `covisits.py` in `../Kaggle-OTTO-Comp`, an editable uv path dependency.

```bash
uv run python 01b_FE_CoVisit_1/benny_covisits.py --sample 0.02 --folds 0 --engine streaming
uv run python 01b_FE_CoVisit_1/benny_covisits.py              # all folds and matrices (A100)
uv run pytest 01b_FE_CoVisit_1
```

The options are `--folds`, `--matrices`, `--sample`, `--engine`, `--session-chunk` and `--force`. Existing matrices are reused unless `--force` is given.

| file | replaces |
|---|---|
| `benny_covisits.py` | `03_CoMatrix-{1-Click2Order,2-CartsOrders,3-buy2buy,4-clicks,6-weighted-clicks-dev,7-weighted-types}.ipynb` |

## Outputs, per fold *k*

- `data_folds/fold_<k>/covisits/benny_<matrix>.parquet`: (`aid`, `cand`, `wgt`, extra columns), each aid's top `cand` by `wgt`. The matrix is built from `folds.fe_events(k)`.
- `data_folds/fold_<k>/candidates/{train,sub}/<source>/cand.parquet`: the matrix's candidates for fold *k*'s held-out sessions (`train`) and the Kaggle test sessions (`sub`). Benny's `01_Split` ranks each source by the column in the last column below and keeps its top 80.

| source | matrix | pairs | weight | top used | ranked by |
|---|---|---|---|---|---|
| `c2o_15` | `c2o` | a click and a cart or order within 7 days (no aid ≠ aid_y filter) | sessions, kept if more than 1 | 15 | `c2o_wgt_15` |
| `caco_15` | `caco` | the last 30 events, within 24 h, aid ≠ aid_y, latest copy per session | type of y: clicks 1, carts 6, orders 3 | 15 | `caco_wgt_15` |
| `b2b_15` | `b2b` | carts and orders: the last 30 of them, within 14 days | sessions | 15 | `b2b_wgt_15` |
| `caco_b2b_15` | `b2b` | empty (see below) | | 15 | `caco_b2b_wgt_15` |
| `clicks_20` | `clicks` | as `caco` | 1 + 3 × the x event's place in time from train start to test end | 20 | `clicks_wgt_20` |
| `weighted_30` | `weighted` | clicks within 1-hour sub-sessions, every pair at different seconds and aids | 1/ln(Δs+2) · 1/ln(Δpos+2) · 1/ln(clicks+2); pairs seen more than once per file chunk; ÷ √(events of x + 1) √(events of y + 1) | 30 | `weighted_wgt_30` |
| `weighted_1_2_20` | `wt_1_2` | a cart and an order of the same session | 1/ln(Δs+2) · 1/ln(Δpos+2); pairs in more than one session; ÷ √(sessions carting x + 1) √(sessions ordering y + 1) | 20 | `weighted_1_2_wgt_sum_2` |

Candidate columns:
- **`c2o`, `caco`, `b2b`, `clicks`:** each event's top matrix rows are summed per (session, cand):
  - `_num` is the number of the session's events that point to the cand;
  - `_rank_min` is the rank of the most recent of them (1 for the session's last event);
  - `_wgt` is the summed weight;
  - `_wgt_2` is the weight divided by that rank, summed;
  - `_wgt_cl`, `_wgt_ca`, `_wgt_or` split `_wgt` by the event's type.
- **`weighted`:** `_num`, `_wgt`, `_wgt_max`, `_count`, `_count_max`.
- **`weighted_1_2`:** computed from the session's carts only. Each cart gets a recency score 2^(0.1 + 0.9 (n − rank)/(n − 1)) − 1, which is 1 for a single cart. The columns are `_dummy`, `_wgt_sum_2`, `_wgt_sum_2_2` (times the score), `_wgt_prob`, `_wgt_prob_2`, `_count` and `_count_2`.

`test_benny_covisits.py` ports each of Benny's matrices and candidate functions to pandas. It checks them against these on random sessions, including the file chunks of `weighted`.

## Checks on a 2% sample of fold 0

The run takes 117 s on the streaming engine with 4.4 GB of RAM. Recall@20 of the `train` candidates, ranked as `01_Split` does:

| source | clicks | carts | orders |
|---|---|---|---|
| `clicks_20` | 0.232 | 0.195 | 0.370 |
| `caco_15` | 0.196 | 0.172 | 0.417 |
| `c2o_15` | 0.174 | 0.200 | 0.369 |
| `weighted_30` | 0.173 | 0.132 | 0.243 |
| `b2b_15` | 0.053 | 0.072 | 0.294 |
| `weighted_1_2_20` | 0.014 | 0.026 | 0.167 |

These are low because the matrices come from 2% of the sessions.

## Differences from Benny's code

- **Kept as Benny had them:**
  - **`caco_b2b_15` is empty.** His filter `df['type'].isin(type_filter)` was given `[['carts', 'orders'], 'caco']` instead of the list of types, and matched nothing. His rankers ignore every `caco_b2b_*` column.
  - **int16 casts.** Every column of `c2o`, `caco`, `b2b` and `clicks` except `_wgt_2` is cast to int16: float weights are truncated and sums above 32767 wrap around. `benny_covisits.py` prints how many sums wrapped. Truncating a float sum that lands exactly on an integer can differ by 1 from cuDF, whose float32 sum order isn't fixed either.
  - **`weighted_1_2_wgt_prob_2`** is `wgt_prob` × score. Benny's loop overwrote the matrix's own `wgt_prob_2` with it.
  - **`weighted` sub-session ids** are session × 100 + k, as his were. A session with 100 or more sub-sessions collides with the next session's ids.
- **File chunks:** only `weighted` depends on how Benny's loop cut the data into files. Its count > 1 filter ran per chunk, and the chunks are rebuilt exactly with `folds.benny_file_chunks(3)`: his train files sorted by name, dealt round-robin into 43 chunks.
- **Ties:** where Benny's sorts had ties, cuDF's order decided. Here:
  - within a session, the event later in the file counts as more recent;
  - within a matrix row, the smaller cand comes first.
- **Saved matrices:** these keep the top used plus some margin (50 for `caco`, `b2b`, `clicks`; 15 for `c2o`, 30 for `weighted`, 20 for `wt_1_2`). Benny saved 50 for some and every pair for others. Only the top used affects the candidates.
- **Dtypes:** float sums are Float32 and counts UInt32. Where Benny didn't cast, his were float64 and int64.
