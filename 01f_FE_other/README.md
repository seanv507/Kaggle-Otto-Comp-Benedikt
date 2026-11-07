# Previous visits, session and item features

Features and candidates that need no model. None of them depends on the fold except the item activity features of `95_Cand_Features.ipynb`, which aren't converted yet.

```bash
uv run python 01f_FE_other/previous_visits.py     # pv and pv_2 candidates (~50 s, ~2 GB RAM)
uv run python 01f_FE_other/session_features.py    # session features (~7 s, ~1 GB RAM)
uv run pytest 01f_FE_other
```

Both scripts take `--sample 0.05` to run on 5% of the sessions; the outputs then go to `data_folds/sample_0.05/`. They also take `--engine gpu`.

| file | replaces | writes |
|---|---|---|
| `previous_visits.py` | `04_PreviousVisits.ipynb`, `04_PreviousVisits-diffWeights.ipynb` | `data_folds/shared/candidates/{train,sub}/{pv,pv_2}/cand.parquet` |
| `session_features.py` | `96_Session_Features.ipynb` | `data_folds/shared/session_features_{train,sub}.parquet` |
| `95_Cand_Features.ipynb` | (not converted yet) | |

`train` is every week-4 session (truncated) and `sub` is every Kaggle test session. Benny copied these files to every fold. Here they're shared, and `folds.read_candidates` filters the `train` candidates to a fold.

## Previous visits (`pv`, `pv_2`)

Every item in the session is a candidate. Each event gets a weight

    (lo + (1 - lo) * (rank - 1) / session_len) * type_weight

with rank 1 for the most recent event, and `lo` = 2^0.1 − 1 (`_wgt_1`) or 2^0.5 − 1 (`_wgt_2`). The weights are summed per (session, item).

| source | type weights (clicks, carts, orders) | columns |
|---|---|---|
| `pv` | 1, 6, 3 | `ts_diff` (s since the last event), `rev_order` (best rank), `pv_wgt_1`, `pv_wgt_2`, `clicks`, `carts`, `orders` |
| `pv_2` | 0.5, 9, 0.5 | `pv_2_wgt_1`, `pv_2_wgt_2` |

Recall@20 of the candidates ranked by `_wgt_1`:

| source | week 4 clicks | carts | orders | Kaggle test clicks | carts | orders |
|---|---|---|---|---|---|---|
| `pv` | 0.322 | 0.308 | 0.589 | 0.317 | 0.315 | 0.598 |
| `pv_2` | 0.322 | 0.308 | 0.591 | 0.317 | 0.315 | 0.599 |

## Session features

These are one row per session: `num_aid`, `num_u_aid`, `diff` (the session's duration in ms), and counts per type (`_cl`, `_ca`, `_co`). The session is cut into segments wherever two events are more than 4 hours apart. `num_sessions` is the number of cuts, and the `_fs` features count only the most recent segment.

## Differences from Benny's code

- **Ties in time (`pv`):** `rank` comes from sorting by time, and events in the same second are common. Benny's order among them was whatever cuDF's sort gave. Here, the event later in the file counts as more recent.
- **Kept as Benny had them:**
  - In `pv`, the most recent event gets the *smallest* weight. That's the reverse of Chris's version.
  - The per-type `_fs` session features (`num_aid_cl_fs`, `num_u_aid_ca_fs`, ...) forget the segment filter, so they equal the per-type features of the whole session. Benny's rankers drop most of them.
- **File names:** Benny wrote `train_sessions.parquet` and `sub_sessions.parquet`. These are `session_features_{train,sub}.parquet`, which avoids a clash with the session split `data_tmp/split/sub_sessions.parquet`.
- **Dtypes:** the counts are UInt32 and `diff` is Int64. Benny's were int64 after his merges.
