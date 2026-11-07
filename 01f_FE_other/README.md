# Previous visits, session and item features

Features and candidates that need no model. None of them depends on the fold.

```bash
uv run python 01f_FE_other/previous_visits.py     # pv and pv_2 candidates (~50 s, ~2 GB RAM)
uv run python 01f_FE_other/session_features.py    # session features (~7 s, ~1 GB RAM)
uv run python 01f_FE_other/item_features.py       # item activity per week and day (~17 s, ~6 GB RAM)
uv run pytest 01f_FE_other
```

Each script takes `--sample 0.05` to run on 5% of the sessions; the outputs then go to `data_folds/sample_0.05/`. They also take `--engine gpu`.

| file | replaces | writes |
|---|---|---|
| `previous_visits.py` | `04_PreviousVisits.ipynb`, `04_PreviousVisits-diffWeights.ipynb` | `data_folds/shared/candidates/{train,sub}/{pv,pv_2}/cand.parquet` |
| `session_features.py` | `96_Session_Features.ipynb` | `data_folds/shared/session_features_{train,sub}.parquet` |
| `item_features.py` | `95_Cand_Features.ipynb` | `data_folds/shared/item_features_{train,sub}.parquet`, `data_folds/shared/item_day_features_{train,sub}.parquet` |

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

## Item activity

These count clicks, carts and orders per item:

| file | key | columns | counts over |
|---|---|---|---|
| `item_features_*` | `aid` | `p0w_*` | the dataset's own week (truncated week 4, or the Kaggle test) |
| | | `p1w_*`, `p2w_*`, `p3w_*` | the 1, 2 and 3 weeks before it, from `TRAIN` |
| `item_day_features_*` | `aid`, `day_` | `sday_*` | each day of the dataset's own week |

The `train` dataset's week starts on 2022-08-22, so its earlier weeks are weeks 1–3 only. The `sub` dataset's week starts on 2022-08-29, so its `p1w` is the whole of week 4. `day_` is `folds.day_index`: 0 for the first two calendar days (the week starts at 22:00 UTC), then 1 to 6. The rankers join `item_day_features` on the session's `day_`.

## Differences from Benny's code

- **Ties in time (`pv`):** `rank` comes from sorting by time, and events in the same second are common. Benny's order among them was whatever cuDF's sort gave. Here, the event later in the file counts as more recent.
- **Kept as Benny had them:**
  - In `pv`, the most recent event gets the *smallest* weight. That's the reverse of Chris's version.
  - The per-type `_fs` session features (`num_aid_cl_fs`, `num_u_aid_ca_fs`, ...) forget the segment filter, so they equal the per-type features of the whole session. Benny's rankers drop most of them.
  - The `p0w_*` item counts are added twice (he appended them to his list both before and after the earlier weeks), so they're double the event counts.
- **`day_`:** Benny looked it up from a table of days of the month. Here it's arithmetic on `ts`, which gives the same values on the real data and runs on the GPU engine.
- **File names:** Benny wrote `train_sessions.parquet`, `sub_sessions.parquet`, `{train,sub}_cand_meta_comb.parquet` and `{train,sub}_cand_day_meta_comb.parquet`. These are `session_features_*`, `item_features_*` and `item_day_features_*`. That also avoids a clash with the session split `data_tmp/split/sub_sessions.parquet`.
- **Dtypes:** the counts are UInt32, `day_` is Int8 and `diff` is Int64. Benny's were int64 after his merges.
