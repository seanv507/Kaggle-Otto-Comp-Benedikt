# Chris's public 0.575 baseline

This is the [candidate rerank model](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575) that Benny's rankers use to fill sessions with fewer than 20 recommendations (`ranking/combine.py`, `fill`).

```bash
uv run python 00_submissions/01_baseline/baseline.py --sample 0.05
uv run python 00_submissions/01_baseline/baseline.py              # the full run (A100: see below)
uv run pytest 00_submissions/01_baseline
```

| file | replaces | writes |
|---|---|---|
| `baseline.py` | `00_1_CoMatrix-Basic.ipynb`, `00_2_CalcSub.ipynb` | `data_tmp/baseline/top_{15_caco,15_b2b,20_clicks}.parquet`, `data_tmp/baseline/{clicks,carts,orders}.parquet` as (`session`, `aid`, `rank`) |

- **Matrices:** three covisitation matrices are built from all of `TRAIN` and `TEST`, once rather than per fold: carts-orders (top 15), buy2buy (top 15) and time-weighted clicks (top 20). Their pairs are those of Benny's `caco`, `b2b` and `clicks`, so they come from `01b_FE_CoVisit_1/benny_covisits.py`.
- **Rules:** for each Kaggle test session, Chris's `suggest_clicks` gives the clicks, and `suggest_buys` gives the carts and orders.
- **Score:** it prints recall@20 against `TEST_LABELS` and the weighted score.

`test_baseline.py` runs Chris's two functions, copied verbatim, against the Polars versions on random sessions. Those include sessions with 20 or more unique aids, which use the weighted history, and shorter ones, which use the covisit lists.

## Checks

| sample | clicks | carts | orders | score |
|---|---|---|---|---|
| 2% | 0.431 | 0.360 | 0.624 | 0.526 |
| 5% | 0.460 | 0.377 | 0.643 | 0.545 |

The matrices come from the same fraction of sessions, so the score rises with the sample. Chris's public LB score was 0.575. The 5% run takes 28 s and 7.4 GB of RAM, so the full run needs a larger machine than the local one, or `--session-chunk`.

## Differences from Chris's code

- **Ties:** `Counter.most_common` breaks ties by first insertion, and the Polars version keeps that order. Among events in the same second, file order counts as time order.
- **Rules:** written as Polars expressions instead of a `pandarallel` loop over sessions.
