# Orders ranker and the submission

Benny's orders reranker is a configuration of the shared `ranking` package. It's a chain: a click model and a cart model add their scores as features, then the orders model trains on them. See [`ranking/README.md`](../ranking/README.md).

```bash
uv run python -m ranking split orders/clicks --sample 0.02 --folds 0 1          # 01a_Split-clicks (+ 02 emb)
uv run python -m ranking split orders/carts_orders --sample 0.02 --folds 0 1    # 01b_Split-carts-orders-sub (+ 02 emb)
uv run python -m ranking stack orders/clicks --sample 0.02 --folds 0 1          # 03_XGB_Clicks: xgb_c_score
uv run python -m ranking stack orders/carts --sample 0.02 --folds 0 1           # 04_XGB_Carts: xgb_ca_score
uv run python -m ranking cv orders --sample 0.02 --folds 0 1                    # 05a_XGB-orders-localCV
uv run python -m ranking sub orders --sample 0.02 --folds 0 1                   # 05b_XGB_orders-submission; 06_Combine-Bags
uv run python -m ranking submission --sample 0.02                               # 06_Combine-Bags: submission.csv
```

- **Click model:** trained without the sessions that have carts or orders labels. Its mean score becomes `xgb_c_score` in the carts and orders chunks and in the submission rows.
- **Cart model:** trained on those carts chunks without the sessions that have orders labels. It adds `xgb_ca_score` to the orders chunks and the submission rows.
- **Orders model:** one model on every orders chunk, with 10 seeds × 345 trees.
- **Submission:** `submission.csv` takes clicks from `02_Clicks`, carts from `03_Carts` and orders from here, as Benny's did. It prints the weighted recall@20 on the Kaggle test.

On a 2% sample of folds 0 and 1, orders CV recall@20 is 0.657, and Kaggle-test recall@20 is 0.644 with 50 trees (Chris's baseline: 0.624). The submission's weighted score is 0.551 (baseline 0.526).
