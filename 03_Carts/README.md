# Carts ranker

Benny's carts reranker is now a configuration of the shared `ranking` package: the split `carts` and the model `carts`. See [`ranking/README.md`](../ranking/README.md).

```bash
uv run python -m ranking split carts --sample 0.02 --folds 0 1     # 01_Split + 02_Split_2_Add_Emb_Features
uv run python -m ranking cv carts --sample 0.02 --folds 0 1        # 03_XGB-v20-carts, bl_sub = False
uv run python -m ranking sub carts --sample 0.02 --folds 0 1       # 03_XGB-v20-carts, bl_sub = True; 04_Combine-Bags
```

It differs from clicks in four ways:
- **Candidates:** every source except the GRU proposes its top 100 per session, and a candidate is kept if at least 2 sources propose it.
- **Train candidates:** only for sessions with carts or orders labels.
- **Train rows:** the sessions with carts labels.
- **Models:** 2 submission models of 5 chunks each, with 10 seeds × 377 trees.

On a 2% sample of folds 0 and 1, CV recall@20 is 0.402, and Kaggle-test recall@20 is 0.395 with 50 trees (Chris's baseline: 0.360).
