# Clicks ranker

Benny's clicks reranker is now a configuration of the shared `ranking` package: the split `clicks` and the model `clicks`. See [`ranking/README.md`](../ranking/README.md) for what each step does and how it differs from Benny's code.

```bash
uv run python -m ranking split clicks --sample 0.02 --folds 0 1    # 01_Split + 02_Split_2_Add_Emb_Features-v2
uv run python -m ranking cv clicks --sample 0.02 --folds 0 1       # 03_XGB-v19-clicks, bl_sub = False
uv run python -m ranking sub clicks --sample 0.02 --folds 0 1      # 03_XGB-v19-clicks, bl_sub = True; 04_Combine-Bags
```

- **Candidates:** the top 80 per session of `c2o_15`, `clicks_20`, `b2b_15`, `caco_15`, `pv`, `weighted_30` and `caco_b2b_15`.
- **Models:** 5 submission models, one per fold, each with 3 seeds × 394 trees.

On a 2% sample of folds 0 and 1, CV recall@20 is 0.474, and Kaggle-test recall@20 is 0.463 with 50 trees (Chris's baseline: 0.431).
