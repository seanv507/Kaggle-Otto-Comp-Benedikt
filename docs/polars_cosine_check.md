# Polars vs numpy cosines in the Word2Vec features

The `polars-einsum` branch replaces numpy in `ranking/emb_features.similarities`. That function computes the cosine of the candidate's and each session event's Word2Vec vectors, keeping values above 0.5:
- **numpy** (`master`): normalise the vectors, gather them per pair, then `np.einsum("ij,ij->i")` in batches of 1M pairs;
- **Polars** (this branch): join each pair's unit vectors as `Array(Float32, 50)` columns, then `(a * b).arr.sum()`, in the same batches.

## What changes

The two use different float32 summation orders.

On the 2% sample's clicks train split, fold 0, chunk 0 (278k rows, 1.5M unique pairs):
- the cosines differ by at most 6.6e-7;
- one pair in 1.3M crosses the 0.5 threshold.

Every `sim_*` feature changes by rounding in 48k–215k rows. `sim_2`, `sim_wgt_1` and `sim_wgt_2` change by up to 0.018, because the crossing pair also changes the divisor `1 + sim_3`. The split's rows and their order are otherwise identical, and the split itself is deterministic: two builds with the same code are byte-identical. The Polars version is a little faster, 0.38 s against 0.45 s for the 1.5M pairs.

## Effect on the clicks ranker

The setup:
- the clicks CV on folds 0 and 1 of each sample, 50 trees;
- the split rebuilt with each version;
- each version run with 5 XGBoost seed offsets (0, 100, 200, 300, 400 added to the CV's seeds 0–2).

The scores are recall@20.

| sample | numpy | Polars | difference | t (Welch) |
|---|---|---|---|---|
| 2% | 0.4748, 0.4747, 0.4743, 0.4743, 0.4747: **0.4746 ± 0.0002** | 0.4732, 0.4751, 0.4733, 0.4745, 0.4752: **0.4743 ± 0.0010** | −0.0003 | −0.6 |
| 5% | 0.5082, 0.5079, 0.5082, 0.5080, 0.5074: **0.5079 ± 0.0003** | 0.5078, 0.5077, 0.5077, 0.5070, 0.5073: **0.5075 ± 0.0003** | −0.0004 | −2.1 |

- **At 2%:** the difference is noise. Two Polars runs beat every numpy run.
- **At 5%:** the seed noise is smaller, and 4 of the 5 Polars runs score below all numpy runs but the lowest. With 5 runs per side, t ≈ −2.1 is borderline, about a 6–7% chance of arising from noise alone.
- **Together:** both samples point the same way, at most about −0.0004 on clicks recall. In the weighted score (0.1 × clicks) that's about 0.00004.

## Decision (2026-11-08)

`master` keeps numpy:
- its results stay bit-identical to the first full run (`docs/full_run.md`);
- the Polars version isn't better on recall, and is only marginally faster.

This branch keeps the Polars version for reference.

**How it was run:** a driver patches `ranking.emb_features.similarities` with each version, rebuilds the train clicks split, and wraps `xgboost.train` to offset the seeds. The 5% sample of folds 0 and 1 took 35 minutes to build on the local machine (RTX 2050, 12 threads), and each version's split and 5 CVs about 10 minutes.
