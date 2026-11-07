"""Train Benny's 6 GRU configurations per fold and write their candidates.

Replaces train_gru.py (run by start.sh), 01_Preprocess_GRU.ipynb and
02_Combine_GRU_Output.ipynb. Per fold k and configuration (gru_model.CONFIGS: a
random and a Word2Vec-initialised model for each target):
- train on the fold's examples with that target type (gru_data), one epoch of at
  most --max-steps batches (Benny: 30,000), Adam;
- print recall@20 on fold k's held-out sessions for the target type;
- predict the top 50 of fold k's held-out sessions and of the Kaggle test sessions,
  with a softmax over the 50 as the score.
The 6 configurations' scores are outer-joined into the candidate source
gru_scores_20 (columns gru_<T|F>_<cl|ca|or>, 0 where a configuration doesn't have
the cand), for fold k's train and sub (folds.write_candidates). The rankers merge it
as features.

Run from the repo root (on the local GPU: a sample, narrow models, few steps):
    uv run python 01d_FE_GRU/gru_train.py --sample 0.02 --folds 0 --max-steps 300 --width 32 --eval-batch-size 64
"""

import argparse
import dataclasses
import sys
import time

import numpy as np
import polars as pl
import torch
from gru_data import (
    OFFSET,
    SEQ,
    inference_inputs,
    training_examples_chunked,
)
from gru_model import CONFIGS, FIRST_ITEM, GRU

import folds
import metrics
from event_types import TYPE_NAMES

SOURCE = "gru_scores_20"
TOP = 50
TYPE_CODES = {name: code for code, name in TYPE_NAMES.items()}


def padded(df: pl.DataFrame, column: str) -> np.ndarray:
    """A list column as an (n, SEQ) array, padded with 0 on the right."""
    filled = (
        pl.col(column)
        .list.concat(pl.lit([0] * SEQ, dtype=pl.List(df.schema[column].inner)))
        .list.head(SEQ)
    )
    return np.stack(df.select(filled.list.to_array(SEQ)).to_series().to_numpy())


def tensors(df: pl.DataFrame, device) -> dict[str, torch.Tensor]:
    out = {
        c: torch.from_numpy(padded(df, c).astype(np.int64))
        for c in ("item", "type", "rank")
    }
    out["length"] = torch.from_numpy(df["item"].list.len().to_numpy().astype(np.int64))
    if "target" in df.columns:
        out["target"] = torch.from_numpy(df["target"].to_numpy().astype(np.int64))
    return {k: v.to(device) for k, v in out.items()}


def pretrained_weights(
    emb: pl.DataFrame, n_items: int, rng: np.random.Generator
) -> np.ndarray:
    """Rows aid + 2 from the Word2Vec vectors; other rows N(0, 1), as Benny's."""
    w = rng.standard_normal((n_items, emb["emb"].dtype.size)).astype(np.float32)
    w[emb["aid"].to_numpy() + OFFSET] = emb["emb"].to_numpy()
    return w


def train(
    model: GRU, examples: pl.DataFrame, max_steps: int, device, seed: int
) -> None:
    c = model.config
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=c.lr)
    order = np.random.default_rng(seed).permutation(examples.height)
    steps = min(max_steps, examples.height // c.batch_size)
    model.train()
    start, total = time.time(), 0.0
    for s in range(steps):
        batch = tensors(
            examples[order[s * c.batch_size : (s + 1) * c.batch_size]], device
        )
        loss = model.loss(
            batch["item"],
            batch["type"],
            batch["rank"],
            batch["length"],
            batch["target"],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item()
        if (s + 1) % 1000 == 0:
            print(
                f"    step {s + 1:,}/{steps:,}  loss {total / (s + 1):.4f}  {(s + 1) / (time.time() - start):.1f} it/s"
            )


def predict(
    model: GRU, inputs: pl.DataFrame, device, batch_size: int = 256
) -> pl.DataFrame:
    """(session, cand, score): each session's top 50, score a softmax over them."""
    model.eval()
    k = min(TOP, model.item_emb.num_embeddings - FIRST_ITEM)
    items, scores = [], []
    for i in range(0, inputs.height, batch_size):
        b = tensors(inputs[i : i + batch_size], device)
        logits, idx = model.predict(b["item"], b["type"], b["rank"], b["length"], k)
        items.append(idx.cpu().numpy())
        scores.append(torch.softmax(logits, dim=1).cpu().numpy())
    items, scores = np.concatenate(items), np.concatenate(scores)
    return pl.DataFrame(
        {
            "session": np.repeat(inputs["session"].to_numpy(), k).astype(np.int32),
            "cand": (items.ravel() - OFFSET).astype(np.int32),
            "score": scores.ravel().astype(np.float32),
        }
    )


def combine(preds: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Benny's 02_Combine_GRU_Output: outer join on (session, cand), 0 where missing."""
    out = None
    for name, df in preds.items():
        df = df.rename({"score": name})
        out = (
            df
            if out is None
            else out.join(df, on=["session", "cand"], how="full", coalesce=True)
        )
    return out.with_columns(pl.exclude("session", "cand").fill_null(0.0)).sort(
        "session", "cand"
    )


def main(args):
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up when piped to a log
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    root = folds.output_root(args.sample)
    configs = [c for c in CONFIGS if not args.configs or c.name in args.configs]
    for k in args.folds:
        start = time.time()
        events = folds.fe_events(k, args.sample)
        examples = training_examples_chunked(events, args.session_chunk or None)
        n_items = (
            int(
                examples.select(pl.col("item").list.max().max(), pl.col("target").max())
                .max_horizontal()
                .item()
            )
            + 1
        )
        n_items = max(
            n_items,
            int(
                pl.scan_parquet(folds.feature_path(k, "word2vec", root))
                .select(pl.col("aid").max())
                .collect()
                .item()
            )
            + OFFSET
            + 1,
        )
        inputs = {
            d: inference_inputs(folds.candidate_events(k, d, args.sample)).collect()
            for d in folds.DATASETS
        }
        emb = pl.read_parquet(folds.feature_path(k, "word2vec", root))
        print(
            f"fold {k}: {examples.height:,} training examples, {n_items:,} item ids ({time.time() - start:.0f} s)"
        )
        preds = {d: {} for d in folds.DATASETS}
        for c in configs:
            if args.width and not c.use_pretrained:
                c = dataclasses.replace(c, width=args.width)
            if args.batch_size:
                c = dataclasses.replace(c, batch_size=args.batch_size)
            start = time.time()
            ex = examples.filter(pl.col("target_type") == TYPE_CODES[c.target] + OFFSET)
            pre = (
                pretrained_weights(emb, n_items, np.random.default_rng(args.seed))
                if c.use_pretrained
                else None
            )
            model = GRU(n_items, c, pre).to(device)
            print(f"  {c.name}: {ex.height:,} examples")
            train(model, ex, args.max_steps, device, args.seed)
            for d in folds.DATASETS:
                preds[d][c.name] = predict(
                    model, inputs[d], device, args.eval_batch_size
                )
            labels = folds.sample_sessions(folds.labels(k), args.sample).filter(
                pl.col("type") == TYPE_CODES[c.target]
            )
            r = metrics.recall(preds["train"][c.name], labels, "score")[c.target]
            print(
                f"  {c.name}: recall@20 {r:.4f} on fold {k}'s held-out sessions ({time.time() - start:.0f} s)"
            )
            del model
            torch.cuda.empty_cache()
        for d in folds.DATASETS:
            path = folds.write_candidates(combine(preds[d]), k, d, SOURCE, root=root)
            print(f"wrote {path.relative_to(root.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(folds.N_FOLDS))
    )
    parser.add_argument(
        "--sample", type=float, default=1.0, help="fraction of sessions"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="e.g. gru_F_cl gru_T_or (default: all 6)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30_000,
        help="batches per model (Benny: 30,000)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="override the random models' width (small GPUs)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="override the batch size"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="sessions per prediction batch (scores against every item)",
    )
    parser.add_argument(
        "--session-chunk",
        type=int,
        default=2_000_000,
        help="session ids per chunk when building the training examples (0: one chunk)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
