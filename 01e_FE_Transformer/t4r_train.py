"""Train the session transformer and write its candidates.

Replaces Benny's 02_T4R_Train-v2.ipynb and 03_Convert.ipynb. Needs the sequences
from t4r_data.py. Benny's schedule: 3 epochs on weeks 1-3 + week 4 truncated +
Kaggle test, then 1 epoch on week 4 truncated + Kaggle test only. After each epoch
it prints recall@20 on week 4 (against XGB_TRAIN_Y, as Benny did) and on the Kaggle
test (against TEST_LABELS, which Benny didn't have), and saves a checkpoint. After
the last epoch it writes the top-k candidates for both, as the shared source `t4r`
(see folds.py):
    data_folds/shared/candidates/train/t4r/cand.parquet  all week-4 sessions
    data_folds/shared/candidates/sub/t4r/cand.parquet    the Kaggle test sessions
with columns session, cand, t4r_rank (0 = best), t4r_score (softmax over the top k).
The model doesn't depend on the fold; Benny copied the same files to every fold.

The full model needs a large GPU (the 1.84M x 256 item table and its Adam state alone
take ~8 GB). On a small GPU, try it with e.g. --d-model 32 --n-head 4 --sample 0.02.
Run from the repo root:
    uv run python 01e_FE_Transformer/t4r_train.py
"""

import argparse
import dataclasses
import math
import sys
import time

import numpy as np
import polars as pl
import torch
from beartype import beartype
from t4r_data import AID_MAP, FIRST_ITEM, PAD, T4R, sequence_path
from t4r_model import Params, SessionTransformer

import folds
import paths
from event_types import TYPE_NAMES
from padding import padded

SOURCE = "t4r"
CAND_TRAIN = folds.candidate_path(None, "train", SOURCE)
CAND_SUB = folds.candidate_path(None, "sub", SOURCE)
CHECKPOINTS = T4R / "checkpoints"


@beartype
def load_sequences(name: str, seq_len: int, min_len: int = 1, sample: float = 1.0, seed: int = 0):
    """(sessions, items): session ids and padded item ids (n, seq_len) as int32."""
    seqs = pl.read_parquet(sequence_path(name), columns=["session", "item"])
    seqs = seqs.filter(pl.col("item").list.len() >= min_len)
    if sample < 1.0:
        seqs = seqs.sample(fraction=sample, seed=seed).sort("session")
    return seqs["session"].to_numpy(), padded(seqs["item"], seq_len, pad=PAD).astype(np.int32)


def cosine_schedule(optimizer, steps: int, cycles: float):
    """HF's get_cosine_schedule_with_warmup without warmup, which T4R used.

    Benny's T4R Trainer kept one schedule across his train() calls; this one restarts
    each epoch.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2.0 * step / max(1, steps))))
    )


def train_epoch(model, optimizer, items: np.ndarray, batch_size: int, device, rng: np.random.Generator, log_every=1000):
    """One shuffled pass, dropping the last incomplete batch (as Benny's loader)."""
    model.train()
    steps = len(items) // batch_size
    schedule = cosine_schedule(optimizer, steps, model.params.cosine_cycles)
    order = rng.permutation(len(items))
    total, start = 0.0, time.time()
    for step in range(steps):
        batch = torch.from_numpy(items[order[step * batch_size : (step + 1) * batch_size]]).to(device).long()
        loss = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        total += loss.item()
        if (step + 1) % log_every == 0:
            print(f"  step {step + 1:,}/{steps:,}  loss {total / (step + 1):.4f}  {(step + 1) / (time.time() - start):.1f} it/s")
    return total / max(1, steps)


@beartype
def predict(
    model: SessionTransformer,
    sessions: np.ndarray,
    items: np.ndarray,
    item_to_aid: np.ndarray,
    k: int,
    batch_size: int,
    device,
) -> pl.DataFrame:
    """Top-k candidates per session: session, cand (aid), t4r_rank, t4r_score."""
    model.eval()
    top_items, top_scores = [], []
    for i in range(0, len(items), batch_size):
        batch = torch.from_numpy(items[i : i + batch_size]).to(device).long()
        scores, idx = model.predict(batch, k)
        top_items.append(idx.cpu().numpy())
        # Benny's t4r_score: softmax over the top k logits
        top_scores.append(torch.softmax(scores, dim=1).cpu().numpy())
    top_items, top_scores = np.concatenate(top_items), np.concatenate(top_scores)
    return pl.DataFrame(
        {
            "session": np.repeat(sessions, k).astype(np.int32),
            "cand": item_to_aid[top_items.ravel()].astype(np.int32),
            "t4r_rank": np.tile(np.arange(k, dtype=np.uint8), len(sessions)),
            "t4r_score": top_scores.ravel().astype(np.float32),
        }
    )


@beartype
def recall_at_20(cands: pl.DataFrame, labels: pl.LazyFrame | pl.DataFrame) -> dict[str, float]:
    """Recall@20 per type, as Benny computed it: hits / min(number of labels, 20).

    Sessions missing from `cands` count as misses.
    """
    labels = labels.lazy()
    top = cands.lazy().filter(pl.col("t4r_rank") < 20).select("session", aid="cand")
    hits = labels.join(top, on=["session", "aid"], how="semi").group_by("type").len()
    totals = labels.group_by("session", "type").len().group_by("type").agg(
        pl.col("len").clip(upper_bound=20).sum().alias("total")
    )
    rows = totals.join(hits, on="type", how="left").fill_null(0).collect().rows(named=True)
    return {TYPE_NAMES[r["type"]]: r["len"] / r["total"] for r in sorted(rows, key=lambda r: r["type"])}


def item_to_aid_lookup(aids: pl.DataFrame, n_items: int) -> np.ndarray:
    lookup = np.full(n_items, -1, dtype=np.int64)  # PAD and UNKNOWN map to -1
    lookup[aids["item"].to_numpy()] = aids["aid"].to_numpy()
    return lookup


def evaluate(model, data, item_to_aid, k, batch_size, device, eval_sample, seed):
    """Recall@20 on a sample of week 4 and of the Kaggle test."""
    out = {}
    for name, (sessions, items), labels in [
        ("week 4", data["xgb_train_x"], paths.XGB_TRAIN_Y),
        ("Kaggle test", data["test"], paths.TEST_LABELS),
    ]:
        if eval_sample < 1.0:
            keep = np.random.default_rng(seed).random(len(sessions)) < eval_sample
            sessions, items = sessions[keep], items[keep]
        cands = predict(model, sessions, items, item_to_aid, k, batch_size, device)
        labels = pl.scan_parquet(labels).join(pl.LazyFrame({"session": sessions}), on="session", how="semi")
        out[name] = recall_at_20(cands, labels)
    return out


def main(args):
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up when piped to a log
    params = dataclasses.replace(
        Params(), **{f.name: getattr(args, f.name) for f in dataclasses.fields(Params) if getattr(args, f.name, None) is not None}
    )
    print(params)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.tf32:
        # TF32 matmuls and convolutions on Ampere and newer: faster, ~10-bit mantissa products
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    aids = pl.read_parquet(AID_MAP)
    n_items = aids.height + FIRST_ITEM
    item_to_aid = item_to_aid_lookup(aids, n_items)

    # prediction uses every session; training only those with a target (length >= 2)
    data = {name: load_sequences(name, params.seq_len) for name in ["xgb_train_x", "test"]}
    recent = np.concatenate([items[(items != PAD).sum(1) >= 2] for _, items in data.values()])
    _, cv_train = load_sequences("cv_train", params.seq_len, min_len=2)
    everything = np.concatenate([cv_train, recent])
    del cv_train
    if args.sample < 1.0:
        everything = everything[rng.random(len(everything)) < args.sample]
        recent = recent[rng.random(len(recent)) < args.sample]
    print(f"training sequences: {len(everything):,} (all), {len(recent):,} (week 4 + test)")

    model = SessionTransformer(n_items, params).to(device)
    # HF/T4R Trainer defaults: AdamW without weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=0.0)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    epochs = [everything] * args.epochs_all + [recent] * args.epochs_recent
    for e, items in enumerate(epochs):
        print(f"epoch {e}: {len(items):,} sequences")
        loss = train_epoch(model, optimizer, items, params.batch_size, device, rng, args.log_every)
        torch.save(model.state_dict(), CHECKPOINTS / f"epoch_{e}.pt")
        recalls = evaluate(model, data, item_to_aid, 20, args.eval_batch_size, device, args.eval_sample, args.seed)
        print(f"epoch {e}: loss {loss:.4f}  recall@20 {recalls}")

    for (sessions, items), dataset in [(data["xgb_train_x"], "train"), (data["test"], "sub")]:
        cands = predict(model, sessions, items, item_to_aid, args.topk, args.eval_batch_size, device)
        path = folds.write_candidates(cands, None, dataset, SOURCE)
        print(f"wrote {path.relative_to(paths.ROOT)}: {cands.height:,} rows")
    print("recall@20, all sessions:")
    print("  week 4:", recall_at_20(pl.read_parquet(CAND_TRAIN), pl.scan_parquet(paths.XGB_TRAIN_Y)))
    print("  Kaggle test:", recall_at_20(pl.read_parquet(CAND_SUB), pl.scan_parquet(paths.TEST_LABELS)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model and training parameters default to Benny's (t4r_model.Params)
    for f in dataclasses.fields(Params):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=f.type if f.type is not bool else int, default=None)
    parser.add_argument("--epochs-all", type=int, default=3, help="epochs on weeks 1-4 + test")
    parser.add_argument("--epochs-recent", type=int, default=1, help="then epochs on week 4 + test only")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-sample", type=float, default=0.1, help="fraction of sessions for per-epoch recall")
    parser.add_argument("--sample", type=float, default=1.0, help="fraction of training sequences (for small GPUs)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (Ampere+ GPUs, e.g. A100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1000)
    main(parser.parse_args())
