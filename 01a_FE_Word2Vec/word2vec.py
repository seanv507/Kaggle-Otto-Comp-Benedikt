"""Word2Vec item embeddings per fold, from the fold's sessions.

Replaces Benny's 000_Word2Vec.ipynb. Each session of folds.fe_events(k) is a
sentence of its aids in time order; gensim's skip-gram Word2Vec with Benny's
parameters (PARAMS) learns a 50-d vector per aid. Writes
    data_folds/fold_<k>/word2vec.parquet   (aid Int32, emb Array(Float32, 50))
which the GRU (01d) and the rankers' embedding features read. Benny pickled a dict
{aid: vector} to data_folds/fold_<k>/word2vec.emb.

The sentences go to gensim as a text file (corpus_file) rather than a Python list:
that needs far less memory for 14.5M sessions, and trains faster with many workers.
gensim with several workers isn't deterministic, in either mode.

Run from the repo root:
    uv run python 01a_FE_Word2Vec/word2vec.py --sample 0.05 --folds 0
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from beartype import beartype
from gensim.models import Word2Vec

import folds

# Benny's Word2Vec(...) arguments, except sentences and workers
PARAMS = {
    "vector_size": 50,
    "epochs": 5,
    "sg": 1,
    "window": 3,
    "sample": 1e-3,
    "ns_exponent": 1,
    "min_count": 1,
}


@beartype
def write_corpus(events: pl.LazyFrame | pl.DataFrame, path: Path) -> int:
    """One line per session: its aids in event order, space-separated. Returns the
    number of sessions.

    The events' row order within a session must be their time order (as
    folds.fe_events keeps it); Benny sorted by (session, ts).
    """
    lines = (
        events.lazy()
        .with_row_index("_i")
        .group_by("session")
        .agg(pl.col("aid").sort_by("_i").cast(pl.String).str.join(" "))
        .sort("session")
        .select("aid")
        .collect(engine="streaming")
    )
    lines.write_csv(path, include_header=False, quote_style="never")
    return lines.height


@beartype
def train(corpus: Path, workers: int, seed: int = 1, **params) -> pl.DataFrame:
    """(aid, emb): the Word2Vec vector of every aid in the corpus, sorted by aid."""
    model = Word2Vec(
        corpus_file=str(corpus), workers=workers, seed=seed, **{**PARAMS, **params}
    )
    wv = model.wv
    aids = np.array([int(a) for a in wv.index_to_key], dtype=np.int32)
    order = np.argsort(aids)
    return pl.DataFrame(
        {
            "aid": aids[order],
            "emb": pl.Series(
                wv.vectors[order].astype(np.float32),
                dtype=pl.Array(pl.Float32, PARAMS["vector_size"]),
            ),
        }
    )


def main(args):
    root = folds.output_root(args.sample)
    for k in args.folds:
        start = time.time()
        path = folds.feature_path(k, "word2vec", root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            corpus = Path(tmp) / "sessions.txt"
            n = write_corpus(folds.fe_events(k, args.sample), corpus)
            print(f"fold {k}: {n:,} sessions ({time.time() - start:.0f} s)")
            emb = train(corpus, args.workers, args.seed)
        emb.write_parquet(path)
        print(
            f"fold {k}: wrote {path.relative_to(root.parent)}: {emb.height:,} aids ({time.time() - start:.0f} s)"
        )


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
        "--workers", type=int, default=os.cpu_count(), help="gensim threads (Benny: 8)"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="gensim's seed (its default)"
    )
    main(parser.parse_args())
