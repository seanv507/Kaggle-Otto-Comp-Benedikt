"""Clicks ranker, step 1: candidates and features per fold (Benny's 01_Split.ipynb).

See ranking/split.py. Needs the candidate sources (01b, 01c, 01e, 01f) and the
shared feature files (01f) of the same --sample.

Run from the repo root:
    uv run python 02_Clicks/split.py --sample 0.02 --folds 0 1
"""

import argparse

import polars as pl

import folds
from ranking import split

TARGET = "clicks"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(folds.N_FOLDS))
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(folds.DATASETS), choices=folds.DATASETS
    )
    parser.add_argument(
        "--sample", type=float, default=1.0, help="fraction of sessions"
    )
    parser.add_argument(
        "--engine", default="streaming", help="streaming, in-memory or gpu"
    )
    args = parser.parse_args()
    engine = pl.GPUEngine(raise_on_fail=True) if args.engine == "gpu" else args.engine
    for k in args.folds:
        for dataset in args.datasets:
            split.build(k, dataset, TARGET, args.sample, engine)
