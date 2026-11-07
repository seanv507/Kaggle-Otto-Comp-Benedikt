"""Clicks ranker, steps 3 and 4: XGBoost and the combined top 20 (Benny's
03_XGB-v19-clicks.ipynb and 04_Combine-Bags.ipynb). See ranking/train.py and
ranking/combine.py. Needs 02_Clicks/split.py's chunks for the same --sample.

    cv    local CV on week 4: recall@20, and the best number of trees
    sub      the submission models (Benny: 394 trees), their scores, and the combined
             top 20 with its recall on the Kaggle test (TEST_LABELS)
    combine  only the last part: the combined top 20, filled from the baseline

Run from the repo root:
    uv run python 02_Clicks/ranker.py cv --sample 0.02 --folds 0 1
    uv run python 02_Clicks/ranker.py sub --sample 0.02 --folds 0 1
"""

import argparse

import folds
from ranking import combine, train

TARGET = "clicks"
N_TREES = 394  # Benny's, from his local CV's best iterations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["cv", "sub", "combine"])
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(folds.N_FOLDS))
    )
    parser.add_argument(
        "--sample", type=float, default=1.0, help="fraction of sessions"
    )
    parser.add_argument(
        "--trees", type=int, default=N_TREES, help="sub: trees per model"
    )
    parser.add_argument("--device", default="cuda", help="XGBoost device: cuda or cpu")
    args = parser.parse_args()
    params = {"device": args.device}
    if args.mode == "cv":
        train.cv(TARGET, args.sample, args.folds, params=params)
    else:
        if args.mode == "sub":
            train.sub(TARGET, args.sample, args.folds, args.trees, params=params)
        combine.main(TARGET, args.sample, args.folds, args.folds)
