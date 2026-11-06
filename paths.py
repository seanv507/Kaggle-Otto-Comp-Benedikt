"""Data file locations, anchored at the repo root.

Benny's notebooks reached these through per-directory `data` symlinks
(see create_symbolic_links.sh). Polars code uses these constants instead.
The inputs are symlinks created by 00_Preprocess/preprocess.py.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"  # Kaggle (LB) data
PREPROCESS = ROOT / "preprocess"  # local CV: weeks 1-3 train, week 4 truncated
DATA_TMP = ROOT / "data_tmp"  # intermediate files shared by all folds
DATA_FOLDS = ROOT / "data_folds"  # per-fold files

# LB: all 4 weeks of train (Benny: data/train/interim/*.parquet) and the Kaggle test week
TRAIN = DATA / "train.parquet"
TEST = DATA / "test.parquet"
# the Kaggle test week untruncated (published after the competition), and its labels
TEST_FULL = DATA / "test_full.parquet"
TEST_LABELS = DATA / "test_labels.parquet"

# local CV (Benny: preprocess/train/interim/*.parquet, preprocess/test*.parquet)
CV_TRAIN = PREPROCESS / "train.parquet"
CV_TEST = PREPROCESS / "test.parquet"
CV_TEST_LABELS = PREPROCESS / "test_labels.parquet"

# Benny's names for the truncated week 4 and its labels
XGB_TRAIN_X = CV_TEST
XGB_TRAIN_Y = CV_TEST_LABELS

# session splits (Benny: sessions_eval.pickle, split/sub_sessoins_10.pickle)
SESSIONS_EVAL = DATA_TMP / "sessions_eval.parquet"
SUB_SESSIONS = DATA_TMP / "split" / "sub_sessions.parquet"
