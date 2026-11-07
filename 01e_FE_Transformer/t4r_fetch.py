"""Install the A100 run's transformer candidates as the shared candidate source `t4r`.

run_a100.sh uploads them to the HF dataset as benny_t4r/cand_{train,sub}.parquet.
This downloads them (or takes an existing download), checks them against the
candidate contract (folds.write_candidates) and writes them to
    data_folds/shared/candidates/{train,sub}/t4r/cand.parquet
with their recall@20, which should match the run's log.

Run from the repo root:
    uv run python 01e_FE_Transformer/t4r_fetch.py
"""

import argparse
import subprocess
from pathlib import Path

import polars as pl

import folds
import metrics
import paths

HF_DATASET = "seanv507/kaggle-otto-multi-objective"
HF_DIR = "benny_t4r"
DOWNLOAD = paths.DATA_TMP / "t4r_a100"
SOURCE = "t4r"


def download(dest: Path) -> None:
    files = [f"{HF_DIR}/cand_{dataset}.parquet" for dataset in folds.DATASETS]
    cmd = [
        "uvx",
        "--from",
        "huggingface_hub",
        "hf",
        "download",
        HF_DATASET,
        "--repo-type",
        "dataset",
    ]
    subprocess.run([*cmd, *files, "--local-dir", str(dest)], check=True)


def main(keep_download: bool):
    download(DOWNLOAD)
    for dataset in folds.DATASETS:
        src = DOWNLOAD / HF_DIR / f"cand_{dataset}.parquet"
        cands = pl.read_parquet(src)
        path = folds.write_candidates(cands, None, dataset, SOURCE)
        r = metrics.recall(
            cands, folds.labels(None, dataset), "t4r_rank", descending=False
        )
        print(
            f"wrote {path.relative_to(paths.ROOT)}: {cands.height:,} rows; recall@20 {metrics.format_recall(r)}"
        )
        if not keep_download:
            src.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--keep-download", action="store_true", help=f"keep the files in {DOWNLOAD}"
    )
    main(parser.parse_args().keep_download)
