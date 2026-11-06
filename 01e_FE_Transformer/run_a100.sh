#!/usr/bin/env bash
# Full transformer run (01e) on a cloud GPU pod: preprocess, sequences, training, candidates.
#
# Needs an NVIDIA driver with CUDA 13 (e.g. an A100; not a V100: CUDA 13 dropped Volta)
# and HF_TOKEN with write access to the HF dataset (for the uploads). The full model needs
# about 8 GB of GPU memory plus activations, and the data steps about 10 GB of RAM. Keep
# WORK on the pod's volume (/workspace) so outputs survive a stop.
#
# As the pod's start command (no SSH needed; the pod stops itself at the end). Set the env
# var HF_TOKEN={{ RUNPOD_SECRET_hf_token }} (a Runpod secret) and the start command to:
#   bash -c '(/start.sh >/workspace/start.log 2>&1 &); mkdir -p /workspace/logs; curl -sSL https://raw.githubusercontent.com/seanv507/Kaggle-Otto-Comp-Benedikt/polars-transformer/01e_FE_Transformer/run_a100.sh | bash 2>&1 | tee -a /workspace/logs/run.log; sleep infinity'
# /start.sh (in Runpod's images) keeps SSH and Jupyter available for debugging; `sleep
# infinity` stops a failed script from restart-looping the container if the self-stop
# below cannot run.
#
# Or by hand over SSH (inside tmux, so a dropped connection does not kill it):
#   export HF_TOKEN=...
#   bash 01e_FE_Transformer/run_a100.sh
#
# Steps (the data steps are skipped if their output exists, so a rerun resumes):
#   1. clone/update the repo, uv sync with CUDA torch
#   2. download train, test, test_full and local_valid/* from HF
#   3. 00_Preprocess/preprocess.py --check  (links, labels, session splits)
#   4. t4r_data.py                           (item ids, sequences)
#   5. t4r_train.py                          (training, recall@20 per epoch, candidates)
#   6. upload the candidates, the last checkpoint and the logs to HF
# On exit, success or failure: upload the logs (if UPLOAD=1) and stop the pod (if
# STOP_POD=1 and on Runpod). A stopped pod keeps /workspace and bills only its storage.
# Extra arguments for t4r_train.py go in TRAIN_ARGS, e.g. TRAIN_ARGS="--eval-batch-size 512".
set -euo pipefail

REPO=${REPO:-https://github.com/seanv507/Kaggle-Otto-Comp-Benedikt.git}
BRANCH=${BRANCH:-polars-transformer}
HF_DATASET=${HF_DATASET:-seanv507/kaggle-otto-multi-objective}
HF_OUT=${HF_OUT:-benny_t4r}  # folder in the HF dataset for the outputs
WORK=${WORK:-/workspace}
UPLOAD=${UPLOAD:-1}  # 0: keep the outputs on the pod only
STOP_POD=${STOP_POD:-1}  # 0: leave the pod running at the end (e.g. to inspect it)
TRAIN_ARGS=${TRAIN_ARGS:-}

export HF_XET_HIGH_PERFORMANCE=1
LOGS=$WORK/logs
mkdir -p "$LOGS"

hf() { uvx --from huggingface_hub hf "$@"; }

stop_pod() {
    # runpodctl and RUNPOD_POD_ID are provided inside Runpod pods
    command -v runpodctl >/dev/null || { echo "runpodctl not found: stop the pod yourself"; return; }
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID" \
        || echo "could not stop pod $RUNPOD_POD_ID: stop it yourself"
}

on_exit() {
    local status=$?
    echo "run_a100.sh exited with status $status at $(date -u +%FT%TZ)" | tee -a "$LOGS/status.txt"
    if [ "$UPLOAD" = 1 ]; then
        hf upload "$HF_DATASET" "$LOGS" "$HF_OUT/logs" --repo-type dataset \
            --commit-message "Logs of the Benny t4r run (exit $status)" || echo "log upload failed"
    fi
    if [ "$STOP_POD" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "stopping pod $RUNPOD_POD_ID"
        stop_pod
    fi
}
trap on_exit EXIT

# 1. code
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
cd "$WORK"
[ -d Kaggle-Otto-Comp-Benedikt ] || git clone --branch "$BRANCH" "$REPO"
cd Kaggle-Otto-Comp-Benedikt
git fetch origin && git checkout "$BRANCH" && git pull --ff-only
uv sync --extra cu132
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$LOGS/gpu.txt"
git rev-parse HEAD | tee "$LOGS/commit.txt"

# 2. inputs, gathered in one directory under the names preprocess.py expects
hf download "$HF_DATASET" --repo-type dataset \
    --include "train.parquet" "test.parquet" "test_full.parquet" "local_valid/*" --local-dir "$WORK/hf"
SRC=$WORK/src
mkdir -p "$SRC"
for f in train test test_full; do ln -sf "$WORK/hf/$f.parquet" "$SRC/$f.parquet"; done
for f in valid_train valid_test valid_test_labels; do ln -sf "$WORK/hf/local_valid/$f.parquet" "$SRC/$f.parquet"; done

# 3. links, labels and session splits (existing splits are kept)
uv run python 00_Preprocess/preprocess.py --src "$SRC" --check 2>&1 | tee -a "$LOGS/preprocess.log"

# 4. sequences
if [ -f data_tmp/t4r/seq_test.parquet ]; then
    echo "sequences exist: skipping t4r_data.py"
else
    uv run python 01e_FE_Transformer/t4r_data.py 2>&1 | tee -a "$LOGS/t4r_data.log"
fi

# 5. training and candidates
if [ -f data_tmp/t4r/cand_sub.parquet ]; then
    echo "candidates exist: skipping t4r_train.py (delete data_tmp/t4r/cand_*.parquet to retrain)"
else
    # shellcheck disable=SC2086  # TRAIN_ARGS is a list of arguments
    uv run python 01e_FE_Transformer/t4r_train.py $TRAIN_ARGS 2>&1 | tee -a "$LOGS/t4r_train.log"
fi

# 6. outputs
if [ "$UPLOAD" = 1 ]; then
    last=$(ls data_tmp/t4r/checkpoints/epoch_*.pt | sort -V | tail -1)
    for f in data_tmp/t4r/cand_train.parquet data_tmp/t4r/cand_sub.parquet data_tmp/t4r/aid_map.parquet "$last"; do
        hf upload "$HF_DATASET" "$f" "$HF_OUT/$(basename "$f")" --repo-type dataset \
            --commit-message "Benny t4r: $(basename "$f") ($(cat "$LOGS/commit.txt"))"
    done
fi
echo "done: recall per epoch is in $LOGS/t4r_train.log"
