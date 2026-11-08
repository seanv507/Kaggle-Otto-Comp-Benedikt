#!/usr/bin/env bash
# Benny's whole pipeline on the full data, on a cloud GPU pod: preprocess, the candidate
# sources and features of all 5 folds, the three rankers, and submission.csv with its
# score on the Kaggle test labels (Benny's own part: LB 0.601).
#
# Needs an NVIDIA driver with CUDA 13 (e.g. an A100 80 GB; not a V100) and HF_TOKEN (read
# access to the HF dataset; write access for the uploads). Sizes scaled linearly from the
# 2% sample (5.3 GB for 2 folds), not measured on the full data:
#   - disk: about 120 GB per fold (candidates, matrices, ranker rows) and 35 GB for the
#     ranker scores, so >= 800 GB on the volume;
#   - RAM: >= 256 GB (Chris's machine; the GRU holds ~150M training examples per fold,
#     and XGBoost loads up to 5 ranker chunks of ~10 GB each).
# Keep WORK on the pod's volume (/workspace) so outputs survive a stop.
#
# As the pod's start command (no SSH needed; the pod stops itself at the end). Set the env
# vars HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }} and NTFY_TOPIC={{ RUNPOD_SECRET_NTFY_TOPIC }}
# (Runpod secrets) and the start command to:
#   bash -c '(/start.sh >/workspace/start.log 2>&1 &); mkdir -p /workspace/logs; curl -sSL https://raw.githubusercontent.com/seanv507/Kaggle-Otto-Comp-Benedikt/master/run_a100.sh | bash 2>&1 | tee -a /workspace/logs/run.log; sleep infinity'
# The branch in the URL and BRANCH below must match.
#
# Or by hand over SSH (inside tmux, so a dropped connection does not kill it):
#   export HF_TOKEN=...
#   bash run_a100.sh
#
# Locally, as a check of the wiring on a sample (the current checkout, no clone, download,
# upload or pod stop; outputs in data_folds/sample_<f>):
#   LOCAL=1 SAMPLE=0.02 FOLDS="0 1" GRU_ARGS="--max-steps 300 --width 32 --eval-batch-size 64" \
#       RANKER_ARGS="--trees 50" WORK=<scratch dir> bash run_a100.sh
#
# Steps (each one leaves $WORK/done/<step> when it succeeds and is skipped on a rerun;
# delete the marker to redo a step):
#   code       clone this repo and Chris's (a path dependency) side by side, uv sync
#   inputs     download train, test, test_full and local_valid/* from HF
#   preprocess 00_Preprocess/preprocess.py --check
#   t4r        01e_FE_Transformer/t4r_fetch.py: the 2026-11-06 run's t4r candidates (shared);
#              with T4R_TRAIN=1 instead: t4r_data, t4r_train (~1.5 h) and t4r_upload
#              (candidates, item map and last checkpoint to HF benny_t4r/)
#   other      01f: pv and pv_2 candidates, session and item features (shared)
#   benny_covisits, chris_covisits, word2vec, gru: 01b, 01c, 01a, 01d for FOLDS
#   baseline   00_submissions/01_baseline/baseline.py (fills short sessions)
#   clicks, carts, orders: the rankers (split, CV, submission models; orders' stacking)
#   submission ranking submission: submission.csv and its weighted recall@20
# Notifications (if NTFY_TOPIC is set): ntfy.sh on each completed step, at the end, and
# with high priority on failure. On exit, success or failure, the logs, the submission and
# the CV results are uploaded (if UPLOAD=1) and the pod stops (if STOP_POD=1 on Runpod).
set -euo pipefail

REPO=${REPO:-https://github.com/seanv507/Kaggle-Otto-Comp-Benedikt.git}
BRANCH=${BRANCH:-master}
# Chris's part, a path dependency (for covisits.py), cloned next to this repo
CHRIS_REPO=${CHRIS_REPO:-https://github.com/seanv507/Kaggle-OTTO-Comp.git}
CHRIS_BRANCH=${CHRIS_BRANCH:-importable-covisits}
HF_DATASET=${HF_DATASET:-seanv507/kaggle-otto-multi-objective}
HF_OUT=${HF_OUT:-benny_full}  # folder in the HF dataset for the outputs
WORK=${WORK:-/workspace}
CODE=$WORK/code
DATA=$WORK/data
FOLDS=${FOLDS:-0 1 2 3 4}
SAMPLE=${SAMPLE:-}  # e.g. 0.02: every step on that fraction of the sessions
LOCAL=${LOCAL:-0}  # 1: use the checkout this script is in (see above)
if [ "$LOCAL" = 1 ]; then UPLOAD=0 STOP_POD=0; fi
UPLOAD=${UPLOAD:-1}  # 0: keep the outputs on the pod only
STOP_POD=${STOP_POD:-1}  # 0: leave the pod running at the end (e.g. to inspect it)
# after a failure, wait this long before stopping: a stopped pod's GPU is soon taken by
# another renter, so a fix is applied by restarting the running pod
FAIL_GRACE_MIN=${FAIL_GRACE_MIN:-20}
TORCH_EXTRA=${TORCH_EXTRA:-cu132}
# extra arguments per step, e.g. BENNY_COVISIT_ARGS="--engine gpu" or
# CHRIS_COVISIT_ARGS="--session-chunk 500000" if a matrix runs out of memory
BENNY_COVISIT_ARGS=${BENNY_COVISIT_ARGS:-}
CHRIS_COVISIT_ARGS=${CHRIS_COVISIT_ARGS:-}
GRU_ARGS=${GRU_ARGS:-}
RANKER_ARGS=${RANKER_ARGS:-}  # e.g. "--trees 100" for a quicker, weaker run
# 1: train the transformer (01e) instead of fetching the candidates of the 2026-11-06 run
# (full data only; t4r_train's --sample means something else)
T4R_TRAIN=${T4R_TRAIN:-0}
T4R_ARGS=${T4R_ARGS:---tf32}  # for t4r_train.py; TF32 matmuls on the A100
HF_T4R_OUT=${HF_T4R_OUT:-benny_t4r}  # where T4R_TRAIN=1 uploads, and t4r_fetch.py reads
NTFY_SERVER=${NTFY_SERVER:-https://ntfy.sh}
# Runpod's /workspace volume is a network filesystem, where uv's many small writes can fail
# ("Stale file handle"); the venv need not survive a stop, so keep it on the container disk
if [ "$LOCAL" != 1 ]; then
    export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/root/benny-venv}
    export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
fi
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
RUN_ID=${RUN_ID:-${RUNPOD_POD_ID:-local}}  # HF folder for this run's logs

LOGS=$WORK/logs
DONE=$WORK/done
mkdir -p "$LOGS" "$DONE" "$CODE"

hf() { uvx --from huggingface_hub hf "$@"; }
run() { uv run --extra "$TORCH_EXTRA" python "$@"; }
S=()  # the sample argument of every step after preprocessing
if [ -n "$SAMPLE" ]; then S=(--sample "$SAMPLE"); fi

upload() {  # upload <local path> <path in the dataset> <message>
    [ -e "$1" ] || return 0
    hf upload "$HF_DATASET" "$1" "$2" --repo-type dataset --commit-message "$3" || echo "upload of $1 failed"
}

notify() {  # notify <message> [priority]
    [ -n "${NTFY_TOPIC:-}" ] || return 0
    curl -fsS -m 15 -o /dev/null -H "Title: Benny full run${RUNPOD_POD_ID:+ ($RUNPOD_POD_ID)}" \
        -H "Priority: ${2:-default}" -d "$1" "$NTFY_SERVER/$NTFY_TOPIC" || echo "ntfy failed"
}

stop_pod() {
    command -v runpodctl >/dev/null || { echo "runpodctl not found: stop the pod yourself"; return; }
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID" \
        || echo "could not stop pod $RUNPOD_POD_ID: stop it yourself"
}

if [ "$LOCAL" = 1 ]; then
    ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
else
    ROOT=$CODE/Kaggle-Otto-Comp-Benedikt
fi
OUT=$ROOT/data_folds${SAMPLE:+/sample_$SAMPLE}
on_exit() {
    local status=$?
    echo "run_a100.sh exited with status $status at $(date -u +%FT%TZ)" | tee -a "$LOGS/status.txt"
    if [ "$status" = 0 ]; then
        notify "Finished: $(cat "$LOGS/summary.txt" 2>/dev/null || echo 'no summary')"
    else
        notify "FAILED (exit $status) in step '${STEP:-?}' after $(( (SECONDS - STEP_T0) / 60 )) min.
Last lines:
$(tail -n 5 "$LOGS/${STEP:-run}.log" 2>/dev/null)" high
    fi
    if [ "$UPLOAD" = 1 ]; then
        upload "$LOGS" "$HF_OUT/logs/$RUN_ID" "Logs of the Benny full run $RUN_ID (exit $status)"
        upload "$OUT/submission.csv" "$HF_OUT/submission.csv" "Benny full run: submission (exit $status)"
        for d in clicks carts orders/orders; do
            upload "$OUT/ranker/$d/sub_top20.parquet" "$HF_OUT/ranker/$d/sub_top20.parquet" \
                "Benny full run: $d top 20 (exit $status)"
        done
    fi
    if [ "$STOP_POD" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        if [ "$status" != 0 ] && [ "$FAIL_GRACE_MIN" -gt 0 ]; then
            echo "waiting $FAIL_GRACE_MIN min before stopping (restart the pod to retry on its GPU)"
            sleep $(( FAIL_GRACE_MIN * 60 ))
        fi
        echo "stopping pod $RUNPOD_POD_ID"
        stop_pod
    fi
}
trap on_exit EXIT

# step <name> <command...>: run the command once, logging to $LOGS/<name>.log, unless
# $DONE/<name> exists; notify with the log's recall lines when it completes
STEP="" STEP_T0=$SECONDS
step() {
    local name=$1; shift
    if [ -e "$DONE/$name" ]; then echo "=== $name: done before, skipping"; return; fi
    STEP=$name STEP_T0=$SECONDS
    echo "=== $name ($(date -u +%T))" | tee -a "$LOGS/steps.txt"
    "$@" 2>&1 | tee -a "$LOGS/$name.log"
    touch "$DONE/$name"
    local min=$(( (SECONDS - STEP_T0) / 60 ))
    echo "=== $name done in $min min" | tee -a "$LOGS/steps.txt"
    notify "Done: $name in $min min
$(grep -hiE "recall|score" "$LOGS/$name.log" | tail -n 12 || true)"
}

# code (always updated, so a rerun picks up fixes)
STEP=code
code() {
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
cd "$CODE"
[ -d Kaggle-OTTO-Comp ] || git clone --branch "$CHRIS_BRANCH" "$CHRIS_REPO"
(cd Kaggle-OTTO-Comp && git fetch origin && git checkout "$CHRIS_BRANCH" && git pull --ff-only)
[ -d Kaggle-Otto-Comp-Benedikt ] || git clone --branch "$BRANCH" "$REPO"
cd "$ROOT"
git fetch origin && git checkout "$BRANCH" && git pull --ff-only
for attempt in 1 2 3; do  # retried: see UV_PROJECT_ENVIRONMENT above
    if uv sync --extra "$TORCH_EXTRA"; then break; fi
    if [ "$attempt" = 3 ]; then exit 1; fi
    echo "uv sync failed (attempt $attempt), retrying"
    sleep 10
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$LOGS/gpu.txt"
}
if [ "$LOCAL" = 1 ]; then cd "$ROOT"; else code; fi
git rev-parse HEAD | tee "$LOGS/commit.txt"
# Polars sizes its thread pool from the host's cores; in a container with a CPU quota that
# oversubscribes, so cap it at the quota (cgroup v2 or v1)
cpu_quota() {
    local quota period
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    fi
    if [ -n "${quota:-}" ] && [ "$quota" != max ] && [ "$quota" -gt 0 ]; then
        echo $(( (quota + period - 1) / period ))
    fi
}
if [ -z "${POLARS_MAX_THREADS:-}" ] && [ -n "$(cpu_quota)" ]; then
    POLARS_MAX_THREADS=$(cpu_quota)
    export POLARS_MAX_THREADS
fi
{ echo "nproc $(nproc), CPU quota $(cpu_quota || true), POLARS_MAX_THREADS ${POLARS_MAX_THREADS:-unset}"
  free -g; df -h "$WORK"; } | tee "$LOGS/machine.txt"

# inputs, gathered in one directory under the names preprocess.py expects
inputs() {
    hf download "$HF_DATASET" --repo-type dataset \
        "train.parquet" "test.parquet" "test_full.parquet" "local_valid/*" --local-dir "$DATA/hf"
    mkdir -p "$DATA/src"
    for f in train test test_full; do ln -sf "$DATA/hf/$f.parquet" "$DATA/src/$f.parquet"; done
    for f in valid_train valid_test valid_test_labels; do
        ln -sf "$DATA/hf/local_valid/$f.parquet" "$DATA/src/$f.parquet"
    done
}
if [ "$LOCAL" = 1 ]; then
    # the data preprocess.py linked before
    step preprocess run 00_Preprocess/preprocess.py
else
    step inputs inputs
    # links, labels and session splits; run every time (it is quick and makes the links)
    STEP=preprocess
    run 00_Preprocess/preprocess.py --src "$DATA/src" --check 2>&1 | tee -a "$LOGS/preprocess.log"
fi

# shared sources and features
t4r_upload() {
    local cands=data_folds/shared/candidates last
    last=$(printf "%s\n" data_tmp/t4r/checkpoints/epoch_*.pt | sort -V | tail -1)
    for f in "$cands/train/t4r/cand.parquet:cand_train.parquet" "$cands/sub/t4r/cand.parquet:cand_sub.parquet" \
             "data_tmp/t4r/aid_map.parquet:aid_map.parquet" "$last:$(basename "$last")"; do
        hf upload "$HF_DATASET" "${f%%:*}" "$HF_T4R_OUT/${f##*:}" --repo-type dataset \
            --commit-message "Benny t4r: ${f##*:} ($(cat "$LOGS/commit.txt"))"
    done
}
if [ "$T4R_TRAIN" = 1 ]; then
    if [ -n "$SAMPLE" ]; then echo "T4R_TRAIN=1 runs on the full data only"; exit 1; fi
    step t4r_data run 01e_FE_Transformer/t4r_data.py
    # shellcheck disable=SC2086  # T4R_ARGS is a list of arguments
    step t4r_train run 01e_FE_Transformer/t4r_train.py $T4R_ARGS
    if [ "$UPLOAD" = 1 ]; then step t4r_upload t4r_upload; fi
else
    step t4r run 01e_FE_Transformer/t4r_fetch.py "${S[@]}"
fi
other() {
    run 01f_FE_other/previous_visits.py "${S[@]}"
    run 01f_FE_other/session_features.py "${S[@]}"
    run 01f_FE_other/item_features.py "${S[@]}"
}
step other other

# per-fold sources and features, one fold at a time, so a failure keeps the folds before it
# shellcheck disable=SC2086  # the *_ARGS are lists of arguments
for k in $FOLDS; do
    step "benny_covisits_$k" run 01b_FE_CoVisit_1/benny_covisits.py --folds "$k" "${S[@]}" $BENNY_COVISIT_ARGS
    step "chris_covisits_$k" run 01c_FE_CoVisit_2_Chris/chris_covisits.py --folds "$k" "${S[@]}" $CHRIS_COVISIT_ARGS
    # gensim's default is os.cpu_count(), the host's cores, not the container's quota
    step "word2vec_$k" run 01a_FE_Word2Vec/word2vec.py --folds "$k" "${S[@]}" --workers "${POLARS_MAX_THREADS:-$(nproc)}"
    step "gru_$k" run 01d_FE_GRU/gru_train.py --folds "$k" "${S[@]}" $GRU_ARGS
done
step baseline run 00_submissions/01_baseline/baseline.py "${S[@]}"

# rankers (see ranking/README.md for the order)
# shellcheck disable=SC2086
r() { run -m ranking "$@" --folds $FOLDS "${S[@]}" $RANKER_ARGS; }
clicks() { r split clicks; r cv clicks; r sub clicks; }
carts() { r split carts; r cv carts; r sub carts; }
orders() {
    r split orders/clicks
    r split orders/carts_orders
    r stack orders/clicks
    r stack orders/carts
    r cv orders
    r sub orders
}
step clicks clicks
step carts carts
step orders orders
STEP=submission
run -m ranking submission "${S[@]}" 2>&1 | tee -a "$LOGS/submission.log"

{ grep -hE "cv recall@20|Kaggle test" "$LOGS"/{clicks,carts,orders,submission}.log 2>/dev/null || true; } \
    | tee "$LOGS/summary.txt"
