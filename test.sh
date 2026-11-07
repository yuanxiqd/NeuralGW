#!/usr/bin/env bash
# Usage:
#   bash test.sh --trainsets PTC_FM BZR --testset MUTAG [--cuda 0] [--seed 0]
#
# --trainsets : one to three training dataset names (passed as --ds_name1/2/3); required
# --testset   : test dataset name (passed as --ds_name{N+1}, always last); required
# --cuda      : CUDA device id (default: 0)
# --seed      : random seed (default: 0)
#
# NOTE: main_cdomain.py sets testset = ds_name_seq[-1], so the testset must be
# the last --ds_name* argument. Trainsets fill --ds_name1..N, testset is --ds_name{N+1}.
set -eo pipefail
cd "$(dirname "$0")"

# Optional: activate the conda environment named by NEURALGW_CONDA_ENV; otherwise use the current Python.
if [ -n "${NEURALGW_CONDA_ENV:-}" ]; then source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$NEURALGW_CONDA_ENV"; fi
export NEURALGW_DATA_ROOT="${NEURALGW_DATA_ROOT:-$HOME/NeuralGW3_data}"

CUDA_ID=0
SEED=0
TESTSET=""
TRAINSETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)   CUDA_ID="$2";  shift 2 ;;
        --seed)   SEED="$2";     shift 2 ;;
        --testset) TESTSET="$2"; shift 2 ;;
        --trainsets)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TRAINSETS+=("$1")
                shift
            done
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TESTSET" ]]; then
    echo "Error: --testset is required." >&2; exit 1
fi
if [[ ${#TRAINSETS[@]} -eq 0 ]]; then
    echo "Error: --trainsets requires at least one dataset name." >&2; exit 1
fi
if [[ ${#TRAINSETS[@]} -gt 3 ]]; then
    echo "Error: at most 3 trainsets are supported (--ds_name1/2/3)." >&2; exit 1
fi

TASK_NAME=neuralgw
log_folder=logs_cdomain_test
mkdir -p "$log_folder"
mkdir -p results/clustering

TRAINSETS_PREFIX=$(IFS=_; echo "${TRAINSETS[*]}")
PREFIX="${TRAINSETS_PREFIX}_to_${TESTSET}"
log_file="$log_folder/log_${TASK_NAME}_${PREFIX}_ngammas9_seed${SEED}_parallel.txt"

DS_ARGS=()
for i in "${!TRAINSETS[@]}"; do
    DS_ARGS+=(--ds_name$((i + 1)) "${TRAINSETS[$i]}")
done
DS_ARGS+=(--ds_name$((${#TRAINSETS[@]} + 1)) "$TESTSET")

nohup env PYTHONUNBUFFERED=1 python -u main_cdomain.py \
    --cuda_id "$CUDA_ID" \
    --task "$TASK_NAME" \
    "${DS_ARGS[@]}" \
    --n_gammas 9 \
    --seed "$SEED" \
    --model_type parallel \
    >> "$log_file" 2>&1 &

echo "NeuralGW cross-domain run (train: ${TRAINSETS_PREFIX}, test: ${TESTSET}) started (PID $!)."
echo "Log: $log_file"
echo "Output matrix: results/clustering/${PREFIX}_seed${SEED}_distance_matrix.npy"
