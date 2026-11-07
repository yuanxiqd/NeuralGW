#!/usr/bin/env bash
# Serial, resumable queue of NeuralGW cross-domain runs for the expanded clustering table.
#
# Targets: PTC_FM, BZR, COX2, DHFR. Each gets two single-source rows + one 2-source
# combo (Table-4 style). 5 seeds each. Runs are serial on a single GPU to avoid OOM,
# and idempotent: a (config, seed) whose output matrix already exists is skipped.
#
# Output matrices: results/clustering/{src[_src2]}_to_{testset}_seed{k}_distance_matrix.npy
# Logs: logs_cdomain_test/clustering_table/{queue.log, <prefix>_seed<k>.log}
set -o pipefail
cd "$(dirname "$0")/.."

# Optional: activate the conda environment named by NEURALGW_CONDA_ENV; otherwise use the current Python.
if [ -n "${NEURALGW_CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$NEURALGW_CONDA_ENV"
fi

# IMPORTANT: point at the complete precomputed root (GW labels + pe_adj_svd cache + TUDataset).
export NEURALGW_DATA_ROOT=${NEURALGW_DATA_ROOT:-./data_300_350}
export PYTHONUNBUFFERED=1

CUDA_ID="${CUDA_ID:-0}"
SEEDS=(0 1 2 3 4)
LOGDIR=logs_cdomain_test/clustering_table
mkdir -p "$LOGDIR" results/clustering
QLOG="$LOGDIR/queue.log"

# "TESTSET|SRC1 [SRC2]"  — source order chosen so cross-pair GW matrices exist on disk.
ENTRIES=(
  "PTC_FM|MUTAG"
  "PTC_FM|BZR"
  "PTC_FM|MUTAG BZR"
  "BZR|DHFR"
  "BZR|COX2"
  "BZR|COX2 DHFR"
  "COX2|BZR"
  "COX2|DHFR"
  "COX2|BZR DHFR"
  "DHFR|BZR"
  "DHFR|COX2"
  "DHFR|BZR COX2"
)

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

log "QUEUE START (cuda:${CUDA_ID}, data_root=${NEURALGW_DATA_ROOT})"
first_run_done=0
total=0; ran=0; skipped=0; failed=0
for entry in "${ENTRIES[@]}"; do
  testset="${entry%%|*}"
  sources="${entry#*|}"
  read -ra SRC <<< "$sources"
  prefix="$(IFS=_; echo "${SRC[*]}")_to_${testset}"
  for s in "${SEEDS[@]}"; do
    total=$((total+1))
    out="results/clustering/${prefix}_seed${s}_distance_matrix.npy"
    if [[ -f "$out" ]]; then log "SKIP (exists): $out"; skipped=$((skipped+1)); continue; fi

    DS_ARGS=(); idx=1
    for src in "${SRC[@]}"; do DS_ARGS+=(--ds_name$idx "$src"); idx=$((idx+1)); done
    DS_ARGS+=(--ds_name$idx "$testset")

    runlog="$LOGDIR/${prefix}_seed${s}.log"
    log "RUN: ${prefix} seed${s} -> $runlog"
    python main_cdomain.py --cuda_id "$CUDA_ID" --task neuralgw "${DS_ARGS[@]}" \
        --n_gammas 9 --seed "$s" --model_type parallel > "$runlog" 2>&1
    rc=$?
    if [[ $rc -ne 0 || ! -f "$out" ]]; then
      log "FAIL (rc=$rc): ${prefix} seed${s} — see $runlog"
      failed=$((failed+1))
      if [[ $first_run_done -eq 0 ]]; then
        log "ABORT: first executed run failed — stopping queue. Inspect $runlog."
        exit 1
      fi
    else
      log "OK: $out"; ran=$((ran+1))
    fi
    first_run_done=1
  done
done
log "QUEUE DONE. total=$total ran=$ran skipped=$skipped failed=$failed"
