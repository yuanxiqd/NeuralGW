#!/usr/bin/env bash
# Cross-domain NeuralGW for the structural clustering table: train on SRC, cluster TGT.
#   bash scripts/run_cdomain_social.sh <SRC> <TGT>
#
# For cross-domain (target appears once) the split is train=SRC-only / test=TGT-only
# (DataPreparationSystem.train_test_split, the "==1" branch), so the SRC x TGT cross-GW
# block is never read during training or eval. We therefore place a ZERO placeholder for
# the cross block to satisfy the precompute existence check and skip the (expensive)
# cross-GW computation. Only the within-domain blocks (SRC_SRC, TGT_TGT) are used.
set -o pipefail
cd "$(dirname "$0")/.."

# Optional: activate the conda environment named by NEURALGW_CONDA_ENV; otherwise use the current Python.
if [ -n "${NEURALGW_CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$NEURALGW_CONDA_ENV"
fi
export NEURALGW_DATA_ROOT=${NEURALGW_DATA_ROOT:-./data_300_350}
export PYTHONUNBUFFERED=1
CUDA_ID="${CUDA_ID:-0}"

SRC="$1"; TGT="$2"
if [[ -z "$SRC" || -z "$TGT" ]]; then echo "usage: $0 <SRC> <TGT>" >&2; exit 1; fi
SEEDS=(0 1 2 3 4)
GWD="$NEURALGW_DATA_ROOT/precomputed/gwd_adj"
LOGDIR=logs_cdomain_test/clustering_table
mkdir -p "$LOGDIR" results/clustering

# Place the zero cross block (unused in cross-domain training; avoids expensive cross-GW).
python - "$SRC" "$TGT" "$GWD" <<'PY'
import sys, os, numpy as np
src, tgt, gwd = sys.argv[1], sys.argv[2], sys.argv[3]
def sz(name):
    return int(np.load(f"{gwd}/{name}_{name}_distance_matrix.npy", mmap_mode="r").shape[0])
ns, nt = sz(src), sz(tgt)
f1 = f"{gwd}/{src}_{tgt}_distance_matrix.npy"; f2 = f"{gwd}/{tgt}_{src}_distance_matrix.npy"
if not os.path.exists(f1) and not os.path.exists(f2):
    np.save(f1, np.zeros((ns, nt), dtype=np.float32))
    print(f"placed ZERO cross {src}_{tgt} {ns}x{nt} (UNUSED: train=SRC-only, test=TGT-only)")
else:
    print("cross block already present; leaving as-is")
PY

PREFIX="${SRC}_to_${TGT}"
echo "[$(date '+%F %T')] cross-domain queue: $PREFIX (cuda:$CUDA_ID)"
for s in "${SEEDS[@]}"; do
  out="results/clustering/${PREFIX}_seed${s}_distance_matrix.npy"
  if [[ -f "$out" ]]; then echo "SKIP (exists): $out"; continue; fi
  echo "RUN $PREFIX seed$s"
  python main_cdomain.py --cuda_id "$CUDA_ID" --task neuralgw \
      --ds_name1 "$SRC" --ds_name2 "$TGT" --n_gammas 9 --seed "$s" --model_type parallel \
      > "$LOGDIR/${PREFIX}_seed${s}.log" 2>&1
  [[ -f "$out" ]] && echo "OK $out" || echo "FAIL $PREFIX seed$s (see $LOGDIR/${PREFIX}_seed${s}.log)"
done
echo "[$(date '+%F %T')] DONE $PREFIX"
