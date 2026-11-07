"""In-domain NeuralGW on a (possibly custom-subset) target for the clustering table.

Generalizes scripts/run_indomain_reddit.py: for each seed, train in-domain on --ds-name
(its subset is whatever DataPreparationSystem._load_graph_ds returns), then predict the
FULL pairwise distance matrix (train+eval+test) and save:

    results/clustering/{ds}_indomain_seed{k}_distance_matrix.npy

Used for COLLAB_BAL (balanced 3-class COLLAB; GW training labels prepared by
scripts/prep_collab_bal.py). Idempotent; aborts if the first trained seed fails.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import json
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import torch
from tqdm import tqdm

from models.neuralgw import NeuralGW


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    wid = torch.initial_seed() % 2 ** 32
    np.random.seed(wid); random.seed(wid)


def _prep_pe(model, idx, device):
    raw = model.dps.pe_for_global_index(model.total_pes, idx)
    d = model.input_dim
    if raw.shape[-1] < d:
        pe = np.pad(raw, ((0, 0), (0, d - raw.shape[-1])))
    elif raw.shape[-1] > d:
        pe = raw[:, :d]
    else:
        pe = raw
    return torch.tensor(pe, dtype=torch.float32, device=device)


def predict_full_matrix(model, device, full_idx):
    model.model.load_state_dict(model.best_model_state)
    model.eval()
    n = len(full_idx)
    D = np.zeros((n, n), dtype=np.float64)
    with torch.no_grad():
        for i in tqdm(range(n)):
            for j in range(i + 1, n):
                idx1, idx2 = full_idx[i], full_idx[j]
                pe1 = _prep_pe(model, idx1, device)
                ds1, g1 = model.dps.map_global_idx_to_local_indices(global_idx=idx1)
                ei1 = model.ds_seq[ds1][g1].edge_index.to(device)
                pe2 = _prep_pe(model, idx2, device)
                ds2, g2 = model.dps.map_global_idx_to_local_indices(global_idx=idx2)
                ei2 = model.ds_seq[ds2][g2].edge_index.to(device)
                v = model.model.forward(X1=pe1, edge_index1=ei1, X2=pe2, edge_index2=ei2).item()
                D[i, j] = D[j, i] = v
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds-name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--treat-undirected", type=int, default=1)
    args = ap.parse_args()

    cuda_id = int(os.environ.get("CUDA_ID", "0"))
    with open("configs/main.json", "r") as f:
        base_config = json.load(f)
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/clustering", exist_ok=True)

    first_done = False
    for seed in args.seeds:
        out = f"results/clustering/{args.ds_name}_indomain_seed{seed}_distance_matrix.npy"
        if os.path.exists(out):
            print(f"SKIP (exists): {out}", flush=True); continue

        config = dict(base_config)
        config.update(seed=seed, cuda_id=cuda_id, model_type="parallel", switch="gin",
                      test_ratio=0.2, pe_mode="svd", treat_as_undirected=bool(args.treat_undirected),
                      n_gammas=9)
        config.setdefault("pe_cost_mode", "adj"); config.setdefault("gw_cost_mode", "adj")

        print(f"\n===== in-domain {args.ds_name} seed {seed} (cuda:{cuda_id}) =====", flush=True)
        set_seed(seed)
        model = NeuralGW(ds_name_seq=[args.ds_name], testset="global", is_naive=False,
                         config=config, device=device).to(device)
        model.train_model(seed_worker=seed_worker)

        full_idx = sorted(list(model.train_idx) + list(model.eval_idx) + list(model.test_idx))
        print(f"[seed {seed}] predicting full matrix over {len(full_idx)} graphs", flush=True)
        D = predict_full_matrix(model, device, full_idx)
        np.save(out, D)
        print(f"[seed {seed}] saved {D.shape} -> {out}", flush=True)

        if not os.path.exists(out) and not first_done:
            print(f"ABORT: seed {seed} produced no output", flush=True)
            sys.exit(1)
        first_done = True


if __name__ == "__main__":
    main()
