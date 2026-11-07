"""Prepare balanced bioinformatics subsets (DD_BAL, ENZYMES_BAL) for the clustering table
and report their exact-GW clustering ceiling (validation-first).

Deterministic subset = first M graphs per class (optionally restricted to num_nodes<=cap,
for tractable GW on large graphs), sorted by global index. Exact GW uses the pipeline's
convention (A+I cost; ot.gromov.gromov_wasserstein2(tol_rel=1e-4, tol_abs=1e-4)).

Writes (so the subset can later serve as NeuralGW training labels, like COLLAB_BAL):
  $NEURALGW_DATA_ROOT/precomputed/gwd_adj/{NAME}_BAL_{NAME}_BAL_distance_matrix.npy
  results/clustering/{NAME}_BAL_exactgw_distance_matrix.npy
  results/clustering/{NAME}_BAL_labels.npy

The subset selection MUST match the {NAME}_BAL branch added to _load_graph_ds.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import ot
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix,
                             normalized_mutual_info_score)
from scipy.optimize import linear_sum_assignment

ROOT = os.environ.get("NEURALGW_DATA_ROOT", "./data_300_350")
GWD = f"{ROOT}/precomputed/gwd_adj"

# name -> (per-class count M, node cap or None)
CONFIG = {
    "ENZYMES": (25, None),   # 6 classes, small graphs (~33 nodes)
    "DD": (30, 150),         # 2 classes, large graphs -> cap for tractable GW
}


def balanced_idx(ds, M, cap):
    byc = defaultdict(list)
    for i in range(len(ds)):
        if cap is None or ds[i].num_nodes <= cap:
            byc[int(ds[i].y)].append(i)
    idx = []
    for c in sorted(byc):
        idx += byc[c][:M]
    return sorted(idx)


def cost(data):
    A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).numpy().astype(np.float32)
    A += np.eye(A.shape[0], dtype=np.float32)
    return A


def hacc(yt, yp):
    cm = confusion_matrix(yt, yp); r, c = linear_sum_assignment(-cm); m = np.zeros_like(yp)
    for a, b in zip(r, c):
        m[yp == b] = a
    return accuracy_score(yt, m)


def main():
    os.makedirs(GWD, exist_ok=True); os.makedirs("results/clustering", exist_ok=True)
    names = sys.argv[1:] or list(CONFIG)
    for name in names:
        M, cap = CONFIG[name]
        ds = TUDataset(root=f"{ROOT}/TUDataset", name=name)
        idx = balanced_idx(ds, M, cap)
        y = np.array([int(ds[i].y) for i in idx])
        sizes = [ds[i].num_nodes for i in idx]
        print(f"{name}_BAL: n={len(idx)} cap={cap} "
              f"classes={dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)]))} "
              f"nodes=({min(sizes)},{max(sizes)})", flush=True)
        Cs = [cost(ds[i]) for i in idx]
        n = len(idx)
        D = np.zeros((n, n), dtype=np.float32)
        for a in range(n):
            for b in range(a + 1, n):
                D[a, b] = D[b, a] = float(ot.gromov.gromov_wasserstein2(C1=Cs[a], C2=Cs[b],
                                                                        tol_rel=1e-4, tol_abs=1e-4))
            if a % 20 == 0:
                print(f"    {name} row {a}/{n}", flush=True)
        np.save(f"{GWD}/{name}_BAL_{name}_BAL_distance_matrix.npy", D)
        np.save(f"results/clustering/{name}_BAL_exactgw_distance_matrix.npy", D)
        np.save(f"results/clustering/{name}_BAL_labels.npy", y)
        DD = 0.5 * (D + D.T); np.fill_diagonal(DD, 0.0)
        s = np.median(DD[np.triu_indices_from(DD, 1)]); s = s if s > 0 else 1.0
        A = np.exp(-(DD ** 2) / (2 * s ** 2))
        p = SpectralClustering(n_clusters=len(np.unique(y)), affinity="precomputed",
                               assign_labels="discretize", n_init=10, random_state=42).fit_predict(A)
        print(f"  {name}_BAL exact-GW ceiling: ACC={hacc(y, p):.3f} "
              f"NMI={normalized_mutual_info_score(y, p):.3f} ARI={adjusted_rand_score(y, p):.3f}", flush=True)


if __name__ == "__main__":
    main()
