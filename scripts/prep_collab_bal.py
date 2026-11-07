"""Prepare a balanced 3-class COLLAB subset ('COLLAB_BAL') for the clustering table.

Deterministic subset = first M graphs per class (sorted by global index). Computes the
exact-GW distance matrix with the SAME convention as the pipeline's _build_graph_cost
('adj': adjacency + self-loops) and ot.gromov.gromov_wasserstein2(tol_rel=1e-4, tol_abs=1e-4,
uniform marginals, square_loss), so the matrix is valid as NeuralGW training labels and the
ceiling is comparable to REDDIT.

Writes:
  $NEURALGW_DATA_ROOT/precomputed/gwd_adj/COLLAB_BAL_COLLAB_BAL_distance_matrix.npy  (training labels)
  results/clustering/COLLAB_BAL_exactgw_distance_matrix.npy                          (ceiling matrix)
  results/clustering/COLLAB_BAL_labels.npy                                           (labels, matrix order)

The subset selection here MUST match the 'COLLAB_BAL' branch in _load_graph_ds.
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

M = 25
ROOT = os.environ.get("NEURALGW_DATA_ROOT", "./data_300_350")


def balanced_idx(ds):
    byc = defaultdict(list)
    for i in range(len(ds)):
        byc[int(ds[i].y)].append(i)
    idx = []
    for c in sorted(byc):
        idx += byc[c][:M]
    return sorted(idx)


def cost(data):
    A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).numpy().astype(np.float32)
    A += np.eye(A.shape[0], dtype=np.float32)  # self-loops -> matches _build_graph_cost('adj')
    return A


def hacc(yt, yp):
    cm = confusion_matrix(yt, yp)
    r, c = linear_sum_assignment(-cm)
    m = np.zeros_like(yp)
    for a, b in zip(r, c):
        m[yp == b] = a
    return accuracy_score(yt, m)


def main():
    ds = TUDataset(root=f"{ROOT}/TUDataset", name="COLLAB")
    idx = balanced_idx(ds)
    y = np.array([int(ds[i].y) for i in idx])
    print(f"COLLAB_BAL: n={len(idx)} classes={dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)]))}",
          flush=True)
    Cs = [cost(ds[i]) for i in idx]
    n = len(idx)
    D = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            D[a, b] = D[b, a] = float(ot.gromov.gromov_wasserstein2(C1=Cs[a], C2=Cs[b],
                                                                    tol_rel=1e-4, tol_abs=1e-4))
    gwd = f"{ROOT}/precomputed/gwd_adj/COLLAB_BAL_COLLAB_BAL_distance_matrix.npy"
    os.makedirs(os.path.dirname(gwd), exist_ok=True)
    np.save(gwd, D)
    os.makedirs("results/clustering", exist_ok=True)
    np.save("results/clustering/COLLAB_BAL_exactgw_distance_matrix.npy", D)
    np.save("results/clustering/COLLAB_BAL_labels.npy", y)
    print(f"saved training labels -> {gwd}  shape={D.shape}", flush=True)

    DD = 0.5 * (D + D.T); np.fill_diagonal(DD, 0.0)
    s = np.median(DD[np.triu_indices_from(DD, 1)]); s = s if s > 0 else 1.0
    A = np.exp(-(DD ** 2) / (2 * s ** 2))
    p = SpectralClustering(n_clusters=len(np.unique(y)), affinity="precomputed",
                           assign_labels="discretize", n_init=10, random_state=42).fit_predict(A)
    print(f"A+I exact-GW ceiling: ACC={hacc(y, p):.3f} "
          f"NMI={normalized_mutual_info_score(y, p):.3f} ARI={adjusted_rand_score(y, p):.3f}", flush=True)


if __name__ == "__main__":
    main()
