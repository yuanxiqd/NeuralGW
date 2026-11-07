"""Precompute within-GW distance matrices for the synthetic cross-domain sources
(ER_SPARSE, BA_SPARSE, ER_DENSE, BA_DENSE), matching the pipeline's convention
(A+I cost, ot.gromov.gromov_wasserstein2(tol_rel=1e-4, tol_abs=1e-4), uniform marginals).

Graphs come from the SAME deterministic generators registered in _load_graph_ds, so the
matrices align with what the cross-domain runs load as training labels. Idempotent.

    python scripts/prep_synth_gw.py [NAME ...]   # default: all four
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import ot
from torch_geometric.utils import to_dense_adj

from models._utils_Processing import _generate_er_sparse_dataset, _generate_ba_sparse_dataset

ROOT = os.environ.get("NEURALGW_DATA_ROOT", "./data_300_350")
GWD = f"{ROOT}/precomputed/gwd_adj"

SOURCES = {
    "ER_SPARSE": lambda: _generate_er_sparse_dataset(50, 300, 350, 0.007),
    "BA_SPARSE": lambda: _generate_ba_sparse_dataset(50, 300, 350, 0.007),
    "ER_DENSE": lambda: _generate_er_sparse_dataset(50, 300, 350, 0.578, seed_base=9000),
    "BA_DENSE": lambda: _generate_ba_sparse_dataset(50, 300, 350, 0.578, seed_base=9500),
}


def cost(d):
    A = to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes).squeeze(0).numpy().astype(np.float32)
    A += np.eye(A.shape[0], dtype=np.float32)
    return A


def within_gw(ds):
    Cs = [cost(d) for d in ds]
    n = len(Cs)
    D = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            D[a, b] = D[b, a] = float(ot.gromov.gromov_wasserstein2(C1=Cs[a], C2=Cs[b],
                                                                    tol_rel=1e-4, tol_abs=1e-4))
        if a % 10 == 0:
            print(f"    row {a}/{n}", flush=True)
    return D


def main():
    os.makedirs(GWD, exist_ok=True)
    names = sys.argv[1:] or list(SOURCES)
    for name in names:
        out = f"{GWD}/{name}_{name}_distance_matrix.npy"
        if os.path.exists(out):
            print(f"SKIP (exists): {out}", flush=True)
            continue
        ds = SOURCES[name]()
        print(f"{name}: {len(ds)} graphs, computing within-GW ...", flush=True)
        D = within_gw(ds)
        np.save(out, D)
        print(f"saved {out}  shape={D.shape}", flush=True)


if __name__ == "__main__":
    main()
