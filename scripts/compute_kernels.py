"""Compute graph-kernel Gram matrices (RW / WL / WL-OA) for the clustering-table baselines.

For each dataset, fits and transforms on the full dataset (self-Gram, used as a
precomputed affinity for spectral clustering) and saves
    results/grakels/{ds}_grakel_{rw|wl|wloa}_kernel.npy
matching the convention in utils_dqiao.py's __main__. Idempotent: skips existing files.

Usage:
    python scripts/compute_kernels.py --datasets BZR COX2 DHFR PTC_FM
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree as pyg_degree

from utils_dqiao import get_kernel_mtx


def ensure_node_features(dataset):
    """Social datasets (REDDIT/COLLAB) have no node features, but get_kernel_mtx's
    grakel conversion needs discrete node labels. Attach a degree one-hot so WL/RW
    use node degree as the label (standard for unlabeled graphs). Molecular datasets
    (data.x present) are returned unchanged."""
    if len(dataset) == 0 or getattr(dataset[0], "x", None) is not None:
        return dataset, False
    maxd = 0
    for d in dataset:
        if d.num_nodes > 0 and d.edge_index.numel() > 0:
            maxd = max(maxd, int(pyg_degree(d.edge_index[0], num_nodes=d.num_nodes).max().item()))
    graphs = []
    for d in dataset:
        deg = pyg_degree(d.edge_index[0], num_nodes=d.num_nodes).long().clamp(max=maxd)
        x = torch.zeros((d.num_nodes, maxd + 1))
        x[torch.arange(d.num_nodes), deg] = 1.0
        nd = d.clone(); nd.x = x
        graphs.append(nd)
    return graphs, True

# --- SciPy>=1.12 compat shim for grakel's random_walk kernel ---
# grakel calls scipy.sparse.linalg.cg(..., tol=1e-6, atol='legacy'); modern SciPy
# removed the `tol` kwarg (use `rtol`) and rejects the 'legacy' atol string. grakel
# binds `cg` at import time, so patch the name inside its module, not just scipy's.
import scipy.sparse.linalg as _spla
import grakel.kernels.random_walk as _grw

_orig_cg = _spla.cg


def _cg_compat(A, b, *a, **kw):
    if "tol" in kw:
        kw["rtol"] = kw.pop("tol")
    if kw.get("atol") == "legacy":
        kw["atol"] = 0.0
    return _orig_cg(A, b, *a, **kw)


_spla.cg = _cg_compat
_grw.cg = _cg_compat

KMAP = {
    "rw": "random_walk",
    "wl": "weisfeiler_lehman",
    "wloa": "weisfeiler_lehman_optimal_assignment",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True, metavar="NAME")
    ap.add_argument("--kernels", nargs="+", default=["rw", "wl", "wloa"], choices=list(KMAP))
    ap.add_argument("--node-range", default=None, metavar="LO,HI",
                    help="Restrict to graphs with LO < num_nodes < HI (matches subset matrices); "
                         "tags output filenames with _sub{LO}_{HI}.")
    default_root = os.environ.get("NEURALGW_DATA_ROOT", "./data_300_350")
    ap.add_argument("--data-root", default=os.path.join(default_root, "TUDataset"))
    ap.add_argument("--out-dir", default="results/grakels")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = ""
    rng = None
    if args.node_range:
        lo, hi = (int(x) for x in args.node_range.split(","))
        rng, tag = (lo, hi), f"_sub{lo}_{hi}"
    for ds_name in args.datasets:
        dataset = TUDataset(root=args.data_root, name=ds_name)
        if rng is not None:
            sub = [i for i in range(len(dataset)) if rng[0] < dataset[i].num_nodes < rng[1]]
            dataset = dataset[sub]
            print(f"[{ds_name}] subset {rng[0]}<n<{rng[1]}: {len(dataset)} graphs", flush=True)
        else:
            print(f"[{ds_name}] {len(dataset)} graphs, {dataset.num_classes} classes", flush=True)
        gdata, degmode = ensure_node_features(dataset)
        if degmode:
            print(f"  (no node features -> using degree one-hot labels)", flush=True)
        for short in args.kernels:
            out = os.path.join(args.out_dir, f"{ds_name}{tag}_grakel_{short}_kernel.npy")
            if os.path.exists(out):
                print(f"  SKIP (exists): {out}", flush=True)
                continue
            print(f"  computing {KMAP[short]} ...", flush=True)
            K = get_kernel_mtx(KMAP[short], gdata, gdata)
            np.save(out, K)
            print(f"  saved {out}  shape={K.shape} range=({K.min():.4f},{K.max():.4f})", flush=True)


if __name__ == "__main__":
    main()
