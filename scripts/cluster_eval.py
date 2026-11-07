"""Unified spectral-clustering evaluator for the clustering-table baselines.

Scores any precomputed N x N matrix over a target TUDataset with the SAME procedure
used for NeuralGW (scripts/spectral_cluster_cross_domain.py): for a distance matrix,
build a median-heuristic Gaussian affinity; for an affinity/kernel matrix, use it
directly. Then SpectralClustering(precomputed, discretize, n_init=10, random_state=42)
and report ACC (Hungarian) / NMI / ARI / RI. One or more matrices (globs) may be
passed; multiple are summarized as mean +/- std (e.g. across seeds).

Examples:
    # exact-GW reference on BZR
    python scripts/cluster_eval.py --kind distance --testset BZR --label GW \
        --matrices ./data_300_350/precomputed/gwd_adj/BZR_BZR_distance_matrix.npy
    # WL kernel on COX2
    python scripts/cluster_eval.py --kind affinity --testset COX2 --label WL \
        --matrices results/grakels/COX2_grakel_wl_kernel.npy
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    rand_score,
)
from torch_geometric.datasets import TUDataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    remapped = np.zeros_like(y_pred)
    for r, c in zip(row_ind, col_ind):
        remapped[y_pred == c] = r
    return accuracy_score(y_true, remapped)


def gw_sigma(D2: np.ndarray) -> float:
    # Bandwidth estimator matching distance_to_kernel (models/_utils_Processing.py):
    # the stored GW matrices are SQUARED distances (GW^2), so sigma = mean of the
    # un-squared distances = mean(sqrt(D2)).
    D2 = np.clip(D2, 0.0, D2.max())
    return float(np.mean(np.sqrt(D2)))


def affinity_from_distance(D2: np.ndarray, sigma: float | None = None) -> np.ndarray:
    # D2 is a squared-distance matrix (GW^2); Gaussian kernel exp(-D2/(2 sigma^2)),
    # used DIRECTLY (not squared again). Mirrors distance_to_kernel.
    D2 = np.clip(D2, 0.0, D2.max())
    if sigma is None:
        sigma = gw_sigma(D2)
    return np.exp(-D2 / (2.0 * sigma**2))


def affinity_from_kernel(K: np.ndarray) -> np.ndarray:
    # symmetrize, drop NaNs, clip negatives -> valid precomputed affinity
    K = 0.5 * (K + K.T)
    K = np.nan_to_num(K, nan=0.0)
    return np.clip(K, 0.0, None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified clustering evaluator for baseline matrices.")
    p.add_argument("--matrices", nargs="+", required=True, metavar="PATH",
                   help="One or more .npy paths or globs (each an N x N matrix in TUDataset order).")
    p.add_argument("--kind", required=True, choices=["distance", "affinity"],
                   help="'distance' -> median-Gaussian affinity; 'affinity' -> use matrix directly.")
    p.add_argument("--testset", required=True, metavar="DATASET",
                   help="Target TUDataset name (for labels + n_clusters).")
    p.add_argument("--label", default=None, help="Display label for this method/row.")
    p.add_argument("--sigma-from", default=None, metavar="PATH",
                   help="For --kind distance: fix the Gaussian bandwidth sigma to the median of "
                        "THIS reference matrix (e.g. the target's exact-GW matrix) and use it for all "
                        "input matrices, instead of a per-matrix median. Matches the 'shared exact-GW "
                        "bandwidth' convention.")
    p.add_argument("--labels-file", default=None, metavar="PATH",
                   help="Load ground-truth labels from a .npy file (in matrix order) instead of a "
                        "TUDataset. n_clusters = number of distinct labels. For custom subsets like "
                        "COLLAB_BAL that aren't TUDatasets.")
    p.add_argument("--node-range", default=None, metavar="LO,HI",
                   help="Subset the target to graphs with LO < num_nodes < HI (exclusive), in dataset "
                        "order, matching the precomputed subset matrices (e.g. '300,350'). n_clusters "
                        "is then the number of distinct labels present in the subset.")
    default_root = os.environ.get("NEURALGW_DATA_ROOT", "./data_300_350")
    p.add_argument("--data-root", default=os.path.join(default_root, "TUDataset"))
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def _score(args, files, labels, n_clusters):
    fixed_sigma = None
    if args.sigma_from and args.kind == "distance":
        fixed_sigma = gw_sigma(np.load(args.sigma_from).astype(float))
        print(f"[sigma-from] fixed bandwidth sigma=mean(sqrt(D2))={fixed_sigma:.4f} (from {os.path.basename(args.sigma_from)})")
    accs, nmis, aris, ris = [], [], [], []
    for fpath in files:
        M = np.load(fpath)
        if M.shape[0] != len(labels):
            print(f"[Warning] {os.path.basename(fpath)}: size {M.shape[0]} != labels {len(labels)}, skipping.",
                  file=sys.stderr)
            continue
        A = affinity_from_distance(M, fixed_sigma) if args.kind == "distance" else affinity_from_kernel(M)
        sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                                assign_labels="discretize", n_init=10, random_state=args.random_state)
        pred = sc.fit_predict(A)
        accs.append(hungarian_accuracy(labels, pred))
        nmis.append(normalized_mutual_info_score(labels, pred))
        aris.append(adjusted_rand_score(labels, pred))
        ris.append(rand_score(labels, pred))

    if not accs:
        sys.exit("No valid matrices were processed.")

    tag = args.label or os.path.basename(files[0])
    n = len(accs)
    print(f"{tag}  [{args.testset}, kind={args.kind}, n_clusters={n_clusters}, {n} matrix/matrices]")
    for name, vals in [("ACC", accs), ("NMI", nmis), ("ARI", aris), ("RI ", ris)]:
        arr = np.array(vals)
        print(f"  {name} : {arr.mean():.4f} +/- {arr.std():.4f}")


def main() -> None:
    args = parse_args()

    files: list[str] = []
    for pat in args.matrices:
        hits = sorted(glob.glob(pat))
        files.extend(hits if hits else ([pat] if os.path.isfile(pat) else []))
    if not files:
        sys.exit(f"No matrices matched: {args.matrices}")

    if args.labels_file:
        labels = np.load(args.labels_file)
        n_clusters = int(len(np.unique(labels)))
        print(f"[labels-file] {len(labels)} labels, {n_clusters} distinct")
    else:
        dataset = TUDataset(root=args.data_root, name=args.testset)
        if args.node_range:
            lo, hi = (int(x) for x in args.node_range.split(","))
            sub = [i for i in range(len(dataset)) if lo < dataset[i].num_nodes < hi]
            labels = dataset.y.numpy()[sub]
            n_clusters = int(len(np.unique(labels)))
            print(f"[subset {lo}<n<{hi}] {len(labels)} graphs, {n_clusters} distinct labels present")
        else:
            labels = dataset.y.numpy()
            n_clusters = int(dataset.num_classes)

    _score(args, files, labels, n_clusters)


if __name__ == "__main__":
    main()
