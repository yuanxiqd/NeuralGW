"""
Spectral clustering on NeuralGW cross-domain distance matrices.

Locates files matching:
  results/clustering/{trainsets}_to_{testset}_{seed-glob}_distance_matrix.npy

Labels are loaded from the testset TUDataset.

Affinity: Gaussian kernel with median-heuristic sigma,
    K = exp(-D^2 / (2 * sigma^2))  where sigma = median of upper-triangle of D.

Example usage:
    python scripts/spectral_cluster_ptc_fm_bzr_to_mutag.py \
        --trainsets PTC_FM BZR --testset MUTAG
    python scripts/spectral_cluster_ptc_fm_bzr_to_mutag.py \
        --trainsets BZR --testset MUTAG --seed-glob seed0
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


def affinity_from_distance(D: np.ndarray) -> np.ndarray:
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    iu = np.triu_indices_from(D, k=1)
    sigma = np.median(D[iu])
    if sigma <= 0.0:
        sigma = 1.0
    return np.exp(-(D**2) / (2.0 * sigma**2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectral clustering on NeuralGW cross-domain distance matrices."
    )
    parser.add_argument(
        "--trainsets",
        nargs="+",
        required=True,
        metavar="DATASET",
        help=(
            "One or more training-set TUDataset names (e.g. PTC_FM BZR). "
            "They are joined with '_' to form the filename prefix."
        ),
    )
    parser.add_argument(
        "--testset",
        required=True,
        metavar="DATASET",
        help="Test-set TUDataset name (e.g. MUTAG). Used for both filename and label loading.",
    )
    parser.add_argument(
        "--clustering-dir",
        default="results/clustering",
        metavar="DIR",
        help="Directory holding the .npy files, relative to repo root. (default: results/clustering)",
    )
    parser.add_argument(
        "--data-root",
        default="data/TUDataset",
        metavar="DIR",
        help="Root directory forwarded to TUDataset(root=...). (default: data/TUDataset)",
    )
    parser.add_argument(
        "--seed-glob",
        default="seed*",
        metavar="PATTERN",
        help=(
            "Glob fragment that matches the seed portion of filenames, "
            "e.g. 'seed*' (default) or 'seed0' for a single seed."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        metavar="INT",
        help="Random state for SpectralClustering. (default: 42)",
    )

    args = parser.parse_args()

    for name in args.trainsets:
        if not name or name.strip() != name:
            parser.error(f"Invalid trainset name (empty or has leading/trailing spaces): {name!r}")

    return args


def main() -> None:
    args = parse_args()

    prefix = "_".join(args.trainsets) + "_to_" + args.testset
    clustering_dir = os.path.join(_REPO_ROOT, args.clustering_dir)
    pattern = os.path.join(clustering_dir, f"{prefix}_{args.seed_glob}_distance_matrix.npy")

    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No files matched: {pattern}")

    data_root = os.path.join(_REPO_ROOT, args.data_root)
    dataset = TUDataset(root=data_root, name=args.testset)
    labels = dataset.y.numpy()
    n_clusters = int(dataset.num_classes)

    accs, nmis, aris, ris = [], [], [], []

    for fpath in files:
        D = np.load(fpath)
        if D.shape[0] != len(labels):
            print(
                f"[Warning] {os.path.basename(fpath)}: matrix size {D.shape[0]} "
                f"!= label count {len(labels)}, skipping.",
                file=sys.stderr,
            )
            continue

        K = affinity_from_distance(D)

        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="discretize",
            n_init=10,
            random_state=args.random_state,
        )
        pred = sc.fit_predict(K)

        accs.append(hungarian_accuracy(labels, pred))
        nmis.append(normalized_mutual_info_score(labels, pred))
        aris.append(adjusted_rand_score(labels, pred))
        ris.append(rand_score(labels, pred))

    if not accs:
        sys.exit("No valid files were processed.")

    n = len(accs)
    print(
        f"Spectral clustering on {prefix} "
        f"({n} seed{'s' if n != 1 else ''}, {args.testset} labels, n_clusters={n_clusters})"
    )
    for name, vals in [("ACC", accs), ("NMI", nmis), ("ARI", aris), ("RI ", ris)]:
        arr = np.array(vals)
        print(f"  {name} : {arr.mean():.4f} +/- {arr.std():.4f}")


if __name__ == "__main__":
    main()
