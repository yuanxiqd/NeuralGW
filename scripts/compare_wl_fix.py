"""Compare WL kernel on MUTAG before vs after the use_label fix.

`results/grakels_1st_for_paper/MUTAG_grakel_wl_kernel.npy` was computed with
the buggy branch (use_label=False -> all node labels set to 1). This script
recomputes the WL kernel with the patched `get_kernel_mtx` (which now uses
discrete argmax(x) labels) and compares spectral-clustering metrics.
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from torch_geometric.datasets import TUDataset
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    accuracy_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    rand_score,
    confusion_matrix,
)
from scipy.optimize import linear_sum_assignment

from utils_dqiao import get_kernel_mtx


def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy after optimal cluster-id -> class permutation."""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    remapped = np.empty_like(y_pred)
    mapping = {c: r for r, c in zip(row_ind, col_ind)}
    for c, r in mapping.items():
        remapped[y_pred == c] = r
    return accuracy_score(y_true, remapped)


def cluster_metrics(K: np.ndarray, y: np.ndarray, n_clusters: int) -> dict:
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=0,
    )
    pred = sc.fit_predict(K)
    return {
        "acc_raw": accuracy_score(y, pred),
        "acc_hung": hungarian_accuracy(y, pred),
        "nmi": normalized_mutual_info_score(y, pred),
        "ari": adjusted_rand_score(y, pred),
        "ri": rand_score(y, pred),
    }


def main() -> None:
    ds_name = "MUTAG"
    dataset = TUDataset(root="data/TUDataset", name=ds_name)
    labels = dataset.y.numpy()
    n_clusters = dataset.num_classes

    old_path = f"results/grakels_1st_for_paper/{ds_name}_grakel_wl_kernel.npy"
    K_wl_old = np.load(old_path)
    print(f"Loaded OLD (buggy) WL kernel: {old_path}  shape={K_wl_old.shape}")

    print("Recomputing WL kernel with the patched use_label branch ...")
    K_wl_new = get_kernel_mtx("weisfeiler_lehman", dataset, dataset)

    out_dir = "results/grakels"
    os.makedirs(out_dir, exist_ok=True)
    new_path = os.path.join(out_dir, f"{ds_name}_grakel_wl_kernel_fixed.npy")
    np.save(new_path, K_wl_new)
    print(f"Saved NEW WL kernel: {new_path}")

    diff = K_wl_new - K_wl_old
    print(
        f"\nKernel-matrix diff (new - old): "
        f"max|diff|={np.abs(diff).max():.4g}, "
        f"mean|diff|={np.abs(diff).mean():.4g}, "
        f"frob={np.linalg.norm(diff):.4g}"
    )
    print(
        f"  old:  min={K_wl_old.min():.4f}  max={K_wl_old.max():.4f}  mean={K_wl_old.mean():.4f}"
    )
    print(
        f"  new:  min={K_wl_new.min():.4f}  max={K_wl_new.max():.4f}  mean={K_wl_new.mean():.4f}"
    )

    m_old = cluster_metrics(K_wl_old, labels, n_clusters)
    m_new = cluster_metrics(K_wl_new, labels, n_clusters)

    print("\nSpectral-clustering on MUTAG (n_classes={}):".format(n_clusters))
    header = (
        f"{'kernel':<24}{'acc_raw':>10}{'acc_hung':>10}{'nmi':>8}{'ari':>8}{'ri':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, m in (
        ("WL (old, all-1 labels)", m_old),
        ("WL (new, atom labels)", m_new),
    ):
        print(
            f"{name:<24}"
            f"{m['acc_raw']:>10.4f}{m['acc_hung']:>10.4f}"
            f"{m['nmi']:>8.4f}{m['ari']:>8.4f}{m['ri']:>8.4f}"
        )


if __name__ == "__main__":
    main()
