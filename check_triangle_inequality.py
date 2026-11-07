"""Check whether a saved pairwise distance matrix satisfies the triangle inequality.

For every ordered triple (i, j, k) with i != j, j != k, i != k we check
    D[i, k] <= D[i, j] + D[j, k]
and report:
  - basic sanity (square, symmetry, zero diagonal, non-negativity)
  - the count and rate of triple violations
  - absolute and relative excess summary
  - the worst-violating triple

Usage:
    python check_triangle_inequality.py [path/to/distance_matrix.npy] [--tol 1e-6]

Default path: results/clustering/COLLAB_to_REDDIT-BINARY_seed0_distance_matrix.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


DEFAULT_PATH = "results/clustering/COLLAB_to_REDDIT-BINARY_seed0_distance_matrix.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", default=DEFAULT_PATH, help=f"path to .npy matrix (default: {DEFAULT_PATH})")
    p.add_argument("--tol", type=float, default=1e-6, help="absolute tolerance for violations (default: 1e-6)")
    p.add_argument("--rel-tol", type=float, default=0.0,
                   help="additional relative tolerance: violation iff D[i,k] > (1+rel_tol)*(D[i,j]+D[j,k]) + tol "
                        "(default: 0.0)")
    return p.parse_args()


def sanity_report(D: np.ndarray) -> None:
    print(f"Matrix shape       : {D.shape}")
    print(f"dtype              : {D.dtype}")
    n = D.shape[0]
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected a square 2-D matrix, got shape {D.shape}.")

    diag = np.diag(D)
    print(f"max |diag|         : {np.max(np.abs(diag)):.6e}")
    asym = float(np.max(np.abs(D - D.T)))
    print(f"max |D - D.T|      : {asym:.6e}  (symmetry residual)")
    neg_count = int(np.sum(D < 0))
    print(f"# negative entries : {neg_count}  (min = {float(D.min()):.6e})")
    finite_mask = np.isfinite(D)
    print(f"# non-finite       : {int((~finite_mask).sum())}")
    print(f"off-diag stats     : min={D[~np.eye(n, dtype=bool)].min():.6e}, "
          f"max={D[~np.eye(n, dtype=bool)].max():.6e}, "
          f"mean={D[~np.eye(n, dtype=bool)].mean():.6e}")


def check_triangle(D: np.ndarray, tol: float, rel_tol: float):
    """Return (n_triples_checked, n_violations, worst_triple_info).

    A triple (i,j,k) is checked only when i, j, k are all distinct.
    Vectorized per-i over the inner pair (j, k).
    """
    n = D.shape[0]
    if n < 3:
        return 0, 0, None

    n_triples = n * (n - 1) * (n - 2)  # ordered with all distinct
    n_violations = 0
    total_abs_excess = 0.0
    total_rel_excess = 0.0
    worst = {"abs_excess": -np.inf, "i": -1, "j": -1, "k": -1,
             "d_ik": 0.0, "d_ij": 0.0, "d_jk": 0.0}

    # For each fixed i, compute S[j, k] = D[i, j] + D[j, k], then compare with D[i, k].
    # Mask out j == i and j == k (and k == i) so that only triples with three distinct
    # indices count.
    eye = np.eye(n, dtype=bool)
    for i in range(n):
        row_i = D[i, :]              # shape (n,)
        S = row_i[:, None] + D       # S[j, k] = D[i,j] + D[j,k]; shape (n, n)
        # invalidate j == i (j equals the source vertex)
        S[i, :] = np.inf
        # invalidate j == k (path through k itself)
        S[eye] = np.inf
        # we still need to skip k == i for the comparison (D[i,i] = 0 trivially satisfies)
        # by masking those columns out of the violation accounting.

        d_ik = row_i[None, :]        # shape (1, n), broadcasts over j
        excess = d_ik - (1.0 + rel_tol) * S - tol   # > 0 means violation
        # exclude k == i column entirely
        excess[:, i] = -np.inf

        violated = excess > 0
        v_count = int(violated.sum())
        if v_count:
            n_violations += v_count
            pos_excess = excess[violated]
            total_abs_excess += float(pos_excess.sum())
            # for the relative metric we report excess / (d_ij + d_jk)
            denom = S[violated]
            rel = pos_excess / np.where(denom > 0, denom, 1.0)
            total_rel_excess += float(rel.sum())

            # locate worst within this i (masked entries are -inf, so argmax is safe)
            flat = np.argmax(excess)
            jj, kk = np.unravel_index(flat, excess.shape)
            local_excess = float(excess[jj, kk])
            if local_excess > worst["abs_excess"]:
                worst.update({
                    "abs_excess": local_excess,
                    "i": int(i), "j": int(jj), "k": int(kk),
                    "d_ik": float(D[i, kk]),
                    "d_ij": float(D[i, jj]),
                    "d_jk": float(D[jj, kk]),
                })

    return n_triples, n_violations, total_abs_excess, total_rel_excess, worst


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.path):
        print(f"[Error] File not found: {args.path}", file=sys.stderr)
        return 2

    D = np.load(args.path)
    D = np.asarray(D, dtype=np.float64)

    print("=" * 72)
    print(f"File: {args.path}")
    print("-" * 72)
    sanity_report(D)
    print("-" * 72)

    n = D.shape[0]
    if n < 3:
        print("Matrix too small for triangle inequality check (need n >= 3).")
        return 0

    print(f"Checking triangle inequality with tol={args.tol}, rel_tol={args.rel_tol} ...")
    n_triples, n_viol, abs_sum, rel_sum, worst = check_triangle(D, args.tol, args.rel_tol)
    rate = n_viol / n_triples if n_triples else 0.0

    print(f"# ordered distinct triples : {n_triples}")
    print(f"# triangle violations      : {n_viol}")
    print(f"violation rate             : {rate:.6%}")
    if n_viol > 0:
        print(f"mean absolute excess       : {abs_sum / n_viol:.6e}")
        print(f"mean relative excess       : {rel_sum / n_viol:.6%}  "
              f"(excess / (D[i,j]+D[j,k]))")
        i, j, k = worst["i"], worst["j"], worst["k"]
        d_ik, d_ij, d_jk = worst["d_ik"], worst["d_ij"], worst["d_jk"]
        excess = d_ik - (d_ij + d_jk)
        print()
        print("Worst-violating triple:")
        print(f"  i={i}, j={j}, k={k}")
        print(f"  D[i,k]            = {d_ik:.6e}")
        print(f"  D[i,j] + D[j,k]   = {d_ij + d_jk:.6e}  "
              f"(D[i,j]={d_ij:.6e}, D[j,k]={d_jk:.6e})")
        print(f"  excess            = {excess:.6e}")
        if (d_ij + d_jk) > 0:
            print(f"  excess / sum      = {excess / (d_ij + d_jk):.6%}")
        print()
        print("Verdict: triangle inequality is VIOLATED.")
    else:
        print()
        print("Verdict: triangle inequality holds (within tolerance).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
