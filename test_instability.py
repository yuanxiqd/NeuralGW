import numpy as np
import torch
import ot

import pandas as pd

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj
from scipy.sparse.csgraph import shortest_path
from tqdm.auto import tqdm


# =========================
# 1. Load MUTAG dataset
# =========================
def load_mutag_graphs(root="data/TUDataset", use_shortest_path=False):
    dataset = TUDataset(root=root, name="MUTAG")
    graphs = []

    for data in dataset:
        A = to_dense_adj(data.edge_index)[0].numpy()

        if use_shortest_path:
            C = shortest_path(A, directed=False)
        else:
            C = A

        graphs.append(C)

    return graphs


# =========================
# 2. GW computation (single run)
# =========================
def compute_gw_once(
    C1,
    C2,
    loss_fun="square_loss",
    max_iter=1000,
    tol_rel=1e-9,
    tol_abs=1e-9,
    random_init=False,
    seed=None,
):
    n1, n2 = C1.shape[0], C2.shape[0]

    p = ot.unif(n1)
    q = ot.unif(n2)

    if random_init:
        rng = np.random.default_rng(seed)
        # POT requires G0 to be a feasible coupling: G0 1 = p and G0^T 1 = q.
        # Build a strictly-positive random matrix then project it onto the
        # transport polytope using Sinkhorn-Knopp / IPFP scaling.
        G0 = rng.random((n1, n2), dtype=np.float64) + 1e-12
        for _ in range(200):
            G0 *= (p / (G0.sum(axis=1) + 1e-18))[:, None]
            G0 *= (q / (G0.sum(axis=0) + 1e-18))[None, :]
    else:
        G0 = np.outer(p, q)  # deterministic initialization

    gw = ot.gromov.gromov_wasserstein2(
        C1=C1,
        C2=C2,
        p=p,
        q=q,
        loss_fun=loss_fun,
        G0=G0,
        max_iter=max_iter,
        tol_rel=tol_rel,
        tol_abs=tol_abs,
        verbose=False,
        log=False,
    )

    return float(gw)


# =========================
# 3. Stability per pair
# =========================
def evaluate_pair_stability(C1, C2, n_runs=10, random_init=False):
    values = []

    for i in range(n_runs):
        val = compute_gw_once(
            C1,
            C2,
            random_init=random_init,
            seed=i,
        )
        values.append(val)

    values = np.array(values)

    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    rel_std = std / (abs(mean) + 1e-12)

    return {
        "values": values,
        "mean": mean,
        "std": std,
        "rel_std": rel_std,
        "min": values.min(),
        "max": values.max(),
    }


# =========================
# 4. Evaluate multiple pairs
# =========================
def evaluate_dataset(graphs, num_pairs=20, n_runs=10, random_init=False):
    rng = np.random.default_rng(0)

    results = []

    for k in range(num_pairs):
        i, j = rng.choice(len(graphs), size=2, replace=False)

        stats = evaluate_pair_stability(
            graphs[i],
            graphs[j],
            n_runs=n_runs,
            random_init=random_init,
        )
        stats["pair"] = f"{i}-{j}"
        results.append(stats)

    return results


def evaluate_all_pairs_directed(graphs, n_runs=10, random_init=False, show_progress=True):
    n = len(graphs)
    total = n * (n - 1)

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, total=n, desc="Graphs (outer loop)")

    results = []
    for i in iterator:
        for j in range(n):
            if i == j:
                continue
            stats = evaluate_pair_stability(
                graphs[i],
                graphs[j],
                n_runs=n_runs,
                random_init=random_init,
            )
            stats["pair"] = f"{i}->{j}"
            results.append(stats)

            if show_progress:
                iterator.set_postfix_str(f"pairs {len(results)}/{total}")

    return results


def evaluate_all_pairs_upper(graphs, n_runs=10, random_init=False, show_progress=True):
    """
    Evaluate all unordered pairs once (upper-triangular): (i, j) with i < j.
    """
    n = len(graphs)
    total = n * (n - 1) // 2

    pbar = None
    if show_progress:
        pbar = tqdm(total=total, desc="Pairs (i<j)")

    results = []
    for i in range(n):
        for j in range(i + 1, n):
            stats = evaluate_pair_stability(
                graphs[i],
                graphs[j],
                n_runs=n_runs,
                random_init=random_init,
            )
            stats["pair"] = f"{i}-{j}"
            results.append(stats)

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return results


# =========================
# 5. Print summary
# =========================
def print_summary(results, title=""):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"{'pair':<10} {'mean':>12} {'std':>12} {'rel_std(%)':>14}")

    rel_stds = []

    for r in results:
        rel_stds.append(r["rel_std"])
        print(
            f"{r['pair']:<10} "
            f"{r['mean']:>12.6e} "
            f"{r['std']:>12.6e} "
            f"{100 * r['rel_std']:>14.4f}"
        )

    rel_stds = np.array(rel_stds)

    print("\nOverall:")
    print(f"Avg rel std (%): {100 * rel_stds.mean():.4f}")
    print(f"Med rel std (%): {100 * np.median(rel_stds):.4f}")
    print(f"Max rel std (%): {100 * rel_stds.max():.4f}")


def print_seed_table(results, n_runs, title="", float_fmt="{:>11.5e}", max_rows=None):
    """
    Print a table with one column per seed/run.
    Each row corresponds to one (directed) pair in `results`.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    seed_cols = [f"seed={seed}" for seed in range(n_runs)]
    header = ["pair", *seed_cols, "mean", "std", "rel_std(%)"]
    widths = [10] + [11] * n_runs + [12, 12, 12]

    def _fmt_cell(x, w):
        s = str(x)
        return s[:w].ljust(w) if len(s) > w else s.ljust(w)

    print(" ".join(_fmt_cell(h, w) for h, w in zip(header, widths)))

    rows = results if max_rows is None else results[:max_rows]
    for r in rows:
        vals = r["values"]
        if len(vals) != n_runs:
            raise ValueError(
                f"Expected {n_runs} runs but got {len(vals)} for pair {r.get('pair')}"
            )

        parts = [
            _fmt_cell(r["pair"], widths[0]),
            *[float_fmt.format(v) for v in vals],
            f"{r['mean']:>12.6e}",
            f"{r['std']:>12.6e}",
            f"{100 * r['rel_std']:>12.4f}",
        ]
        print(" ".join(parts))

    if max_rows is not None and len(results) > max_rows:
        print(f"... ({len(results) - max_rows} more rows)")

    rel_stds = np.array([r["rel_std"] for r in results], dtype=float)
    print("\nOverall:")
    print(f"Avg rel std (%): {100 * rel_stds.mean():.4f}")
    print(f"Med rel std (%): {100 * np.median(rel_stds):.4f}")
    print(f"Max rel std (%): {100 * rel_stds.max():.4f}")


def results_to_dataframe(results, n_runs):
    rows = []
    for r in results:
        row = {"pair": r["pair"]}
        vals = r["values"]
        if len(vals) != n_runs:
            raise ValueError(
                f"Expected {n_runs} runs but got {len(vals)} for pair {r.get('pair')}"
            )
        for seed, v in enumerate(vals):
            row[f"seed={seed}"] = float(v)
        row["mean"] = float(r["mean"])
        row["std"] = float(r["std"])
        row["rel_std(%)"] = float(100 * r["rel_std"])
        row["min"] = float(r["min"])
        row["max"] = float(r["max"])
        rows.append(row)

    df = pd.DataFrame(rows)
    seed_cols = [f"seed={seed}" for seed in range(n_runs)]
    ordered_cols = ["pair", *seed_cols, "mean", "std", "rel_std(%)", "min", "max"]
    return df[ordered_cols]


def save_results_xlsx(path, results_det, results_rand, n_runs):
    df_det = results_to_dataframe(results_det, n_runs=n_runs)
    df_rand = results_to_dataframe(results_rand, n_runs=n_runs)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_det.to_excel(writer, sheet_name="deterministic", index=False)
        df_rand.to_excel(writer, sheet_name="random", index=False)


# =========================
# 6. Main
# =========================
if __name__ == "__main__":
    # ---- settings ----
    USE_SHORTEST_PATH = False   # set True if your paper uses SP distance
    NUM_PAIRS = 10
    N_RUNS = 5
    ALL_PAIRS_UPPER = False  # True: run all i<j (upper-triangular) with tqdm
    PRINT_PER_SEED_TABLE = True
    PER_SEED_MAX_ROWS = 30  # None prints all rows (can be huge)
    SAVE_XLSX = True
    OUT_XLSX = "mutag_gw_instability.xlsx"

    # ---- load data ----
    graphs = load_mutag_graphs(use_shortest_path=USE_SHORTEST_PATH)

    print(f"Loaded {len(graphs)} graphs from MUTAG")

    # ---- deterministic init ----
    if ALL_PAIRS_UPPER:
        results_det = evaluate_all_pairs_upper(
            graphs,
            n_runs=N_RUNS,
            random_init=False,
            show_progress=True,
        )
    else:
        results_det = evaluate_dataset(
            graphs,
            num_pairs=NUM_PAIRS,
            n_runs=N_RUNS,
            random_init=False,
        )

    if PRINT_PER_SEED_TABLE:
        print_seed_table(
            results_det,
            n_runs=N_RUNS,
            title="Deterministic Initialization (per-seed values)",
            max_rows=PER_SEED_MAX_ROWS,
        )

    # ---- random init ----
    if ALL_PAIRS_UPPER:
        results_rand = evaluate_all_pairs_upper(
            graphs,
            n_runs=N_RUNS,
            random_init=True,
            show_progress=True,
        )
    else:
        results_rand = evaluate_dataset(
            graphs,
            num_pairs=NUM_PAIRS,
            n_runs=N_RUNS,
            random_init=True,
        )

    if PRINT_PER_SEED_TABLE:
        print_seed_table(
            results_rand,
            n_runs=N_RUNS,
            title="Random Initialization (per-seed values)",
            max_rows=PER_SEED_MAX_ROWS,
        )

    if SAVE_XLSX:
        save_results_xlsx(
            OUT_XLSX,
            results_det=results_det,
            results_rand=results_rand,
            n_runs=N_RUNS,
        )
        print(f"\nSaved results to: {OUT_XLSX}")