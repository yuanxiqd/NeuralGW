import argparse
import itertools
import os
import timeit
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import ot
from tqdm.auto import tqdm


@dataclass(frozen=True)
class GraphSpec:
    n: int
    seed: int


def _largest_connected_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    if nx.is_connected(G):
        return G
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()


def generate_er_adj(
    n: int,
    avg_degree: float,
    seed: int,
    ensure_connected: bool = True,
    dtype=np.float64,
) -> np.ndarray:
    """
    Generate an undirected ER graph adjacency matrix (dense).

    Notes:
    - Returns a (n', n') dense adjacency matrix where n' may be < n if
      `ensure_connected=True` and the largest connected component is taken.
    - Adds no self-loops here; do that in the caller if desired.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if avg_degree <= 0:
        raise ValueError("avg_degree must be > 0")

    p = min(1.0, float(avg_degree) / max(1, n - 1))
    G = nx.fast_gnp_random_graph(n=n, p=p, seed=seed, directed=False)
    if ensure_connected:
        G = _largest_connected_component(G)

    A = nx.to_numpy_array(G, dtype=dtype)
    return A


def compute_gw_distance(
    C1: np.ndarray,
    C2: np.ndarray,
    tol_rel: float = 1e-4,
    tol_abs: float = 1e-4,
    max_iter: int = 1000,
) -> float:
    return float(
        ot.gromov.gromov_wasserstein2(
            C1=C1,
            C2=C2,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
            max_iter=max_iter,
            verbose=False,
        )
    )


def _pair_indices(num_graphs: int, pairing: str) -> Iterable[Tuple[int, int]]:
    if pairing == "all":
        yield from itertools.combinations(range(num_graphs), 2)
        return
    if pairing == "consecutive":
        yield from ((i, i + 1) for i in range(num_graphs - 1))
        return
    raise ValueError(f"Unknown pairing: {pairing}")


def save_graphs_as_tudataset(
    adjs_no_self_loops: Sequence[np.ndarray],
    dataset_name: str = "LARGE_GW",
    tu_root: str = "data/TUDataset",
) -> str:
    """
    Save generated graphs in TU Dortmund raw format.

    Files written:
    - <name>_A.txt
    - <name>_graph_indicator.txt
    - <name>_graph_labels.txt
    - <name>_node_labels.txt
    """
    raw_dir = os.path.join(tu_root, dataset_name, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    a_lines: List[str] = []
    graph_indicator_lines: List[str] = []
    graph_labels_lines: List[str] = []
    node_labels_lines: List[str] = []

    node_offset = 0
    for g_idx, A in enumerate(adjs_no_self_loops, start=1):
        n = int(A.shape[0])
        graph_labels_lines.append("1\n")
        graph_indicator_lines.extend([f"{g_idx}\n"] * n)
        node_labels_lines.extend(["1\n"] * n)

        rows, cols = np.nonzero(A > 0)
        keep = rows != cols
        rows = rows[keep]
        cols = cols[keep]
        for r, c in zip(rows.tolist(), cols.tolist()):
            a_lines.append(f"{node_offset + r + 1}, {node_offset + c + 1}\n")

        node_offset += n

    with open(os.path.join(raw_dir, f"{dataset_name}_A.txt"), "w", encoding="utf-8") as f:
        f.writelines(a_lines)
    with open(
        os.path.join(raw_dir, f"{dataset_name}_graph_indicator.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(graph_indicator_lines)
    with open(
        os.path.join(raw_dir, f"{dataset_name}_graph_labels.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(graph_labels_lines)
    with open(
        os.path.join(raw_dir, f"{dataset_name}_node_labels.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(node_labels_lines)

    return raw_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 10 large random graphs (2000-5000 nodes) and compute GW "
            "distances with tol_rel=1e-4 and tol_abs=1e-4."
        )
    )
    parser.add_argument("--num_graphs", type=int, default=10)
    parser.add_argument("--n_min", type=int, default=2000)
    parser.add_argument("--n_max", type=int, default=5000)
    parser.add_argument("--avg_degree", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensure_connected", action="store_true", default=True)
    parser.add_argument(
        "--pairing",
        choices=("all", "consecutive"),
        default="all",
        help="Which graph pairs to compute GW for.",
    )
    parser.add_argument("--tol_rel", type=float, default=1e-4)
    parser.add_argument("--tol_abs", type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument(
        "--tu_name",
        type=str,
        default=None,
        help=(
            "Dataset name for TU-format export under data/TUDataset/<name>/raw. "
            "Default: LGW{num_graphs} (e.g. LGW10 when --num_graphs 10)."
        ),
    )
    parser.add_argument(
        "--tu_root",
        type=str,
        default="data/TUDataset",
        help="Root folder for TU-format export.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help=(
            "Output path for the pairwise GW distance matrix (.npy). "
            "Default: data/LGW{num_graphs}.npy."
        ),
    )
    parser.add_argument(
        "--save_graphs",
        action="store_true",
        help="If set, save generated adjacency matrices to a compressed .npz file.",
    )
    parser.add_argument(
        "--graphs_out_path",
        type=str,
        default=None,
        help=(
            "Optional output path for generated graph adjacencies (.npz). "
            "Default: derived from --out_path."
        ),
    )
    parser.add_argument(
        "--meta_out_path",
        type=str,
        default=None,
        help=(
            "Optional output path for generation metadata (.npz). "
            "Default: derived from --out_path."
        ),
    )
    args = parser.parse_args(argv)

    stem = f"LGW{args.num_graphs}"
    if args.tu_name is None:
        args.tu_name = stem
    if args.out_path is None:
        args.out_path = os.path.join("data", f"{stem}.npy")

    rng = np.random.default_rng(args.seed)
    specs: List[GraphSpec] = []
    for i in range(args.num_graphs):
        n = int(rng.integers(args.n_min, args.n_max + 1))
        specs.append(GraphSpec(n=n, seed=int(args.seed + i)))

    print(
        f"[Info] Generating {args.num_graphs} graphs with n in "
        f"[{args.n_min}, {args.n_max}] (avg_degree={args.avg_degree})."
    )

    t_gen0 = timeit.default_timer()
    adjs: List[np.ndarray] = []
    adjs_no_self_loops: List[np.ndarray] = []
    for s in tqdm(specs, desc="Generating graphs"):
        A = generate_er_adj(
            n=s.n,
            avg_degree=args.avg_degree,
            seed=s.seed,
            ensure_connected=args.ensure_connected,
            dtype=np.float64,
        )
        adjs_no_self_loops.append(A.copy())
        # match other scripts: add self-loops
        A = A + np.eye(A.shape[0], dtype=A.dtype)
        adjs.append(A)
    t_gen1 = timeit.default_timer()

    sizes = [A.shape[0] for A in adjs]
    print(f"[Info] Generated graph sizes: {sizes}")
    print(f"[Info] Graph generation time: {t_gen1 - t_gen0:.2f} s")

    out_path = args.out_path
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_root, _ = os.path.splitext(out_path)

    # Save graph artifacts immediately after generation for robustness.
    meta_out_path = args.meta_out_path or f"{out_root}_meta.npz"
    meta_dir = os.path.dirname(meta_out_path) or "."
    os.makedirs(meta_dir, exist_ok=True)

    planned_sizes = np.array([s.n for s in specs], dtype=np.int64)
    used_sizes = np.array([A.shape[0] for A in adjs], dtype=np.int64)
    seeds = np.array([s.seed for s in specs], dtype=np.int64)
    np.savez_compressed(
        meta_out_path,
        planned_sizes=planned_sizes,
        used_sizes=used_sizes,
        seeds=seeds,
        num_graphs=np.array(args.num_graphs, dtype=np.int64),
        n_min=np.array(args.n_min, dtype=np.int64),
        n_max=np.array(args.n_max, dtype=np.int64),
        avg_degree=np.array(args.avg_degree, dtype=np.float64),
        seed=np.array(args.seed, dtype=np.int64),
        ensure_connected=np.array(int(args.ensure_connected), dtype=np.int64),
        pairing=np.array(args.pairing),
        tol_rel=np.array(args.tol_rel, dtype=np.float64),
        tol_abs=np.array(args.tol_abs, dtype=np.float64),
        max_iter=np.array(args.max_iter, dtype=np.int64),
    )
    print(f"[Info] Saved generation metadata to: {meta_out_path}")

    if args.save_graphs:
        graphs_out_path = args.graphs_out_path or f"{out_root}_graphs.npz"
        graphs_dir = os.path.dirname(graphs_out_path) or "."
        os.makedirs(graphs_dir, exist_ok=True)
        graph_dict = {f"graph_{i:03d}": A for i, A in enumerate(adjs)}
        np.savez_compressed(graphs_out_path, **graph_dict)
        print(f"[Info] Saved generated adjacency matrices to: {graphs_out_path}")

    tu_raw_dir = save_graphs_as_tudataset(
        adjs_no_self_loops=adjs_no_self_loops,
        dataset_name=args.tu_name,
        tu_root=args.tu_root,
    )
    print(f"[Info] Saved graphs in TU raw format to: {tu_raw_dir}")

    pairs = list(_pair_indices(args.num_graphs, args.pairing))
    print(
        f"[Info] Computing GW distances for {len(pairs)} pair(s) "
        f"(pairing={args.pairing})."
    )
    print(
        f"[Info] GW params: tol_rel={args.tol_rel:g}, tol_abs={args.tol_abs:g}, "
        f"max_iter={args.max_iter}"
    )

    t_gw0 = timeit.default_timer()
    D = np.full((args.num_graphs, args.num_graphs), np.nan, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    results: List[Tuple[int, int, float]] = []
    for i, j in tqdm(pairs, desc="GW pairs"):
        d = compute_gw_distance(
            adjs[i],
            adjs[j],
            tol_rel=args.tol_rel,
            tol_abs=args.tol_abs,
            max_iter=args.max_iter,
        )
        results.append((i, j, d))
        D[i, j] = d
        D[j, i] = d
    t_gw1 = timeit.default_timer()

    for i, j, d in results:
        print(f"GW({i},{j}) = {d:.8e}   [n={adjs[i].shape[0]} vs {adjs[j].shape[0]}]")

    print(f"[Info] Total GW time: {t_gw1 - t_gw0:.2f} s")
    if len(results) > 0:
        print(f"[Info] Avg time per pair: {(t_gw1 - t_gw0) / len(results):.2f} s")

    np.save(out_path, D)
    print(f"[Info] Saved pairwise GW matrix to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

