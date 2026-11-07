"""
Build reproducible directed variants of TU datasets from PyG TUDataset.

Supported sources:

- **MUTAG** (with edge attributes): writes ``*_A.txt``, ``*_graph_indicator.txt``,
  ``*_graph_labels.txt``, ``*_node_labels.txt``, ``*_edge_labels.txt``.

- **REDDIT-BINARY** (no node/edge attributes in raw): writes only ``*_A.txt``,
  ``*_graph_indicator.txt``, ``*_graph_labels.txt`` (matches the original TU layout).

Load with::

    TUDataset(root=tu_root, name=out_name, use_edge_attr=True)  # MUTAG
    TUDataset(root=tu_root, name=out_name)  # REDDIT-BINARY

NeuralGW uses ``to_dense_adj(edge_index, ...)``, which keeps asymmetric adjacency
when only one orientation per undirected edge is listed.

Orientation modes
-----------------
- ``random``: for each undirected edge, choose ``(min,max)`` or ``(max,min)`` with
  probability 1/2 using a single NumPy RNG stream (graphs in index order, edges
  sorted by ``(min(u,v), max(u,v))``).
- ``low_to_high``: always orient ``min -> max`` (fixed DAG-like orientation, no RNG).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

OrientationMode = Literal["random", "low_to_high"]

# Default root for exported TU datasets (directed MUTAG variant lives under <this>/<out_name>/).
DEFAULT_TU_OUTPUT_ROOT = os.path.join(os.environ.get("NEURALGW_DATA_ROOT", "./data"), "TUDataset")


def _canonical_undirected_pairs(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    """Unique undirected edges {u,v} from ``edge_index`` (may list both directions)."""
    ei = edge_index.cpu()
    seen: set[Tuple[int, int]] = set()
    for k in range(ei.size(1)):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        seen.add((a, b))
    return sorted(seen)


def _orient_undirected_pairs(
    pairs: Sequence[Tuple[int, int]],
    rng: np.random.Generator | None,
    orientation: OrientationMode,
) -> Tuple[List[int], List[int]]:
    src: List[int] = []
    dst: List[int] = []
    for a, b in pairs:
        if orientation == "low_to_high":
            u, v = a, b
        elif orientation == "random":
            assert rng is not None
            u, v = (a, b) if rng.random() < 0.5 else (b, a)
        else:
            raise ValueError(f"Unknown orientation: {orientation}")
        src.append(u)
        dst.append(v)
    return src, dst


def _canonical_pairs(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Map each undirected bond {u,v} to one-hot edge attributes (must match both directions)."""
    ei = edge_index.cpu()
    ea = edge_attr.cpu()
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for k in range(ei.size(1)):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        row = ea[k]
        if key not in out:
            out[key] = row
        else:
            if not torch.equal(out[key], row):
                raise ValueError(
                    f"Inconsistent edge attributes for undirected pair {key}: "
                    f"{out[key]} vs {row}"
                )
    return out


def _orient_graph(
    bond_attr: Dict[Tuple[int, int], torch.Tensor],
    rng: np.random.Generator | None,
    orientation: OrientationMode,
) -> Tuple[List[int], List[int], List[torch.Tensor]]:
    src: List[int] = []
    dst: List[int] = []
    attrs: List[torch.Tensor] = []
    for a, b in sorted(bond_attr.keys()):
        attr = bond_attr[(a, b)]
        if orientation == "low_to_high":
            u, v = a, b
        elif orientation == "random":
            assert rng is not None
            u, v = (a, b) if rng.random() < 0.5 else (b, a)
        else:
            raise ValueError(f"Unknown orientation: {orientation}")
        src.append(u)
        dst.append(v)
        attrs.append(attr.clone())
    return src, dst, attrs


def _one_hot_to_int_labels(t: torch.Tensor) -> np.ndarray:
    if t.numel() == 0:
        return np.zeros(0, dtype=np.int64)
    return t.argmax(dim=-1).numpy().astype(np.int64)


def export_directed_mutag(
    *,
    source_root: str = "data/TUDataset",
    out_name: str = "MUTAG_DIR",
    tu_root: str | None = None,
    seed: int = 42,
    orientation: OrientationMode = "random",
    clean_processed: bool = True,
) -> str:
    """
    Create ``<tu_root>/<out_name>/raw/<out_name>_*.txt`` from MUTAG.

    Returns path to the ``raw`` directory.
    """
    if tu_root is None:
        tu_root = source_root

    ds = TUDataset(root=source_root, name="MUTAG", use_edge_attr=True)
    rng = np.random.default_rng(seed) if orientation == "random" else None

    raw_dir = os.path.join(tu_root, out_name, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    a_lines: List[str] = []
    edge_label_lines: List[str] = []
    graph_indicator_lines: List[str] = []
    graph_labels_lines: List[str] = []
    node_labels_lines: List[str] = []

    node_offset = 0
    stats: List[dict] = []

    for g_idx in range(len(ds)):
        data = ds[g_idx]
        n = int(data.num_nodes)
        bonds = _canonical_pairs(data.edge_index, data.edge_attr)
        src, dst, attrs = _orient_graph(bonds, rng, orientation)

        graph_labels_lines.append(f"{int(data.y.item())}\n")
        graph_indicator_lines.extend([f"{g_idx + 1}\n"] * n)

        nl = _one_hot_to_int_labels(data.x)
        if nl.shape[0] != n:
            raise RuntimeError(f"Graph {g_idx}: node label count {nl.shape[0]} != num_nodes {n}")
        for lab in nl.tolist():
            node_labels_lines.append(f"{lab}\n")

        bond_types = _one_hot_to_int_labels(torch.stack(attrs, dim=0))
        for u, v, bt in zip(src, dst, bond_types.tolist()):
            gu, gv = node_offset + u + 1, node_offset + v + 1
            a_lines.append(f"{gu}, {gv}\n")
            edge_label_lines.append(f"{bt}\n")

        ei = torch.tensor([src, dst], dtype=torch.long)
        ea = torch.stack(attrs, dim=0)
        A = to_dense_adj(ei, max_num_nodes=n, edge_attr=None)[0].numpy()
        sym = bool(np.allclose(A, A.T))
        stats.append(
            {
                "graph_index": g_idx,
                "num_nodes": n,
                "num_directed_edges": len(src),
                "num_undirected_bonds": len(bonds),
                "adj_symmetric": sym,
            }
        )

        node_offset += n

    prefix = out_name
    with open(os.path.join(raw_dir, f"{prefix}_A.txt"), "w", encoding="utf-8") as f:
        f.writelines(a_lines)
    with open(os.path.join(raw_dir, f"{prefix}_graph_indicator.txt"), "w", encoding="utf-8") as f:
        f.writelines(graph_indicator_lines)
    with open(os.path.join(raw_dir, f"{prefix}_graph_labels.txt"), "w", encoding="utf-8") as f:
        f.writelines(graph_labels_lines)
    with open(os.path.join(raw_dir, f"{prefix}_node_labels.txt"), "w", encoding="utf-8") as f:
        f.writelines(node_labels_lines)
    with open(os.path.join(raw_dir, f"{prefix}_edge_labels.txt"), "w", encoding="utf-8") as f:
        f.writelines(edge_label_lines)

    meta = {
        "source": "MUTAG",
        "source_root": os.path.abspath(source_root),
        "out_name": out_name,
        "tu_root": os.path.abspath(tu_root),
        "seed": seed,
        "orientation": orientation,
        "num_graphs": len(ds),
        "per_graph": stats,
    }
    meta_path = os.path.join(tu_root, out_name, "BUILD_META.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    proc = os.path.join(tu_root, out_name, "processed")
    if clean_processed and os.path.isdir(proc):
        shutil.rmtree(proc)

    return raw_dir


def export_directed_reddit_binary(
    *,
    source_root: str = "data/TUDataset",
    out_name: str = "REDDIT-BINARY_DIR",
    tu_root: str | None = None,
    seed: int = 42,
    orientation: OrientationMode = "random",
    clean_processed: bool = True,
) -> str:
    """
    Create ``<tu_root>/<out_name>/raw/`` TU files from REDDIT-BINARY (no node/edge label files).

    Returns path to the ``raw`` directory.
    """
    if tu_root is None:
        tu_root = source_root

    ds = TUDataset(root=source_root, name="REDDIT-BINARY", use_edge_attr=False)
    rng = np.random.default_rng(seed) if orientation == "random" else None

    raw_dir = os.path.join(tu_root, out_name, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    a_lines: List[str] = []
    graph_indicator_lines: List[str] = []
    graph_labels_lines: List[str] = []

    node_offset = 0
    stats: List[dict] = []

    for g_idx in range(len(ds)):
        data = ds[g_idx]
        n = int(data.num_nodes)
        pairs = _canonical_undirected_pairs(data.edge_index)
        src, dst = _orient_undirected_pairs(pairs, rng, orientation)

        graph_labels_lines.append(f"{int(data.y.item())}\n")
        graph_indicator_lines.extend([f"{g_idx + 1}\n"] * n)

        for u, v in zip(src, dst):
            gu, gv = node_offset + u + 1, node_offset + v + 1
            a_lines.append(f"{gu}, {gv}\n")

        ei = torch.tensor([src, dst], dtype=torch.long)
        A = to_dense_adj(ei, max_num_nodes=n, edge_attr=None)[0].numpy()
        sym = bool(np.allclose(A, A.T))
        stats.append(
            {
                "graph_index": g_idx,
                "num_nodes": n,
                "num_directed_edges": len(src),
                "num_undirected_edges": len(pairs),
                "adj_symmetric": sym,
            }
        )

        node_offset += n

    prefix = out_name
    with open(os.path.join(raw_dir, f"{prefix}_A.txt"), "w", encoding="utf-8") as f:
        f.writelines(a_lines)
    with open(os.path.join(raw_dir, f"{prefix}_graph_indicator.txt"), "w", encoding="utf-8") as f:
        f.writelines(graph_indicator_lines)
    with open(os.path.join(raw_dir, f"{prefix}_graph_labels.txt"), "w", encoding="utf-8") as f:
        f.writelines(graph_labels_lines)

    meta = {
        "source": "REDDIT-BINARY",
        "source_root": os.path.abspath(source_root),
        "out_name": out_name,
        "tu_root": os.path.abspath(tu_root),
        "seed": seed,
        "orientation": orientation,
        "num_graphs": len(ds),
        "per_graph": stats,
    }
    meta_path = os.path.join(tu_root, out_name, "BUILD_META.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    proc = os.path.join(tu_root, out_name, "processed")
    if clean_processed and os.path.isdir(proc):
        shutil.rmtree(proc)

    return raw_dir


def _verify_load(tu_root: str, name: str, *, use_edge_attr: bool = True) -> None:
    ds = TUDataset(root=tu_root, name=name, use_edge_attr=use_edge_attr, force_reload=True)
    asymmetric = 0
    for i in range(len(ds)):
        d = ds[i]
        A = to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes)[0].numpy()
        if not np.allclose(A, A.T):
            asymmetric += 1
    print(
        f"[verify] Loaded {len(ds)} graphs; {asymmetric} have non-symmetric adjacency "
        f"(expected {len(ds)} whenever each bond is a single directed edge)."
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export a directed TU dataset (MUTAG or REDDIT-BINARY) from PyG TUDataset."
    )
    p.add_argument(
        "--source_name",
        type=str,
        choices=("MUTAG", "REDDIT-BINARY"),
        default="MUTAG",
        help="Which TU dataset to read and orient",
    )
    p.add_argument(
        "--source_root",
        type=str,
        default="data/TUDataset",
        help="Root containing the source dataset folder (e.g. data/TUDataset/MUTAG/)",
    )
    p.add_argument(
        "--tu_root",
        type=str,
        default=DEFAULT_TU_OUTPUT_ROOT,
        help=f"Output TU root (default: {DEFAULT_TU_OUTPUT_ROOT})",
    )
    p.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="New dataset folder name under tu_root (default: MUTAG_DIR or REDDIT-BINARY_DIR)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (random orientation only)")
    p.add_argument(
        "--orientation",
        choices=("random", "low_to_high"),
        default="random",
        help="How to orient each undirected edge",
    )
    p.add_argument("--no_clean_processed", action="store_true", help="Keep stale processed/ if present")
    p.add_argument("--verify", action="store_true", help="Reload with PyG and print adjacency symmetry stats")
    args = p.parse_args(argv)

    out_name = args.out_name
    if out_name is None:
        out_name = "MUTAG_DIR" if args.source_name == "MUTAG" else "REDDIT-BINARY_DIR"

    clean = not args.no_clean_processed
    if args.source_name == "MUTAG":
        raw_dir = export_directed_mutag(
            source_root=args.source_root,
            out_name=out_name,
            tu_root=args.tu_root,
            seed=args.seed,
            orientation=args.orientation,
            clean_processed=clean,
        )
        verify_edge_attr = True
    else:
        raw_dir = export_directed_reddit_binary(
            source_root=args.source_root,
            out_name=out_name,
            tu_root=args.tu_root,
            seed=args.seed,
            orientation=args.orientation,
            clean_processed=clean,
        )
        verify_edge_attr = False

    print(f"[ok] Wrote TU raw files to: {raw_dir}")
    print(f"[ok] Metadata: {os.path.join(os.path.dirname(raw_dir), 'BUILD_META.json')}")
    if args.verify:
        _verify_load(args.tu_root, out_name, use_edge_attr=verify_edge_attr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
