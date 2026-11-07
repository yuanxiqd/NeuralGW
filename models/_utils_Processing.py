import os
import pickle
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import torch_geometric.utils as gutils
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd
import scipy as sp
from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurtosis
from scipy.sparse import csr_matrix as _scipy_csr_matrix
from scipy.sparse.linalg import eigsh as _scipy_eigsh
from scipy.linalg import eigh as _scipy_eigh
import ot
from tqdm import tqdm


def _normalize_shortest_path_cost(C: np.ndarray) -> np.ndarray:
    """
    Per-graph scale for GW / SVD-PE: finite shortest-path values to [0, 1] by max finite hop
    count; unreachable pairs (``inf``) become ``1.0`` (same upper scale as a diameter-normalized
    path). Safe when ``d_max == 0`` (e.g. isolated nodes: only diagonal zeros, rest ``inf``).
    """
    out = np.asarray(C, dtype=np.float32, copy=True)
    finite = np.isfinite(out)
    if not finite.any():
        out.fill(1.0)
        np.fill_diagonal(out, 0.0)
        return out
    d_max = float(np.max(out[finite]))
    if d_max > 0:
        out[finite] = (out[finite] / d_max).astype(np.float32, copy=False)
    out[~finite] = 1.0
    return out


def subsample_TUDataset(ds: TUDataset, ratio: int = 0.2, num_bins: int =10):
    num_nodes_seq = np.array([data.num_nodes for data in ds])
    if True:
        quantiles = np.linspace(0, 1, num_bins + 1) # 10 equal-mass bins
        bins = np.quantile(num_nodes_seq, quantiles)
    else:
        bins = np.histogram_bin_edges(num_nodes_seq, bins=num_bins)
    
    # print(f'bins: {bins}')
    # print(100*'-')
    
    # count, edges = np.histogram(num_nodes_seq, bins=bins)
    # print(f'(count, edges) = \n({count},\n {edges})')
    # print(100*'-')
    
    bin_ids = np.digitize(num_nodes_seq, bins=bins)
    # print(f'bin_ids: {bin_ids}')
    # print(100*'-')
    
    # for b in np.unique(bin_ids):
    #     print(b, np.sum(bin_ids == b))
    
    # print(100*'-')
    if num_nodes_seq[num_nodes_seq >= bins[-1]].any():
        # print(f'I have reset the idx of entries which are outside the range of bins.')
        bin_ids = np.clip(bin_ids, 1, len(bins) - 1)
    
    # for b in np.unique(bin_ids):
    #     print(b, np.sum(bin_ids == b))
    
    indices = np.arange(len(ds))
    train_idx, sample_idx = train_test_split(indices, train_size=1-ratio, stratify=bin_ids, random_state=0)
    # print(f'sample_idx: {sample_idx}')
    return sample_idx

def distance_to_kernel(D2, sigma=None, output_sigma=False):
    """
    Convert a distance matrix to a kernel matrix using the Gaussian kernel.
    Args:
        D (numpy.ndarray): distance matrix
        sigma (float): bandwidth parameter for the Gaussian kernel
    Returns:
        numpy.ndarray: kernel matrix
    """
    if sigma is None:
        # sigma = np.median(np.sqrt(D2))
        # sigma = np.mean(np.sqrt(D2))
        sigma = np.mean(D2)
        print(f'[Info] Using median of D as sigma: {sigma}')
    K = np.exp(-D2 / (2 * sigma**2))
    if output_sigma:
        return K, sigma
    else:
        return K

def flip_pe_sign(x, method='l0'):
    assert method in ['none', 'pos', 'sum', 'l1', 'l2', 'linf', 'random']
    if method == 'none':
        return x
    elif method == 'random':
        sign = torch.randint(0, 2, (x.shape[1],))*2 - 1
        return x * sign
    fliped_x = -1 * x
    if method == 'pos':
        x_v = torch.sum(x >= 0, dim=0)
        fliped_x_v = torch.sum(fliped_x >= 0, dim=0)
    elif method == 'sum':
        x_v = torch.sum(x, dim=0)
        fliped_x_v = torch.sum(fliped_x, dim=0)
    elif method == 'linf':
        x_v, _ = torch.max(x, dim=0)
        fliped_x_v, _ = torch.max(fliped_x, dim=0)
    elif method == 'l1':
        x_pos_mask = x >= 0
        x_v = torch.norm(x * x_pos_mask, dim=0, p=1)

        fliped_x_pos_mask = fliped_x >= 0
        fliped_x_v = torch.norm(fliped_x * fliped_x_pos_mask, dim=0, p=1)
    elif method == 'l2':
        x_pos_mask = x >= 0
        x_v = torch.norm(x * x_pos_mask, dim=0, p=2)

        fliped_x_pos_mask = fliped_x >= 0
        fliped_x_v = torch.norm(fliped_x * fliped_x_pos_mask, dim=0, p=2)
    else:
        raise NotImplementedError
    diff = x_v - fliped_x_v
    sign = torch.sign(diff)
    return x * sign

def generate_variable_sbm_graph_with_label(
    n_min=50, n_max=200,
    k_min=2, k_max=6,
    p_in_range=(0.2, 0.8),
    p_out_range=(0.01, 0.1),
    label_type="num_communities",
    seed = 42
    ) -> Data:
    """
    Generate an SBM graph with graph-level labels.
    label_type options:
      - 'num_communities'
      - 'density_regime'
      - 'community_balance'
    """
    rng = np.random.default_rng(seed=seed)

    # number of nodes
    # n = np.random.randint(n_min, n_max + 1)
    n = rng.integers(n_min, n_max + 1)

    # number of communities
    # k = np.random.randint(k_min, k_max + 1)
    k = rng.integers(k_min, k_max + 1)

    # random block sizes summing to n
    # raw_sizes = np.random.randint(1, 10, size=k)
    raw_sizes = rng.integers(1, 10, size=k)
    sizes = (raw_sizes / raw_sizes.sum() * n).astype(int)
    sizes[-1] += n - sizes.sum()

    # sampling intra/inter probabilities
    # p_in = np.random.uniform(*p_in_range)
    p_in = rng.uniform(*p_in_range)
    # p_out = np.random.uniform(*p_out_range)
    p_out = rng.uniform(*p_out_range)

    P = np.full((k, k), p_out)
    np.fill_diagonal(P, p_in)

    # NetworkX SBM
    # G = nx.stochastic_block_model(sizes, P, seed=np.random.randint(1e9))
    G = nx.stochastic_block_model(sizes, P, seed=np.random.randint(1e9))
    data = from_networkx(G)

    # Simple node features
    data.x = torch.ones((G.number_of_nodes(), 1))

    # Node-level community assignment
    community_labels = []
    for block_id, block_nodes in enumerate(G.graph["partition"]):
        for node in block_nodes:
            community_labels.append((node, block_id))
    community_labels = sorted(community_labels)
    data.node_y = torch.tensor([c for _, c in community_labels], dtype=torch.long)

    # ---- Graph-level labels ----
    if label_type == "num_communities":
        graph_y = k

    elif label_type == "density_regime":
        # classify based on separation p_in - p_out
        sep = p_in - p_out
        if sep > 0.4:
            graph_y = 2   # easy
        elif sep > 0.2:
            graph_y = 1   # medium
        else:
            graph_y = 0   # hard

    elif label_type == "community_balance":
        ratio = max(sizes) / min(sizes)
        if ratio < 1.5:
            graph_y = 0   # balanced
        elif ratio < 3:
            graph_y = 1   # somewhat skewed
        else:
            graph_y = 2   # highly unbalanced

    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    data.y = torch.tensor([graph_y], dtype=torch.long)

    # store metadata
    data.k = k
    data.p_in = p_in
    data.p_out = p_out
    data.seed = seed

    return data


def generate_sbm_dataset(
    num_graphs=200,
    n_min=200, n_max=200,
    k_min=5, k_max=5,
    label_type="num_communities"
    ) -> List[Data]:
    seeds_base = [42, 1, 3, 5, 7]
    seeds = [seed for seed in seeds_base for n_graphs in range(num_graphs//len(seeds_base))]
    dataset = []
    for seed in seeds:
        g = generate_variable_sbm_graph_with_label(
            n_min=n_min, n_max=n_max,
            k_min=k_min, k_max=k_max,
            label_type=label_type,
            seed=seed
        )
        dataset.append(g)
    return dataset


def _generate_er_sparse_dataset(num_graphs: int = 50, n_min: int = 300, n_max: int = 350,
                                density: float = 0.007, seed_base: int = 2025) -> List[Data]:
    """Erdős–Rényi G(n, p) graphs with n in [n_min, n_max] and p=density.
    Designed to match the empirical REDDIT-BINARY density (~0.007) at the same
    node-count range, so the training pool can be augmented with sparse examples
    that look like the cross-domain test distribution in scale (not structure)."""
    dataset = []
    for i in range(num_graphs):
        rng = np.random.default_rng(seed=seed_base + i)
        n = int(rng.integers(n_min, n_max + 1))
        # Upper-triangular Bernoulli edges, then symmetrize.
        ut = (rng.random((n, n)) < density).astype(np.uint8)
        ut = np.triu(ut, k=1)
        A = ut + ut.T
        rows, cols = np.where(A > 0)
        if rows.size == 0:
            # Degenerate edge case: force a single edge so the graph isn't isolated.
            rng2 = np.random.default_rng(seed=seed_base + i + 100000)
            u, v = int(rng2.integers(0, n)), int(rng2.integers(0, n))
            while u == v:
                v = int(rng2.integers(0, n))
            rows = np.array([u, v], dtype=np.int64)
            cols = np.array([v, u], dtype=np.int64)
        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=n)
        data.x = torch.ones((n, 1))
        data.y = torch.tensor([0], dtype=torch.long)
        dataset.append(data)
    return dataset


def _generate_ba_sparse_dataset(num_graphs: int = 50, n_min: int = 300, n_max: int = 350,
                                density: float = 0.007, seed_base: int = 7000) -> List[Data]:
    """Tree/hub-structured sparse graphs at the REDDIT-BINARY density (~0.007).

    Barabási–Albert (m=1) preferential-attachment **tree backbone** gives hub
    structure resembling REDDIT discussion threads (popular nodes accrue many
    replies); random edges are then added until the edge count hits the target
    density. Same *density* as ER_SPARSE but different *structure* — so comparing
    COLLAB+BA_SPARSE vs COLLAB+ER_SPARSE isolates 'does structure help on top of
    density?' for training-pool augmentation."""
    dataset = []
    for i in range(num_graphs):
        seed = seed_base + i
        rng = np.random.default_rng(seed=seed)
        n = int(rng.integers(n_min, n_max + 1))
        G = nx.barabasi_albert_graph(n, 1, seed=int(seed))  # n-1 edges, hub/tree structure
        target_edges = int(round(density * n * (n - 1) / 2.0))
        attempts = 0
        max_attempts = 50 * max(target_edges, 1)
        while G.number_of_edges() < target_edges and attempts < max_attempts:
            u = int(rng.integers(0, n)); v = int(rng.integers(0, n))
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
            attempts += 1
        edges = np.array(list(G.edges()), dtype=np.int64)
        if edges.size == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            both = np.concatenate([edges.T, edges[:, ::-1].T], axis=1)
            edge_index = torch.tensor(both, dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=n)
        data.x = torch.ones((n, 1))
        data.y = torch.tensor([0], dtype=torch.long)
        dataset.append(data)
    return dataset


def _edge_set_to_data(edges, n: int) -> Data:
    """Build an undirected ``Data`` from an iterable of (u,v) edges (no self/dup)."""
    e = [(int(min(u, v)), int(max(u, v))) for u, v in edges if int(u) != int(v)]
    e = sorted(set(e))
    if not e:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        arr = np.array(e, dtype=np.int64)
        both = np.concatenate([arr.T, arr[:, ::-1].T], axis=1)
        edge_index = torch.tensor(both, dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=n)
    data.x = torch.ones((n, 1))
    data.y = torch.tensor([0], dtype=torch.long)
    return data


def _generate_clique_dense_dataset(num_graphs: int = 50, n_min: int = 300, n_max: int = 350,
                                   density: float = 0.578, seed_base: int = 11000) -> List[Data]:
    """Union-of-cliques graphs at COLLAB density (~0.578). Models COLLAB's actual
    structure: collaboration ego-networks are unions of co-authorship cliques, giving
    very high clustering (~0.9) and heterogeneous degrees. Plants random cliques (size
    sampled per clique) until the target edge count is reached; trims a few edges to
    hit the density exactly."""
    dataset = []
    for i in range(num_graphs):
        rng = np.random.default_rng(seed_base + i)
        n = int(rng.integers(n_min, n_max + 1))
        target = int(round(density * n * (n - 1) / 2.0))
        edges = set()
        while len(edges) < target:
            k = int(rng.integers(120, 260))        # large varied cliques -> high clustering + hubs
            k = min(k, n)
            nodes = rng.choice(n, size=k, replace=False)
            for a in range(k):
                for b in range(a + 1, k):
                    u, v = int(nodes[a]), int(nodes[b])
                    edges.add((min(u, v), max(u, v)))
        edges = list(edges)
        if len(edges) > target:                    # trim overshoot to exact density
            keep = rng.choice(len(edges), size=target, replace=False)
            edges = [edges[j] for j in keep]
        dataset.append(_edge_set_to_data(edges, n))
    return dataset


def _generate_cfg_dense_dataset(num_graphs: int = 50, seed_base: int = 12000) -> List[Data]:
    """Configuration-model graphs matching COLLAB's per-graph DEGREE SEQUENCES.
    Preserves the degree distribution (heterogeneity, hubs) but randomizes edges
    otherwise — isolates the effect of matching degree structure. Uses COLLAB degree
    sequences as a summary statistic (no raw COLLAB edges are copied)."""
    root = os.environ.get('NEURALGW_DATA_ROOT', './data_300_350')
    collab = TUDataset(root=f'{root}/TUDataset', name='COLLAB')
    sub = [i for i in range(len(collab)) if 300 < collab[i].num_nodes < 350]
    collab = collab[sub]
    dataset = []
    for i in range(num_graphs):
        g = collab[i % len(collab)]
        n = g.num_nodes
        A = gutils.to_dense_adj(edge_index=g.edge_index, max_num_nodes=n).squeeze(0).cpu().numpy()
        A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
        deg_seq = (A > 0).sum(axis=1).astype(int).tolist()
        if sum(deg_seq) % 2 == 1:
            deg_seq[int(np.argmax(deg_seq))] += 1     # make degree sum even
        G = nx.configuration_model(deg_seq, seed=seed_base + i)
        G = nx.Graph(G)                                # collapse parallel edges
        G.remove_edges_from(nx.selfloop_edges(G))      # drop self-loops
        dataset.append(_edge_set_to_data(list(G.edges()), n))
    return dataset


def _generate_sbm_dense_dataset(num_graphs: int = 50, n_min: int = 300, n_max: int = 350,
                                density: float = 0.578, ratio: float = 2.2,
                                seed_base: int = 13000) -> List[Data]:
    """Stochastic-block-model graphs at COLLAB density with planted communities
    (within/between edge-prob ratio = ``ratio``). p_in is solved so the overall
    density hits the target; gives community structure and elevated clustering.
    Note: at density 0.578 strong communities are infeasible (p_in would exceed 1),
    so ratio is capped low — high density geometrically limits community contrast."""
    dataset = []
    for i in range(num_graphs):
        rng = np.random.default_rng(seed_base + i)
        n = int(rng.integers(n_min, n_max + 1))
        k = int(rng.integers(3, 9))                    # number of communities
        cuts = np.sort(rng.choice(np.arange(1, n), size=k - 1, replace=False)) if k > 1 else np.array([])
        bounds = np.concatenate(([0], cuts, [n]))
        sizes = np.diff(bounds).astype(int).tolist()
        W = sum(s * (s - 1) / 2.0 for s in sizes)      # within-community node pairs
        total = n * (n - 1) / 2.0
        B = total - W                                   # between-community pairs
        target_edges = density * total
        p_in = target_edges / (W + B / ratio) if (W + B / ratio) > 0 else density
        p_in = float(min(p_in, 1.0)); p_out = float(min(p_in / ratio, 1.0))
        P = np.full((k, k), p_out); np.fill_diagonal(P, p_in)
        G = nx.stochastic_block_model(sizes, P, seed=int(seed_base + i))
        dataset.append(_edge_set_to_data(list(G.edges()), n))
    return dataset


META_FEATURE_GROUPS = ('size', 'degree', 'clustering', 'spectral')
META_GROUP_KEYS: Dict[str, List[str]] = {
    'size':       ['n_nodes', 'n_edges', 'edge_density'],
    'degree':     ['deg_mean', 'deg_std', 'deg_max', 'deg_min', 'deg_skew', 'deg_kurt'],
    'clustering': ['transitivity', 'mean_clustering', 'degree_assortativity'],
    'spectral':   ['lap_eig_1', 'lap_eig_2', 'lap_eig_3', 'lap_eig_4', 'lap_eig_5',
                   'spectral_gap', 'trA2_over_n', 'trA3_over_n'],
}
META_FULL_KEYS: List[str] = [k for g in META_FEATURE_GROUPS for k in META_GROUP_KEYS[g]]


def _meta_safe(x) -> float:
    return float(np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))


def _meta_binary_adj(data: Data, treat_as_undirected: bool = True) -> np.ndarray:
    A = gutils.to_dense_adj(edge_index=data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).cpu().numpy()
    np.fill_diagonal(A, 0)
    A = (A > 0).astype(np.float64)
    if treat_as_undirected:
        A = np.maximum(A, A.T)
    return A


def _size_transform_fn():
    """Transform applied to n_nodes / n_edges, selected by NEURALGW_SIZE_TRANSFORM
    (default 'sqrt'): raw | log (log1p) | sqrt. edge_density is never transformed.

    Default is 'sqrt': across COLLAB -> REDDIT-BINARY it gave the best 3-seed mean
    (2.666 vs log 2.680, raw 2.763); compressing the cross-domain edge-count gap
    (COLLAB ~5000 vs REDDIT ~360) helps FiLM transfer."""
    t = os.environ.get('NEURALGW_SIZE_TRANSFORM', 'sqrt').lower()
    if t == 'raw':
        return (lambda x: float(x)), t
    if t == 'log':
        return (lambda x: float(np.log1p(x))), t
    if t == 'sqrt':
        return (lambda x: float(np.sqrt(max(x, 0.0)))), t
    raise ValueError(f"Unknown NEURALGW_SIZE_TRANSFORM: {t!r}. Choose from: raw, log, sqrt.")


def _meta_group_size(A: np.ndarray) -> Dict[str, float]:
    n = int(A.shape[0])
    m_undirected = float(A.sum()) / 2.0
    denom = n * (n - 1) / 2.0 if n > 1 else 1.0
    fn, _ = _size_transform_fn()
    return {
        'n_nodes':      _meta_safe(fn(n)),            # node count (transform per NEURALGW_SIZE_TRANSFORM)
        'n_edges':      _meta_safe(fn(m_undirected)), # undirected edge count (same transform)
        'edge_density': _meta_safe(m_undirected / denom),
    }


def _meta_group_degree(A: np.ndarray) -> Dict[str, float]:
    deg = A.sum(axis=1)
    if deg.size == 0 or float(deg.std()) < 1e-12:
        skew_v, kurt_v = 0.0, 0.0
    else:
        skew_v = _scipy_skew(deg, bias=False, nan_policy='omit')
        kurt_v = _scipy_kurtosis(deg, bias=False, nan_policy='omit')
    return {
        'deg_mean': _meta_safe(deg.mean() if deg.size else 0.0),
        'deg_std':  _meta_safe(deg.std() if deg.size else 0.0),
        'deg_max':  _meta_safe(deg.max() if deg.size else 0.0),
        'deg_min':  _meta_safe(deg.min() if deg.size else 0.0),
        'deg_skew': _meta_safe(skew_v),
        'deg_kurt': _meta_safe(kurt_v),
    }


def _meta_group_clustering(A: np.ndarray) -> Dict[str, float]:
    G = nx.from_numpy_array(A)
    try:
        trans = nx.transitivity(G)
    except Exception:
        trans = 0.0
    try:
        clust_vals = list(nx.clustering(G).values())
        mean_clust = float(np.mean(clust_vals)) if clust_vals else 0.0
    except Exception:
        mean_clust = 0.0
    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = 0.0
    return {
        'transitivity':         _meta_safe(trans),
        'mean_clustering':      _meta_safe(mean_clust),
        'degree_assortativity': _meta_safe(assort),
    }


def _meta_group_spectral(A: np.ndarray, k_eigs: int = 5) -> Dict[str, float]:
    n = int(A.shape[0])
    out = {f'lap_eig_{i+1}': 0.0 for i in range(k_eigs)}
    out['spectral_gap'] = 0.0
    out['trA2_over_n'] = 0.0
    out['trA3_over_n'] = 0.0
    if n == 0:
        return out
    deg = A.sum(axis=1)
    d_inv_sqrt = np.zeros(n, dtype=np.float64)
    np.divide(1.0, np.sqrt(deg), out=d_inv_sqrt, where=deg > 0)
    Dmh = np.diag(d_inv_sqrt)
    L_sym = np.eye(n, dtype=np.float64) - Dmh @ A @ Dmh
    L_sym = (L_sym + L_sym.T) / 2.0
    try:
        if n < 100:
            evals = _scipy_eigh(L_sym, eigvals_only=True)
            evals = np.asarray(evals, dtype=np.float64)
            evals.sort()
        else:
            evals = _scipy_eigsh(_scipy_csr_matrix(L_sym), k=min(k_eigs + 1, n - 1),
                                 which='SA', return_eigenvectors=False)
            evals = np.asarray(evals, dtype=np.float64)
            evals.sort()
    except Exception:
        evals = np.zeros(k_eigs + 1, dtype=np.float64)
    for i in range(k_eigs):
        out[f'lap_eig_{i+1}'] = _meta_safe(evals[i] if i < evals.size else 0.0)
    if evals.size >= 2:
        out['spectral_gap'] = _meta_safe(evals[1] - evals[0])
    A2 = A @ A
    out['trA2_over_n'] = _meta_safe(np.trace(A2) / max(n, 1))
    out['trA3_over_n'] = _meta_safe(float(np.einsum('ij,jk,ki->', A, A, A)) / max(n, 1))
    return out


def _extract_graph_meta(data: Data, treat_as_undirected: bool = True) -> Dict[str, float]:
    """Compute the full meta-feature superset (18 scalars) for one graph. NaN-safe."""
    A = _meta_binary_adj(data, treat_as_undirected=treat_as_undirected)
    out: Dict[str, float] = {}
    out.update(_meta_group_size(A))
    out.update(_meta_group_degree(A))
    out.update(_meta_group_clustering(A))
    out.update(_meta_group_spectral(A))
    return out


def active_meta_keys() -> List[str]:
    """Subset of META_FULL_KEYS selected by NEURALGW_META_FEATURES env var."""
    raw = os.environ.get('NEURALGW_META_FEATURES', ','.join(META_FEATURE_GROUPS))
    groups = [g.strip() for g in raw.split(',') if g.strip()]
    for g in groups:
        if g not in META_GROUP_KEYS:
            raise ValueError(f"Unknown meta group: {g!r}. Choose from {list(META_GROUP_KEYS)}.")
    return [k for g in groups for k in META_GROUP_KEYS[g]]


class DataPreparationSystem:
    """
    Data preparation system for graph datasets and precomputed caches.
    Precomputed PE files live under ``precomputed/pe_{pe_cost_mode}_{pe_mode}/`` (with ``_dir`` / ``_nrm``
    suffixes as before): per-dataset ``{ds_name}_pe_dict.pkl`` only — a ``dict[local_graph_idx -> ndarray]``
    of shape ``(num_nodes, feature_dim)``. ``compose_pe_from_multiple_ds`` returns
    ``dict[dataset_name -> dict[local_graph_idx -> ndarray]]``. Legacy ``{ds_name}_pe.npy`` files are removed when new caches are written.
    GW / transport-plan caches use ``precomputed/gwd_{gw_cost_mode}/`` (and matching
    ``tp_*``).

    **``pe_cost_mode`` / ``gw_cost_mode``** (each ``'adj'`` or ``'sp'``): matrix used for SVD-PE vs for
    Gromov–Wasserstein / transport plans. ``'adj'`` uses symmetrized (if undirected) adjacency plus
    self-loops where applicable; ``'sp'`` uses per-graph normalized unweighted shortest-path lengths
    (``inf`` → ``1``). With ``treat_as_undirected=False``, shortest paths follow SciPy
    ``directed=True`` for whichever side is ``'sp'``.

    **Undirected (default):** ``treat_as_undirected=True`` symmetrizes adjacency with ``max(A, A.T)``.
    Shortest-path on undirected graphs uses ``directed=False``.

    ``LapPE`` / ``RWSE`` ignore ``pe_cost_mode`` (they use adjacency structure only).
    """

    def __init__(
        self,
        ds_name_seq: List[str],
        pe_cost_mode: str = 'adj',
        gw_cost_mode: str = 'adj',
        pe_mode: str = 'svd',
        treat_as_undirected: bool = True,
    ):
        self._ds_name_seq = ds_name_seq
        for name, m in (
            ('pe_cost_mode', pe_cost_mode),
            ('gw_cost_mode', gw_cost_mode),
        ):
            if m not in ('adj', 'sp'):
                raise ValueError(f"{name} must be 'adj' or 'sp', got {m!r}")
        self._pe_cost_mode = pe_cost_mode
        self._gw_cost_mode = gw_cost_mode
        self._data_root = os.environ.get('NEURALGW_DATA_ROOT', './data')
        self._pe_mode = pe_mode
        self._treat_as_undirected = treat_as_undirected
        # create the precomputed directory if it doesn't exist
        os.makedirs(f'{self._data_root}/precomputed/pe_{self._precomputed_cost_pe_tag()}', exist_ok=True)
        os.makedirs(f'{self._data_root}/precomputed/gwd_{self._gwd_tp_cache_tag()}', exist_ok=True)
        os.makedirs(f'{self._data_root}/precomputed/tp_{self._gwd_tp_cache_tag()}', exist_ok=True)
        os.makedirs(f'{self._data_root}/precomputed/meta', exist_ok=True)

    def _precomputed_cost_pe_tag(self) -> str:
        """Subfolder name for PE caches (legacy path when undirected; ``_dir`` suffix when directed)."""
        base = f'{self._pe_cost_mode}_{self._pe_mode}'
        tagged = base if self._treat_as_undirected else f'{base}_dir'
        if self._pe_cost_mode == 'sp':
            tagged = f'{tagged}_nrm'
        return tagged

    def _gwd_tp_cache_tag(self) -> str:
        """Subfolder for GW distance and transport-plan caches."""
        base = self._gw_cost_mode if self._treat_as_undirected else f'{self._gw_cost_mode}_dir'
        if self._gw_cost_mode == 'sp':
            return f'{base}_nrm'
        return base

    def _load_graph_ds(self, name: str ='MUTAG'):
        # load the specified graph dataset from TUDataset
        if name == 'DD':
            ds = TUDataset(root=f'{self._data_root}/TUDataset', name=name)
            sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
            # sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes <= 80]
            ds = ds[sub_idx]
        elif name == 'ENZYMES':
            ds = TUDataset(root=f'{self._data_root}/TUDataset', name=name)
            if False:
                sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes <= 80]
                ds = ds[sub_idx]

                sub_idx = subsample_TUDataset(ds=ds, ratio=0.3)
            else:
                sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes >= 40]
            ds = ds[sub_idx]
        elif name == 'COLLAB':
            # ds = TUDataset(root='data/TUDataset', name=ds_name)
            # sub_idx = subsample_TUDataset(ds=ds, ratio=0.2)
            # ds = ds[sub_idx]
            ds = TUDataset(root=f'{self._data_root}/TUDataset', name=name)
            sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
            # sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes <= 80]
            ds = ds[sub_idx]

            # sub_idx = subsample_TUDataset(ds=ds, ratio=0.05)
            # ds = ds[sub_idx]
        elif name == 'COLLAB_BAL':
            # Balanced 3-class COLLAB subset for the structural clustering table:
            # first M graphs per class, sorted by global index. Must match
            # scripts/prep_collab_bal.py (which precomputes the matching GW labels).
            ds = TUDataset(root=f'{self._data_root}/TUDataset', name='COLLAB')
            M = 25
            byc = {}
            for i in range(len(ds)):
                byc.setdefault(int(ds[i].y), []).append(i)
            sub_idx = []
            for c in sorted(byc):
                sub_idx += byc[c][:M]
            sub_idx = sorted(sub_idx)
            ds = ds[sub_idx]
        elif name == 'IMDB-MULTI':
            ds = TUDataset(root=f'{self._data_root}/TUDataset', name=name)
            sub_idx = subsample_TUDataset(ds=ds, ratio=0.2)
            ds = ds[sub_idx]
        elif name == 'REDDIT-BINARY':
            ds = TUDataset(root=f'{self._data_root}/TUDataset', name=name)
            sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
            # sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes >= 1000]
            ds = ds[sub_idx]
        elif name == 'SBM':
            ds = generate_sbm_dataset(num_graphs=200, n_min=100, n_max=100, k_min=5, k_max=5, label_type="num_communities")
            self._sbm_seeds = list(dict.fromkeys([g.seed for g in ds]))
        elif name == 'ER_SPARSE':
            # Erdős–Rényi sparse graphs matching REDDIT-BINARY's node range (300–350)
            # and density (~0.007). Used for training-pool augmentation in cross-domain
            # transfer to REDDIT-BINARY.
            ds = _generate_er_sparse_dataset(num_graphs=50, n_min=300, n_max=350, density=0.007)
        elif name == 'BA_SPARSE':
            # Tree/hub-structured sparse graphs at REDDIT-BINARY density (~0.007):
            # BA(m=1) backbone + random edges. Same density as ER_SPARSE, different
            # structure (isolates the effect of structure in augmentation).
            ds = _generate_ba_sparse_dataset(num_graphs=50, n_min=300, n_max=350, density=0.007)
        elif name == 'ER_DENSE':
            # Erdős–Rényi graphs at COLLAB's mean density (~0.578), node range 300–350.
            # The dense analog of ER_SPARSE — used for the reverse-direction transfer
            # test (train on density-matched random graphs, EVALUATE on COLLAB).
            ds = _generate_er_sparse_dataset(num_graphs=50, n_min=300, n_max=350,
                                             density=0.578, seed_base=9000)
        elif name == 'BA_DENSE':
            # Hub-structured dense graphs at COLLAB density (~0.578); BA(m≈95) backbone.
            ds = _generate_ba_sparse_dataset(num_graphs=50, n_min=300, n_max=350,
                                             density=0.578, seed_base=9500)
        elif name == 'CLIQUE_DENSE':
            # Structure-matched: union-of-cliques at COLLAB density (high clustering).
            ds = _generate_clique_dense_dataset(num_graphs=50, n_min=300, n_max=350, density=0.578)
        elif name == 'SBM_DENSE':
            # Structure-matched: SBM with planted communities at COLLAB density.
            ds = _generate_sbm_dense_dataset(num_graphs=50, n_min=300, n_max=350, density=0.578)
        elif name in ('ENZYMES_BAL', 'DD_BAL'):
            # Balanced bioinformatics subsets for the clustering table: first M graphs
            # per class (optionally num_nodes<=cap), sorted by global index. MUST match
            # scripts/prep_bio_bal.py (which precomputes the matching GW labels).
            base, M, cap = ('ENZYMES', 25, None) if name == 'ENZYMES_BAL' else ('DD', 30, 150)
            ds = TUDataset(root=f'{self._data_root}/TUDataset', name=base)
            byc = {}
            for i in range(len(ds)):
                if cap is None or ds[i].num_nodes <= cap:
                    byc.setdefault(int(ds[i].y), []).append(i)
            sub_idx = []
            for c in sorted(byc):
                sub_idx += byc[c][:M]
            ds = ds[sorted(sub_idx)]
        # NOTE: CFG_DENSE (configuration model on COLLAB degree sequences) is NOT
        # registered — dense degree sequences are not realizable as simple graphs
        # (parallel edges collapse, dropping density 0.578 -> ~0.10). The generator
        # function is kept for reference but excluded from the structure-match ladder.
        else:
            ds = TUDataset(root=f'{self._data_root}/TUDataset', name=name)
        return(ds)
    
    def load_all_graph_ds(self) -> List[TUDataset]:
        # load all specified graph datasets from TUDataset
        ds_seq = [self._load_graph_ds(name=ds_name) for ds_name in self._ds_name_seq]
        return ds_seq
    
    def get_num_nodes_seq(self) -> List[int]:
        # get the number of nodes in each graph dataset
        num_nodes_seq = []
        for ds_name in self._ds_name_seq:
            ds = self._load_graph_ds(name=ds_name)
            num_nodes_seq.extend([g.num_nodes for g in ds])
        return num_nodes_seq
    
    def _get_num_graphs_seq(self) -> List[int]:
        # get the number of graphs in each dataset
        num_graphs_seq = []
        for ds_name in self._ds_name_seq:
            ds = self._load_graph_ds(name=ds_name)
            num_graphs_seq.append(len(ds))
        return num_graphs_seq
    
    def _create_mapping_table(self, num_graphs_seq: List[int]):
        # create a mapping table from global graph indices to (dataset_index, local_graph_index)
        mapping_table = {}
        for i, num_graphs in enumerate(num_graphs_seq):
            for j in range(num_graphs):
                global_index = sum(num_graphs_seq[:i]) + j
                mapping_table[global_index] = (i, j)
        return mapping_table
    
    def train_test_split(self, testset: str, test_ratio: float = 0.2, random_seed: int = 42) -> Tuple[List[int], List[int]]:
        if testset == 'global':
            rng = np.random.default_rng(random_seed)
            num_graphs_seq = self._get_num_graphs_seq()
            total_num_graphs = sum(num_graphs_seq)
            all_indices = np.arange(total_num_graphs)
            rng.shuffle(all_indices)
            num_test = int(np.ceil(total_num_graphs * test_ratio))
            test_indices = all_indices[:num_test].tolist()
            train_indices = all_indices[num_test:].tolist()
            return train_indices, test_indices
        if testset == 'SBM':
            if self._ds_name_seq != ['SBM']:
                raise ValueError('When testset is "sbm", the only dataset in ds_name_seq should be "SBM".')
            
            from sklearn import model_selection
            num_graphs_seq = self._get_num_graphs_seq()
            total_num_graphs = sum(num_graphs_seq)
            all_indices = np.arange(total_num_graphs)
            indices_by_class = np.split(all_indices, indices_or_sections=5)
            train_idx, eval_idx, test_idx = [], [], []
            for i in range(len(indices_by_class)):
                train_eval_idx_by_each, test_idx_by_each = model_selection.train_test_split(indices_by_class[i], test_size=test_ratio, random_state=random_seed)
                train_idx_by_each, eval_idx_by_each = model_selection.train_test_split(train_eval_idx_by_each, test_size=test_ratio, random_state=random_seed)
                train_idx.extend(train_idx_by_each.tolist())
                eval_idx.extend(eval_idx_by_each.tolist())
                test_idx.extend(test_idx_by_each.tolist())
            return train_idx, eval_idx, test_idx
        elif len([ds_name for ds_name in self._ds_name_seq if testset == ds_name]) == 1:
            # In this case, we use the specified dataset as the test set while the rest as the training set
            mapping_table = self._create_mapping_table(num_graphs_seq=self._get_num_graphs_seq())
            test_indices = [global_idx for global_idx, (ds_idx, _) in mapping_table.items() if self._ds_name_seq[ds_idx] == testset]
            train_indices = [global_idx for global_idx, (ds_idx, _) in mapping_table.items() if self._ds_name_seq[ds_idx] != testset]
            return train_indices, test_indices
        elif len([ds_name for ds_name in self._ds_name_seq if testset == ds_name]) > 1:
            # preprocess ds_name_seq when a dataset arise over one time
            testset_ptr = [i for i, ds_name in enumerate(self._ds_name_seq) if ds_name == testset]
            self._ds_name_seq = [ds_name for i, ds_name in enumerate(self._ds_name_seq) if not (ds_name == testset and i != testset_ptr[0])]
            mapping_table = self._create_mapping_table(num_graphs_seq=self._get_num_graphs_seq())

            # retrieve the global idx of graphs that are in the test as extra training idx
            extra_train_idx = [global_idx for global_idx, (ds_idx, _) in mapping_table.items() if self._ds_name_seq[ds_idx] != testset]

            # retrieve the global idx of graphs in testset and split it into training, validation, test sub_idx
            base_train_eval_test_idx = [global_idx for global_idx, (ds_idx, _) in mapping_table.items() if self._ds_name_seq[ds_idx] == testset]

            # use another idx system to split base_train_eval_test_idx (In fact, this another idx system acts as a pointer system)
            rng = np.random.default_rng(random_seed)
            base_total_num_graph = len(base_train_eval_test_idx)
            print(f'dps.train_test_split: base_total_num_graph = {base_total_num_graph}')
            all_pointers = np.arange(base_total_num_graph)
            rng.shuffle(all_pointers)
            num_test = int(np.ceil(base_total_num_graph * test_ratio))
            test_pointers = all_pointers[:num_test].tolist()
            train_eval_pointers = all_pointers[num_test:].tolist()
            base_train_eval_idx = [global_idx for i, global_idx in enumerate(base_train_eval_test_idx) if i in train_eval_pointers]
            test_idx = [global_idx for i, global_idx in enumerate(base_train_eval_test_idx) if i in test_pointers]
            assert sorted(base_train_eval_idx + test_idx) == sorted(base_train_eval_test_idx), f'dps.train_test_split: train_eval_test split error for the test that arises in ds_name_seq for over one time'
            base_train_idx, eval_idx = train_test_split(base_train_eval_idx, test_size=0.2, random_state=random_seed)
            
            # merge base_train_idx from testset and extra_train_idx from other datasets into the train_idx
            train_idx = base_train_idx + extra_train_idx

            return train_idx, eval_idx, test_idx
        else:
            raise ValueError(f'Invalid testset: {testset}. Must be "global" or one of {self._ds_name_seq}.')

    def train_test_split_based_on_distance_matrix(self, testset: str, test_ratio: float = 0.2, random_seed: int = 42) -> Tuple[List[int], List[int]]:
        """
        This function is created to use index system directly on the pairwise distance matrix
        Given a pairwise distance matrix of n x n,
            all_pair_global_idx = np.stack(np.triu_indices(n, k=1), axis=1) 
            # In each row of pair_global_idx, (global_idx_i, global_idx_j) represents the pair of graphs indexed by global_idx_i and global_idx_j
            , and either global_idx_i or global_idx_j can be used to find the ds_id and graph_id they belong to by self._create_mapping_table.
            total_idx = np.range(pair_global_idx.shape[0]) # this is the total indices that are connected to the distance value of the graph pair (upper triangular diatance matrix)
            rng = np.random.default_rng(random_seed)
            rng.shuffle(total_idx)
            num_test = int(np.ceil(len(total_idx) * test_ratio))
            test_idx = total_idx[:num_test].tolist()
            train_eval_idx = total_idx[num_test:].tolist()
        """
        if testset == 'global':
            rng = np.random.default_rng(random_seed)
            num_graphs_seq = self._get_num_graphs_seq()
            total_num_graphs = sum(num_graphs_seq) # this indicates that the composed distance matrix will have a shape toal_num_graphs x toal_num_graphs
            # all_indices = np.arange(total_num_graphs)
            all_pair_global_idx = np.stack(np.triu_indices(total_num_graphs, k=1), axis=1)
            all_indices = np.arange(all_pair_global_idx.shape[0])
            rng.shuffle(all_indices)
            num_test = int(np.ceil(len(all_indices) * test_ratio))
            test_indices = all_indices[:num_test].tolist()
            train_eval_indices = all_indices[num_test:].tolist()
            return train_eval_indices, test_indices
        elif testset == 'SBM':
            if self._ds_name_seq != ['SBM']:
                raise ValueError('When testset is "sbm", the only dataset in ds_name_seq should be "SBM".')
                
        elif len([ds_name for ds_name in self._ds_name_seq if testset == ds_name]) == 1:
            mapping_table = self._create_mapping_table(num_graphs_seq=self._get_num_graphs_seq())

        elif len([ds_name for ds_name in self._ds_name_seq if testset == ds_name]) > 1:
            pass
        else:
            raise ValueError(f'Invalid testset: {testset}. Must be "global" or one of {self._ds_name_seq}.')

    
    def _pe_dict_cache_path(self, ds_name: str) -> str:
        """Path to the pickle dict cache ``{ds_name}_pe_dict.pkl`` for one dataset."""
        base = f'{self._data_root}/precomputed/pe_{self._precomputed_cost_pe_tag()}'
        return f'{base}/{ds_name}_pe_dict.pkl'

    def _pe_legacy_npy_path(self, ds_name: str) -> str:
        """Deprecated stacked-array path (deleted after recomputing to dict format)."""
        base = f'{self._data_root}/precomputed/pe_{self._precomputed_cost_pe_tag()}'
        return f'{base}/{ds_name}_pe.npy'

    @staticmethod
    def max_pe_feature_dim(pes_by_dataset: Dict[str, Dict[int, np.ndarray]]) -> int:
        """Maximum PE width (last axis) over all graphs in all datasets; for model ``input_dim``."""
        mx = 0
        for ds_pe in pes_by_dataset.values():
            for arr in ds_pe.values():
                mx = max(mx, int(arr.shape[1]))
        return mx

    def pe_for_global_index(
        self,
        pes_by_dataset: Dict[str, Dict[int, np.ndarray]],
        global_idx: int,
    ) -> np.ndarray:
        """PE array for ``global_idx`` using ``map_global_idx_to_local_indices`` and ``_ds_name_seq``."""
        ds_idx, g_idx = self.map_global_idx_to_local_indices(global_idx)
        ds_name = self._ds_name_seq[ds_idx]
        return pes_by_dataset[ds_name][g_idx]

    def map_global_idx_to_local_indices(self, global_idx: int) -> Tuple[int, int]:
        # fetch the (ds_idx, graph_idx) tuple for a given global index
        if not hasattr(self, '_mapping_table'):
            num_graphs_seq = self._get_num_graphs_seq()
            self._mapping_table = self._create_mapping_table(num_graphs_seq=num_graphs_seq)
        return self._mapping_table[global_idx]
    
    def get_global_index(self, ds_idx: int, graph_idx: int) -> int:
        # fetch the global index for a given (ds_idx, graph_idx) tuple
        if not hasattr(self, '_mapping_table'):
            num_graphs_seq = self._get_num_graphs_seq()
            self._mapping_table = self._create_mapping_table(num_graphs_seq=num_graphs_seq)
        for global_idx, (d_idx, g_idx) in self._mapping_table.items():
            if d_idx == ds_idx and g_idx == graph_idx:
                return global_idx
        raise ValueError(f'No global index found for dataset index {ds_idx} and graph index {graph_idx}.')
    
    def _dense_adj_from_data(self, data: Data) -> np.ndarray:
        """Unweighted dense adjacency from ``edge_index``; symmetrized when ``treat_as_undirected`` is True."""
        A = gutils.to_dense_adj(edge_index=data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).cpu().numpy()
        if self._treat_as_undirected:
            A = np.maximum(A, A.T)
        return A

    def _build_graph_cost(self, data: Data, matrix_mode: str = 'adj') -> np.ndarray:
        """
        Builds the graph cost matrix from the given data.
        :param data: The input data containing edge index and node features.
        :param matrix_mode: ``'adj'`` or ``'sp'`` — adjacency (with self-loops) vs normalized shortest-path.
        :return: The graph cost matrix.
        """
        adj = self._dense_adj_from_data(data).astype(np.float32, copy=False)
        adj += np.eye(adj.shape[0], dtype=np.float32)  # add self-loops
        if matrix_mode == 'adj':
            C = adj
        elif matrix_mode == 'sp':
            adj_sp = sp.sparse.csr_matrix(adj)
            directed = not self._treat_as_undirected
            C_raw = sp.sparse.csgraph.shortest_path(adj_sp, directed=directed, unweighted=True)
            C = _normalize_shortest_path_cost(C_raw)
        else:
            raise ValueError(f"Unknown matrix_mode: {matrix_mode}. Use 'adj' or 'sp'.")
        return C

    def _binary_adjacency_for_pe(self, data: Data) -> np.ndarray:
        """
        Binary {0,1} adjacency without self-loops. Symmetrized when ``treat_as_undirected``; otherwise
        keeps edge orientation for RWSE / directed LapPE features.
        """
        A = self._dense_adj_from_data(data)
        np.fill_diagonal(A, 0)
        A = (A > 0).astype(np.float64)
        if self._treat_as_undirected:
            A = np.maximum(A, A.T)
        return A

    def _extract_graph_pe(self, data: Data, matrix_mode: str = 'adj', pe_mode: str = 'svd', n_components: int = 100) -> np.ndarray:
        """
        Extracts the graph positional encoding from the edge index.
        :param data: The input data containing edge index with known number of nodes.
        :param matrix_mode: For ``pe_mode=='svd'``: ``'adj'`` or ``'sp'`` cost matrix (see ``_build_graph_cost``).
        :param pe_mode: Positional encoding type: ``'svd'``, ``'LapPE'``, or ``'RWSE'``.
        :param n_components: For ``svd``: randomized SVD rank when num_nodes >= 100; for ``matrix_mode=='sp'``
            the embedding concatenates ``U sqrt(Σ)`` and ``V sqrt(Σ)`` (twice the column count before padding).
            For ``LapPE``: undirected: smallest eigenpairs of the symmetric normalized Laplacian; directed: leading left
            singular vectors of the row-normalized adjacency (random-walk matrix), scaled by ``sqrt(s)``.
            For ``RWSE``: number of random-walk steps (features are ``diag(P^k)`` for ``k = 1..n_components``)
            with out-degree normalization (directed when ``treat_as_undirected`` is False).
        :return: The graph positional encoding (feature width depends on mode and graph size; callers may pad/truncate).
        """
        if pe_mode == 'svd':
            C = self._build_graph_cost(data=data, matrix_mode=matrix_mode)
            cost_symmetric = self._treat_as_undirected
            if C.shape[0] < 100:
                if cost_symmetric:
                    u, s, vh = np.linalg.svd(C, hermitian=True)
                else:
                    u, s, vh = np.linalg.svd(C, full_matrices=False)
            else:
                assert n_components <= C.shape[0], f"n_components={n_components} must be less than or equal to the number of nodes: {C.shape[0]}"
                u, s, vh = randomized_svd(C, n_components=n_components, n_iter=5, random_state=None)

            sqrt_s = np.sqrt(np.maximum(s, 0))
            if matrix_mode == 'sp':
                pe = np.concatenate((u * sqrt_s, vh.conj().T * sqrt_s), axis=1)
            else:
                pe = vh.conj().T * sqrt_s
        elif pe_mode == 'LapPE':
            # print(f'Computing LapPE for graph with {data.num_nodes} nodes...')
            A = self._binary_adjacency_for_pe(data)
            n = A.shape[0]
            k = min(n_components, n)
            if self._treat_as_undirected:
                deg = A.sum(axis=1)
                d_inv_sqrt = np.zeros(n, dtype=np.float64)
                np.divide(1.0, np.sqrt(deg), out=d_inv_sqrt, where=deg > 0)
                Dmh = np.diag(d_inv_sqrt)
                L_sym = np.eye(n, dtype=np.float64) - Dmh @ A @ Dmh
                _, V = np.linalg.eigh(L_sym)
                pe = V[:, :k].astype(np.float32)
            else:
                deg_out = A.sum(axis=1)
                P = np.zeros((n, n), dtype=np.float64)
                np.divide(A, deg_out[:, np.newaxis], out=P, where=deg_out[:, np.newaxis] > 0)
                if k == 0:
                    pe = np.zeros((n, 0), dtype=np.float32)
                else:
                    U, s_svd, _ = np.linalg.svd(P, full_matrices=False)
                    take = min(k, U.shape[1])
                    pe = (U[:, :take] * np.sqrt(np.maximum(s_svd[:take], 0))).astype(np.float32)
        elif pe_mode == 'RWSE':
            # print(f'Computing RWSE for graph with {data.num_nodes} nodes...')
            A = self._binary_adjacency_for_pe(data)
            n = A.shape[0]
            deg = A.sum(axis=1)
            P = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                if deg[i] > 0:
                    P[i] = A[i] / deg[i]
            pe_cols = []
            Pk = np.eye(n, dtype=np.float64)
            for _ in range(n_components):
                Pk = Pk @ P
                pe_cols.append(np.diag(Pk))
            pe = (
                np.column_stack(pe_cols).astype(np.float32)
                if pe_cols
                else np.zeros((n, 0), dtype=np.float32)
            )
        else:
            raise ValueError(f"Unknown pe_mode: {pe_mode}. Use 'svd', 'LapPE', or 'RWSE'.")
        pe = flip_pe_sign(torch.tensor(pe, dtype=torch.float32), method='sum').cpu().numpy()  # flip the sign of the positional encoding
        return pe

    def _check_the_existence_of_precomputed_pe(self, ds_name_seq: List[str]):
        # check if the graph positional encodings have been precomputed (dict pickle only)
        for ds_name in ds_name_seq:
            fname_pkl = self._pe_dict_cache_path(ds_name)
            fname_npy = self._pe_legacy_npy_path(ds_name)
            if os.path.exists(fname_pkl):
                continue
            print(f'Graph positional encodings for dataset {ds_name} have not been precomputed. The system will precompute them...')
            ds = self._load_graph_ds(name=ds_name)
            pe_dict: Dict[int, np.ndarray] = {}
            for g_idx, g in enumerate(ds):
                pe_dict[g_idx] = np.asarray(
                    self._extract_graph_pe(
                        data=g,
                        matrix_mode=self._pe_cost_mode,
                        pe_mode=self._pe_mode,
                    ),
                    dtype=np.float32,
                )
            with open(fname_pkl, 'wb') as f:
                pickle.dump(pe_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            if os.path.exists(fname_npy):
                os.remove(fname_npy)
                print(f'Removed legacy PE cache {fname_npy}.')
            max_pe_cols = max(arr.shape[1] for arr in pe_dict.values()) if pe_dict else 0
            print(
                f'Precomputed graph positional encodings for dataset {ds_name} and saved to {fname_pkl} '
                f'({len(pe_dict)} graphs; max PE columns {max_pe_cols}).'
            )

    def compose_pe_from_multiple_ds(self) -> Dict[str, Dict[int, np.ndarray]]:
        # step 1: check if the graph positional encodings have been precomputed
        self._check_the_existence_of_precomputed_pe(ds_name_seq=self._ds_name_seq)

        # step 2: load per-dataset pickle dicts: dataset_name -> { local_graph_idx -> ndarray }
        out: Dict[str, Dict[int, np.ndarray]] = {}
        for ds_name in self._ds_name_seq:
            fname_pkl = self._pe_dict_cache_path(ds_name)
            if not os.path.exists(fname_pkl):
                raise FileNotFoundError(
                    f'Missing PE dict cache after precheck: {fname_pkl}. '
                    f'Delete stale caches and retry, or ensure _check_the_existence_of_precomputed_pe ran.'
                )
            with open(fname_pkl, 'rb') as f:
                local = pickle.load(f)
            dataset_pe: Dict[int, np.ndarray] = {}
            for k, arr in sorted(local.items(), key=lambda kv: int(kv[0])):
                dataset_pe[int(k)] = np.asarray(arr, dtype=np.float32)
            out[ds_name] = dataset_pe # why not out[ds_name] = local?
            n_here = len(dataset_pe)
            print(
                f'Loaded graph positional encodings for dataset {ds_name} from {fname_pkl}: '
                f'{n_here} graphs (local indices 0..{n_here - 1}).'
            )
            legacy_npy = self._pe_legacy_npy_path(ds_name)
            if os.path.exists(legacy_npy):
                os.remove(legacy_npy)
                print(f'Removed legacy PE cache {legacy_npy}.')

        return out

    def _meta_dict_cache_path(self, ds_name: str) -> str:
        """Path to the pickle dict cache for one dataset (full superset).

        Meta-features only depend on the binary adjacency; cache is shared across pe_/gw_ modes.
        Directedness affects symmetrization, so split caches by treat_as_undirected. The
        size-transform (raw/log/sqrt) changes n_nodes/n_edges, so it is part of the key too.
        """
        tag = 'undirected' if self._treat_as_undirected else 'directed'
        sz = os.environ.get('NEURALGW_SIZE_TRANSFORM', 'sqrt').lower()
        return f'{self._data_root}/precomputed/meta/{ds_name}_meta_dict_{tag}_{sz}.pkl'

    def _check_the_existence_of_precomputed_meta(self, ds_name_seq: List[str]) -> None:
        for ds_name in ds_name_seq:
            fname_pkl = self._meta_dict_cache_path(ds_name)
            if os.path.exists(fname_pkl):
                continue
            print(f'Graph meta-features for dataset {ds_name} have not been precomputed. The system will precompute them...')
            ds = self._load_graph_ds(name=ds_name)
            meta_dict: Dict[int, Dict[str, float]] = {}
            for g_idx, g in enumerate(tqdm(ds, desc=f'meta:{ds_name}')):
                meta_dict[g_idx] = _extract_graph_meta(data=g, treat_as_undirected=self._treat_as_undirected)
            with open(fname_pkl, 'wb') as f:
                pickle.dump(meta_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Precomputed graph meta-features for dataset {ds_name} and saved to {fname_pkl} ({len(meta_dict)} graphs, {len(META_FULL_KEYS)} features each).')

    def compose_meta_from_multiple_ds(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Return ``{dataset_name -> {local_graph_idx -> {feature_name -> float}}}`` (full superset)."""
        self._check_the_existence_of_precomputed_meta(ds_name_seq=self._ds_name_seq)
        out: Dict[str, Dict[int, Dict[str, float]]] = {}
        for ds_name in self._ds_name_seq:
            fname_pkl = self._meta_dict_cache_path(ds_name)
            if not os.path.exists(fname_pkl):
                raise FileNotFoundError(
                    f'Missing meta dict cache after precheck: {fname_pkl}. '
                    f'Delete stale caches and retry.'
                )
            with open(fname_pkl, 'rb') as f:
                local = pickle.load(f)
            dataset_meta: Dict[int, Dict[str, float]] = {}
            for k, d in sorted(local.items(), key=lambda kv: int(kv[0])):
                dataset_meta[int(k)] = {kk: float(vv) for kk, vv in d.items()}
            out[ds_name] = dataset_meta
            print(f'Loaded meta-features for dataset {ds_name} from {fname_pkl}: {len(dataset_meta)} graphs.')
        return out

    def meta_for_global_index(
        self,
        metas_by_dataset: Dict[str, Dict[int, Dict[str, float]]],
        global_idx: int,
    ) -> Dict[str, float]:
        ds_idx, g_idx = self.map_global_idx_to_local_indices(global_idx)
        ds_name = self._ds_name_seq[ds_idx]
        return metas_by_dataset[ds_name][g_idx]

    def _generate_the_path_of_precomputed_distance_matrices(self, ds_name_1, ds_name_2) -> str:
        return f'{self._data_root}/precomputed/gwd_{self._gwd_tp_cache_tag()}/{ds_name_1}_{ds_name_2}_distance_matrix.npy'
    
    def _check_if_precomputed_distance_matrices_exist(self, ds_name_seq: List[str]) -> bool:
        # check if the precomputed distance matrices exist for all datasets
        for i in range(len(ds_name_seq)):
            ds_name_i = ds_name_seq[i]
            for j in range(i, len(ds_name_seq)):
                ds_name_j = ds_name_seq[j]
                fname1 = self._generate_the_path_of_precomputed_distance_matrices(ds_name_i, ds_name_j)
                fname2 = self._generate_the_path_of_precomputed_distance_matrices(ds_name_j, ds_name_i)
                if os.path.exists(fname1):
                    continue
                elif os.path.exists(fname2):
                    continue
                else: # need to precompute the distance matrix
                    print(f'Precomputed distance matrix for dataset ({ds_name_i},{ds_name_j}) does not exist. The system will precompute them ...')
                    ds_i = self._load_graph_ds(name=ds_name_i)
                    ds_j = self._load_graph_ds(name=ds_name_j)
                    D_ij = np.zeros((len(ds_i), len(ds_j)), dtype=np.float32)
                    for gi in tqdm(range(len(ds_i))):
                        C1 = self._build_graph_cost(data=ds_i[gi], matrix_mode=self._gw_cost_mode)
                        gj_start = gi + 1 if i == j else 0
                        for gj in range(gj_start, len(ds_j)):
                            C2 = self._build_graph_cost(data=ds_j[gj], matrix_mode=self._gw_cost_mode)
                            D_ij[gi, gj] = ot.gromov.gromov_wasserstein2(C1=C1, C2=C2, tol_rel=1e-4, tol_abs=1e-4)
                            if i == j:
                                D_ij[gj, gi] = D_ij[gi, gj]
                    # save the precomputed distance matrix
                    os.makedirs(os.path.dirname(fname1), exist_ok=True)
                    with open(fname1, 'wb') as f:
                        np.save(f, D_ij)
                        print(f'Precomputed distance matrix for dataset {ds_name_i}_{ds_name_j} and saved to {fname1}, shape: {D_ij.shape}.')

    def _load_distance_matrix_with_ordered_pair(self, ds_name_1: str, ds_name_2: str) -> np.ndarray:
        fname1 = self._generate_the_path_of_precomputed_distance_matrices(ds_name_1, ds_name_2)
        fname2 = self._generate_the_path_of_precomputed_distance_matrices(ds_name_2, ds_name_1)
        if os.path.exists(fname1):
            with open(fname1, 'rb') as f:
                D_ij = np.load(f, allow_pickle=True)
                print(f'Loaded distance matrix for dataset pair {ds_name_1}_{ds_name_2} from {fname1}, shape: {D_ij.shape}.')
                return D_ij
        elif os.path.exists(fname2):
            with open(fname2, 'rb') as f:
                D_ij = np.load(f, allow_pickle=True)
                print(f'Loaded distance matrix for dataset pair {ds_name_2}_{ds_name_1} from {fname2}, shape: {D_ij.shape}.')
                return D_ij.T
        else:
            raise FileNotFoundError(f'No precomputed distance matrix found for dataset pair ({ds_name_1}, {ds_name_2}).')

    def compose_pairwise_distances_from_multiple_ds(self) -> np.ndarray:
        # step 1: check if the distance matrices have been precomputed
        self._check_if_precomputed_distance_matrices_exist(ds_name_seq=self._ds_name_seq)

        # step 2: compose the distance matrices into a big distance matrix
        composed_D_list = []
        for i in range(len(self._ds_name_seq)):
            ds_name_i = self._ds_name_seq[i]
            composed_D_row = []
            for j in range(len(self._ds_name_seq)):
                ds_name_j = self._ds_name_seq[j]
                composed_D_row.append(self._load_distance_matrix_with_ordered_pair(ds_name_i, ds_name_j))
            composed_D_list.append(composed_D_row)
        composed_D = np.block(composed_D_list)
        print(f'Composed distance matrix shape: {composed_D.shape}.')
        return composed_D
    
    def _generate_the_path_of_precomputed_transport_plans(self, ds_name_1, ds_name_2) -> str:
        return f'{self._data_root}/precomputed/tp_{self._gwd_tp_cache_tag()}/{ds_name_1}_{ds_name_2}_transport_plans.npy'

    def _check_if_precomputed_transport_plans_exist(self, ds_name_seq: List[str]) -> None:
        """
        Check if the precomputed transport plans exist for all pairs of datasets.
        If not, precompute them and save to disk.
        args:
            ds_name_seq: List of dataset names.
        returns:
            bool: True if all precomputed transport plans exist, False otherwise.
        Note:
            The transport plans are stored in a dictionary with keys as (gi, gj) tuples,
            where gi and gj are the graph indices in their respective datasets.
            The values are the transport plan matrices.
        """
        #  check if the precomputed transport plans exist for all pairs of datasets
        for i in range(len(ds_name_seq)):
            ds_name_i = ds_name_seq[i]
            for j in range(i, len(ds_name_seq)):
                ds_name_j = ds_name_seq[j]
                fname1 = self._generate_the_path_of_precomputed_transport_plans(ds_name_i, ds_name_j)
                fname2 = self._generate_the_path_of_precomputed_transport_plans(ds_name_j, ds_name_i)
                if os.path.exists(fname1):
                    continue
                elif os.path.exists(fname2):
                    continue
                else: # need to precompute the transport plans
                    print(f'Precomputed transport plans for dataset ({ds_name_i},{ds_name_j}) do not exist. The system will precompute them ...')
                    ds_i = self._load_graph_ds(name=ds_name_i)
                    ds_j = self._load_graph_ds(name=ds_name_j)
                    tp_ij = {}
                    for gi in tqdm(range(len(ds_i))):
                        C1 = self._build_graph_cost(data=ds_i[gi], matrix_mode=self._gw_cost_mode)
                        gj_start = gi if i == j else 0
                        for gj in range(gj_start, len(ds_j)):
                            C2 = self._build_graph_cost(data=ds_j[gj], matrix_mode=self._gw_cost_mode)
                            T_ij = ot.gromov.gromov_wasserstein(C1=C1, C2=C2, tol_rel=1e-4, tol_abs=1e-4)
                            tp_ij[(gi, gj)] = T_ij
                            if i == j and gi != gj:
                                tp_ij[(gj, gi)] = T_ij.T
                    # save the precomputed transport plans
                    os.makedirs(os.path.dirname(fname1), exist_ok=True)
                    with open(fname1, 'wb') as f:
                        np.save(f, tp_ij)
                        print(f'Precomputed transport plans for dataset {ds_name_i}_{ds_name_j} and saved to {fname1}, number of transport plans: {len(tp_ij)}.')
    
    def _load_transport_plans_with_ordered_pair(self, ds_name_1: str, ds_name_2: str) -> Dict[Tuple[int, int], np.ndarray]:
        fname1 = self._generate_the_path_of_precomputed_transport_plans(ds_name_1, ds_name_2)
        fname2 = self._generate_the_path_of_precomputed_transport_plans(ds_name_2, ds_name_1)
        if os.path.exists(fname1):
            with open(fname1, 'rb') as f:
                tp_ij = np.load(f, allow_pickle=True).item()
                print(f'Loaded transport plans for dataset pair {ds_name_1}_{ds_name_2} from {fname1}, number of transport plans: {len(tp_ij)}.')
                return tp_ij
        elif os.path.exists(fname2):
            with open(fname2, 'rb') as f:
                tp_ji = np.load(f, allow_pickle=True).item()
                print(f'Loaded transport plans for dataset pair {ds_name_2}_{ds_name_1} from {fname2}, number of transport plans: {len(tp_ji)}.')
                # need to transpose the transport plans
                tp_ij = {(gi, gj): T_ji.T for (gj, gi), T_ji in tp_ji.items()}
                return tp_ij
        else:
            raise FileNotFoundError(f'No precomputed transport plans found for dataset pair ({ds_name_1}, {ds_name_2}).')
    
    def compose_transport_plans_from_multiple_ds(self) -> List[List[np.ndarray]]:
        # step 1: check if the transport plans have been precomputed
        self._check_if_precomputed_transport_plans_exist(ds_name_seq=self._ds_name_seq)

        # step 2: compose the transport plans into a big list of lists
        composed_tp = {}
        for i in range(len(self._ds_name_seq)):
            ds_name_i = self._ds_name_seq[i]
            for j in range(i, len(self._ds_name_seq)):
                ds_name_j = self._ds_name_seq[j]
                tp_ij = self._load_transport_plans_with_ordered_pair(ds_name_i, ds_name_j)
                composed_tp[(i,j)] = tp_ij
                if i != j:
                    tp_ji = {(gj, gi): T_ij.T for (gi, gj), T_ij in tp_ij.items()}
                    composed_tp[(j,i)] = tp_ji

        return composed_tp
    
    def compose_labels_from_multiple_ds(self) -> np.ndarray:
        # step 1: load the datasets and get the number of graphs in each dataset
        ds_seq = [self._load_graph_ds(name=ds_name) for ds_name in self._ds_name_seq]

        # step 2: create the labels for each graph in each dataset
        labels_seq = [g.y.item() for ds in ds_seq for g in ds]

        labels_seq = np.asarray(labels_seq, dtype=np.int64)

        return labels_seq
