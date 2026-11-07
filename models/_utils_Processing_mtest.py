import os
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
import ot
from tqdm import tqdm

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

class DataPreparationSystem:
    """
    Data preparation system for graph datasets (mtest variant)
    """
    def __init__(self, ds_name_seq: List[str], pe_cost_mode: str = 'adj', gw_cost_mode: str = 'adj'):
        self._ds_name_seq = ds_name_seq
        self._pe_cost_mode = pe_cost_mode
        self._gw_cost_mode = gw_cost_mode

    def _load_graph_ds(self, name: str ='MUTAG'):
        # load the specified graph dataset from TUDataset
        if name == 'DD':
            ds = TUDataset(root='data/TUDataset', name=name)
            sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
            # sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes <= 80]
            ds = ds[sub_idx]
        elif name == 'ENZYMES':
            ds = TUDataset(root='data/TUDataset', name=name)
            if False:
                sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes <= 80]
                ds = ds[sub_idx]

                sub_idx = subsample_TUDataset(ds=ds, ratio=0.3)
            else:
                sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes >= 70]
            ds = ds[sub_idx]
        elif name == 'COLLAB':
            # ds = TUDataset(root='data/TUDataset', name=ds_name)
            # sub_idx = subsample_TUDataset(ds=ds, ratio=0.2)
            # ds = ds[sub_idx]
            ds = TUDataset(root='data/TUDataset', name=name)
            sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
            # sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes <= 80]
            ds = ds[sub_idx]

            # sub_idx = subsample_TUDataset(ds=ds, ratio=0.05)
            # ds = ds[sub_idx]
        elif name == 'IMDB-MULTI':
            ds = TUDataset(root='data/TUDataset', name=name)
            sub_idx = subsample_TUDataset(ds=ds, ratio=0.2)
            ds = ds[sub_idx]
        elif name == 'REDDIT-BINARY':
            ds = TUDataset(root='data/TUDataset', name=name)
            sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
            ds = ds[sub_idx]
        elif name == 'SBM':
            ds = generate_sbm_dataset(num_graphs=200, n_min=100, n_max=100, k_min=5, k_max=5, label_type="num_communities")
            self._sbm_seeds = list(dict.fromkeys([g.seed for g in ds]))
        else:
            ds = TUDataset(root='data/TUDataset', name=name)
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
        elif len([ds_name for ds_name in self._ds_name_seq if testset == ds_name or ds_name in testset]) == 1:
            # In this case, we use the specified dataset as the test set while the rest as the training set
            mapping_table = self._create_mapping_table(num_graphs_seq=self._get_num_graphs_seq())
            test_indices = [global_idx for global_idx, (ds_idx, _) in mapping_table.items() if self._ds_name_seq[ds_idx] == testset]
            train_indices = [global_idx for global_idx, (ds_idx, _) in mapping_table.items() if self._ds_name_seq[ds_idx] != testset]
            return train_indices, test_indices
        elif len([ds_name for ds_name in self._ds_name_seq if testset == ds_name or ds_name in testset]) > 1:
            print(f'[Info] testset corresponds to multiple datasets: {testset}')
            assert isinstance(testset, list) or isinstance(testset, tuple), f'When testset corresponds to multiple datasets, it should be a list or tuple of dataset names.'
            # In this case, we use all graphs from the specified dataset(s) as the test set while the rest as the training set
            mapping_table = self._create_mapping_table(num_graphs_seq=self._get_num_graphs_seq())
            test_indices = [global_idx for global_idx, (ds_idx, _) in mapping_table.items() if self._ds_name_seq[ds_idx] in testset]
            train_indices = [global_idx for global_idx, (ds_idx, _) in mapping_table.items() if self._ds_name_seq[ds_idx] not in testset]
            return train_indices, test_indices
        elif len([ds_name for ds_name in self._ds_name_seq if testset == ds_name or ds_name in testset]) < -1:
            # Warning: this case is created to handle the situation where a dataset arises over one time in ds_name_seq
            #############################################################################
            # I gave a negative threshold (-1) to avoid conflict with the previous case #
            #############################################################################
            testset_ptr = [i for i, ds_name in enumerate(self._ds_name_seq) if ds_name == testset or ds_name in testset]
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
    
    def _build_graph_cost(self, data: Data, matrix_mode: str = 'adj') -> np.ndarray:
        """
        Builds the graph cost matrix from the given data.
        :param data: The input data containing edge index and node features.
        :param matrix_mode: ``'adj'`` or ``'sp'``.
        :return: The graph cost matrix.
        """
        adj = gutils.to_dense_adj(edge_index=data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).cpu().numpy()
        adj += np.eye(adj.shape[0], dtype=np.float32)  # add self-loops
        if matrix_mode == 'adj':
            C = adj
        elif matrix_mode == 'sp':
            adj = sp.sparse.csr_matrix(adj)
            C = sp.sparse.csgraph.shortest_path(adj, directed=False, unweighted=True)
        else:
            raise ValueError(f"Unknown matrix_mode: {matrix_mode}. Use 'adj' or 'sp'.")
        return C

    def _extract_graph_pe(self, data: Data, matrix_mode: str = 'adj', n_components: int = 100) -> np.ndarray:
        """
        Extracts the graph positional encoding from the edge index.
        :param data: The input data containing edge index with known number of nodes.
        :param matrix_mode: ``'adj'`` or ``'sp'`` for the SVD input matrix.
        :param n_components: The number of components for Randomized SVD (only applicable for num_nodes > 100).
        :return: The graph positional encoding (callers may pad columns to a shared width).
        """
        C = self._build_graph_cost(data=data, matrix_mode=matrix_mode)
        if C.shape[0] < 100:
            u, s, vh = np.linalg.svd(C, hermitian=True)
        else:
            assert n_components <= C.shape[0], f"n_components={n_components} must be less than or equal to the number of nodes: {C.shape[0]}"
            u, s, vh = randomized_svd(C, n_components=n_components, n_iter=5, random_state=None)

        pe = vh.conj().T@np.diag(s**0.5)
        pe = flip_pe_sign(torch.tensor(pe, dtype=torch.float32), method='l2').cpu().numpy()  # flip the sign of the positional encoding
        return pe

    def _check_the_existence_of_precomputed_pe(self, ds_name_seq: List[str]):
        # check if the graph positional encodings have been precomputed
        for ds_name in ds_name_seq:
            fname = f'data/precomputed/pe/{ds_name}_pe.npy'
            if not os.path.exists(fname):
                print(f'Graph positional encodings for dataset {ds_name} have not been precomputed. The system will precompute them...')
                ds = self._load_graph_ds(name=ds_name)
                num_nodes_seq = [g.num_nodes for g in ds]
                max_dim = max(num_nodes_seq)
                def _pe_row_padded(g):
                    pe_g = self._extract_graph_pe(data=g, matrix_mode=self._pe_cost_mode)
                    w = pe_g.shape[1]
                    if w < max_dim:
                        pe_g = np.pad(pe_g, ((0, 0), (0, max_dim - w)), mode='constant', constant_values=0)
                    elif w > max_dim:
                        pe_g = pe_g[:, :max_dim]
                    return np.pad(
                        pe_g,
                        pad_width=((0, max_dim - g.num_nodes), (0, 0)),
                        mode='constant',
                        constant_values=0,
                    )

                pe = np.asarray([_pe_row_padded(g) for g in ds])
                with open(fname, 'wb') as f:
                    np.save(f, pe)
                    print(f'Precomputed graph positional encodings for dataset {ds_name} and saved to {fname}, shape: {pe.shape}.')

    def compose_pe_from_multiple_ds(self) -> np.ndarray:
        # step 1: check if the graph positional encodings have been precomputed
        self._check_the_existence_of_precomputed_pe(ds_name_seq=self._ds_name_seq)
        
        # step 2: load the precomputed graph positional encodings
        # step 2.1: determine the max sequence length and the max embedding dimension
        pe_seq = []
        for ds_name in self._ds_name_seq:
            fname = f'data/precomputed/pe/{ds_name}_pe.npy'
            with open(fname, 'rb') as f:
                pc = np.load(f, allow_pickle=True)
                pe_seq.append(pc)
                print(f'Loaded graph positional encodings for dataset {ds_name} from {fname}, shape: {pc.shape}.')

        num_nodes_seq = self.get_num_nodes_seq()
        max_seq_len, max_embed_dim = max(num_nodes_seq), max(num_nodes_seq)
        # step 2.2: pad the point clouds to the same size
        pe_seq_pad = []
        pe_seq_pad = np.concatenate([
            np.pad(pc, pad_width=((0,0), (0, max_seq_len - pc.shape[1]), (0, max_embed_dim - pc.shape[2])) , mode='constant', constant_values=0)
            for pc in pe_seq
        ], axis=0)

        return pe_seq_pad
    
    def _generate_the_path_of_precomputed_distance_matrices(self, ds_name_1, ds_name_2) -> str:
        return f'data/precomputed/gwd/{ds_name_1}_{ds_name_2}_distance_matrix.npy'
    
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
        return f'data/precomputed/tp/{ds_name_1}_{ds_name_2}_transport_plans.npy'

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
