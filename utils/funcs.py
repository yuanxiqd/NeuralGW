import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Dataset
import torch_geometric.utils as gutils
import scipy as sp
from sklearn.utils.extmath import randomized_svd
from typing import List, Tuple

def custom_train_test_split(idx_seq: List[int], test_ratio: float = 0.2, random_seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(random_seed)
    rng.shuffle(idx_seq)
    split_idx = int(len(idx_seq) * (1 - test_ratio))
    # test_idx = idx_seq[:split_idx]
    # train_idx = idx_seq[split_idx:]
    train_idx = idx_seq[:split_idx]
    test_idx = idx_seq[split_idx:]
    return train_idx, test_idx

def filp_pe_sign(x, method='l0'):
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

def build_graph_cost(data: Data, mode: str = 'adj') -> torch.Tensor:
    """
    Builds the graph cost matrix from the given data.
    :param data: The input data containing edge index and node features.
    :param mode: The mode to use for adjacency matrix ('adj' or 'sp')
      - 'adj' means that we use the adjacency matrix from edge_index to build graph cost matrix 
      - 'sp' means that we use the shortest_path matrix to build graph cost matrix.
    :return: The graph cost matrix.
    """
    adj = gutils.to_dense_adj(edge_index=data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).cpu().numpy()
    adj += np.eye(adj.shape[0], dtype=np.float32)  # add self-loops
    if mode == 'adj':
        C = adj
    elif mode == 'sp':
        adj = sp.sparse.csr_matrix(adj)
        C = sp.sparse.csgraph.shortest_path(adj, directed=False, unweighted=True)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'adj' or 'sp'.")
    return C

def extract_graph_pe(data: Data, N: int, mode: str = 'adj', n_components: int = 100) -> torch.Tensor:
    """
    Extracts the graph positional encoding from the edge index.
    :param data: The input data containing edge index with known number of nodes.
    :param N: The predefined number of dimensions for the positional encoding.
    :param mode: The mode to use for positional encoding ('adj' or 'sp')
      - 'adj' means that we use the adjacency matrix from edge_index to build graph positional encoding
      - 'sp' means that we use the shortest_path matrix to build graph positional encoding.
    :param n_components: The number of components for Randomized SVD (only applicable for num_nodes > 100).
    :return: The graph positional encoding.
    """
    C = build_graph_cost(data=data, mode=mode)
    if C.shape[0] < 100:
        u, s, vh = np.linalg.svd(C, hermitian=True)
    else:
        assert n_components <= C.shape[0], f"n_components={n_components} must be less than or equal to the number of nodes: {C.shape[0]}"
        u, s, vh = randomized_svd(C, n_components=n_components, n_iter=5, random_state=None)

    pe = vh.conj().T@np.diag(s**0.5)
    pe = filp_pe_sign(torch.tensor(pe, dtype=torch.float32), method='sum').cpu().numpy()  # flip the sign of the positional encoding
    # padding the positional encoding to match the predefined dimension N
    if pe.shape[1] < N:
        pe = np.pad(pe, ((0, 0), (0, N - pe.shape[1])), mode='constant', constant_values=0)
    elif pe.shape[1] > N:
        pe = pe[:, :N]

    return pe

def get_ds(name: str, path: str='../data') -> Dataset:
    if name == 'COLLAB':
        ds = TUDataset(root=path+'/'+name, name=name)
        subset_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
        print(f'{name} -> Subset indices: {subset_idx}')
        ds = ds[subset_idx]
    elif name == 'REDDIT-BINARY':
        ds = TUDataset(root=path+'/'+name, name=name)
        subset_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
        print(f'{name} -> Subset indices: {subset_idx}')
        ds = ds[subset_idx]
    else:
        ds = TUDataset(root=path+'/'+name, name=name)
    return ds

def compute_graph_ptr(ds_name: str) -> None:
    """
    Generates a list of graph pointers indicating the start and end indices of each graph in a batched representation.
    :param ds: The input dataset containing multiple graphs.
    :return: A list of graph pointers.
    """
    ds = get_ds(name=ds_name, path='../data')
    ptr = [0]
    for data in ds:
        ptr.append(ptr[-1] + data.num_nodes)
    # save the graph pointers to a file
    ds_name = ds.name
    save_path = f'precomputed/graph_ptr/{ds_name}_graph_ptr.npy'
    with open(save_path, 'wb') as f:
        np.save(f, np.array(ptr))
    return None

def load_graph_ptr(ds_name: str) -> List:
    """
    Loads the precomputed graph pointers for the specified dataset.
    :param ds_name: The name of the dataset.
    :return: A list of graph pointers.
    """
    file_path = f'precomputed/graph_ptr/{ds_name}_graph_ptr.npy'
    with open(file_path, 'rb') as f:
        ptr = np.load(f).tolist()
    return ptr

def compute_graph_pe(ds_name: str, N: int, mode: str = 'adj', n_components: int = 100) -> None:
    """
    Computes the graph positional encoding for all graphs in the dataset.
    :param ds_name: The name of the dataset.
    :param N: The predefined number of dimensions for the positional encoding.
    :param mode: The mode to use for positional encoding ('adj' or 'sp').
    :param n_components: The number of components for Randomized SVD (only applicable for num_nodes > 100).
    :return: A 2D numpy array representing the graph positional encodings.
    """
    dataset = get_ds(name=ds_name, path='../data')
    pe_list = []
    for data in dataset:
        pe = extract_graph_pe(data=data, N=N, mode=mode, n_components=n_components)
        pe_list.append(pe)
    
    # save the positional encodings to a file
    pe_all = np.vstack(pe_list)
    save_path = f'precomputed/graph_pe_based_{mode}/{ds_name}_graph_pe_based_{mode}.npy'
    with open(save_path, 'wb') as f:
        np.save(f, pe_all)
    return None

def load_graph_pe(ds_name: str, mode: str = 'adj') -> np.ndarray:
    """
    Loads the precomputed graph positional encoding for the specified dataset and mode.
    :param ds_name: The name of the dataset.
    :param mode: The mode to use for positional encoding ('adj' or 'sp').
    :return: A 2D numpy array representing the graph positional encodings.
    """
    file_path = f'precomputed/graph_pe_based_{mode}/{ds_name}_graph_pe_based_{mode}.npy'
    with open(file_path, 'rb') as f:
        pe = np.load(f)
    return pe

def compute_pairwise_distances(ds_name: str, metric: str = 'gw', mode: str = 'adj') -> None:
    """
    Computes the pairwise distances between all graphs in the dataset.
    :param dataset: The input dataset containing graph data.
    :return: A 2D numpy array representing the pairwise distances.
    """
    import ot
    from tqdm import tqdm
    dataset = get_ds(name=ds_name, path='../data')
    n = len(dataset)
    D = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n)):
        C1 = build_graph_cost(data=dataset[i], mode=mode)
        for j in range(i + 1, n):
            C2 = build_graph_cost(data=dataset[j], mode=mode)
            if metric == 'gw':
                # Here you can implement the actual Gromov-Wasserstein distance calculation
                dist = ot.gromov.gromov_wasserstein2(C1=C1, C2=C2, loss_fun='square_loss')
            elif metric == 'egw_ppa':
                # Here you can implement the actual Entropic Gromov-Wasserstein distance calculation
                dist = ot.gromov.entropic_gromov_wasserstein2(C1=C1, C2=C2, loss_fun='square_loss', solver='PPA', tol=1e-4)
            elif metric == 'sampled_gw':
                p = ot.unif(C1.shape[0], type_as=C1.dtype)
                q = ot.unif(C2.shape[0], type_as=C2.dtype)
                _, log = ot.gromov.sampled_gromov_wasserstein(C1=C1, C2=C2, p=p, q=q, loss_fun='square_loss')
                dist = log['gw_dist']
            D[i, j] = dist
            D[j, i] = dist

    # save the distance matrix to a file
    save_path = f'precomputed/gw_dist_mat_based_{mode}/{ds_name}_{metric}_dist_mat_based_{mode}.npy' if metric == 'gw' else f'precomputed/en_gw_dist_mat_based_{mode}/{ds_name}_{metric}_dist_mat_based_{mode}.npy'
    with open(save_path, 'wb') as f:
        np.save(f, D)
    return None

def load_distance_matrix(ds_name: str, metric: str = 'gw', mode: str = 'adj') -> np.ndarray:
    """
    Loads the precomputed distance matrix for the specified dataset and metric.
    :param ds_name: The name of the dataset.
    :param metric: The metric to use ('gw' or 'en_gw').
    :param mode: The mode to use for distance computation ('adj' or 'sp').
    :return: A 2D numpy array representing the distance matrix.
    """
    file_path = f'precomputed/gw_dist_mat_based_{mode}/{ds_name}_{metric}_dist_mat_based_{mode}.npy'
    with open(file_path, 'rb') as f:
        D = np.load(f)
    return D

def compute_pairwise_tp(ds_name: str, metric: str = 'gw', mode: str = 'adj') -> None:
    """
    Computes the pairwise transport plan between all graphs in the dataset.
    :param ds_name: The name of the dataset.
    :param metric: The metric to use ('gw' or 'en_gw').
    :param mode: The mode to use for distance computation ('adj' or 'sp').
    :return: A 2D numpy array representing the pairwise transport plans.
    """
    import ot
    from tqdm import tqdm
    import pickle
    dataset = get_ds(name=ds_name, path='../data')
    n = len(dataset)
    P = []
    for i in tqdm(range(n)):
        P_row = []
        C1 = build_graph_cost(data=dataset[i], mode=mode)
        for j in range(i + 1, n):
            C2 = build_graph_cost(data=dataset[j], mode=mode)
            if metric == 'gw':
                # Here you can implement the actual Gromov-Wasserstein distance calculation
                tp = ot.gromov.gromov_wasserstein(C1=C1, C2=C2, loss_fun='square_loss')
            elif metric == 'en_gw':
                # Here you can implement the actual Entropic Gromov-Wasserstein distance calculation
                tp = ot.gromov.entropic_gromov_wasserstein(C1=C1, C2=C2, loss_fun='square_loss', solver='PPA', tol=1e-4)
            P_row.append(tp)
        P.append(P_row)

    # save the transport plans to a file
    save_path = f'precomputed/gw_tp_based_{mode}/{ds_name}_{metric}_tp_based_{mode}.npy' if metric == 'gw' else f'precomputed/en_gw_tp_based_{mode}/{ds_name}_{metric}_tp_based_{mode}.npy'
    with open(save_path, 'wb') as f:
        pickle.dump(P, f)
    return None

def load_transport_plans(ds_name: str, metric: str = 'gw', mode: str = 'adj') -> List:
    """
    Loads the precomputed transport plans for the specified dataset and metric.
    :param ds_name: The name of the dataset.
    :param metric: The metric to use ('gw' or 'en_gw').
    :param mode: The mode to use for distance computation ('adj' or 'sp').
    :return: A list of transport plans.
    """
    import pickle
    if metric == 'gw':
        file_path = f'precomputed/gw_tp_based_{mode}/{ds_name}_{metric}_tp_based_{mode}.npy'
    elif metric == 'en_gw':
        file_path = f'precomputed/en_gw_tp_based_{mode}/{ds_name}_{metric}_tp_based_{mode}.npy'
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'gw' or 'en_gw'.")
    
    with open(file_path, 'rb') as f:
        P = pickle.load(f)
    return P

def compute_MMD_from_Kernel_matrix(K: torch.Tensor, seg_ptr: int, device: torch.device = torch.device('cpu'), is_biased: bool = False) -> torch.Tensor:
    """
    Computes the Maximum Mean Discrepancy (MMD) from the kernel matrix.
    :param K: The kernel matrix.
    :param seg_ptr: The segmentation pointer.
    :param device: The device to perform the computation on.
    :return: The MMD value.
    """
    # Move the kernel matrix to the specified device
    K = K.to(device)
    K11, K12, K22 = K[:seg_ptr, :seg_ptr], K[:seg_ptr, seg_ptr:], K[seg_ptr:, seg_ptr:]
    if is_biased:
        mmd = K11.sum()/seg_ptr**2 - 2*K12.sum()/(seg_ptr*(K.shape[0] - seg_ptr)) + K22.sum()/(K.shape[0] - seg_ptr)**2
    else:
        mmd = (K11.sum() - seg_ptr)/(seg_ptr * (seg_ptr - 1)) - 2 * K12.sum() / (seg_ptr * (K.shape[0] - seg_ptr)) + (K22.sum() - (K.shape[0] - seg_ptr)) / ((K.shape[0] - seg_ptr) * (K.shape[0] - seg_ptr - 1))

    return mmd