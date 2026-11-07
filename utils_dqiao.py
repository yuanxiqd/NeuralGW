import numpy as np

# GraKeL still does `from numpy import ComplexWarning`, which NumPy 2 removed
# from the top-level namespace; restore it before importing grakel.
if not hasattr(np, "ComplexWarning"):
    from numpy.exceptions import ComplexWarning as _NumpyComplexWarning

    np.ComplexWarning = _NumpyComplexWarning


import torch
from torch_geometric.utils import to_dense_adj
from grakel import Graph, GraphKernel

kernel_configs = {
    'random_walk': {
        'kernel': {'name': 'random_walk'},
        'normalize': True
    },
    'weisfeiler_lehman': {
        'kernel': {'name': 'weisfeiler_lehman', 'n_iter': 5},
        'normalize': True
    },
    'weisfeiler_lehman_optimal_assignment': {
        'kernel': {'name': 'weisfeiler_lehman_optimal_assignment', 'n_iter': 5},
        'normalize': True
    },
}

def to_grakel_graph(dataset, use_label=True, as_vector=False):
    print(f'[echo] dataset: {dataset}')
    graphs = []
    for i, data in enumerate(dataset):
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).numpy()
        # print(f'[echo] {i} data has node_label') if hasattr(data, 'node_label') else print(f'[echo] data has no node_label')
        # print(f'[echo] {i} data has x') if hasattr(data, 'x') else print(f'[echo] data has no x')
        if hasattr(data, 'x') and use_label:
            print(f'[echo] I came 1') if i == len(dataset)-1 else None
            if as_vector:
                node_attributes = data.x.numpy() if hasattr(data.x, 'numpy') else data.x
                node_attributes = {i: attr.tolist() if hasattr(attr, 'tolist') else attr for i, attr in enumerate(node_attributes)}
            else:
                node_attributes = torch.argmax(data.x, dim=1).numpy()
                node_attributes = {i: attr for i, attr in enumerate(node_attributes)}
        # elif hasattr(data, 'x'):
        #     node_attributes = data.x.numpy() if hasattr(data.x, 'numpy') else data.x
        #     node_attributes = {i: attr.tolist() if hasattr(attr, 'tolist') else attr for i, attr in enumerate(node_attributes)} 
        elif as_vector:
            print(f'[echo] I came 2') if i == len(dataset)-1 else None
            node_labels = [1.0] * data.num_nodes
            node_labels = {i: [label, 0] for i, label in enumerate(node_labels)}
            node_attributes = node_labels           
        else:
            print(f'[echo] I came 3') if i == len(dataset)-1 else None
            node_labels = [1] * data.num_nodes
            node_labels = {i: label for i, label in enumerate(node_labels)}
            node_attributes = node_labels
            
        g = Graph(adj, node_labels=node_attributes)
        graphs.append(g)
    return graphs

def get_kernel_mtx(kernel_name, fit_graphs, tran_graphs):
    if kernel_name == "MFD":
        raise NotImplementedError(
            "MFD kernel has no kernel_config in this module; add it to kernel_configs first."
        )
    try:
        kernel_config = kernel_configs[kernel_name]
    except KeyError as err:
        raise ValueError(
            f"Unknown kernel_name={kernel_name!r}; "
            f"expected one of {sorted(kernel_configs)}."
        ) from err
    # WL (and WL-OA) need discrete node labels to be meaningful; only kernels
    # that truly operate on unlabeled graphs should set use_label=False.
    if kernel_name in ['svm_theta']:
        use_label = False
    else:
        use_label = True
        
    # if kernel_name == 'random_walk':
    #     fit_maxdegrs = list(map(lambda graph: torch.max(degree(graph.edge_index[0], graph.num_nodes)).item(), fit_graphs))
    #     tran_maxdegrs = list(map(lambda graph: torch.max(degree(graph.edge_index[0], graph.num_nodes)).item(), tran_graphs))
    #     max_dgr = max(fit_maxdegrs + tran_maxdegrs)
    #     # kernel_config['kernel']['lamda'] = 1/(max_dgr+1)
    #     kernel_config['kernel']['lamda'] = (0.05)
    as_vector = (kernel_name == 'graph_hopper')
    if as_vector:
        print('!')
    fit_graphs = to_grakel_graph(fit_graphs, use_label, as_vector=as_vector)
    tran_graphs = to_grakel_graph(tran_graphs, use_label, as_vector=as_vector)
    kernel = GraphKernel(**kernel_config)
    kernel.fit(fit_graphs)
    K = kernel.transform(tran_graphs)

    nan_count = np.isnan(K).sum()
    total_elems = K.size
    nan_ratio = float(nan_count) / float(total_elems) if total_elems > 0 else 0.0
    print(f'{kernel_name} kernel NaN ratio: {nan_ratio:.6%} ({nan_count}/{total_elems})')
    if nan_count:
        print(f'Warning: Kernel matrix contains NaN!')
        K = np.nan_to_num(K) # handle NaN values
    return K

if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    ds_name = 'MUTAG'
    dataset = TUDataset(root='data/TUDataset', name=ds_name)
    print(f'Number of graphs: {len(dataset)}')
    K_rw = get_kernel_mtx('random_walk', dataset, dataset)
    K_wl = get_kernel_mtx('weisfeiler_lehman', dataset, dataset)
    K_wloa = get_kernel_mtx('weisfeiler_lehman_optimal_assignment', dataset, dataset)
    print(f'Random Walk Kernel: \n{K_rw} \n with (min, max) = ({K_rw.min()}, {K_rw.max()})')
    print(f'Weisfeiler-Lehman Kernel: \n{K_wl} \n with (min, max) = ({K_wl.min()}, {K_wl.max()})')
    print(f'Weisfeiler-Lehman Optimal Assignment Kernel: \n{K_wloa} \n with (min, max) = ({K_wloa.min()}, {K_wloa.max()})')

    # save kernel matrices
    fname = f'results/grakels/{ds_name}_grakel_rw_kernel.npy'
    with open(fname, 'wb') as f:
        np.save(f, K_rw)
        print(f'Saved Random Walk kernel matrix to {fname}')
    fname = f'results/grakels/{ds_name}_grakel_wl_kernel.npy'
    with open(fname, 'wb') as f:
        np.save(f, K_wl)
        print(f'Saved Weisfeiler-Lehman kernel matrix to {fname}')
    fname = f'results/grakels/{ds_name}_grakel_wloa_kernel.npy'
    with open(fname, 'wb') as f:
        np.save(f, K_wloa)
        print(f'Saved Weisfeiler-Lehman Optimal Assignment kernel matrix to {fname}')

    # # load kernel matrices
    # fname = f'results/grakels_1st_for_paper/{ds_name}_grakel_rw_kernel.npy'
    # with open(fname, 'rb') as f:
    #     K_rw = np.load(f)
    # fname = f'results/grakels_1st_for_paper/{ds_name}_grakel_wl_kernel.npy'
    # with open(fname, 'rb') as f:
    #     K_wl = np.load(f)
    # fname = f'results/grakels_1st_for_paper/{ds_name}_grakel_wloa_kernel.npy'
    # with open(fname, 'rb') as f:
    #     K_wloa = np.load(f)
    # print(f'Loaded Random Walk kernel matrix from {fname}')
    # print(f'Loaded Weisfeiler-Lehman kernel matrix from {fname}')
    # print(f'Loaded Weisfeiler-Lehman Optimal Assignment kernel matrix from {fname}')

    print(100*'-')
    # perform spectral clustering on the kernel matrices in terms of accuracy, nmi, ari, and ri
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score, rand_score, confusion_matrix
    from scipy.optimize import linear_sum_assignment
    labels = dataset.y.numpy()

    def hungarian_accuracy(y_true, y_pred):
        """Accuracy after optimal label permutation via the Hungarian algorithm."""
        cm = confusion_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)  # maximise matched counts
        remapped = np.zeros_like(y_pred)
        for r, c in zip(row_ind, col_ind):
            remapped[y_pred == c] = r
        return accuracy_score(y_true, remapped)

    for K, name in zip([K_rw, K_wl, K_wloa], ['Random Walk', 'Weisfeiler-Lehman', 'Weisfeiler-Lehman Optimal Assignment']):
        sc = SpectralClustering(n_init=10, n_clusters=dataset.num_classes, affinity='precomputed', assign_labels='discretize', random_state=42)
        pred_labels = sc.fit_predict(K)
        acc = hungarian_accuracy(labels, pred_labels)
        # acc = accuracy_score(labels, pred_labels)
        nmi = normalized_mutual_info_score(labels, pred_labels)
        ari = adjusted_rand_score(labels, pred_labels)
        ri = rand_score(labels, pred_labels)
        print(f'{name} Kernel Clustering Results:')
        print(f'Accuracy: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, RI: {ri:.4f}\n')