import numpy as np
import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as gutils
import ot
import timeit
import json
from tqdm import tqdm
# I want to import random_svd from sklearn
from sklearn.utils.extmath import randomized_svd

if __name__ == "__main__":
    ds_name = 'COLLAB'  # change to other datasets as needed
    flag = [True, False, False, False]  # GW, entropic GW, sampled GW, NeuralGW
    if ds_name == 'DD':
        ds = TUDataset(root='data/TUDataset', name=ds_name)
        sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
        ds = ds[sub_idx]
    elif ds_name == 'ENZYMES':
        ds = TUDataset(root='data/TUDataset', name=ds_name)
        sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 70]
        ds = ds[sub_idx]
    elif ds_name == 'COLLAB':
        ds = TUDataset(root='data/TUDataset', name=ds_name)
        # sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
        sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes >= 300 and ds[i].num_nodes <= 306]
        ds = ds[sub_idx]
    elif ds_name == 'REDDIT-BINARY':
        ds = TUDataset(root='data/TUDataset', name=ds_name)
        sub_idx = [i for i in range(len(ds)) if ds[i].num_nodes > 300 and ds[i].num_nodes < 350]
        ds = ds[sub_idx]
    else:
        ds = TUDataset(root='data/TUDataset', name=ds_name)
    print(f'Dataset: {ds_name}, Number of graphs: {len(ds)}')
    max_num_nodes = max([data.num_nodes for data in ds])
    print(f'[Info] max_num_nodes in the dataset: {max_num_nodes}')

    time_seq = []

    if flag[0]:
        # calculate GW
        t0 = timeit.default_timer()
        for i in tqdm(range(len(ds))):
            C1 = gutils.to_dense_adj(edge_index=ds[i].edge_index, max_num_nodes=ds[i].num_nodes).squeeze(0).cpu().numpy()
            C1 += np.eye(C1.shape[0], dtype=C1.dtype)  # add self-loops
            for j in range(i + 1, len(ds)):
                C2 = gutils.to_dense_adj(edge_index=ds[j].edge_index, max_num_nodes=ds[j].num_nodes).squeeze(0).cpu().numpy()
                C2 += np.eye(C2.shape[0], dtype=C2.dtype)  # add self-loops
                d = ot.gromov.gromov_wasserstein2(C1=C1, C2=C2, tol_abs=1e-4, tol_rel=1e-4)

        t1 = timeit.default_timer()
        time_seq.append(t1 - t0)
        print(f'Total time for computing GW on {len(ds)} graphs: {t1 - t0:.4f} seconds')
        print(f'Average time per graph pair: {(t1 - t0) / (len(ds) * (len(ds) - 1) / 2):.4f} seconds')

    if flag[1]:
        # calculate entropic GW
        t0 = timeit.default_timer()
        for i in tqdm(range(len(ds))):
            C1 = gutils.to_dense_adj(edge_index=ds[i].edge_index, max_num_nodes=ds[i].num_nodes).squeeze(0).cpu().numpy()
            C1 += np.eye(C1.shape[0], dtype=C1.dtype)  # add self-loops
            for j in range(i + 1, len(ds)):
                C2 = gutils.to_dense_adj(edge_index=ds[j].edge_index, max_num_nodes=ds[j].num_nodes).squeeze(0).cpu().numpy()
                C2 += np.eye(C2.shape[0], dtype=C2.dtype)  # add self-loops
                d = ot.gromov.entropic_gromov_wasserstein2(C1=C1, C2=C2, epsilon=0.1, tol=1e-4, max_iter=100)

        t1 = timeit.default_timer()
        time_seq.append(t1 - t0)
        print(f'Total time for computing entropic GW on {len(ds)} graphs: {t1 - t0:.4f} seconds')
        print(f'Average time per graph pair: {(t1 - t0) / (len(ds) * (len(ds) - 1) / 2):.4f} seconds')

    if flag[2]:
        # calculate sampled GW
        def loss_fun(C1, C2):
            return (C1 - C2)**2
        t0 = timeit.default_timer()
        for i in tqdm(range(len(ds))):
            C1 = gutils.to_dense_adj(edge_index=ds[i].edge_index, max_num_nodes=ds[i].num_nodes).squeeze(0).cpu().numpy()
            C1 += np.eye(C1.shape[0], dtype=C1.dtype)  # add self-loops
            p = ot.unif(C1.shape[0], type_as=C1)
            for j in range(i + 1, len(ds)):
                C2 = gutils.to_dense_adj(edge_index=ds[j].edge_index, max_num_nodes=ds[j].num_nodes).squeeze(0).cpu().numpy()
                C2 += np.eye(C2.shape[0], dtype=C2.dtype)  # add self-loops
                q = ot.unif(C2.shape[0], type_as=C2)
                _, log = ot.gromov.sampled_gromov_wasserstein(C1=C1, C2=C2, p=p, q=q, loss_fun=loss_fun, epsilon=0.1, nb_samples_grad=100, max_iter=10, log=True, verbose=True)
                d = log['gw_dist_estimated']
        t1 = timeit.default_timer()
        time_seq.append(t1 - t0)
        print(f'Total time for computing sampled GW on {len(ds)} graphs: {t1 - t0:.4f} seconds')
        print(f'Average time per graph pair: {(t1 - t0) / (len(ds) * (len(ds) - 1) / 2):.4f} seconds')

    
    if flag[3]:
        # calculate the inference time of the proposed NeuralGW model
        fname = f'checkpoints/exp_indomain/BZR_PTC_FM_MUTAG_model_gwidth128_epoch1_cuda0.pth'
        adjusted_size = 64 # for BZR_PTC_FM_MUTAG
        from models.neuralgw_ddp import NeuralGW
        task = 'neuralgw'
        # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        print(f'Using device: {device}')
        with open(f'configs/main.json', "r") as f:
            config = json.load(f)
        model = NeuralGW(
            ds_name_seq=['BZR','PTC_FM','MUTAG'],
            testset='MUTAG',
            is_naive=False,
            config=config,
            device=device
        ).to(device)
        model.model.load_state_dict(torch.load(fname, weights_only=True, map_location=device))
        model.eval()
        t0 = timeit.default_timer()
        for i in tqdm(range(len(ds))):
            C1 = gutils.to_dense_adj(edge_index=ds[i].edge_index, max_num_nodes=ds[i].num_nodes).squeeze(0).cpu().numpy()
            C1 += np.eye(C1.shape[0], dtype=C1.dtype)  # add self-loops
            # u, s, vh = np.linalg.svd(C1)
            u, s, vh = randomized_svd(C1, n_components=max_num_nodes, n_iter=5, random_state=None)
            pe1 = u@np.diag(s**0.5)
            pe1 = np.pad(pe1, ((0, 0), (0, max_num_nodes - pe1.shape[1])), mode='constant', constant_values=0)  # pad to 28 dimensions
            pe1 = torch.tensor(pe1, dtype=torch.float32, device=device)
            # I have to pad the pe to 64 because the model input dimension is fixed to 352
            if pe1.shape[1] > adjusted_size:
                pe1 = pe1[:, :adjusted_size]  # truncate to adjusted_size dimensions
            else:
                pe1 = torch.cat([pe1, torch.zeros((pe1.shape[0], adjusted_size - pe1.shape[1]), device=device)], dim=1)
            edge_index1 = ds[i].edge_index.to(device)
            for j in range(i + 1, len(ds)):
                C2 = gutils.to_dense_adj(edge_index=ds[j].edge_index, max_num_nodes=ds[j].num_nodes).squeeze(0).cpu().numpy()
                C2 += np.eye(C2.shape[0], dtype=C2.dtype)  # add self-loops
                # u, s, vh = np.linalg.svd(C2)
                u, s, vh = randomized_svd(C2, n_components=max_num_nodes, n_iter=5, random_state=None)
                pe2 = u@np.diag(s**0.5)
                pe2 = np.pad(pe2, ((0, 0), (0, max_num_nodes - pe2.shape[1])), mode='constant', constant_values=0)  # pad to 28 dimensions
                pe2 = torch.tensor(pe2, dtype=torch.float32, device=device)
                # I have to pad the pe to 64 because the model input dimension is fixed to 352
                # pe2 = torch.cat([pe2, torch.zeros((pe2.shape[0], 64 - pe2.shape[1]), device=device)], dim=1)
                if pe2.shape[1] > adjusted_size:
                    pe2 = pe2[:, :adjusted_size]  # truncate to adjusted_size dimensions
                else:
                    pe2 = torch.cat([pe2, torch.zeros((pe2.shape[0], adjusted_size - pe2.shape[1]), device=device)], dim=1)
                edge_index2 = ds[j].edge_index.to(device)
                with torch.no_grad():
                    # print(f'[Debug] pe1 shape: {pe1.shape}, pe2 shape: {pe2.shape}')
                    d = model.model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2)

        t1 = timeit.default_timer()
        time_seq.append(t1 - t0)
        print(f'Total time for computing NeuralGW on {len(ds)} graphs: {t1 - t0:.4f} seconds')
        print(f'Average time per graph pair: {(t1 - t0) / (len(ds) * (len(ds) - 1) / 2):.4f} seconds')

        print(f'Time sequence (seconds): {time_seq}')
        print(f'All tasks completed.')
