import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Tuple, Dict, Callable

from utils import funcs
from ._utils_Processing import DataPreparationSystem
# from ._utils_Architectures import GINTransformer, GINTransformer_Naive

import timeit
from copy import deepcopy
from tqdm import tqdm


class NeuralGW(nn.Module):
    def __init__(self, ds_name_seq: List[str], testset: str = 'global', is_naive: bool = False,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(NeuralGW, self).__init__()
        self.config = config
        self.device = device

        # prepare for train/eval/test datasets
        self.dps = DataPreparationSystem(ds_name_seq=ds_name_seq, pe_cost_mode='adj', gw_cost_mode='adj')
        self.ds_seq = self.dps.load_all_graph_ds()
        self.total_pes = self.dps.compose_pe_from_multiple_ds()
        self.total_pairwise_distances = self.dps.compose_pairwise_distances_from_multiple_ds()
        # End prepare for train/eval/test datasets

        # train_eval_idx, self.test_idx = train_test_split(range(self.total_pes.shape[0]), test_size=0.2, random_state=42)
        # self.train_idx, self.eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)
        if testset == 'global':
            train_eval_idx, self.test_idx = self.dps.train_test_split(testset=testset, test_ratio=0.2, random_seed=42)
            self.train_idx, self.eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)
        elif testset in ds_name_seq:
            self.train_idx, test_eval_idx = self.dps.train_test_split(testset=testset, test_ratio=0.2, random_seed=42)
            self.test_idx, self.eval_idx = train_test_split(test_eval_idx, test_size=0.2, random_state=42)
        else:
            raise ValueError(f"testset {testset} not recognized. It should be 'global' or one of {ds_name_seq}.")

        # instantiate the model
        self.input_dim = int(np.ceil(self.dps.max_pe_feature_dim(self.total_pes) / 4)) * 4  # ensure the input_dim is a multiple of 4 because of the multi-head attention mechanism
        print(f'Input dimension adjusted to: {self.input_dim}')
        if not is_naive:
            from ._utils_Architectures import GINTransformer
            self.model = GINTransformer(input_dim=self.input_dim,
                                        hidden_dim=config["gwidth"],
                                        gin_layers=config["num_gin_layers"],
                                        transformer_layers=config["num_transformer_layers"],
                                        n_gammas=config['n_gammas'],
                                        config=config,
                                        device=device).to(device)
        else:
            from ._utils_Architectures import GINTransformer_Naive
            self.model = GINTransformer_Naive(input_dim=self.input_dim,
                                             hidden_dim=config["gwidth"],
                                             gin_layers=config["num_gin_layers"],
                                             transformer_layers=config["num_transformer_layers"],
                                             n_gammas=config['n_gammas'],
                                             config=config,
                                             device=device).to(device)
    
    def compute_loss(self, gwd_pred: torch.Tensor, gwd_gt: np.floating) -> torch.Tensor:
        loss = (gwd_pred - gwd_gt) ** 2
        # gwd_pred = gwd_pred.sigmoid()
        # loss = -(gwd_gt * torch.log(gwd_pred) + (1 - gwd_gt) * torch.log(1 - gwd_pred))  # Binary Cross Entropy Loss
        return loss
    
    def create_train_state(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], betas=(0.9, 0.999), weight_decay=self.config["weight_decay"])

    def train_step(self, batch_idx: List[int]) -> Tuple[torch.Tensor, Tuple[float, float]]:
        tri_u_ind = np.stack(np.triu_indices(len(batch_idx), k=1), axis=1)
        tmp_gwd_pred_seq_per_batch = []
        tmp_gwd_gt_seq_per_batch = []
        batch_loss_seq = []
        trange = tqdm(range(tri_u_ind.shape[0])) if len(batch_idx) > 10 else range(tri_u_ind.shape[0])
        for i in trange:
            idx1, idx2 = batch_idx[tri_u_ind[i,0]], batch_idx[tri_u_ind[i,1]]
            idx1 = batch_idx[tri_u_ind[i, 0]].item() if hasattr(batch_idx[tri_u_ind[i, 0]], 'item') else batch_idx[tri_u_ind[i, 0]]
            idx2 = batch_idx[tri_u_ind[i, 1]].item() if hasattr(batch_idx[tri_u_ind[i, 1]], 'item') else batch_idx[tri_u_ind[i, 1]]
            
            raw_pe1 = self.dps.pe_for_global_index(self.total_pes, idx1)
            if raw_pe1.shape[-1] < self.input_dim:
                pe1 = torch.tensor(np.pad(raw_pe1, ((0, 0), (0, self.input_dim - raw_pe1.shape[-1]))), dtype=torch.float32, device=self.device)
            elif raw_pe1.shape[-1] > self.input_dim:
                pe1 = torch.tensor(raw_pe1[:, : self.input_dim], dtype=torch.float32, device=self.device)
            else:
                pe1 = torch.tensor(raw_pe1, dtype=torch.float32, device=self.device)
            # edge_index1 = self.ds[idx1].edge_index.to(self.device)
            ds_idx1, g_idx1 = self.dps.map_global_idx_to_local_indices(global_idx=idx1)
            edge_index1 = self.ds_seq[ds_idx1][g_idx1].edge_index.to(self.device)

            raw_pe2 = self.dps.pe_for_global_index(self.total_pes, idx2)
            if raw_pe2.shape[-1] < self.input_dim:
                pe2 = torch.tensor(np.pad(raw_pe2, ((0, 0), (0, self.input_dim - raw_pe2.shape[-1]))), dtype=torch.float32, device=self.device)
            elif raw_pe2.shape[-1] > self.input_dim:
                pe2 = torch.tensor(raw_pe2[:, : self.input_dim], dtype=torch.float32, device=self.device)
            else:
                pe2 = torch.tensor(raw_pe2, dtype=torch.float32, device=self.device)
            # edge_index2 = self.ds[idx2].edge_index.to(self.device)
            ds_idx2, g_idx2 = self.dps.map_global_idx_to_local_indices(global_idx=idx2)
            edge_index2 = self.ds_seq[ds_idx2][g_idx2].edge_index.to(self.device)

            # forward pass
            gwd_pred = self.model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2)
            gwd_gt = self.total_pairwise_distances[idx1, idx2]
            loss = self.compute_loss(gwd_pred, gwd_gt)

            # keep track of the loss, gwd_pred, and gwd_gt
            batch_loss_seq.append(loss)
            tmp_gwd_pred_seq_per_batch.append(gwd_pred.item())
            tmp_gwd_gt_seq_per_batch.append(gwd_gt.item())

        # backward pass
        batch_loss = sum(batch_loss_seq) / len(batch_loss_seq)
        # tmp_res_per_batch = np.linalg.vector_norm(np.array(tmp_gwd_pred_seq_per_batch) - np.array(tmp_gwd_gt_seq_per_batch), ord=2) / np.linalg.vector_norm(np.array(tmp_gwd_gt_seq_per_batch), ord=2)
        return batch_loss, (batch_loss_seq, tmp_gwd_pred_seq_per_batch, tmp_gwd_gt_seq_per_batch)

    def train_model(self, seed_worker: Callable[[int], None]) -> None:
        g = torch.Generator()
        g.manual_seed(self.config["seed"])
        # sampler = torch.utils.data.distributed.DistributedSampler(self.train_idx, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=True, seed=self.config["seed"])
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_idx, shuffle=True, drop_last=True, seed=self.config["seed"])
        trainloader = torch.utils.data.DataLoader(self.train_idx, batch_size=self.config["batch_size"], drop_last=True, sampler=sampler)
        # trainloader = torch.utils.data.DataLoader(self.train_idx, batch_size=self.config["batch_size"], shuffle=True, drop_last=True, num_workers=4, generator=g, worker_init_fn=seed_worker)

        self.create_train_state()

        best_eval_res = float('inf')
        train_loss_seq = []
        train_res_seq = []
        ctime = timeit.default_timer()
        for epoch in range(self.config["num_epochs"]):
            sampler.set_epoch(epoch)
            self.train()
            train_loss_seq_per_epoch = []
            train_res_seq_per_epoch = []
            for batch_idx in trainloader:
                self.zero_grad()
                batch_loss, (batch_loss_seq, gwd_pred_seq_per_batch, gwd_gt_seq_per_batch) = self.train_step(batch_idx=batch_idx)
                batch_loss.backward()
                self.optimizer.step()

                # keep track of the average loss and the whole relative residual of error under the current batch
                batch_loss_mean = sum(batch_loss_seq).item() / len(batch_loss_seq)
                batch_gwd_res_mean = np.linalg.vector_norm(np.array(gwd_pred_seq_per_batch) - np.array(gwd_gt_seq_per_batch), ord=2) / np.linalg.vector_norm(np.array(gwd_gt_seq_per_batch), ord=2)
                train_loss_seq_per_epoch.append(batch_loss_mean)
                train_res_seq_per_epoch.append(batch_gwd_res_mean)

            # keep track of the average loss and the whole relative residual of error under the current epoch
            train_loss_seq.append(sum(train_loss_seq_per_epoch) / len(train_loss_seq_per_epoch))
            train_res_seq.append(sum(train_res_seq_per_epoch) / len(train_res_seq_per_epoch))
            time_elapsed = timeit.default_timer() - ctime
            print(f'[{epoch:-3d}/{self.config["num_epochs"]}]({time_elapsed/60:.2f} m) [running_train_loss running_train_res] = [{train_loss_seq[-1]:.8f} {train_res_seq[-1]:.8f}]')

            # eval the model
            stop_flag = torch.tensor(0, device=self.device)
            if epoch % self.config["eval_interval"] == 0 and dist.get_rank() == 0:
                # eval_loss, eval_res = model.eval_model(config=config, model=model, ds=ds, eval_idx=eval_idx, ptr=ptr, pe=pe, D=D, device=device)
                eval_loss, eval_res = self.eval_model(eval_idx=self.eval_idx)
                time_elapsed = timeit.default_timer() - ctime
                print(f'[{epoch:-3d}/{self.config["num_epochs"]}]({dist.get_rank()})({time_elapsed/60:.2f} m) [running_train_loss eval_loss train_res eval_res] = [{train_loss_seq[-1]:.8f} {eval_loss:.8f} {train_res_seq[-1]:.8f} {eval_res:.8f}]')
                if eval_res < best_eval_res:
                    print(f'[Info] New best eval_res: {eval_res:.8f} < {best_eval_res:.8f}, updating the best model...')
                    stop_count = 0
                    best_eval_res = eval_res
                    print(f'[Info] Best_eval_res has been updated to {best_eval_res:.8f}')
                    self.best_model_state = deepcopy(self.model.state_dict())
                    print(f'[Info] Best_model_state has been recorded.')
                    print(f'[Info] Stop_count has been reset to 0.')
                else:
                    stop_count += 1
                    print(f'[Info] eval_res: {eval_res:.8f} >= best_eval_res: {best_eval_res:.8f}. Stop_count has been increased to {stop_count}.')
                    
                if stop_count >= self.config["early_stop_patience"]:
                    print(f'[Info] Early stopping triggered. Stopping training...')
                    stop_flag.fill_(1)

            #  broadcast the stop_flag to all processes
            if dist.is_initialized():
                dist.broadcast(stop_flag, src=0)

            if stop_flag.item() == 1:
                print(f'[Info] Rank {dist.get_rank()} received stop signal. Stopping training...')
                break

        # test the model
        if dist.get_rank() == 0:
            test_loss, test_res = self.test_model(test_idx=self.test_idx)
            print(f'Test loss: {test_loss:.8f}, Test result: {test_res:.8f}')
            
            # # save the best model state
            checkpoint_path = f'checkpoints/exp_indomain/{self.config["ds_name"]}_model_gwidth{self.config["gwidth"]}_epoch{epoch}_cuda{self.config["cuda_id"]}.pth'
            torch.save(self.best_model_state, checkpoint_path)

    def eval_model(self, eval_idx: List[int]) -> Tuple[float, float]:
        # model.eval()
        self.eval()
        with torch.no_grad():
            sampler = torch.utils.data.distributed.DistributedSampler(eval_idx, shuffle=False, drop_last=True, seed=self.config["seed"])
            dataloader = torch.utils.data.DataLoader(eval_idx, batch_size=self.config["batch_size"], drop_last=True, sampler=sampler)
            eval_loss_seq = []
            eval_gwd_pred_seq = []
            eval_gwd_gt_seq = []
            for batch_idx in dataloader:
                _, (batch_loss_seq, gwd_pred_seq_per_batch, gwd_gt_seq_per_batch) = self.train_step(batch_idx=batch_idx)
                eval_loss_seq.extend(batch_loss_seq)
                eval_gwd_pred_seq.extend(gwd_pred_seq_per_batch)
                eval_gwd_gt_seq.extend(gwd_gt_seq_per_batch)

            eval_loss = sum(eval_loss_seq).item() / len(eval_loss_seq)
            eval_res = np.linalg.vector_norm(np.array(eval_gwd_pred_seq) - np.array(eval_gwd_gt_seq), ord=2) / np.linalg.vector_norm(np.array(eval_gwd_gt_seq), ord=2)

        return eval_loss, eval_res

    def test_model(self, test_idx: List[int]) -> Tuple[float, float]:
        self.model.load_state_dict(self.best_model_state)
        # self.load_state_dict(self.best_model_state)
        test_loss, test_res = self.eval_model(eval_idx=test_idx)
        return test_loss, test_res
    
    def average_metric(val: float, device: torch.device) -> float:
        t = torch.tensor(val, dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / dist.get_world_size()).item()
    

class NeuralGW_TP(nn.Module):
    def __init__(self, ds_name_seq: List[str], testset: str = 'global', is_naive: bool = False,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(NeuralGW_TP, self).__init__()
        self.config = config
        self.device = device

        # prepare for train/eval/test datasets
        self.dps = DataPreparationSystem(ds_name_seq=ds_name_seq, pe_cost_mode='adj', gw_cost_mode='adj')
        self.ds_seq = self.dps.load_all_graph_ds()
        self.total_pes = self.dps.compose_pe_from_multiple_ds()
        self.total_transport_plans = self.dps.compose_transport_plans_from_multiple_ds()
        # End prepare for train/eval/test datasets

        # train_eval_idx, self.test_idx = train_test_split(range(self.total_pes.shape[0]), test_size=0.2, random_state=42)
        # self.train_idx, self.eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)
        train_eval_idx, self.test_idx = self.dps.train_test_split(testset=testset, test_ratio=0.2, random_seed=42)
        self.train_idx, self.eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)

        # instantiate the model
        self.input_dim = int(np.ceil(self.dps.max_pe_feature_dim(self.total_pes) / 4)) * 4  # ensure the input_dim is a multiple of 4 because of the multi-head attention mechanism
        print(f'Input dimension adjusted to: {self.input_dim}')
        if not is_naive:
            from ._utils_Architectures import GINTransformer_TP
            self.model = GINTransformer_TP(input_dim=self.input_dim,
                                           hidden_dim=config["gwidth"],
                                           gin_layers=config["num_gin_layers"],
                                           transformer_layers=config["num_transformer_layers"],
                                           n_gammas=config['n_gammas'],
                                           config=config,
                                           device=device).to(device)
        else:
            from ._utils_Architectures import GINTransformer_TP_Naive
            self.model = GINTransformer_TP_Naive(input_dim=self.input_dim,
                                                 hidden_dim=config["gwidth"],
                                                 gin_layers=config["num_gin_layers"],
                                                 transformer_layers=config["num_transformer_layers"],
                                                 n_gammas=config['n_gammas'],
                                                 config=config,
                                                 device=device).to(device)
    
    def compute_loss(self, tp_pred: torch.Tensor, tp_gt: np.ndarray) -> torch.Tensor:
        loss = nn.functional.binary_cross_entropy_with_logits(tp_pred, tp_gt)
        return loss
    
    def create_train_state(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], betas=(0.9, 0.999), weight_decay=self.config["weight_decay"])

    def train_step(self, batch_idx: List[int]) -> Tuple[torch.Tensor, Tuple[float, float]]:
        tri_u_ind = np.stack(np.triu_indices(len(batch_idx), k=1), axis=1)
        batch_loss = []
        train_res_seq_per_batch = []
        for i in range(tri_u_ind.shape[0]):
            idx1, idx2 = batch_idx[tri_u_ind[i,0]], batch_idx[tri_u_ind[i,1]]
            idx1 = batch_idx[tri_u_ind[i, 0]].item() if hasattr(batch_idx[tri_u_ind[i, 0]], 'item') else batch_idx[tri_u_ind[i, 0]]
            idx2 = batch_idx[tri_u_ind[i, 1]].item() if hasattr(batch_idx[tri_u_ind[i, 1]], 'item') else batch_idx[tri_u_ind[i, 1]]
            
            raw_pe1 = self.dps.pe_for_global_index(self.total_pes, idx1)
            if raw_pe1.shape[-1] < self.input_dim:
                pe1 = torch.tensor(np.pad(raw_pe1, ((0, 0), (0, self.input_dim - raw_pe1.shape[-1]))), dtype=torch.float32, device=self.device)
            elif raw_pe1.shape[-1] > self.input_dim:
                pe1 = torch.tensor(raw_pe1[:, : self.input_dim], dtype=torch.float32, device=self.device)
            else:
                pe1 = torch.tensor(raw_pe1, dtype=torch.float32, device=self.device)
            # edge_index1 = self.ds[idx1].edge_index.to(self.device)
            ds_idx1, g_idx1 = self.dps.map_global_idx_to_local_indices(global_idx=idx1)
            edge_index1 = self.ds_seq[ds_idx1][g_idx1].edge_index.to(self.device)

            raw_pe2 = self.dps.pe_for_global_index(self.total_pes, idx2)
            if raw_pe2.shape[-1] < self.input_dim:
                pe2 = torch.tensor(np.pad(raw_pe2, ((0, 0), (0, self.input_dim - raw_pe2.shape[-1]))), dtype=torch.float32, device=self.device)
            elif raw_pe2.shape[-1] > self.input_dim:
                pe2 = torch.tensor(raw_pe2[:, : self.input_dim], dtype=torch.float32, device=self.device)
            else:
                pe2 = torch.tensor(raw_pe2, dtype=torch.float32, device=self.device)
            # edge_index2 = self.ds[idx2].edge_index.to(self.device)
            ds_idx2, g_idx2 = self.dps.map_global_idx_to_local_indices(global_idx=idx2)
            edge_index2 = self.ds_seq[ds_idx2][g_idx2].edge_index.to(self.device)

            # forward pass
            tp_pred = self.model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2)
            tp_gt = torch.tensor(self.total_transport_plans[(ds_idx1, ds_idx2)][(g_idx1, g_idx2)], dtype=torch.float32, device=self.device)
            loss = self.compute_loss(tp_pred, tp_gt)
            tp_pred = tp_pred.sigmoid()
            res = torch.linalg.vector_norm(tp_pred - tp_gt, ord=1) / torch.linalg.vector_norm(tp_gt, ord=1)

            # keep track of the loss, tp_pred, and tp_gt
            batch_loss.append(loss)
            train_res_seq_per_batch.append(res.item())

        # backward pass
        batch_loss = sum(batch_loss) / len(batch_loss)
        train_res_per_batch = sum(train_res_seq_per_batch) / len(train_res_seq_per_batch)
        return batch_loss, (batch_loss.item(), train_res_per_batch)

    def train_model(self, seed_worker: Callable[[int], None]) -> None:
        g = torch.Generator()
        g.manual_seed(self.config["seed"])
        trainloader = torch.utils.data.DataLoader(self.train_idx, batch_size=self.config["batch_size"], shuffle=True, drop_last=True, num_workers=4, generator=g, worker_init_fn=seed_worker)

        self.create_train_state()

        best_eval_res = float('inf')
        train_loss_seq = []
        train_res_seq = []
        ctime = timeit.default_timer()
        for epoch in range(self.config["num_epochs"]):
            self.train()
            train_loss_seq_per_epoch = []
            train_res_seq_per_epoch = []
            for batch_idx in trainloader:
                self.zero_grad()
                batch_loss, (train_loss_per_batch, train_res_per_batch) = self.train_step(batch_idx=batch_idx)
                batch_loss.backward()
                self.optimizer.step()

                # keep track of the average loss and the whole relative residual of error under the current batch
                train_loss_seq_per_epoch.append(train_loss_per_batch)
                train_res_seq_per_epoch.append(train_res_per_batch)

            # keep track of the average loss and the whole relative residual of error under the current epoch
            train_loss_seq.append(sum(train_loss_seq_per_epoch) / len(train_loss_seq_per_epoch))
            train_res_seq.append(sum(train_res_seq_per_epoch) / len(train_res_seq_per_epoch))
            time_elapsed = timeit.default_timer() - ctime
            print(f'[{epoch:-3d}/{self.config["num_epochs"]}]({time_elapsed/60:.2f} m) [running_train_loss running_train_res] = [{train_loss_seq[-1]:.8f} {train_res_seq[-1]:.8f}]')

            # eval the model
            if epoch % self.config["eval_interval"] == 0:
                # eval_loss, eval_res = model.eval_model(config=config, model=model, ds=ds, eval_idx=eval_idx, ptr=ptr, pe=pe, D=D, device=device)
                eval_loss, eval_res = self.eval_model(eval_idx=self.eval_idx)
                time_elapsed = timeit.default_timer() - ctime
                print(f'[{epoch:-3d}/{self.config["num_epochs"]}]({time_elapsed/60:.2f} m) [running_train_loss eval_loss train_res eval_res] = [{train_loss_seq[-1]:.8f} {eval_loss:.8f} {train_res_seq[-1]:.8f} {eval_res:.8f}]')
                if eval_res < best_eval_res:
                    print(f'[Info] New best eval_res: {eval_res:.8f} < {best_eval_res:.8f}, updating the best model...')
                    stop_count = 0
                    best_eval_res = eval_res
                    print(f'[Info] Best_eval_res has been updated to {best_eval_res:.8f}')
                    self.best_model_state = deepcopy(self.state_dict())
                    print(f'[Info] Best_model_state has been recorded.')
                    print(f'[Info] Stop_count has been reset to 0.')
                else:
                    stop_count += 1
                    print(f'[Info] eval_res: {eval_res:.8f} >= best_eval_res: {best_eval_res:.8f}. Stop_count has been increased to {stop_count}.')
                    
                if stop_count >= self.config["early_stop_patience"]:
                    print(f'[Info] Early stopping triggered. Stopping training...')
                    break

        # test the model
        test_loss, test_res = self.test_model(test_idx=self.test_idx)
        print(f'Test loss: {test_loss:.8f}, Test result: {test_res:.8f}')
        
        # save the best model state
        checkpoint_path = f'checkpoints/exp_indomain/{self.config["ds_name"]}_model_gwidth{self.config["gwidth"]}_epoch{epoch}_cuda{self.config["cuda_id"]}.pth'
        torch.save(self.best_model_state, checkpoint_path)

    def eval_model(self, eval_idx: List[int]) -> Tuple[float, float]:
        # model.eval()
        self.eval()
        with torch.no_grad():
            _, (eval_loss, eval_res) = self.train_step(batch_idx=eval_idx)

        return eval_loss, eval_res

    def test_model(self, test_idx: List[int]) -> Tuple[float, float]:
        # model.load_state_dict(best_model_state)
        self.load_state_dict(self.best_model_state)
        test_loss, test_res = self.eval_model(eval_idx=test_idx)
        return test_loss, test_res