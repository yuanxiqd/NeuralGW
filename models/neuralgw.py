import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score, rand_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Tuple, Dict, Callable

from utils import funcs
from ._utils_Processing import DataPreparationSystem, distance_to_kernel, active_meta_keys
# from ._utils_Architectures import GINTransformer, GINTransformer_Naive

import timeit
from copy import deepcopy
from tqdm import tqdm
import pickle


class NeuralGW(nn.Module):
    def __init__(self, ds_name_seq: List[str], testset: str = 'global', is_naive: bool = False,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(NeuralGW, self).__init__()
        self._ds_name_seq = ds_name_seq
        self.testset = testset
        self.config = config
        self.device = device

        # prepare for train/eval/test datasets
        self.dps = DataPreparationSystem(
            ds_name_seq=ds_name_seq,
            pe_cost_mode=config['pe_cost_mode'],
            gw_cost_mode=config['gw_cost_mode'],
            pe_mode=config['pe_mode'],
            treat_as_undirected=config.get('treat_as_undirected', True),
        )
        self.ds_seq = self.dps.load_all_graph_ds()
        self.total_pes = self.dps.compose_pe_from_multiple_ds()
        self.total_pairwise_distances = self.dps.compose_pairwise_distances_from_multiple_ds()

        # Meta-feature injection (FiLM at backbone exits, env-var gated; default OFF
        # so legacy behavior is preserved). Cache is loaded eagerly when ON;
        # normalizer is fit at the end of __init__ once self.train_idx is known.
        self.meta_injection = os.environ.get('NEURALGW_META_INJECTION', 'off').lower()
        if self.meta_injection not in ('off', 'film', 'mmd', 'film+mmd', 'concat', 'film+concat', 'mlpconcat', 'mmdmlp', 'mmdpair'):
            raise ValueError(
                f"Unknown NEURALGW_META_INJECTION: {self.meta_injection!r}. "
                f"Choose from: off, film, mmd, film+mmd, concat, film+concat, mlpconcat, mmdmlp, mmdpair."
            )
        if self.meta_injection != 'off':
            self.meta_active_keys = active_meta_keys()
            self.meta_dim = len(self.meta_active_keys)
            self.total_metas = self.dps.compose_meta_from_multiple_ds()
        else:
            self.meta_active_keys = []
            self.meta_dim = 0
        # End prepare for train/eval/test datasets

        # train_eval_idx, self.test_idx = train_test_split(range(self.total_pes.shape[0]), test_size=0.2, random_state=42)
        # self.train_idx, self.eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)
        if testset == 'global':
            train_eval_idx, self.test_idx = self.dps.train_test_split(testset=testset, test_ratio=config['test_ratio'], random_seed=42)
            self.train_idx, self.eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)
            print(f'[Info] test_idx length: {len(self.test_idx)}')
            print(f'[Info] eval_idx length: {len(self.eval_idx)}')
            print(f'[Info] train_idx length: {len(self.train_idx)}')
        elif testset == 'SBM':
            self.train_idx, self.eval_idx, self.test_idx = self.dps.train_test_split(testset=testset, test_ratio=0.2, random_seed=42)
            print(f'[Info] test_idx length: {len(self.test_idx)}')
            print(f'[Info] eval_idx length: {len(self.eval_idx)}')
            print(f'[Info] train_idx length: {len(self.train_idx)}')
            print(f'[Info] train_idx: {self.train_idx}')
            print(f'[Info] eval_idx: {self.eval_idx}')
            print(f'[Info] test_idx: {self.test_idx}')
        elif len([ds_name for ds_name in ds_name_seq if testset == ds_name]) == 1:
            self.train_idx, self.test_eval_idx = self.dps.train_test_split(testset=testset, test_ratio=0.2, random_seed=42)
            self.test_idx, self.eval_idx = train_test_split(self.test_eval_idx, test_size=0.2, random_state=42)
            # self.test_idx = self.test_idx[:188]
            # self.eval_idx = self.eval_idx[:188]
            print(f'[Info] train_idx length: {len(self.train_idx)}')
            print(f'[Info] test_idx length: {len(self.test_idx)}')
            print(f'[Info] eval_idx length: {len(self.eval_idx)}')
        elif len([ds_name for ds_name in ds_name_seq if testset == ds_name]) > 1:
            self.train_idx, self.eval_idx, self.test_idx = self.dps.train_test_split(testset=testset, test_ratio=0.2, random_seed=42)
            # print(f'[Warning] I will set self.test_idx = self.train_idx')
            # self.test_idx = self.train_idx
            print(f'[Info] train_idx length: {len(self.train_idx)}')
            print(f'[Info] test_idx length: {len(self.test_idx)}')
            print(f'[Info] eval_idx length: {len(self.eval_idx)}')
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

        # Finalize meta-feature wiring now that self.train_idx exists.
        # Normalizer is fit on the training pool only (for COLLAB -> REDDIT-BINARY,
        # self.train_idx is exactly the COLLAB training subset — see _utils_Processing
        # train_test_split, len(==testset) branch).
        if self.meta_injection != 'off':
            self._fit_meta_normalizer()
            if 'film' in self.meta_injection and hasattr(self.model, 'attach_film'):
                self.model.attach_film(meta_dim=self.meta_dim)
                # Ensure FiLM weights are on the same device as the rest of the model.
                self.model.to(device)
                print(f'[Info] FiLM heads attached: film_gin(out={self.model.hidden_dim}), '
                      f'film_attn(out={self.model.input_dim}); identity-at-init.')
            if 'concat' in self.meta_injection and 'mlpconcat' not in self.meta_injection \
                    and hasattr(self.model, 'attach_meta_concat'):
                self.model.attach_meta_concat(meta_dim=self.meta_dim)
                self.model.to(device)
                print(f'[Info] Meta-concat enabled: fc input widened by k={self.meta_dim} '
                      f'(merged dim {self.model.hidden_dim + 2*self.model.input_dim} -> '
                      f'{self.model.hidden_dim + 2*self.model.input_dim + self.meta_dim}).')
            if 'mlpconcat' in self.meta_injection and hasattr(self.model, 'attach_meta_mlpconcat'):
                mlp_out = int(os.environ.get('NEURALGW_META_MLP_DIM', '16'))
                self.model.attach_meta_mlpconcat(meta_dim=self.meta_dim, out_dim=mlp_out)
                self.model.to(device)
                print(f'[Info] Meta-MLP-concat enabled: meta {self.meta_dim} -> MLP -> {mlp_out}, '
                      f'fc input widened to {self.model.hidden_dim + 2*self.model.input_dim + mlp_out}; '
                      f'identity-at-init.')
            if 'mmdpair' in self.meta_injection and hasattr(self.model, 'attach_meta_mmd_pair'):
                self.model.attach_meta_mmd_pair(num_pair_scalars=2)
                self.model.to(device)
                print(f'[Info] Meta-MMD-pair enabled: head input widened '
                      f'{self.config["n_gammas"]} -> {self.config["n_gammas"] + 2} '
                      f'(append [node_sum, density_sum]).')
            if 'mmd' in self.meta_injection and 'mmdmlp' not in self.meta_injection \
                    and 'mmdpair' not in self.meta_injection \
                    and hasattr(self.model, 'attach_meta_mmd'):
                self.model.attach_meta_mmd(meta_dim=self.meta_dim)
                self.model.to(device)
                _dens_on = os.environ.get('NEURALGW_META_MMD_DENSITY_SUM', '1') != '0'
                _head_in = self.config["n_gammas"] + 2 * self.meta_dim + (1 if _dens_on else 0)
                print(f'[Info] Meta-MMD-inject enabled: head input widened '
                      f'{self.config["n_gammas"]} -> {_head_in} '
                      f'(append [|d_meta|, mean_meta]{" , pair_density_sum" if _dens_on else ""}).')
            if 'mmdmlp' in self.meta_injection and hasattr(self.model, 'attach_meta_mmd_mlp'):
                mlp_out = int(os.environ.get('NEURALGW_META_MLP_DIM', '16'))
                self.model.attach_meta_mmd_mlp(meta_dim=self.meta_dim, out_dim=mlp_out)
                self.model.to(device)
                _dens_on = os.environ.get('NEURALGW_META_MMD_DENSITY_SUM', '1') != '0'
                _head_in = self.config["n_gammas"] + 2 * mlp_out + (1 if _dens_on else 0)
                print(f'[Info] Meta-MMD-MLP-inject enabled: meta {self.meta_dim} -> MLP -> {mlp_out}, '
                      f'head input widened {self.config["n_gammas"]} -> {_head_in} '
                      f'(append [|d_enc|, mean_enc]{" , pair_density_sum" if _dens_on else ""}).')
            print(f'[Info] Meta injection: {self.meta_injection}, k={self.meta_dim}, '
                  f'mu/sigma fit on |train_idx|={len(self.train_idx)} graphs '
                  f'(features: {self.meta_active_keys}).')

    def _meta_vec_for(self, idx: int) -> np.ndarray:
        d = self.dps.meta_for_global_index(self.total_metas, idx)
        return np.array([float(d[k]) for k in self.meta_active_keys], dtype=np.float64)

    def _fit_meta_normalizer(self) -> None:
        """Compute (mu, sigma) per meta-feature on self.train_idx and stash on device.

        Mode selected by NEURALGW_META_NORM (default ``zscore``):
          zscore  – mu=mean, sigma=std.clamp_min(1e-6)
          robust  – mu=median, sigma=(p75 - p25) / 1.349, clamp_min(1e-6)
          none    – mu=0, sigma=1 (raw values)
        """
        mode = os.environ.get('NEURALGW_META_NORM', 'zscore').lower()
        rows = np.stack([self._meta_vec_for(idx) for idx in self.train_idx], axis=0)
        if mode == 'zscore':
            mu = rows.mean(axis=0)
            sigma = rows.std(axis=0)
            sigma = np.where(sigma < 1e-6, 1.0, sigma)
        elif mode == 'robust':
            mu = np.median(rows, axis=0)
            q25, q75 = np.percentile(rows, [25, 75], axis=0)
            sigma = (q75 - q25) / 1.349
            sigma = np.where(sigma < 1e-6, 1.0, sigma)
        elif mode == 'none':
            mu = np.zeros(rows.shape[1], dtype=np.float64)
            sigma = np.ones(rows.shape[1], dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown NEURALGW_META_NORM: {mode!r}. "
                f"Choose from: zscore, robust, none."
            )
        self.meta_mu = torch.tensor(mu, dtype=torch.float32, device=self.device)
        self.meta_sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        print(f'[Info] meta normalizer mode={mode} mu={mu} sigma={sigma}')

        # Pair-sum normalizers (used by the mmd-inject family). For a statistic x_i+x_j
        # with both graphs from the training pool: E=2*E[x], Var=2*Var[x] (independent).
        dens = np.array([self._raw_density(idx) for idx in self.train_idx], dtype=np.float64)
        self.dsum_mu = float(2.0 * dens.mean())
        ssig = float(np.sqrt(2.0) * dens.std())
        self.dsum_sigma = ssig if ssig > 1e-6 else 1.0
        nodes = np.array([self._num_nodes(idx) for idx in self.train_idx], dtype=np.float64)
        self.nsum_mu = float(2.0 * nodes.mean())
        nsig = float(np.sqrt(2.0) * nodes.std())
        self.nsum_sigma = nsig if nsig > 1e-6 else 1.0
        print(f'[Info] pair-sum normalizers: density mu={self.dsum_mu:.6f} sigma={self.dsum_sigma:.6f}; '
              f'nodes mu={self.nsum_mu:.4f} sigma={self.nsum_sigma:.4f}')

    def _raw_density(self, idx: int) -> float:
        """Un-normalized edge density for a graph; always present in the meta cache
        (full superset) regardless of NEURALGW_META_FEATURES."""
        d = self.dps.meta_for_global_index(self.total_metas, idx)
        return float(d.get('edge_density', 0.0))

    def _num_nodes(self, idx: int) -> int:
        """Node count for a graph (read from the loaded dataset)."""
        ds_idx, g_idx = self.dps.map_global_idx_to_local_indices(global_idx=idx)
        return int(self.ds_seq[ds_idx][g_idx].num_nodes)

    def _pair_scalar(self, idx1: int, idx2: int):
        """Mode-dependent symmetric per-pair scalar tensor appended at the MMD vector:
          mmdpair : [ norm(n1+n2), norm(d1+d2) ]    (2-dim)
          mmd/mmdmlp : [ norm(d1+d2) ]              (1-dim, unless density-sum disabled)
          otherwise : None
        """
        if 'mmdpair' in self.meta_injection:
            ns = (self._num_nodes(idx1) + self._num_nodes(idx2) - self.nsum_mu) / self.nsum_sigma
            ds = (self._raw_density(idx1) + self._raw_density(idx2) - self.dsum_mu) / self.dsum_sigma
            return torch.tensor([ns, ds], dtype=torch.float32, device=self.device)
        if 'mmd' in self.meta_injection:
            if os.environ.get('NEURALGW_META_MMD_DENSITY_SUM', '1') == '0':
                return None
            s = (self._raw_density(idx1) + self._raw_density(idx2) - self.dsum_mu) / self.dsum_sigma
            return torch.tensor([s], dtype=torch.float32, device=self.device)
        return None

    def _prepare_meta(self, idx: int):
        """Return a length-k normalized meta tensor, or None when meta is off."""
        if self.meta_injection == 'off':
            return None
        d = self.dps.meta_for_global_index(self.total_metas, idx)
        vec = torch.tensor(
            [float(d[k]) for k in self.meta_active_keys],
            dtype=torch.float32, device=self.device,
        )
        return (vec - self.meta_mu) / self.meta_sigma

    def _prepare_pair(self, idx1: int, idx2: int):
        pe1 = self._prepare_pe(idx1)
        ds_idx1, g_idx1 = self.dps.map_global_idx_to_local_indices(global_idx=idx1)
        edge_index1 = self.ds_seq[ds_idx1][g_idx1].edge_index.to(self.device)
        meta1 = self._prepare_meta(idx1)

        pe2 = self._prepare_pe(idx2)
        ds_idx2, g_idx2 = self.dps.map_global_idx_to_local_indices(global_idx=idx2)
        edge_index2 = self.ds_seq[ds_idx2][g_idx2].edge_index.to(self.device)
        meta2 = self._prepare_meta(idx2)
        pair_scalar = self._pair_scalar(idx1, idx2)
        return pe1, edge_index1, meta1, pe2, edge_index2, meta2, pair_scalar

    def _prepare_pe(self, idx: int) -> torch.Tensor:
        """Load PE for global graph index, pad/trim to input_dim, then normalize.

        Normalization is controlled by NEURALGW_PE_NORM (default: none,
        selected by the post-hoc calibration no-fix ablations):
          zscore_col    – per-column z-score per graph (original behavior)
          none          – raw PE, no normalization
          center_col    – subtract per-column mean only (no scaling)
          scale_col     – divide by per-column std only (no centering)
          zscore_scalar – one scalar mu/sigma over the entire PE matrix per graph
        """
        raw_pe = self.dps.pe_for_global_index(self.total_pes, idx)
        if raw_pe.shape[-1] < self.input_dim:
            raw_pe = np.pad(raw_pe, ((0, 0), (0, self.input_dim - raw_pe.shape[-1])))
        elif raw_pe.shape[-1] > self.input_dim:
            raw_pe = raw_pe[:, : self.input_dim]
        pe = torch.tensor(raw_pe, dtype=torch.float32, device=self.device)
        norm = os.environ.get('NEURALGW_PE_NORM', 'none').lower()
        if norm == 'none':
            return pe
        elif norm == 'zscore_col':
            mu = pe.mean(dim=0, keepdim=True)
            sigma = pe.std(dim=0, keepdim=True).clamp_min(1e-6)
            return (pe - mu) / sigma
        elif norm == 'center_col':
            return pe - pe.mean(dim=0, keepdim=True)
        elif norm == 'scale_col':
            sigma = pe.std(dim=0, keepdim=True).clamp_min(1e-6)
            return pe / sigma
        elif norm == 'zscore_scalar':
            return (pe - pe.mean()) / pe.std().clamp_min(1e-6)
        else:
            raise ValueError(
                f"Unknown NEURALGW_PE_NORM: {norm!r}. "
                f"Choose from: zscore_col, none, center_col, scale_col, zscore_scalar."
            )

    def compute_loss(self, gwd_pred: torch.Tensor, gwd_gt: np.floating) -> torch.Tensor:
        loss = (gwd_pred - gwd_gt) ** 2
        # gwd_pred = gwd_pred.sigmoid()
        # loss = -(gwd_gt * torch.log(gwd_pred) + (1 - gwd_gt) * torch.log(1 - gwd_pred))  # Binary Cross Entropy Loss
        return loss
    
    def create_train_state(self):
        # Allow NEURALGW_WEIGHT_DECAY to override the config (handy for ablations).
        _wd = float(os.environ.get('NEURALGW_WEIGHT_DECAY', self.config["weight_decay"]))
        print(f'[Info] AdamW weight_decay = {_wd:g} (lr={self.config["lr"]:g})')
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], betas=(0.9, 0.999), weight_decay=_wd)

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

            pe1, edge_index1, meta1, pe2, edge_index2, meta2, pair_scalar = self._prepare_pair(idx1, idx2)

            # forward pass
            gwd_pred = self.model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2, meta1=meta1, meta2=meta2, pair_scalar=pair_scalar)
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
        # sampler = torch.utils.data.distributed.DistributedSampler(self.train_idx, shuffle=True, drop_last=True, seed=self.config["seed"])
        # trainloader = torch.utils.data.DataLoader(self.train_idx, batch_size=self.config["batch_size"], drop_last=True, sampler=sampler)
        trainloader = torch.utils.data.DataLoader(self.train_idx, batch_size=self.config["batch_size"], shuffle=True, drop_last=True, num_workers=4, generator=g, worker_init_fn=seed_worker)

        self.create_train_state()

        best_eval_res = float('inf')
        stop_count = 0
        train_loss_seq = []
        train_res_seq = []
        ctime = timeit.default_timer()
        for epoch in range(self.config["num_epochs"]):
            # sampler.set_epoch(epoch)
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
            # if epoch % self.config["eval_interval"] == 0 and dist.get_rank() == 0:
            if epoch % self.config["eval_interval"] == 0:
                # eval_loss, eval_res = model.eval_model(config=config, model=model, ds=ds, eval_idx=eval_idx, ptr=ptr, pe=pe, D=D, device=device)
                eval_loss, eval_res = self.eval_model(eval_idx=self.eval_idx)
                time_elapsed = timeit.default_timer() - ctime
                # print(f'[{epoch:-3d}/{self.config["num_epochs"]}]({dist.get_rank()})({time_elapsed/60:.2f} m) [running_train_loss eval_loss train_res eval_res] = [{train_loss_seq[-1]:.8f} {eval_loss:.8f} {train_res_seq[-1]:.8f} {eval_res:.8f}]')
                print(f'[{epoch:-3d}/{self.config["num_epochs"]}]({self.device.index})({time_elapsed/60:.2f} m) [running_train_loss eval_loss train_res eval_res] = [{train_loss_seq[-1]:.8f} {eval_loss:.8f} {train_res_seq[-1]:.8f} {eval_res:.8f}]')
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
                    
                # Early stopping (toggleable). Set NEURALGW_DISABLE_EARLY_STOP=1 to
                # train the full num_epochs regardless of patience. Best-model-by-eval
                # selection still applies, so test uses the best checkpoint seen.
                if stop_count >= self.config["early_stop_patience"]:
                    if os.environ.get('NEURALGW_DISABLE_EARLY_STOP', '0') == '1':
                        print(f'[Info] Early-stop criterion met (stop_count={stop_count}) but '
                              f'NEURALGW_DISABLE_EARLY_STOP=1 — continuing to full {self.config["num_epochs"]} epochs.')
                    else:
                        print(f'[Info] Early stopping triggered. Stopping training...')
                        break

        # test the model
        # if dist.get_rank() == 0:
        test_loss, test_res = self.test_model(test_idx=self.test_idx)
        print(f'Test loss: {test_loss:.8f}, Test result: {test_res:.8f}')

        # self.save_distance_matrix_on_testset()
        
        # # save the best model state
        trainset_str = '_'.join([ds_name for ds_name in self._ds_name_seq if ds_name != self.testset])
        testset_str = self.testset
        checkpoint_path = f'checkpoints/{trainset_str}_{testset_str}_model_gwidth{self.config["gwidth"]}_epoch{epoch}_cuda{self.config["cuda_id"]}.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.best_model_state, checkpoint_path)

    def eval_model(self, eval_idx: List[int]) -> Tuple[float, float]:
        # model.eval()
        self.eval()
        with torch.no_grad():
            _, (eval_loss_seq, eval_gwd_pred_seq, eval_gwd_gt_seq) = self.train_step(batch_idx=eval_idx)
            eval_loss = sum(eval_loss_seq).item() / len(eval_loss_seq)
            preds_arr = np.array(eval_gwd_pred_seq)
            if hasattr(self, 'calib_a'):
                preds_arr = self.calib_a * preds_arr + self.calib_b
            eval_res = np.linalg.vector_norm(preds_arr - np.array(eval_gwd_gt_seq), ord=2) / np.linalg.vector_norm(np.array(eval_gwd_gt_seq), ord=2)

            # sampler = torch.utils.data.distributed.DistributedSampler(eval_idx, shuffle=False, drop_last=True, seed=self.config["seed"])
            # dataloader = torch.utils.data.DataLoader(eval_idx, batch_size=self.config["batch_size"], drop_last=True, sampler=sampler)
            # eval_loss_seq = []
            # eval_gwd_pred_seq = []
            # eval_gwd_gt_seq = []
            # for batch_idx in tqdm(dataloader):
            #     _, (batch_loss_seq, gwd_pred_seq_per_batch, gwd_gt_seq_per_batch) = self.train_step(batch_idx=batch_idx)
            #     # print(f'[Debug] batch_loss_seq: {batch_loss_seq}')
            #     eval_loss_seq.extend(batch_loss_seq)
            #     eval_gwd_pred_seq.extend(gwd_pred_seq_per_batch)
            #     eval_gwd_gt_seq.extend(gwd_gt_seq_per_batch)

            # # print(f'[Debug] eval_loss_seq: {eval_loss_seq}')
            # # print(f'[Debug] sum(eval_loss_seq)={sum(eval_loss_seq).item()}, len(eval_loss_seq)={len(eval_loss_seq)}')
            # eval_loss = sum(eval_loss_seq).item() / len(eval_loss_seq)
            # eval_res = np.linalg.vector_norm(np.array(eval_gwd_pred_seq) - np.array(eval_gwd_gt_seq), ord=2) / np.linalg.vector_norm(np.array(eval_gwd_gt_seq), ord=2)

        return eval_loss, eval_res

    def _fit_calibration(self) -> None:
        """Fit post-hoc calibration controlled by NEURALGW_POSTHOC_CALIB.

        Modes:
          off             - use raw predictions (default, as in the paper)
          shift           - fit y = pred + b on eval_idx
          scale           - fit y = a * pred on eval_idx
          affine          - fit y = a * pred + b on eval_idx
          affine_trainfit - fit y = a * pred + b on a deterministic train subset
        """
        mode = os.environ.get('NEURALGW_POSTHOC_CALIB', 'off').lower()
        if mode == 'off':
            if hasattr(self, 'calib_a'):
                del self.calib_a
            if hasattr(self, 'calib_b'):
                del self.calib_b
            print('[Info] Post-hoc calibration: OFF (raw predictions).')
            return

        if mode == 'affine_trainfit':
            rng = np.random.RandomState(self.config['seed'])
            k = min(len(self.eval_idx), len(self.train_idx))
            fit_idx = list(rng.choice(self.train_idx, size=k, replace=False))
        else:
            fit_idx = self.eval_idx

        # strict=False so a legacy (meta-off) checkpoint loads cleanly into a
        # meta-on model; FiLM weights stay at their identity-init values.
        self.model.load_state_dict(self.best_model_state, strict=False)
        self.eval()
        with torch.no_grad():
            _, (_, preds, gts) = self.train_step(batch_idx=fit_idx)
        preds = np.asarray(preds, dtype=np.float64)
        gts = np.asarray(gts, dtype=np.float64)

        if mode == 'shift':
            a, b = 1.0, float(np.mean(gts - preds))
        elif mode == 'scale':
            # how to compute the scale?
            # we can use the std of the preds
            # 
            denom = float(np.sum(preds * preds))
            a = float(np.sum(preds * gts) / denom) if denom > 1e-12 else 1.0
            b = 0.0
        elif mode in ('affine', 'affine_trainfit'):
            if len(preds) >= 2 and np.std(preds) > 1e-12:
                a, b = np.polyfit(preds, gts, 1)
            else:
                a, b = 1.0, 0.0
            a, b = float(a), float(b)
        else:
            raise ValueError(
                f"Unknown NEURALGW_POSTHOC_CALIB: {mode!r}. "
                f"Choose from: off, shift, scale, affine, affine_trainfit."
            )

        self.calib_a, self.calib_b = a, b
        print(f'[Info] Post-hoc calibration ({mode}): a={a:.6f}, b={b:.6f}')

    def test_model(self, test_idx: List[int]) -> Tuple[float, float]:
        self.model.load_state_dict(self.best_model_state, strict=False)
        self._fit_calibration()
        test_loss, test_res = self.eval_model(eval_idx=test_idx)
        return test_loss, test_res

    def save_distance_matrix_on_testset(self):
        # load the best model state
        self.model.load_state_dict(self.best_model_state, strict=False)
        self.eval()
        # prepare the pairwise distance matrix on the test set
        if len([ds_name for ds_name in self._ds_name_seq if self.testset == ds_name]) == 1:
            pass
        elif len([ds_name for ds_name in self._ds_name_seq if self.testset == ds_name]) > 1:
            self.test_eval_idx = self.test_idx
        else:
            self.test_eval_idx = self.test_idx

        ################################################ 
        # Warning: I will add a new piece of code here to save the distance matrix on the whole dataset
        ################################################ 
        if self._ds_name_seq == ['SBM']:
            self.test_eval_idx = sorted(self.train_idx + self.eval_idx + self.test_idx)
        ################################################
        # End of the new piece of code
        ################################################

        test_size = len(self.test_eval_idx)
        pairwise_distance_matrix = np.zeros((test_size, test_size))
        with torch.no_grad():
            for i in tqdm(range(test_size)):
                for j in range(i+1, test_size):
                    idx1, idx2 = self.test_eval_idx[i], self.test_eval_idx[j]
                    pe1, edge_index1, meta1, pe2, edge_index2, meta2, pair_scalar = self._prepare_pair(idx1, idx2)

                    # forward pass; apply linear calibration if available
                    gwd_pred_val = self.model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2, meta1=meta1, meta2=meta2, pair_scalar=pair_scalar).item()
                    if hasattr(self, 'calib_a'):
                        gwd_pred_val = self.calib_a * gwd_pred_val + self.calib_b
                    pairwise_distance_matrix[i, j] = gwd_pred_val
                    pairwise_distance_matrix[j, i] = gwd_pred_val

        # save the pairwise distance matrix on the test set
        trainset = '_'.join([ds_name for ds_name in self._ds_name_seq if ds_name != self.testset])
        fname = f'results/clustering/{trainset}_to_{self.testset}_seed{self.config["seed"]}_distance_matrix.npy'
        with open(fname, 'wb') as f:
            np.save(f, pairwise_distance_matrix)
            print(f'[Info] Distance matrix on the test set under {self._ds_name_seq} has been saved to {fname}.')


################################################################################################################################
# I want to realize another NeuralGW_GPS under the GraphGPS architecture variants here
################################################################################################################################
class NeuralGW_GPS(NeuralGW):
    def __init__(self, ds_name_seq: List[str], testset: str = 'global', is_naive: bool = False,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(NeuralGW_GPS, self).__init__(ds_name_seq=ds_name_seq, testset=testset, is_naive=is_naive, config=config, device=device)
        if not is_naive:
            from ._utils_Architectures import GINTransformer_GPS
            self.model = GINTransformer_GPS(input_dim=self.input_dim,
                                            hidden_dim=config["gwidth"],
                                            gin_layers=config["num_gin_layers"],
                                            transformer_layers=config["num_transformer_layers"],
                                            num_gps_layers=config["num_gps_layers"],
                                            n_gammas=config['n_gammas'],
                                            config=config,
                                            device=device).to(device)
        else:
            from ._utils_Architectures import GINTransformer_GPS_Naive
            self.model = GINTransformer_GPS_Naive(input_dim=self.input_dim,
                                                  hidden_dim=config["gwidth"],
                                                  gin_layers=config["num_gin_layers"],
                                                  transformer_layers=config["num_transformer_layers"],
                                                  num_gps_layers=config["num_gps_layers"],
                                                  n_gammas=config['n_gammas'],
                                                  config=config,
                                                  device=device).to(device)

################################################################################################################################
# End of another NeuralGW_GPS
################################################################################################################################
    
class NeuralGW_Ablation(NeuralGW):
    def __init__(self, ds_name_seq: List[str], testset: str = 'global', is_naive: bool = False,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(NeuralGW_Ablation, self).__init__(ds_name_seq=ds_name_seq, testset=testset, is_naive=is_naive, config=config, device=device)
        from ._utils_Architectures import GINTransformer_Ablation
        self.model = GINTransformer_Ablation(input_dim=self.input_dim,
                                                hidden_dim=config["gwidth"],
                                                gin_layers=config["num_gin_layers"],
                                                transformer_layers=config["num_transformer_layers"],
                                                n_gammas=config['n_gammas'],
                                                config=config,
                                                device=device).to(device)

class NeuralGW_TP(nn.Module):
    def __init__(self, ds_name_seq: List[str], testset: str = 'global', is_naive: bool = False,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(NeuralGW_TP, self).__init__()
        self._ds_name_seq = ds_name_seq
        self.testset = testset
        self.config = config
        self.device = device

        # prepare for train/eval/test datasets
        self.dps = DataPreparationSystem(
            ds_name_seq=ds_name_seq,
            pe_cost_mode=config['pe_cost_mode'],
            gw_cost_mode=config['gw_cost_mode'],
            pe_mode=config['pe_mode'],
            treat_as_undirected=config.get('treat_as_undirected', True),
        )
        self.ds_seq = self.dps.load_all_graph_ds()
        self.total_pes = self.dps.compose_pe_from_multiple_ds()
        self.total_transport_plans = self.dps.compose_transport_plans_from_multiple_ds()
        # End prepare for train/eval/test datasets

        # train_eval_idx, self.test_idx = train_test_split(range(self.total_pes.shape[0]), test_size=0.2, random_state=42)
        # self.train_idx, self.eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)
        if testset == 'global':
            train_eval_idx, self.test_idx = self.dps.train_test_split(testset=testset, test_ratio=config['test_ratio'], random_seed=42)
            self.train_idx, self.eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)
            self.eval_idx = train_eval_idx + self.test_idx  # use more eval data for better model selection
            print(f'[Info] train_idx length: {len(self.train_idx)}')
            print(f'[Info] eval_idx length: {len(self.eval_idx)}')
            print(f'[Info] test_idx length: {len(self.test_idx)}')
        elif testset in ds_name_seq:
            self.train_idx, test_eval_idx = self.dps.train_test_split(testset=testset, test_ratio=0.2, random_seed=42)
            self.test_idx, self.eval_idx = train_test_split(test_eval_idx, test_size=0.2, random_state=42)
        else:
            raise ValueError(f"testset {testset} not recognized. It should be 'global' or one of {ds_name_seq}.")

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
        # loss = torch.linalg.vector_norm(tp_pred - tp_gt, ord=2)
        return loss
    
    def create_train_state(self):
        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], betas=(0.9, 0.999), weight_decay=self.config["weight_decay"])
        gamma_group = {"params": self.model.kernel_layer.parameters(), "lr": self.config["lr"] * 1000}
        main_group = {"params": [param for name, param in self.model.named_parameters() if "kernel_layer" not in name], "lr": self.config["lr"], "weight_decay": self.config["weight_decay"]}
        self.optimizer = torch.optim.AdamW([gamma_group, main_group], betas=(0.9, 0.999))

    def train_step(self, batch_idx: List[int]) -> Tuple[torch.Tensor, Tuple[float, float]]:
        tri_u_ind = np.stack(np.triu_indices(len(batch_idx), k=1), axis=1)
        batch_loss_seq = []
        train_res_seq_per_batch = []
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
            tp_pred = self.model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2)
            tp_gt = torch.tensor(self.total_transport_plans[(ds_idx1, ds_idx2)][(g_idx1, g_idx2)], dtype=torch.float32, device=self.device)
            # tp_pred = tp_pred.sigmoid()
            # tp_pred = tp_pred / tp_pred.sum(dim=1, keepdim=True)
            # tp_pred = tp_pred / tp_pred.sum(dim=0, keepdim=True)
            # tp_pred = tp_pred / tp_pred.sum(dim=1, keepdim=True)
            # tp_pred = tp_pred / tp_pred.sum(dim=0, keepdim=True)
            loss = self.compute_loss(tp_pred, tp_gt)
            tp_pred = tp_pred.sigmoid()
            # tp_pred = tp_pred / tp_pred.sum(dim=1, keepdim=True)
            # tp_pred = tp_pred / tp_pred.sum(dim=0, keepdim=True)
            res = torch.linalg.vector_norm(tp_pred - tp_gt, ord=1) / torch.linalg.vector_norm(tp_gt, ord=1)

            # keep track of the loss, tp_pred, and tp_gt
            batch_loss_seq.append(loss)
            train_res_seq_per_batch.append(res.item())

        # backward pass
        batch_loss = sum(batch_loss_seq) / len(batch_loss_seq)
        # train_res_per_batch = sum(train_res_seq_per_batch) / len(train_res_seq_per_batch)
        return batch_loss, (batch_loss_seq, train_res_seq_per_batch)

    def train_model(self, seed_worker: Callable[[int], None]) -> None:
        g = torch.Generator()
        g.manual_seed(self.config["seed"])
        trainloader = torch.utils.data.DataLoader(self.train_idx, batch_size=self.config["batch_size"], shuffle=True, drop_last=True, num_workers=4, generator=g, worker_init_fn=seed_worker)
        # sampler = torch.utils.data.distributed.DistributedSampler(self.train_idx, shuffle=True, drop_last=True, seed=self.config["seed"])
        # trainloader = torch.utils.data.DataLoader(self.train_idx, batch_size=self.config["batch_size"], drop_last=True, sampler=sampler)

        self.create_train_state()

        best_eval_res = float('inf')
        train_loss_seq = []
        train_res_seq = []
        ctime = timeit.default_timer()
        for epoch in range(self.config["num_epochs"]):
            # sampler.set_epoch(epoch)
            self.train()
            train_loss_seq_per_epoch = []
            train_res_seq_per_epoch = []
            for batch_idx in trainloader:
                self.zero_grad()
                batch_loss, (batch_loss_seq, train_res_seq_per_batch) = self.train_step(batch_idx=batch_idx)
                batch_loss.backward()
                self.optimizer.step()

                
                # keep track of the average loss and the whole relative residual of error under the current batch
                batch_loss_mean = sum(batch_loss_seq).item() / len(batch_loss_seq)
                batch_tp_res_mean = sum(train_res_seq_per_batch) / len(train_res_seq_per_batch)
                train_loss_seq_per_epoch.append(batch_loss_mean)
                train_res_seq_per_epoch.append(batch_tp_res_mean)

            # keep track of the average loss and the whole relative residual of error under the current epoch
            train_loss_seq.append(sum(train_loss_seq_per_epoch) / len(train_loss_seq_per_epoch))
            train_res_seq.append(sum(train_res_seq_per_epoch) / len(train_res_seq_per_epoch))
            time_elapsed = timeit.default_timer() - ctime
            print(f'[{epoch:-3d}/{self.config["num_epochs"]}]({time_elapsed/60:.2f} m) [running_train_loss running_train_res] = [{train_loss_seq[-1]:.8f} {train_res_seq[-1]:.8f}]')

            # eval the model
            stop_flag = torch.tensor(0, device=self.device)
            # if epoch % self.config["eval_interval"] == 0 and dist.get_rank() == 0:
            if epoch % self.config["eval_interval"] == 0:
                # eval_loss, eval_res = model.eval_model(config=config, model=model, ds=ds, eval_idx=eval_idx, ptr=ptr, pe=pe, D=D, device=device)
                eval_loss, eval_res = self.eval_model(eval_idx=self.eval_idx)
                time_elapsed = timeit.default_timer() - ctime
                print(f'[{epoch:-3d}/{self.config["num_epochs"]}]({time_elapsed/60:.2f} m) [running_train_loss eval_loss train_res eval_res] = [{train_loss_seq[-1]:.8f} {eval_loss:.8f} {train_res_seq[-1]:.8f} {eval_res:.8f}]')
                if hasattr(self.model, 'kernel_layer'):
                    print(f'[Info] current raw_gammas: {self.model.kernel_layer._raw_gammas.detach().cpu().numpy()}')
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
                    # stop_flag.fill_(1)

            # #  broadcast the stop_flag to all processes
            # if dist.is_initialized():
            #     dist.broadcast(stop_flag, src=0)

            # if stop_flag.item() == 1:
            #     print(f'[Info] Rank {dist.get_rank()} received stop signal. Stopping training...')
            #     break

        # test the model
        # if dist.get_rank() == 0:
        test_loss, test_res = self.test_model(test_idx=self.test_idx)
        print(f'Test loss: {test_loss:.8f}, Test result: {test_res:.8f}')
            
        # save the best model state
        trainset_str = '_'.join([ds_name for ds_name in self._ds_name_seq if ds_name != self.testset])
        testset_str = self.testset
        checkpoint_path = f'checkpoints/exp_indomain_tp/{trainset_str}_{testset_str}_model_gwidth{self.config["gwidth"]}_epoch{epoch}_cuda{self.config["cuda_id"]}.pth'
        torch.save(self.best_model_state, checkpoint_path)

    def eval_model(self, eval_idx: List[int]) -> Tuple[float, float]:
        # model.eval()
        self.eval()
        with torch.no_grad():
            _, (eval_loss_seq, eval_res_seq) = self.train_step(batch_idx=eval_idx)
            eval_loss = sum(eval_loss_seq).item() / len(eval_loss_seq)
            eval_res = sum(eval_res_seq) / len(eval_res_seq)
            # sampler = torch.utils.data.distributed.DistributedSampler(eval_idx, shuffle=False, drop_last=True, seed=self.config["seed"])
            # dataloader = torch.utils.data.DataLoader(eval_idx, batch_size=self.config["batch_size"], drop_last=True, sampler=sampler)
            # eval_loss_seq = []
            # eval_res_seq = []
            # # eval_loss = torch.tensor(0.0, device=self.device)
            # # eval_res = torch.tensor(0.0, device=self.device)
            # for batch_idx in tqdm(dataloader):
            #     _, (batch_loss_seq, eval_res_seq_per_batch) = self.train_step(batch_idx=batch_idx)
            #     eval_loss_seq.extend(batch_loss_seq)
            #     eval_res_seq.extend(eval_res_seq_per_batch)
            #     # eval_loss += sum(batch_loss_seq) / len(batch_loss_seq)
            #     # eval_res += sum(eval_res_seq_per_batch) / len(eval_res_seq_per_batch)

            # # if dist.is_initialized():
            # #     dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
            # #     dist.all_reduce(eval_res, op=dist.ReduceOp.SUM)

            # eval_loss = sum(eval_loss_seq).item() / len(eval_loss_seq)
            # eval_res = sum(eval_res_seq) / len(eval_res_seq)
            # # eval_loss = eval_loss.item() / len(dataloader)
            # # eval_res = eval_res.item() / len(dataloader)

        return eval_loss, eval_res

    def test_model(self, test_idx: List[int]) -> Tuple[float, float]:
        # model.load_state_dict(best_model_state)
        self.model.load_state_dict(self.best_model_state)
        test_loss, test_res = self.eval_model(eval_idx=test_idx)
        return test_loss, test_res