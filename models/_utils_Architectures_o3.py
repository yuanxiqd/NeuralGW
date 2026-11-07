import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
import timeit
from typing import List, Dict, Callable, Tuple
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np
from torch_geometric.data import Dataset
from utils import funcs
from ._utils_Processing import DataPreparationSystem

class GINTransformer(nn.Module):
    def __init__(self,
                 input_dim: int = None, hidden_dim: int = None,
                 gin_layers: int = 3, transformer_layers: int = 1,
                 n_gammas=9,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(GINTransformer, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gin_layers = gin_layers
        self.transformer_layers = transformer_layers
        self.n_gammas = n_gammas
        self.device = device

        # define the GIN layers
        self.gin_layers_list = nn.ModuleList()
        for _ in range(gin_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim if _ == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers_list.append(GINConv(mlp))

        # define the transformer layers
        self.attn_layer_for_serial = nn.TransformerEncoderLayer(
            d_model=hidden_dim+input_dim,
            nhead=4
        )

        self.attn_layer_for_parallel = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4
        )

        self.fc = nn.Linear(hidden_dim + 2*input_dim, 128)

        # define the intermediate MMD layers
        self.mmd_layer = MMDLayer(n_gammas=n_gammas)

        # define the last MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(n_gammas, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

        # initialize weights
        self._init_weights(self.gin_layers_list)
        self._init_weights(self.attn_layer_for_serial)
        self._init_weights(self.attn_layer_for_parallel)
        self._init_weights(self.mlp_head)

    def build_serial_model(self):
        pass

    def build_parallel_model(self):
        pass

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward_gin(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # apply GIN layers:
        for gin_layer in self.gin_layers_list:
            if gin_layer == self.gin_layers_list[0]:
                gin_z = gin_layer(X, edge_index)
                gin_z = F.relu(gin_z)
            else:
                gin_z = gin_layer(gin_z, edge_index)
                gin_z = F.relu(gin_z)
        return gin_z

    def forward_siamese_serial(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor, is_whole_attn: bool = True) -> torch.Tensor:
        # feedforward two graphs through GIN layers, respectively
        gin_Z1 = self.forward_gin(X=X1, edge_index=edge_index1)
        gin_Z2 = self.forward_gin(X=X2, edge_index=edge_index2)
        gin_Z1 = torch.cat((gin_Z1, X1), dim=1)
        gin_Z2 = torch.cat((gin_Z2, X2), dim=1)
        
        #  apply transformer layers:
        if is_whole_attn:
            attn_Z = F.relu(self.attn_layer_for_serial(torch.cat((gin_Z1, gin_Z2), dim=0)))
            # attn_Z = torch.cat((attn_Z, torch.cat((X1, X2), dim=0)), dim=1)
            # retrieve the attention scores
            attn_Z1 = attn_Z[:gin_Z1.size(0)]
            attn_Z2 = attn_Z[gin_Z1.size(0):]
        else:
            attn_Z1 = F.relu(self.attn_layer_for_serial(gin_Z1))
            attn_Z2 = F.relu(self.attn_layer_for_serial(gin_Z2))

        # combine outputs from GIN and transformer layers
        # Z1 = torch.cat((gin_Z1, attn_Z1), dim=1)
        # Z2 = torch.cat((gin_Z2, attn_Z2), dim=1)
        Z1, Z2 = attn_Z1, attn_Z2
        return Z1, Z2
    
    def forward_siamese_parallel(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor, is_whole_attn: bool = False) -> torch.Tensor:
        # feedforward two graphs through GIN layers, respectively
        gin_Z1 = self.forward_gin(X=X1, edge_index=edge_index1)
        gin_Z2 = self.forward_gin(X=X2, edge_index=edge_index2)
        gin_Z1 = torch.cat((gin_Z1, X1), dim=1)
        gin_Z2 = torch.cat((gin_Z2, X2), dim=1)
        
        #  apply transformer layers:
        if is_whole_attn:
            attn_Z = F.relu(self.attn_layer_for_parallel(torch.cat((X1, X2), dim=0)))
            # retrieve the attention scores
            attn_Z1 = attn_Z[:gin_Z1.size(0)]
            attn_Z2 = attn_Z[gin_Z1.size(0):]
        else:
            attn_Z1 = F.relu(self.attn_layer_for_parallel(X1))
            attn_Z2 = F.relu(self.attn_layer_for_parallel(X2))

        # combine outputs from GIN and transformer layers
        Z1 = torch.cat((gin_Z1, attn_Z1), dim=1)
        Z2 = torch.cat((gin_Z2, attn_Z2), dim=1)
        Z1, Z2 = self.fc(Z1), self.fc(Z2)
        return Z1, Z2

    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        # feedforward it into the backbone - A hybrid model of GIN and Transformer
        # Z1, Z2 = self.forward_siamese_serial(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)

        # feedforward it into MMD layer
        o = self.mmd_layer.forward(X1=Z1, X2=Z2)

        # feedforward it into MLP head
        o = self.mlp_head(o)

        return o
    

class MMDLayer(nn.Module):
    def __init__(self, n_gammas: int = 9):
        super(MMDLayer, self).__init__()
        assert n_gammas%2 == 1, f"The width of MMD layer: 'n_gammas={n_gammas}' must be odd."
        self.n_gammas = n_gammas
        # self._raw_gammas = nn.Linear(n_gammas, 1, bias=False).weight.view(-1)
        self._raw_gamma_base = nn.Parameter(torch.tensor(0.001))
        # self._raw_gamma_bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, X1: torch.Tensor, X2: torch.Tensor, is_biased: bool = False) -> torch.Tensor:
        assert X1.dim() == 2 and X2.dim() == 2, "Input 'X1' and 'X2' must be 2D Tensors"
        n1, n2 = X1.shape[0], X2.shape[0]
        M = torch.cat((X1, X2), dim=0)
        D = torch.cdist(M, M, compute_mode='use_mm_for_euclid_dist')
        # print(f'Estimated gamma of gaussian kernel: {1/(2* D.mean().item()**2):.4f}')
        D2 = D**2
        # self._raw_gamma_base = 1/(2 * D2.mean().item()**2)
        self._raw_gammas = [self._raw_gamma_base * c**2 for c in range(self.n_gammas)]
        # self._raw_gammas = [self._raw_gamma_base * 2**(i + 1) for i in range(-self.n_gammas//2, self.n_gammas//2)]
        # self._raw_gammas = [self._raw_gamma_base/70, self._raw_gamma_base/50, self._raw_gamma_base/30, self._raw_gamma_base/10, self._raw_gamma_base, self._raw_gamma_base*10, self._raw_gamma_base*30, self._raw_gamma_base*50, self._raw_gamma_base*70]
        # self._raw_gammas = [self._raw_gamma_base/50, self._raw_gamma_base/30, self._raw_gamma_base/10, self._raw_gamma_base, self._raw_gamma_base*10, self._raw_gamma_base*30, self._raw_gamma_base*50]
        # self._raw_gammas = [self._raw_gamma_base/30, self._raw_gamma_base/10, self._raw_gamma_base, self._raw_gamma_base*10, self._raw_gamma_base*30]
        # self._raw_gammas = [self._raw_gamma_base/10, self._raw_gamma_base, self._raw_gamma_base*10]
        mmd_seq = []
        for raw_gamma in self._raw_gammas:
            # positive_gamma = F.softplus(raw_gamma)
            positive_gamma = raw_gamma
            K = torch.exp(-positive_gamma * D2)
            K11, K12, K22 = K[:n1, :n1], K[:n1, n1:], K[n1:, n1:]
            if is_biased:
                mmd = K11.sum()/n1**2 - 2*K12.sum()/(n1*n2) + K22.sum()/n2**2
            else:
                mmd = (K11.sum() - n1)/(n1*(n1 - 1)) - 2*K12.sum()/(n1*n2) + (K22.sum() - n2)/(n2*(n2 - 1))
            mmd_seq.append(mmd)
        return torch.stack(mmd_seq)
    
class GINTransformer_Naive(GINTransformer):
    def __init__(self,
                 input_dim: int, hidden_dim: int,
                 gin_layers: int = 3, transformer_layers: int = 1,
                 n_gammas=9,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(GINTransformer_Naive, self).__init__(input_dim=input_dim,
                                                   hidden_dim=hidden_dim,
                                                   gin_layers=gin_layers,
                                                   transformer_layers=transformer_layers,
                                                   n_gammas=n_gammas,
                                                   config=config,
                                                   device=device)
        from torch_geometric.nn import global_mean_pool
        self.readout = global_mean_pool
        
    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        # feedforward it into the backbone - A hybrid model of GIN and Transformer
        # Z1, Z2 = self.forward_siamese_serial(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)

        z1 = self.readout(Z1)
        z2 = self.readout(Z2)
        o = torch.linalg.vector_norm(z1 - z2)**2

        return o
    

class GINTransformer_TP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, gin_layers: int = 3, transformer_layers: int = 1, n_gammas=9):
        super(GINTransformer_TP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gin_layers = gin_layers
        self.transformer_layers = transformer_layers
        self.n_gammas = n_gammas

        # define the GIN layers
        self.gin_layers_list = nn.ModuleList()
        for _ in range(gin_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim if _ == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers_list.append(GINConv(mlp))

        # define the transformer layers
        self.attn_layer_for_serial = nn.TransformerEncoderLayer(
            d_model=hidden_dim + input_dim,
            nhead=4
        )

        self.attn_layer_for_parallel = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4
        )

        self.fc = nn.Linear(hidden_dim + input_dim, 128)

        # define the intermediate kernel layers
        self.kernel_layer = KernelLayer(n_gammas=n_gammas)

        # define the last MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(n_gammas, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # initialize weights
        # self._init_weights(self.gin_layers_list)
        # self._init_transformer_weights(self.attn_layer_for_serial)
        # self._init_transformer_weights(self.attn_layer_for_parallel)
        # self._init_weights(self.mlp_head)

    @staticmethod
    def _init_weights(layer):
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    @staticmethod
    def _init_transformer_weights(layer):
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_gin(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # apply GIN layers:
        for gin_layer in self.gin_layers_list:
            if gin_layer == self.gin_layers_list[0]:
                gin_z = gin_layer(X, edge_index)
                gin_z = F.relu(gin_z)
            else:
                gin_z = gin_layer(gin_z, edge_index)
                gin_z = F.relu(gin_z)
        return gin_z

    def forward_siamese_serial(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor, is_whole_attn: bool = True) -> torch.Tensor:
        # feedforward two graphs through GIN layers, respectively
        gin_Z1 = self.forward_gin(X=X1, edge_index=edge_index1)
        gin_Z2 = self.forward_gin(X=X2, edge_index=edge_index2)
        gin_Z1 = torch.cat((gin_Z1, X1), dim=1)
        gin_Z2 = torch.cat((gin_Z2, X2), dim=1)
        
        #  apply transformer layers:
        if is_whole_attn:
            attn_Z = F.relu(self.attn_layer_for_serial(torch.cat((gin_Z1, gin_Z2), dim=0)))
            # attn_Z = torch.cat((attn_Z, torch.cat((X1, X2), dim=0)), dim=1)
            # retrieve the attention scores
            attn_Z1 = attn_Z[:gin_Z1.size(0)]
            attn_Z2 = attn_Z[gin_Z1.size(0):]
        else:
            attn_Z1 = F.relu(self.attn_layer_for_serial(gin_Z1))
            attn_Z2 = F.relu(self.attn_layer_for_serial(gin_Z2))

        # combine outputs from GIN and transformer layers
        # Z1 = torch.cat((gin_Z1, attn_Z1), dim=1)
        # Z2 = torch.cat((gin_Z2, attn_Z2), dim=1)
        Z1, Z2 = attn_Z1, attn_Z2
        return Z1, Z2
    
    def forward_siamese_parallel(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor, is_whole_attn: bool = False) -> torch.Tensor:
        # feedforward two graphs through GIN layers, respectively
        gin_Z1 = self.forward_gin(X=X1, edge_index=edge_index1)
        gin_Z2 = self.forward_gin(X=X2, edge_index=edge_index2)
        # gin_Z1 = torch.cat((gin_Z1, X1), dim=1)
        # gin_Z2 = torch.cat((gin_Z2, X2), dim=1)
        
        #  apply transformer layers:
        if is_whole_attn:
            attn_Z = F.relu(self.attn_layer_for_parallel(torch.cat((X1, X2), dim=0)))
            # retrieve the attention scores
            attn_Z1 = attn_Z[:gin_Z1.size(0)]
            attn_Z2 = attn_Z[gin_Z1.size(0):]
        else:
            attn_Z1 = F.relu(self.attn_layer_for_parallel(X1))
            attn_Z2 = F.relu(self.attn_layer_for_parallel(X2))

        # combine outputs from GIN and transformer layers
        Z1 = torch.cat((gin_Z1, attn_Z1), dim=1)
        Z2 = torch.cat((gin_Z2, attn_Z2), dim=1)
        Z1, Z2 = self.fc(Z1), self.fc(Z2)
        return Z1, Z2

    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        # feedforward it into the backbone - A hybrid model of GIN and Transformer
        Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)

        # feedforward it into kernel layer
        Kernel_Z = self.kernel_layer.forward(X1=Z1, X2=Z2)

        # feedforward it into MLP head
        o = self.mlp_head(Kernel_Z)
        tp_pred = o.reshape(Z1.shape[0], Z2.shape[0])

        return tp_pred

    def train_model(self, config: Dict, model: nn.Module, ds: Dataset, optimizer: torch.optim.Optimizer, device: torch.device, seed_worker: Callable[[int], None]) -> None:
        g = torch.Generator()
        g.manual_seed(config["seed"])
        model.train()
        # create a DataLoader for the dataset
        train_eval_idx, test_idx = train_test_split(range(len(ds)), test_size=0.2, random_state=42)
        train_idx, eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)
        trainloader = torch.utils.data.DataLoader(train_idx, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=4, generator=g, worker_init_fn=seed_worker)
        # load the graph pointers
        ptr = funcs.load_graph_ptr(ds_name=config["ds_name"])
        # load the graph positional encodings
        pe = funcs.load_graph_pe(ds_name=config["ds_name"], mode=config["pe_mode"])
        # load the ground truth distance matrix
        # D = funcs.load_distance_matrix(ds_name=config["ds_name"], metric=config["metric"], mode=config["mode"])
        P = funcs.load_transport_plans(ds_name=config["ds_name"], metric=config["metric"], mode=config["mode"])
        best_eval_res = float('inf')
        train_loss_seq = []
        train_res_seq = []
        ctime = timeit.default_timer()
        for epoch in range(config["num_epochs"]):
            train_loss_seq_per_epoch = []
            train_res_seq_per_epoch = []
            for batch_idx in trainloader:
                model.zero_grad()
                train_loss_seq_per_batch = []
                train_res_seq_per_batch = []
                batch_loss = 0.0
                for i in range(len(batch_idx)):
                    idx1 = batch_idx[i].item()
                    # pe1 = torch.tensor(funcs.extract_graph_pe(data=ds[idx1], N=config["pe_dim"], mode=config["pe_mode"]), dtype=torch.float32, device=device)
                    pe1 = torch.tensor(pe[ptr[idx1]:ptr[idx1 + 1],:], dtype=torch.float32, device=device)
                    edge_index1 = ds[idx1].edge_index.to(device)
                    for j in range(i + 1, len(batch_idx)):
                        idx2 = batch_idx[j].item()
                        # pe2 = torch.tensor(funcs.extract_graph_pe(data=ds[idx2], N=config["pe_dim"], mode=config["pe_mode"]), dtype=torch.float32, device=device)
                        pe2 = torch.tensor(pe[ptr[idx2]:ptr[idx2 + 1],:], dtype=torch.float32, device=device)
                        edge_index2 = ds[idx2].edge_index.to(device)

                        # forward pass
                        tp_pred = model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2)
                        tp_gt = torch.tensor(P[idx1][idx2 - idx1 - 1] if idx1 < idx2 else P[idx2][idx1 - idx2 - 1].T, dtype=torch.float32, device=device)
                        loss = nn.functional.binary_cross_entropy_with_logits(tp_pred, tp_gt)
                        tp_pred = torch.sigmoid(tp_pred)
                        res = torch.linalg.vector_norm(tp_pred - tp_gt, ord=1) / torch.linalg.vector_norm(tp_gt, ord=1)


                        # keep track of the loss, gwd_pred, and gwd_gt
                        batch_loss += loss
                        train_loss_seq_per_batch.append(loss.item())
                        train_res_seq_per_batch.append(res.item())

                # backward pass
                batch_loss.backward()
                optimizer.step()

                # keep track of the average loss and the whole relative residual of error under the current batch
                train_loss_seq_per_epoch.append(sum(train_loss_seq_per_batch) / len(train_loss_seq_per_batch))
                train_res_seq_per_epoch.append(sum(train_res_seq_per_batch) / len(train_res_seq_per_batch))

            # keep track of the average loss and the whole relative residual of error under the current epoch
            train_loss_seq.append(sum(train_loss_seq_per_epoch) / len(train_loss_seq_per_epoch))
            train_res_seq.append(sum(train_res_seq_per_epoch) / len(train_res_seq_per_epoch))
            time_elapsed = timeit.default_timer() - ctime
            print(f'[{epoch:-3d}/{config["num_epochs"]}]({time_elapsed/60:.2f} m) [running_train_loss running_train_res] = [{train_loss_seq[-1]:.8f} {train_res_seq[-1]:.8f}]')

            # eval the model
            if epoch % config["eval_interval"] == 0:
                eval_loss, eval_res = model.eval_model(config=config, model=model, ds=ds, eval_idx=eval_idx, P=P, device=device)
                time_elapsed = timeit.default_timer() - ctime
                print(f'[{epoch:-3d}/{config["num_epochs"]}]({time_elapsed/60:.2f} m) [running_train_loss eval_loss train_res eval_res] = [{train_loss_seq[-1]:.8f} {eval_loss:.8f} {train_res_seq[-1]:.8f} {eval_res:.8f}]')
                if eval_res < best_eval_res:
                    print(f'[Info] New best eval_res: {eval_res:.8f} < {best_eval_res:.8f}, updating the best model...')
                    stop_count = 0
                    best_eval_res = eval_res
                    print(f'[Info] Best_eval_res has been updated to {best_eval_res:.8f}')
                    best_model_state = deepcopy(model.state_dict())
                    print(f'[Info] Best_model_state has been recorded.')
                    print(f'[Info] Stop_count has been reset to 0.')
                else:
                    stop_count += 1
                    print(f'[Info] eval_res: {eval_res:.8f} >= best_eval_res: {best_eval_res:.8f}. Stop_count has been increased to {stop_count}.')
                    
                if stop_count >= config["early_stop_patience"]:
                    print(f'[Info] Early stopping triggered. Stopping training...')
                    break

        # test the model
        test_loss, test_res = self.test_model(config=config, model=model, best_model_state=best_model_state, ds=ds, test_idx=test_idx, P=P, device=device)
        print(f'Test loss: {test_loss:.8f}, Test result: {test_res:.8f}')
        
        # save the best model state
        checkpoint_path = f'checkpoints/exp_indomain/{config["ds_name"]}_model_gwidth{config["gwidth"]}_epoch{epoch}_cuda{config["cuda_id"]}.pth'
        torch.save(best_model_state, checkpoint_path)

    def eval_model(self, config: Dict, model: nn.Module, ds: Dataset, eval_idx: List, P: List, device: torch.device) -> Tuple[float, float]:
        model.eval()
        with torch.no_grad():
            eval_loss_seq = []
            eval_res_seq = []
            for i in range(len(eval_idx)):
                idx1 = eval_idx[i]
                pe1 = torch.tensor(funcs.extract_graph_pe(data=ds[idx1], N=config["pe_dim"], mode=config["pe_mode"]), dtype=torch.float32, device=device)
                edge_index1 = ds[idx1].edge_index.to(device)
                for j in range(i + 1, len(eval_idx)):
                    idx2 = eval_idx[j]
                    pe2 = torch.tensor(funcs.extract_graph_pe(data=ds[idx2], N=config["pe_dim"], mode=config["pe_mode"]), dtype=torch.float32, device=device)
                    edge_index2 = ds[idx2].edge_index.to(device)

                    # forward pass
                    tp_pred = model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2)
                    tp_gt = torch.tensor(P[idx1][idx2 - idx1 - 1] if idx1 < idx2 else P[idx2][idx1 - idx2 - 1].T, dtype=torch.float32, device=device)
                    loss = nn.functional.binary_cross_entropy_with_logits(tp_pred, tp_gt)
                    tp_pred = torch.sigmoid(tp_pred)
                    res = torch.linalg.vector_norm(tp_pred - tp_gt, ord=1) / torch.linalg.vector_norm(tp_gt, ord=1)

                    # keep track of the loss, tp_pred, and tp_gt
                    eval_loss_seq.append(loss.item())
                    eval_res_seq.append(res.item())

            eval_loss = sum(eval_loss_seq) / len(eval_loss_seq)
            eval_res = sum(eval_res_seq) / len(eval_res_seq)

            model.train()
        return eval_loss, eval_res

    def test_model(self, config: Dict, model: nn.Module, best_model_state: Dict, ds: Dataset, test_idx: List, P: List, device: torch.device) -> Tuple[float, float]:
        model.load_state_dict(best_model_state)
        test_loss, test_res = self.eval_model(config=config, model=model, ds=ds, eval_idx=test_idx, P=P, device=device)
        return test_loss, test_res
    
class KernelLayer(nn.Module):
    def __init__(self, n_gammas: int = 9):
        super(KernelLayer, self).__init__()
        assert n_gammas%2 == 1, f"The width of Kernel layer: 'n_gammas={n_gammas}' must be odd."
        self.n_gammas = n_gammas
        self._raw_gammas = nn.Linear(n_gammas, 1, bias=False).weight.view(-1)
    
    def forward(self, X1: torch.Tensor, X2: torch.Tensor, is_biased: bool = False) -> torch.Tensor:
        assert X1.dim() == 2 and X2.dim() == 2, "Input 'X1' and 'X2' must be 2D Tensors"
        n1, n2 = X1.shape[0], X2.shape[0]
        D2 = torch.cdist(X1, X2, compute_mode='use_mm_for_euclid_dist')**2
        # self._raw_gammas = [self._raw_gamma_base * 2**(i + 1) for i in range(-self.n_gammas//2, self.n_gammas//2)]
        K_seq = []
        for raw_gamma in self._raw_gammas:
            positive_gamma = F.softplus(raw_gamma)
            K = torch.exp(-positive_gamma * D2).reshape(-1, 1)
            K_seq.append(K)
        return torch.cat(K_seq, dim=1)
    

class GINTransformer_TP_Naive(GINTransformer_TP):
    def __init__(self, input_dim: int, hidden_dim: int, gin_layers: int = 3, transformer_layers: int = 1, n_gammas=9):
        super(GINTransformer_TP_Naive, self).__init__(input_dim=input_dim, hidden_dim=hidden_dim, gin_layers=gin_layers, transformer_layers=transformer_layers, n_gammas=n_gammas)
        self._raw_gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)

    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        # feedforward it into the backbone - A hybrid model of GIN and Transformer
        Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)

        # compute the kernel matrix of latent embeddings (Z1, Z2) as the predicted transport plan
        D2 = torch.cdist(Z1, Z2, compute_mode='use_mm_for_euclid_dist')**2
        tp_pred = torch.exp(-self._raw_gamma*D2)

        return tp_pred
    
    def train_model(self, config: Dict, model: nn.Module, ds: Dataset, optimizer: torch.optim.Optimizer, device: torch.device, seed_worker: Callable[[int], None]) -> None:
        g = torch.Generator()
        g.manual_seed(config["seed"])
        model.train()
        # create a DataLoader for the dataset
        train_eval_idx, test_idx = train_test_split(range(len(ds)), test_size=0.2, random_state=42)
        train_idx, eval_idx = train_test_split(train_eval_idx, test_size=0.2, random_state=42)
        trainloader = torch.utils.data.DataLoader(train_idx, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=4, generator=g, worker_init_fn=seed_worker)
        # load the graph pointers
        ptr = funcs.load_graph_ptr(ds_name=config["ds_name"])
        # load the graph positional encodings
        pe = funcs.load_graph_pe(ds_name=config["ds_name"], mode=config["pe_mode"])
        # load the ground truth distance matrix
        # D = funcs.load_distance_matrix(ds_name=config["ds_name"], metric=config["metric"], mode=config["mode"])
        P = funcs.load_transport_plans(ds_name=config["ds_name"], metric=config["metric"], mode=config["mode"])
        best_eval_res = float('inf')
        train_loss_seq = []
        train_res_seq = []
        ctime = timeit.default_timer()
        for epoch in range(config["num_epochs"]):
            train_loss_seq_per_epoch = []
            train_res_seq_per_epoch = []
            for batch_idx in trainloader:
                model.zero_grad()
                train_loss_seq_per_batch = []
                train_res_seq_per_batch = []
                batch_loss = 0.0
                for i in range(len(batch_idx)):
                    idx1 = batch_idx[i].item()
                    # pe1 = torch.tensor(funcs.extract_graph_pe(data=ds[idx1], N=config["pe_dim"], mode=config["pe_mode"]), dtype=torch.float32, device=device)
                    pe1 = torch.tensor(pe[ptr[idx1]:ptr[idx1 + 1],:], dtype=torch.float32, device=device)
                    edge_index1 = ds[idx1].edge_index.to(device)
                    for j in range(i + 1, len(batch_idx)):
                        idx2 = batch_idx[j].item()
                        # pe2 = torch.tensor(funcs.extract_graph_pe(data=ds[idx2], N=config["pe_dim"], mode=config["pe_mode"]), dtype=torch.float32, device=device)
                        pe2 = torch.tensor(pe[ptr[idx2]:ptr[idx2 + 1],:], dtype=torch.float32, device=device)
                        edge_index2 = ds[idx2].edge_index.to(device)

                        # forward pass
                        tp_pred = model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2)
                        tp_gt = torch.tensor(P[idx1][idx2 - idx1 - 1] if idx1 < idx2 else P[idx2][idx1 - idx2 - 1].T, dtype=torch.float32, device=device)
                        loss = nn.functional.binary_cross_entropy_with_logits(tp_pred, tp_gt)
                        # tp_pred = torch.sigmoid(tp_pred)
                        res = torch.linalg.vector_norm(tp_pred - tp_gt, ord=1) / torch.linalg.vector_norm(tp_gt, ord=1)


                        # keep track of the loss, gwd_pred, and gwd_gt
                        batch_loss += loss
                        train_loss_seq_per_batch.append(loss.item())
                        train_res_seq_per_batch.append(res.item())

                # backward pass
                batch_loss.backward()
                optimizer.step()

                # keep track of the average loss and the whole relative residual of error under the current batch
                train_loss_seq_per_epoch.append(sum(train_loss_seq_per_batch) / len(train_loss_seq_per_batch))
                train_res_seq_per_epoch.append(sum(train_res_seq_per_batch) / len(train_res_seq_per_batch))

            # keep track of the average loss and the whole relative residual of error under the current epoch
            train_loss_seq.append(sum(train_loss_seq_per_epoch) / len(train_loss_seq_per_epoch))
            train_res_seq.append(sum(train_res_seq_per_epoch) / len(train_res_seq_per_epoch))
            time_elapsed = timeit.default_timer() - ctime
            print(f'[{epoch:-3d}/{config["num_epochs"]}]({time_elapsed/60:.2f} m) [running_train_loss running_train_res] = [{train_loss_seq[-1]:.8f} {train_res_seq[-1]:.8f}]')

            # eval the model
            if epoch % config["eval_interval"] == 0:
                eval_loss, eval_res = model.eval_model(config=config, model=model, ds=ds, eval_idx=eval_idx, P=P, device=device)
                time_elapsed = timeit.default_timer() - ctime
                print(f'[{epoch:-3d}/{config["num_epochs"]}]({time_elapsed/60:.2f} m) [running_train_loss eval_loss train_res eval_res] = [{train_loss_seq[-1]:.8f} {eval_loss:.8f} {train_res_seq[-1]:.8f} {eval_res:.8f}]')
                if eval_res < best_eval_res:
                    print(f'[Info] New best eval_res: {eval_res:.8f} < {best_eval_res:.8f}, updating the best model...')
                    stop_count = 0
                    best_eval_res = eval_res
                    print(f'[Info] Best_eval_res has been updated to {best_eval_res:.8f}')
                    best_model_state = deepcopy(model.state_dict())
                    print(f'[Info] Best_model_state has been recorded.')
                    print(f'[Info] Stop_count has been reset to 0.')
                else:
                    stop_count += 1
                    print(f'[Info] eval_res: {eval_res:.8f} >= best_eval_res: {best_eval_res:.8f}. Stop_count has been increased to {stop_count}.')
                    
                if stop_count >= config["early_stop_patience"]:
                    print(f'[Info] Early stopping triggered. Stopping training...')
                    break

        # test the model
        test_loss, test_res = self.test_model(config=config, model=model, best_model_state=best_model_state, ds=ds, test_idx=test_idx, P=P, device=device)
        print(f'Test loss: {test_loss:.8f}, Test result: {test_res:.8f}')
        
        # save the best model state
        checkpoint_path = f'checkpoints/exp_indomain/{config["ds_name"]}_model_gwidth{config["gwidth"]}_epoch{epoch}_cuda{config["cuda_id"]}.pth'
        torch.save(best_model_state, checkpoint_path)

    def eval_model(self, config: Dict, model: nn.Module, ds: Dataset, eval_idx: List, P: List, device: torch.device) -> Tuple[float, float]:
        model.eval()
        with torch.no_grad():
            eval_loss_seq = []
            eval_res_seq = []
            for i in range(len(eval_idx)):
                idx1 = eval_idx[i]
                pe1 = torch.tensor(funcs.extract_graph_pe(data=ds[idx1], N=config["pe_dim"], mode=config["pe_mode"]), dtype=torch.float32, device=device)
                edge_index1 = ds[idx1].edge_index.to(device)
                for j in range(i + 1, len(eval_idx)):
                    idx2 = eval_idx[j]
                    pe2 = torch.tensor(funcs.extract_graph_pe(data=ds[idx2], N=config["pe_dim"], mode=config["pe_mode"]), dtype=torch.float32, device=device)
                    edge_index2 = ds[idx2].edge_index.to(device)

                    # forward pass
                    tp_pred = model.forward(X1=pe1, edge_index1=edge_index1, X2=pe2, edge_index2=edge_index2)
                    tp_gt = torch.tensor(P[idx1][idx2 - idx1 - 1] if idx1 < idx2 else P[idx2][idx1 - idx2 - 1].T, dtype=torch.float32, device=device)
                    loss = nn.functional.binary_cross_entropy_with_logits(tp_pred, tp_gt)
                    # tp_pred = torch.sigmoid(tp_pred)
                    res = torch.linalg.vector_norm(tp_pred - tp_gt, ord=1) / torch.linalg.vector_norm(tp_gt, ord=1)

                    # keep track of the loss, tp_pred, and tp_gt
                    eval_loss_seq.append(loss.item())
                    eval_res_seq.append(res.item())

            eval_loss = sum(eval_loss_seq) / len(eval_loss_seq)
            eval_res = sum(eval_res_seq) / len(eval_res_seq)

            model.train()
        return eval_loss, eval_res