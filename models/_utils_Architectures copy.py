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
        self.model_type = config['model_type']
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
        if self.model_type == 'serial':
            print(f'[Info] Using serial model architecture.')
            self.build_serial_model()
        elif self.model_type == 'parallel':
            print(f'[Info] Using parallel model architecture.')
            self.build_parallel_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")

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
        if self.model_type == 'serial':
            self._init_weights(self.attn_layer_for_serial)
        elif self.model_type == 'parallel':
            self._init_weights(self.attn_layer_for_parallel)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")
        self._init_weights(self.mlp_head)

    def build_serial_model(self):
        self.attn_layer_for_serial = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim+self.input_dim,
            nhead=4
        )

        self.fc = nn.Linear(self.hidden_dim + self.input_dim, 128)

    def build_parallel_model(self):
        self.attn_layer_for_parallel = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=4
        )

        self.fc = nn.Linear(self.hidden_dim + 2*self.input_dim, 128)

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
        if self.model_type == 'serial':
            Z1, Z2 = self.forward_siamese_serial(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        elif self.model_type == 'parallel':
            Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")

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
    
################################################################################################################################
# I want to realize the GraphGPS architecture variants here
################################################################################################################################
# class GraphGPS(nn.Module):


################################################################################################################################
# End of GraphGPS architecture variants
################################################################################################################################
    
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
        del self._modules['mmd_layer'], self.mlp_head
        from torch_geometric.nn import global_mean_pool
        self.readout = global_mean_pool
        
    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        # feedforward it into the backbone - A hybrid model of GIN and Transformer
        if self.model_type == 'serial':
            Z1, Z2 = self.forward_siamese_serial(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        elif self.model_type == 'parallel':
            Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")

        z1 = self.readout(Z1, batch=None)
        z2 = self.readout(Z2, batch=None)
        o = torch.linalg.vector_norm(z1 - z2)**2

        return o
    

class GINTransformer_Ablation(GINTransformer):
    def __init__(self,
                 input_dim: int, hidden_dim: int,
                 gin_layers: int = 3, transformer_layers: int = 1,
                 n_gammas=9,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(GINTransformer_Ablation, self).__init__(input_dim=input_dim,
                                                     hidden_dim=hidden_dim,
                                                     gin_layers=gin_layers,
                                                     transformer_layers=transformer_layers,
                                                     n_gammas=n_gammas,
                                                     config=config,
                                                     device=device)
        self.switch = self.config["switch"]  # 'gin' or 'mha'
        if self.switch == 'mha':
            print(f'[Info] Ablation study: removing GIN layers.')
            del self._modules['gin_layers_list']
        elif self.switch == 'gin':
            print(f'[Info] Ablation study: removing Transformer layers.')
            if self.model_type == 'serial':
                del self._modules['attn_layer_for_serial']
            elif self.model_type == 'parallel':
                del self._modules['attn_layer_for_parallel']
            else:
                raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")
        else:
            raise ValueError(f"Unknown ablation switch: {self.switch}. Supported switches are 'no_gin' and 'no_transformer'.")
        
    def forward_siamese_gin_only(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        gin_Z1 = self.forward_gin(X=X1, edge_index=edge_index1)
        gin_Z2 = self.forward_gin(X=X2, edge_index=edge_index2)
        gin_Z1 = torch.cat((gin_Z1, X1), dim=1)
        gin_Z2 = torch.cat((gin_Z2, X2), dim=1)
        return gin_Z1, gin_Z2
    
    def forward_siamese_attention_only(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor, is_whole_attn: bool = True) -> torch.Tensor:
        if self.model_type == 'serial':
            raise ValueError(f"Ablation study with 'no_gin' does not support 'serial' model type.")
        elif self.model_type == 'parallel':
            #  apply transformer layers:
            if is_whole_attn:
                attn_Z = F.relu(self.attn_layer_for_parallel(torch.cat((X1, X2), dim=0)))
                # retrieve the attention scores
                attn_Z1 = attn_Z[:X1.size(0)]
                attn_Z2 = attn_Z[X1.size(0):]
            else:
                attn_Z1 = F.relu(self.attn_layer_for_parallel(X1))
                attn_Z2 = F.relu(self.attn_layer_for_parallel(X2))

            Z1, Z2 = attn_Z1, attn_Z2
            return Z1, Z2
        
    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        # feedforward it into the backbone - A hybrid model of GIN and Transformer
        if self.switch == 'mha':
            Z1, Z2 = self.forward_siamese_attention_only(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        elif self.switch == 'gin':
            Z1, Z2 = self.forward_siamese_gin_only(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2)
        else:
            raise ValueError(f"Unknown ablation switch: {self.switch}. Supported switches are 'no_gin' and 'no_transformer'.")
        
        # feedforward it into MMD layer
        o = self.mmd_layer.forward(X1=Z1, X2=Z2)

        # feedforward it into MLP head
        o = self.mlp_head(o)

        return o

class GINTransformer_TP(nn.Module):
    def __init__(self,
                 input_dim: int, hidden_dim: int,
                 gin_layers: int = 3, transformer_layers: int = 1,
                 n_gammas=9,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(GINTransformer_TP, self).__init__()
        self.config = config
        self.model_type = config['model_type']
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
        if self.model_type == 'serial':
            self.build_serial_model()
        elif self.model_type == 'parallel':
            self.build_parallel_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")

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
        self._init_weights(self.gin_layers_list)
        if self.model_type == 'serial':
            self._init_transformer_weights(self.attn_layer_for_serial)
        elif self.model_type == 'parallel':
            self._init_transformer_weights(self.attn_layer_for_parallel)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")
        self._init_weights(self.mlp_head)

    def build_serial_model(self):
        self.attn_layer_for_serial = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim + self.input_dim,
            nhead=4
        )

        self.fc = nn.Linear(self.hidden_dim + self.input_dim, 128)

    def build_parallel_model(self):
        self.attn_layer_for_parallel = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=4
        )

        self.fc = nn.Linear(self.hidden_dim + 2*self.input_dim, 128)

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
        if self.model_type == 'serial':
            Z1, Z2 = self.forward_siamese_serial(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        elif self.model_type == 'parallel':
            Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")

        # feedforward it into kernel layer
        Kernel_Z = self.kernel_layer.forward(X1=Z1, X2=Z2)

        # feedforward it into MLP head
        o = self.mlp_head(Kernel_Z)
        tp_pred = o.reshape(Z1.shape[0], Z2.shape[0])

        return tp_pred
    
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
    def __init__(self, input_dim: int, hidden_dim: int,
                 gin_layers: int = 3, transformer_layers: int = 1,
                 n_gammas=9,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(GINTransformer_TP_Naive, self).__init__(input_dim=input_dim,
                                                      hidden_dim=hidden_dim,
                                                      gin_layers=gin_layers,
                                                      transformer_layers=transformer_layers,
                                                      n_gammas=n_gammas,
                                                      config=config,
                                                      device=device)
        del self._modules['kernel_layer'], self.mlp_head
        self._raw_gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)

    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        # feedforward it into the backbone - A hybrid model of GIN and Transformer
        if self.model_type == 'serial':
            Z1, Z2 = self.forward_siamese_serial(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        elif self.model_type == 'parallel':
            Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")

        # compute the kernel matrix of latent embeddings (Z1, Z2) as the predicted transport plan
        D2 = torch.cdist(Z1, Z2, compute_mode='use_mm_for_euclid_dist')**2
        # tp_pred = torch.exp(-self._raw_gamma*D2)
        gamma = 1/(2 * D2.mean().item()**2)
        tp_pred = torch.exp(-gamma*D2)
        # tp_pred = D2

        return tp_pred