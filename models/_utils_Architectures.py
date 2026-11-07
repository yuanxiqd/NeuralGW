import os
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

        # # define the last MLP head
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(n_gammas, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 10),
        #     nn.ReLU(),
        #     nn.Linear(10, 1),
        #     # nn.Sigmoid()
        # )

        # define the last MLP head (smaller + optional LayerNorm/Dropout)
        # Config keys (with sensible defaults): head_use_layernorm (bool), head_dropout_p (float).
        # Env vars NEURALGW_HEAD_LAYERNORM / NEURALGW_HEAD_DROPOUT_P override the config (handy for ablations).
        _use_ln = bool(config.get('head_use_layernorm', False))
        _do_p = float(config.get('head_dropout_p', 0.0))
        if 'NEURALGW_HEAD_LAYERNORM' in os.environ:
            _use_ln = os.environ['NEURALGW_HEAD_LAYERNORM'] != '0'
        if 'NEURALGW_HEAD_DROPOUT_P' in os.environ:
            _do_p = float(os.environ['NEURALGW_HEAD_DROPOUT_P'])
        # Optional ReLU on the final output (NEURALGW_HEAD_OUTPUT_RELU=1) — clamps the
        # GW-distance prediction to >= 0, a valid inductive bias since GW >= 0.
        _out_relu = os.environ.get('NEURALGW_HEAD_OUTPUT_RELU', '0') == '1'
        _head_layers = []
        if _use_ln:
            _head_layers.append(nn.LayerNorm(n_gammas))
        _head_layers.append(nn.Linear(n_gammas, 64))
        _head_layers.append(nn.ReLU())
        if _do_p > 0:
            _head_layers.append(nn.Dropout(p=_do_p))
        _head_layers.append(nn.Linear(64, 16))
        _head_layers.append(nn.ReLU())
        if _do_p > 0:
            _head_layers.append(nn.Dropout(p=_do_p))
        _head_layers.append(nn.Linear(16, 1))
        if _out_relu:
            _head_layers.append(nn.ReLU())
        self.mlp_head = nn.Sequential(*_head_layers)
        print(f'[Info] MLP head: LayerNorm={_use_ln}, Dropout_p={_do_p}, OutputReLU={_out_relu}')

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

        self.fc = nn.Linear(self.hidden_dim + self.input_dim, self.config['embedding_dim_for_mmd_layer'])

    def build_parallel_model(self):
        # self.attn_layer_for_parallel = nn.TransformerEncoderLayer(
        #     d_model=self.input_dim,
        #     nhead=4
        # )

        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=4
        )
        self.attn_layer_for_parallel = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config['num_transformer_layers']
        )

        self.fc = nn.Linear(self.hidden_dim + 2*self.input_dim, self.config['embedding_dim_for_mmd_layer'])

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def attach_film(self, meta_dim: int) -> None:
        """Attach two FiLM heads (one per backbone branch) — identity at init.

        ``film_gin`` modulates the GIN output (channel dim = ``hidden_dim``) BEFORE
        the PE residual concat, so the raw PE stream stays pristine (no double
        modulation). ``film_attn`` modulates the Transformer-branch output
        (channel dim = ``input_dim``).
        """
        self.film_gin = FiLM(meta_dim=meta_dim, out_dim=self.hidden_dim).to(self.device)
        self.film_attn = FiLM(meta_dim=meta_dim, out_dim=self.input_dim).to(self.device)
        self.meta_dim = meta_dim
        self.use_film = True

    def attach_meta_concat(self, meta_dim: int) -> None:
        """Concat per-graph meta features (broadcast over nodes) onto the merged
        GIN+Transformer output, then re-project with a resized ``fc`` (parallel path).

        The merge ``Z = [gin_Z | attn_Z]`` has width ``hidden_dim + 2*input_dim``;
        appending ``meta_dim`` columns requires the projection to accept the wider
        input, so ``fc`` is rebuilt here (this discards the init-time ``fc`` weights,
        which is fine because attach happens before training).
        """
        self.meta_concat_dim = meta_dim
        self.use_meta_concat = True
        new_in = self.hidden_dim + 2 * self.input_dim + meta_dim
        self.fc = nn.Linear(new_in, self.config['embedding_dim_for_mmd_layer']).to(self.device)
        self._init_weights(self.fc)

    def attach_meta_mlpconcat(self, meta_dim: int, out_dim: int = 16, hidden: int = None) -> None:
        """Encode per-graph meta features with a small MLP, then concat its output
        (broadcast over nodes) onto the merged GIN+Transformer embedding (parallel).

        Difference vs ``attach_meta_concat``: raw meta values are first transformed
        by a learnable MLP, so the model can re-weight / suppress noisy groups (e.g.
        the OOD degree features) instead of pushing raw values straight into the
        MMD embedding. The MLP's final layer is zero-initialized, so the meta
        embedding is 0 at init -> identity (≡ off) at the start of training, which
        gives a clean ablation and a stable warm-up.
        """
        if hidden is None:
            hidden = max(32, 2 * out_dim)
        self.meta_mlp_out = out_dim
        self.use_meta_mlpconcat = True
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        ).to(self.device)
        # Identity-at-init: zero the final layer so meta embedding starts at 0.
        # No output ReLU — otherwise relu'(0)=0 would block gradients to the head.
        nn.init.zeros_(self.meta_encoder[-1].weight)
        nn.init.zeros_(self.meta_encoder[-1].bias)
        new_in = self.hidden_dim + 2 * self.input_dim + out_dim
        self.fc = nn.Linear(new_in, self.config['embedding_dim_for_mmd_layer']).to(self.device)
        self._init_weights(self.fc)

    def attach_meta_mmd(self, meta_dim: int, use_pair_density: bool = True) -> None:
        """Append a symmetric per-pair meta encoding to the MMD vector before the head.

        The MMD vector is symmetric under swapping the two graphs (GW distance is
        symmetric), so the appended encoding must be too:
            pair_meta = [ |meta1 - meta2| , 0.5*(meta1 + meta2) ]   (2*meta_dim)
        When ``use_pair_density`` is set, one extra scalar — the (separately
        normalized) sum of the two graphs' edge densities, supplied at call time as
        ``pair_scalar`` — is appended as a pair-level scale cue (denser pairs tend
        to have larger GW costs). That term is also swap-symmetric (a sum).
        The head is rebuilt to accept ``n_gammas + 2*meta_dim (+1)`` inputs,
        replicating the LayerNorm/Dropout config used at construction.
        """
        self.meta_mmd_dim = 2 * meta_dim
        self.use_meta_mmd = True
        self.use_pair_density = bool(use_pair_density)
        new_in = self.n_gammas + self.meta_mmd_dim + (1 if self.use_pair_density else 0)
        self._rebuild_head(new_in)

    def attach_meta_mmd_mlp(self, meta_dim: int, out_dim: int = 16, hidden: int = None,
                            use_pair_density: bool = True) -> None:
        """Encode meta with a small MLP, then concat (symmetrically) onto the MMD vector.

        Mirrors ``attach_meta_mmd`` but inserts a learnable encoder: a shared MLP maps
        each graph's meta to ``e = MLP(meta)``; the appended block is the swap-symmetric
        ``[ |e1 - e2| , 0.5*(e1 + e2) ]`` (2*out_dim). Final MLP layer is zero-init so the
        meta block starts at 0 (stable warm-up). Optional raw pair-density-sum scalar is
        appended after, identical to the raw mmd-inject variant.
        """
        if hidden is None:
            hidden = max(32, 2 * out_dim)
        self.meta_mmdmlp_out = out_dim
        self.use_meta_mmdmlp = True
        self.use_pair_density = bool(use_pair_density)
        self.meta_mmd_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        ).to(self.device)
        nn.init.zeros_(self.meta_mmd_encoder[-1].weight)
        nn.init.zeros_(self.meta_mmd_encoder[-1].bias)
        new_in = self.n_gammas + 2 * out_dim + (1 if self.use_pair_density else 0)
        self._rebuild_head(new_in)

    def attach_meta_mmd_pair(self, num_pair_scalars: int = 2) -> None:
        """Minimal mmd-inject: append a few interpretable symmetric pair-level scalars
        (e.g. node-count sum, density sum) to the MMD vector before the head.

        No full meta vector and no |Δmeta| — just ``num_pair_scalars`` scalars supplied
        at call time as ``pair_scalar``. Sums are swap-symmetric, so pair invariance holds.
        """
        self.use_meta_mmdpair = True
        self.n_pair_scalars = int(num_pair_scalars)
        new_in = self.n_gammas + self.n_pair_scalars
        self._rebuild_head(new_in)

    def _rebuild_head(self, new_in: int) -> None:
        """Rebuild the MLP head to accept ``new_in`` features, replicating the
        LayerNorm/Dropout config used at construction (env overrides honored)."""
        _use_ln = bool(self.config.get('head_use_layernorm', False))
        _do_p = float(self.config.get('head_dropout_p', 0.0))
        if 'NEURALGW_HEAD_LAYERNORM' in os.environ:
            _use_ln = os.environ['NEURALGW_HEAD_LAYERNORM'] != '0'
        if 'NEURALGW_HEAD_DROPOUT_P' in os.environ:
            _do_p = float(os.environ['NEURALGW_HEAD_DROPOUT_P'])
        layers = []
        if _use_ln:
            layers.append(nn.LayerNorm(new_in))
        layers.append(nn.Linear(new_in, 64))
        layers.append(nn.ReLU())
        if _do_p > 0:
            layers.append(nn.Dropout(p=_do_p))
        layers.append(nn.Linear(64, 16))
        layers.append(nn.ReLU())
        if _do_p > 0:
            layers.append(nn.Dropout(p=_do_p))
        layers.append(nn.Linear(16, 1))
        if os.environ.get('NEURALGW_HEAD_OUTPUT_RELU', '0') == '1':
            layers.append(nn.ReLU())
        self.mlp_head = nn.Sequential(*layers).to(self.device)

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

    def forward_siamese_serial(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor, is_whole_attn: bool = True, meta1: torch.Tensor = None, meta2: torch.Tensor = None) -> torch.Tensor:
        # FiLM is not implemented for the serial path (parallel is the cdomain default);
        # meta1/meta2 accepted for signature compat and intentionally ignored here.
        del meta1, meta2
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

    def forward_siamese_parallel(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor, is_whole_attn: bool = False, meta1: torch.Tensor = None, meta2: torch.Tensor = None) -> torch.Tensor:
        # feedforward two graphs through GIN layers, respectively
        gin_z1 = self.forward_gin(X=X1, edge_index=edge_index1)
        gin_z2 = self.forward_gin(X=X2, edge_index=edge_index2)
        # FiLM modulation on the GIN branch (channel dim = hidden_dim), BEFORE the
        # PE residual concat — leaves the raw PE stream untouched.
        if getattr(self, 'use_film', False) and meta1 is not None and meta2 is not None:
            gin_z1 = self.film_gin(gin_z1, meta1)
            gin_z2 = self.film_gin(gin_z2, meta2)
        gin_Z1 = torch.cat((gin_z1, X1), dim=1)
        gin_Z2 = torch.cat((gin_z2, X2), dim=1)

        #  apply transformer layers:
        if is_whole_attn:
            attn_Z = F.relu(self.attn_layer_for_parallel(torch.cat((X1, X2), dim=0)))
            # retrieve the attention scores
            attn_Z1 = attn_Z[:gin_Z1.size(0)]
            attn_Z2 = attn_Z[gin_Z1.size(0):]
        else:
            attn_Z1 = F.relu(self.attn_layer_for_parallel(X1))
            attn_Z2 = F.relu(self.attn_layer_for_parallel(X2))

        # FiLM modulation on the Transformer branch (channel dim = input_dim).
        if getattr(self, 'use_film', False) and meta1 is not None and meta2 is not None:
            attn_Z1 = self.film_attn(attn_Z1, meta1)
            attn_Z2 = self.film_attn(attn_Z2, meta2)

        # combine outputs from GIN and transformer layers
        Z1 = torch.cat((gin_Z1, attn_Z1), dim=1)
        Z2 = torch.cat((gin_Z2, attn_Z2), dim=1)
        # Alternative meta-injection: append per-graph meta features, broadcast over
        # nodes, onto the merged embedding before the fc projection.
        if getattr(self, 'use_meta_concat', False) and meta1 is not None and meta2 is not None:
            Z1 = torch.cat((Z1, meta1.unsqueeze(0).expand(Z1.size(0), -1)), dim=1)
            Z2 = torch.cat((Z2, meta2.unsqueeze(0).expand(Z2.size(0), -1)), dim=1)
        # MLP-encoded variant: pass meta through a small MLP first, then concat its
        # (broadcast) output. Zero-init final layer => 0 at init (identity).
        if getattr(self, 'use_meta_mlpconcat', False) and meta1 is not None and meta2 is not None:
            me1 = self.meta_encoder(meta1)
            me2 = self.meta_encoder(meta2)
            Z1 = torch.cat((Z1, me1.unsqueeze(0).expand(Z1.size(0), -1)), dim=1)
            Z2 = torch.cat((Z2, me2.unsqueeze(0).expand(Z2.size(0), -1)), dim=1)
        Z1, Z2 = F.relu(self.fc(Z1)), F.relu(self.fc(Z2))
        return Z1, Z2

    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor, meta1: torch.Tensor = None, meta2: torch.Tensor = None, pair_scalar: torch.Tensor = None) -> torch.Tensor:
        # feedforward it into the backbone - A hybrid model of GIN and Transformer
        if self.model_type == 'serial':
            Z1, Z2 = self.forward_siamese_serial(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True, meta1=meta1, meta2=meta2)
        elif self.model_type == 'parallel':
            Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True, meta1=meta1, meta2=meta2)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")

        # feedforward it into MMD layer
        o = self.mmd_layer.forward(X1=Z1, X2=Z2)

        # Optional meta-injection at the MMD vector: append a symmetric per-pair
        # encoding (|Δmeta|, mean meta) so the head sees graph-level descriptors
        # without breaking the pair-swap symmetry of the MMD statistics.
        if getattr(self, 'use_meta_mmd', False) and meta1 is not None and meta2 is not None:
            pair_meta = torch.cat(((meta1 - meta2).abs(), 0.5 * (meta1 + meta2)), dim=0)
            o = torch.cat((o, pair_meta), dim=0)
            # Extra pair-level scale cue: normalized sum of the two graph densities.
            if getattr(self, 'use_pair_density', False) and pair_scalar is not None:
                o = torch.cat((o, pair_scalar), dim=0)

        # MLP-encoded variant of the MMD-vector injection: meta -> small MLP, then
        # append the swap-symmetric combination of the two per-graph encodings.
        if getattr(self, 'use_meta_mmdmlp', False) and meta1 is not None and meta2 is not None:
            e1 = self.meta_mmd_encoder(meta1)
            e2 = self.meta_mmd_encoder(meta2)
            pair_enc = torch.cat(((e1 - e2).abs(), 0.5 * (e1 + e2)), dim=0)
            o = torch.cat((o, pair_enc), dim=0)
            if getattr(self, 'use_pair_density', False) and pair_scalar is not None:
                o = torch.cat((o, pair_scalar), dim=0)

        # Minimal variant: append only a few interpretable pair scalars (node-sum,
        # density-sum) to the MMD vector.
        if getattr(self, 'use_meta_mmdpair', False) and pair_scalar is not None:
            o = torch.cat((o, pair_scalar), dim=0)

        # feedforward it into MLP head
        o = self.mlp_head(o)

        return o

class FiLM(nn.Module):
    """Per-graph FiLM (Feature-wise Linear Modulation).

    Maps a per-graph meta vector ``m_g`` of shape ``(meta_dim,)`` to a per-channel
    scale ``gamma`` and shift ``beta`` of shape ``(out_dim,)``. Modulation is applied
    by broadcasting over the node axis:  ``Z' = (1 + gamma) * Z + beta``.

    The final linear layer is zero-initialized so that ``gamma=0, beta=0`` at start,
    which makes ``Z' == Z`` — i.e. the layer is the identity at initialization. This
    is the smoke-check guarantee referenced in the plan: turning the feature ON
    must not perturb the first-batch loss vs OFF.
    """

    def __init__(self, meta_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(meta_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * out_dim),
        )
        # Identity-at-init: zero the final layer's weights and biases.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, Z: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # Z: (n_nodes, out_dim); meta: (meta_dim,)
        gb = self.net(meta)
        gamma, beta = gb[: self.out_dim], gb[self.out_dim :]
        return Z * (1.0 + gamma) + beta


class MMDLayer(nn.Module):
    def __init__(self, n_gammas: int = 9):
        super(MMDLayer, self).__init__()
        assert n_gammas%2 == 1, f"The width of MMD layer: 'n_gammas={n_gammas}' must be odd."
        self.n_gammas = n_gammas
        # self._raw_gammas = nn.Linear(n_gammas, 1, bias=False).weight.view(-1)
        # self._raw_gamma_base = nn.Parameter(torch.tensor(0.001))
        self._raw_gamma_base = nn.Parameter(torch.log(torch.expm1(torch.tensor(0.001))))
        # self._raw_gamma_base = nn.Parameter(torch.log(torch.tensor(0.001)))
        # self._raw_gamma_bias = nn.Parameter(torch.tensor(0.0))
        # self._raw_gamma_base = torch.tensor(0.001)
        print(f'[Info] The initial value of _raw_gamma_base: {self._raw_gamma_base.item():.8f}')

    @property
    def gamma_base(self) -> torch.Tensor:
        # print(f'[Info] The value of _raw_gamma_base: {self._raw_gamma_base.item():.8f}')
        # o = nn.functional.softplus(torch.log(torch.expm1(self._raw_gamma_base)))
        # print(f'[Info] The value of o: {o.item():.8f}')
        # print(100*'*')
        # return nn.functional.softplus(torch.log(torch.expm1(self._raw_gamma_base)))
        # return torch.exp(self._raw_gamma_base)
        return nn.functional.softplus(self._raw_gamma_base)
    
    def forward(self, X1: torch.Tensor, X2: torch.Tensor, is_biased: bool = False) -> torch.Tensor:
        assert X1.dim() == 2 and X2.dim() == 2, "Input 'X1' and 'X2' must be 2D Tensors"
        n1, n2 = X1.shape[0], X2.shape[0]
        M = torch.cat((X1, X2), dim=0)
        D = torch.cdist(M, M, compute_mode='use_mm_for_euclid_dist')
        # print(f'Estimated gamma of gaussian kernel: {1/(2* D.mean().item()**2):.4f}')
        D2 = D**2
        # Bandwidth ablation knobs. Defaults keep the fixed gamma schedule; adaptive
        # per-batch scaling can still be enabled from the environment.
        # Ablation knobs (env vars, optional):
        #   NEURALGW_MMD_BW_ESTIMATOR : {none (default), median, mean, max}
        #   NEURALGW_MMD_BW_SCALE     : positive float multiplied into the divisor (default 1.0)
        bw_estimator = os.environ.get('NEURALGW_MMD_BW_ESTIMATOR', 'none').lower()
        bw_scale = float(os.environ.get('NEURALGW_MMD_BW_SCALE', '1.0'))
        if bw_estimator == 'none':
            divisor = torch.tensor(bw_scale, device=D2.device, dtype=D2.dtype).clamp_min(1e-12)
        elif bw_estimator == 'median':
            divisor = (bw_scale * D2.detach().median()).clamp_min(1e-12)
        elif bw_estimator == 'mean':
            divisor = (bw_scale * D2.detach().mean()).clamp_min(1e-12)
        elif bw_estimator == 'max':
            divisor = (bw_scale * D2.detach().max()).clamp_min(1e-12)
        else:
            raise ValueError(f"Unknown NEURALGW_MMD_BW_ESTIMATOR: {bw_estimator!r}. "
                             f"Choose from: median, mean, max, none.")
        self._raw_gammas = [self.gamma_base * (c + 1) ** 2 / divisor for c in range(self.n_gammas)]
        # self._raw_gammas = [self._raw_gamma_base * (c+1)**2 for c in range(self.n_gammas)]
        # self._raw_gammas = [self._raw_gamma_base * c**2 for c in range(self.n_gammas)]
        # self._raw_gammas = [self._raw_gamma_base * 2**(i + 1) for i in range(-self.n_gammas//2, self.n_gammas//2)]
        # self._raw_gammas = [self._raw_gamma_base/70, self._raw_gamma_base/50, self._raw_gamma_base/30, self._raw_gamma_base/10, self._raw_gamma_base, self._raw_gamma_base*10, self._raw_gamma_base*30, self._raw_gamma_base*50, self._raw_gamma_base*70]
        # self._raw_gammas = [self._raw_gamma_base/50, self._raw_gamma_base/30, self._raw_gamma_base/10, self._raw_gamma_base, self._raw_gamma_base*10, self._raw_gamma_base*30, self._raw_gamma_base*50]
        # self._raw_gammas = [self._raw_gamma_base/30, self._raw_gamma_base/10, self._raw_gamma_base, self._raw_gamma_base*10, self._raw_gamma_base*30]
        # self._raw_gammas = [self._raw_gamma_base/10, self._raw_gamma_base, self._raw_gamma_base*10]
        mmd_seq = []
        for raw_gamma in self._raw_gammas:
            # positive_gamma = F.softplus(raw_gamma)
            positive_gamma = raw_gamma
            # positive_gamma = torch.relu(raw_gamma) + 1e-6
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
class GINTransformer_GPS(nn.Module):
    def __init__(self,
                 input_dim: int = None, hidden_dim: int = None,
                 gin_layers: int = 3, transformer_layers: int = 1,
                 num_gps_layers: int = 3,
                 n_gammas=9,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(GINTransformer_GPS, self).__init__()
        self.config = config
        self.model_type = config['model_type']
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gin_layers = gin_layers
        self.transformer_layers = transformer_layers
        self.num_gps_layers = num_gps_layers
        self.n_gammas = n_gammas
        self.device = device

        # define the GraphGPS architecture here
        self.build_GraphGPS_block()

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

        self._init_weights(self.gps_layers)
        self._init_weights(self.mlp_head)

    def build_GraphGPS_block(self):
        from torch_geometric.nn import GPSConv
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.gps_layers = nn.ModuleList()
        for _ in range(self.num_gps_layers):
            gin_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.gps_layers.append(GPSConv(channels=self.hidden_dim,
                                           conv=GINConv(gin_mlp),
                                           heads=4))
            
        self.fc = nn.Linear(self.hidden_dim, self.config['embedding_dim_for_mmd_layer'])
            
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        # feedforward two graphs through GraphGPS layers, respectively
        Z1 = F.relu(self.input_embedding(X1))
        Z2 = F.relu(self.input_embedding(X2))
        for gps_layer in self.gps_layers:
            Z1 = F.relu(gps_layer(Z1, edge_index1))
            Z2 = F.relu(gps_layer(Z2, edge_index2))

        # embed to 128-dim
        Z1, Z2 = self.fc(Z1), self.fc(Z2)

        # feedforward it into MMD layer
        o = self.mmd_layer.forward(X1=Z1, X2=Z2)

        # feedforward it into MLP head
        o = self.mlp_head(o)

        return o


class GINTransformer_GPS_Naive(GINTransformer_GPS):
    def __init__(self,
                 input_dim: int = None, hidden_dim: int = None,
                 gin_layers: int = 3, transformer_layers: int = 1,
                 num_gps_layers: int = 3,
                 n_gammas=9,
                 config: Dict = None,
                 device = torch.device('cpu')):
        super(GINTransformer_GPS_Naive, self).__init__(input_dim=input_dim,
                                                      hidden_dim=hidden_dim,
                                                      gin_layers=gin_layers,
                                                      transformer_layers=transformer_layers,
                                                      num_gps_layers=num_gps_layers,
                                                      n_gammas=n_gammas,
                                                      config=config,
                                                      device=device)
        del self._modules['mmd_layer'], self.mlp_head
        self.fc = nn.Linear(self.hidden_dim, 32)
        from torch_geometric.nn import global_mean_pool
        self.readout = global_mean_pool
        
    def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        # feedforward two graphs through GraphGPS layers, respectively
        Z1 = F.relu(self.input_embedding(X1))
        Z2 = F.relu(self.input_embedding(X2))
        for gps_layer in self.gps_layers:
            Z1 = F.relu(gps_layer(Z1, edge_index1))
            Z2 = F.relu(gps_layer(Z2, edge_index2))

        Z1 = F.relu(self.fc(Z1))
        Z2 = F.relu(self.fc(Z2))
        z1 = self.readout(Z1, batch=None)
        z2 = self.readout(Z2, batch=None)
        o = torch.linalg.vector_norm(z1 - z2)**2

        return o
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
            self.fc = nn.Linear(self.input_dim, self.config['embedding_dim_for_mmd_layer'])
        elif self.switch == 'gin':
            print(f'[Info] Ablation study: removing Transformer layers.')
            if self.model_type == 'serial':
                del self._modules['attn_layer_for_serial']
            elif self.model_type == 'parallel':
                del self._modules['attn_layer_for_parallel']
            else:
                raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")
            
            self.fc = nn.Linear(self.hidden_dim + self.input_dim, self.config['embedding_dim_for_mmd_layer'])
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
        O1, O2 = self.fc(Z1), self.fc(Z2)
        o = self.mmd_layer.forward(X1=O1, X2=O2)
        # o = self.mmd_layer.forward(X1=Z1, X2=Z2)

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
        self.kernel_layer = KernelLayer(n_gammas=n_gammas, device=self.device)

        # define the last MLP head
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(n_gammas, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )

        # For MUTAG
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(n_gammas, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )

        # For BZR
        self.mlp_head = nn.Sequential(
            nn.Linear(n_gammas, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
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

        self.fc = nn.Linear(self.hidden_dim + self.input_dim, self.config['embedding_dim_for_mmd_layer'])

    def build_parallel_model(self):
        # self.attn_layer_for_parallel = nn.TransformerEncoderLayer(
        #     d_model=self.input_dim,
        #     nhead=4
        # )

        encoder = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=4
        )

        self.attn_layer_for_parallel = nn.TransformerEncoder(
            encoder_layer=encoder,
            num_layers=self.config['num_transformer_layers']
        )

        self.fc = nn.Linear(self.hidden_dim + 2*self.input_dim, self.config['embedding_dim_for_mmd_layer'])

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
    def __init__(self, n_gammas: int = 9, device = torch.device('cpu')):
        super(KernelLayer, self).__init__()
        assert n_gammas%2 == 1, f"The width of Kernel layer: 'n_gammas={n_gammas}' must be odd."
        self.n_gammas = n_gammas
        # self._raw_gammas = nn.Linear(n_gammas, 1, bias=False).weight.view(-1)
        # self._raw_gamma_base = torch.log(torch.expm1(torch.tensor(0.001)))
        # self._raw_gamma_base = torch.tensor(0.001)
        self.gamma_layer = nn.Linear(n_gammas, 1, bias=False).to(device)
        # nn.init.constant_(self.gamma_layer.weight, val=torch.log(torch.expm1(torch.tensor(0.1))).item())
        self._raw_gammas = self.gamma_layer.weight.view(-1)
        print(f'[Info] The initial value of _raw_gammas: {self._raw_gammas.detach().cpu().numpy()}')
    
    def forward(self, X1: torch.Tensor, X2: torch.Tensor, is_biased: bool = False) -> torch.Tensor:
        assert X1.dim() == 2 and X2.dim() == 2, "Input 'X1' and 'X2' must be 2D Tensors"
        n1, n2 = X1.shape[0], X2.shape[0]
        D2 = torch.cdist(X1, X2, compute_mode='use_mm_for_euclid_dist')**2
        # self._raw_gammas = [self._raw_gamma_base * 2**(i + 1) for i in range(-self.n_gammas//2, self.n_gammas//2)]
        # self._raw_gammas = [self._raw_gamma_base * (c+1)**2 for c in range(self.n_gammas)]
        K_seq = []
        for raw_gamma in self._raw_gammas:
            # positive_gamma = F.relu(raw_gamma) + 1e-6
            positive_gamma = F.softplus(raw_gamma)
            # positive_gamma = raw_gamma
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
    

# class GINTransformer_TP(nn.Module):
#     def __init__(self,
#                  input_dim: int, hidden_dim: int,
#                  gin_layers: int = 3, transformer_layers: int = 1,
#                  n_gammas=9,
#                  config: Dict = None,
#                  device = torch.device('cpu')):
#         super(GINTransformer_TP, self).__init__()
#         self.config = config
#         self.model_type = config['model_type']
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.gin_layers = gin_layers
#         self.transformer_layers = transformer_layers
#         self.num_gps_layers = config['num_gps_layers']
#         self.n_gammas = n_gammas
#         self.device = device

#         # define the GraphGPS architecture here
#         self.build_GraphGPS_block()

#         # define the intermediate kernel layers
#         self.kernel_layer = KernelLayer(n_gammas=n_gammas, device=self.device)

#         # define the last MLP head
#         self.mlp_head = nn.Sequential(
#             nn.Linear(n_gammas, 512),
#             nn.ReLU(),
#             nn.Linear(512, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             # nn.Sigmoid()
#         )

#         self._init_weights(self.gps_layers)
#         self._init_weights(self.mlp_head)

#     def build_GraphGPS_block(self):
#         from torch_geometric.nn import GPSConv
#         self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
#         self.gps_layers = nn.ModuleList()
#         for _ in range(self.num_gps_layers):
#             gin_mlp = nn.Sequential(
#                 nn.Linear(self.hidden_dim, self.hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(self.hidden_dim, self.hidden_dim)
#             )
#             self.gps_layers.append(GPSConv(channels=self.hidden_dim,
#                                            conv=GINConv(gin_mlp),
#                                            heads=4))
            
#         self.fc = nn.Linear(self.hidden_dim, self.config['embedding_dim_for_mmd_layer'])
    
#     @staticmethod
#     def _init_weights(layer):
#         for m in layer.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)


#     @staticmethod
#     def _init_transformer_weights(layer):
#         for m in layer.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, X1: torch.Tensor, edge_index1: torch.Tensor, X2: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
#         # feedforward it into the backbone - A hybrid model of GIN and Transformer
#         if self.model_type == 'serial':
#             Z1, Z2 = self.forward_siamese_serial(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
#         elif self.model_type == 'parallel':
#             Z1, Z2 = self.forward_siamese_parallel(X1=X1, edge_index1=edge_index1, X2=X2, edge_index2=edge_index2, is_whole_attn=True)
#         else:
#             raise ValueError(f"Unknown model type: {self.model_type}. Supported types are 'serial' and 'parallel'.")

#         # feedforward it into kernel layer
#         Kernel_Z = self.kernel_layer.forward(X1=Z1, X2=Z2)

#         # feedforward it into MLP head
#         o = self.mlp_head(Kernel_Z)
#         tp_pred = o.reshape(Z1.shape[0], Z2.shape[0])

#         return tp_pred