import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
from typing import Tuple, Optional

def build_line_graph_adj_matrix(connections: pd.DataFrame, dtype=torch.float32):
    """
    Build line-graph adjacency matrix A_S (edges-as-nodes) from `connections` DataFrame
    with columns ['C1', 'C2'] or from a networkx Graph.
    Returns (A_s_sparse: torch.sparse_coo_tensor, edge_list: List[Tuple(u,v)])
    """
    # Step 1: Build the base undirected graph from connections
    G = nx.from_pandas_edgelist(connections, 'C1', 'C2', create_using=nx.Graph())
    orig_edges = list(G.edges())
    nE = len(orig_edges)

    # Map original edge -> index in line-graph space
    edge2idx = {e: i for i, e in enumerate(orig_edges)}
    rows, cols = [], []

    # Step 2: Build connectivity between edges that share a node
    for (u, v) in nx.line_graph(G).edges():
        i = edge2idx[u]
        j = edge2idx[v]
        # Add symmetric connections (since graph is undirected)
        rows.append(i); cols.append(j)
        rows.append(j); cols.append(i)

    # Step 3: Add self-loops to stabilize GCN normalization
    for i in range(nE):
        rows.append(i); cols.append(i)

    # Step 4: Convert to sparse COO format
    if len(rows) == 0:
        indices = torch.empty((2,0), dtype=torch.int64)
        values = torch.empty((0,), dtype=dtype)
    else:
        indices = torch.tensor([rows, cols], dtype=torch.int64)
        values = torch.ones(indices.shape[1], dtype=dtype)

    A = torch.sparse_coo_tensor(indices, values, (nE, nE))
    return A.coalesce(), orig_edges

def normalize_sparse_adj(A: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    """
    Apply symmetric normalization:  A_hat = D^{-1/2} * A * D^{-1/2}.
    This ensures numerical stability and equal weighting in message passing.
    """
    A = A.coalesce()
    indices = A.indices()
    values = A.values()
    n = A.shape[0]

    # Compute node degrees from sparse indices
    deg = torch.zeros(n, dtype=values.dtype)
    for idx, val in zip(indices.t(), values):
        deg[idx[0]] += val

    # Invert sqrt of degree (avoid div by zero)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0.0

    # Apply normalization elementwise
    row, col = indices[0, :], indices[1, :]
    vals = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(indices, vals, A.shape).coalesce()

def sparse_mm(A: torch.sparse_coo_tensor, X: torch.Tensor) -> torch.Tensor:
    """Sparse-dense matrix multiplication wrapper for readability."""
    return torch.spmm(A, X)

class EdgeLineGraphBuilder:
    """
    Class to construct and hold normalized line-graph adjacency.
    """
    def __init__(self, connections_df: pd.DataFrame):
        self.connections_df = connections_df
        self.A_s, self.edge_list = build_line_graph_adj_matrix(connections_df)
        self.A_s = normalize_sparse_adj(self.A_s)
        self.n_edges = len(self.edge_list)
    def get_adj(self) -> torch.sparse_coo_tensor:
        return self.A_s
    def get_edge_list(self):
        return self.edge_list

class MLP(nn.Module):
    """
    Feedforward network used for embedding edge & weather features.
    """

    def __init__(self, in_dim, out_dim, hidden_dims=(), activation=nn.ReLU, final_activation=None):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(activation())
        if final_activation is not None:
            layers.append(final_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SimpleGraphConv(nn.Module):
    """
    Sparse GCN-style graph convolution layer:
      H' = σ(A_hat * H * W + b)
    """
    def __init__(self, in_dim, out_dim, activation=nn.ReLU, use_bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)
        self.activation = activation()
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights uniformly for stable early training
        stdv = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, A_sparsed: torch.sparse_coo_tensor, H: torch.Tensor):
        # (N, F_in) * (F_in, F_out)
        HW = H.matmul(self.W)
        out = sparse_mm(A_sparsed, HW)
        if self.bias is not None:
            out = out + self.bias
        return self.activation(out)

class TemporalGatedConv(nn.Module):
    """
    Temporal modeling via gated 1D convolution (WaveNet-style):
      output = tanh(Conv_f(X)) * sigmoid(Conv_g(X))
    Captures nonlinear temporal interactions over hours.
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, dilation=1):
        super().__init__()
        # Use causal padding to preserve sequence length
        padding = (kernel_size-1) * dilation
        self.conv_f = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, dilation=dilation)
        self.conv_g = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, dilation=dilation)
        self.out_dim = out_dim

    def forward(self, X):
        # Expect shape [N, T, D] -> [N, D, T] for Conv1d
        x = X.transpose(1, 2)
        # Slice to ensure original temporal length (drop padding artifacts)
        f = self.conv_f(x)[..., :X.shape[1]]
        g = self.conv_g(x)[..., :X.shape[1]]
        out = torch.tanh(f) * torch.sigmoid(g)
        out = out.transpose(1, 2) # restore [N, T, D]
        return out

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model
    def forward(self, X, src_mask: Optional[torch.Tensor]=None):
        return self.transformer(X, mask=src_mask)

class TemporalAdjLearner(nn.Module):
    def __init__(self, in_dim, key_dim=64, topk=8, sparsify=True):
        super().__init__()
        self.key = nn.Linear(in_dim, key_dim)
        self.query = nn.Linear(in_dim, key_dim)
        self.topk = topk
        self.sparsify = sparsify

    def forward(self, U: torch.Tensor) -> torch.sparse_coo_tensor:
        U_pool = U.mean(dim=1)
        Q = self.query(U_pool)
        K = self.key(U_pool)
        scores = torch.matmul(Q, K.transpose(0,1)) / math.sqrt(Q.size(1))
        attn = F.softmax(scores, dim=1)
        if not self.sparsify:
            indices = torch.nonzero(attn > 0, as_tuple=False).t().contiguous()
            values = attn[indices[0], indices[1]]
            return torch.sparse_coo_tensor(indices, values, (attn.size(0), attn.size(1))).coalesce()
        topk_vals, topk_idx = torch.topk(attn, k=min(self.topk, attn.size(1)), dim=1)
        rows = torch.arange(attn.size(0)).unsqueeze(1).repeat(1, topk_idx.size(1)).flatten()
        cols = topk_idx.flatten()
        vals = topk_vals.flatten()
        indices = torch.stack([rows, cols], dim=0)
        A_t = torch.sparse_coo_tensor(indices, vals, (attn.size(0), attn.size(1)))
        return A_t.coalesce()

class FusionAdjacency(nn.Module):
    def __init__(self, n_nodes, learnable_scalar=True):
        super().__init__()
        if learnable_scalar:
            self.gamma = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('gamma', torch.tensor(0.0))
        self.n = n_nodes
    def forward(self, A_s: torch.sparse_coo_tensor, A_t: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        As_dense = A_s.to_dense()
        At_dense = A_t.to_dense()
        alpha = torch.sigmoid(self.gamma)
        Af_dense = alpha * As_dense + (1.0 - alpha) * At_dense
        row_sum = Af_dense.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        Af_dense = Af_dense / row_sum
        return Af_dense.to_sparse().coalesce()

class SpatioTemporalFusionBlock(nn.Module):
    def __init__(self, n_nodes, in_dim, hidden_dim, temporal_mode='gated'):
        super().__init__()
        self.gconv = SimpleGraphConv(in_dim, hidden_dim)
        self.temporal_mode = temporal_mode
        if temporal_mode == 'gated':
            self.temporal = TemporalGatedConv(hidden_dim, hidden_dim, kernel_size=3)
        elif temporal_mode == 'transformer':
            self.temporal = TemporalTransformerEncoder(d_model=hidden_dim, n_heads=4, n_layers=1)
        else:
            raise ValueError("Unknown temporal_mode")
        self.res_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H: torch.Tensor, A_f: torch.sparse_coo_tensor):
        N, T, D = H.shape
        H_out_time = []
        for t in range(T):
            h_t = H[:, t, :]
            g_t = self.gconv(A_f, h_t)
            H_out_time.append(g_t.unsqueeze(1))
        H_gc = torch.cat(H_out_time, dim=1)
        H_temp = self.temporal(H_gc)
        residual = self.res_proj(H).to(H_temp.dtype)
        out = self.norm(H_temp + residual)
        return out

class E_STFGNN(nn.Module):
    def __init__(self, n_edges:int, in_feat_dim:int, weather_dim:int,
                 d_model=128, n_blocks=3, temporal_mode='gated', topk=8):
        super().__init__()
        self.n_edges = n_edges
        self.d_model = d_model
        self.edge_embed = MLP(in_feat_dim, d_model, hidden_dims=(d_model//2,))
        self.weather_embed = MLP(weather_dim, d_model, hidden_dims=(d_model//2,))
        self.combine = nn.Linear(d_model*2, d_model)
        self.temporal_learner = TemporalAdjLearner(in_dim=d_model, key_dim=d_model//2, topk=topk, sparsify=True)
        self.fusion = FusionAdjacency(n_nodes=n_edges, learnable_scalar=True)
        self.blocks = nn.ModuleList([SpatioTemporalFusionBlock(n_edges, d_model, d_model, temporal_mode=temporal_mode) for _ in range(n_blocks)])
        self.pred_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))

    def forward(self, X_edges: torch.Tensor, X_weather_edges: torch.Tensor, A_s: torch.sparse_coo_tensor) -> torch.Tensor:
        N, T, _ = X_edges.shape
        Xe = self.edge_embed(X_edges.view(-1, X_edges.size(-1))).view(N, T, -1)
        Xw = self.weather_embed(X_weather_edges.view(-1, X_weather_edges.size(-1))).view(N, T, -1)
        H0 = torch.cat([Xe, Xw], dim=-1)
        H0 = torch.relu(self.combine(H0))
        A_t = self.temporal_learner(H0)
        A_t = normalize_sparse_adj(A_t)
        A_s = A_s.coalesce()
        A_f = self.fusion(A_s, A_t)
        H = H0
        for block in self.blocks:
            H = block(H, A_f)
        H_last = H[:, -1, :]
        logits = self.pred_head(H_last).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs.unsqueeze(-1)

def tiny_sanity_check():
    df_conn = pd.DataFrame({'C1': ['A', 'A', 'B', 'C'], 'C2': ['B', 'C', 'D', 'D']})
    builder = EdgeLineGraphBuilder(df_conn)
    A_s = builder.get_adj()
    nE = builder.n_edges
    T = 6
    in_feat = 5
    weather_feat = 4
    X_edges = torch.randn(nE, T, in_feat)
    X_weather_edges = torch.randn(nE, T, weather_feat)
    model = E_STFGNN(n_edges=nE, in_feat_dim=in_feat, weather_dim=weather_feat, d_model=64, n_blocks=2, temporal_mode='gated', topk=3)
    with torch.no_grad():
        out = model(X_edges, X_weather_edges, A_s)
    print(f"n_edges={nE}, output shape={out.shape} (format [N,1])")
    return model, builder, out

model, builder, out = tiny_sanity_check()
