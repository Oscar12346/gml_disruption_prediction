import torch
import torch.nn as nn
import math

def sparse_mm(A: torch.sparse_coo_tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Multiply sparse matrix A (N x N) with dense matrix X (N x D) -> (N x D)
    """
    return torch.spmm(A, X)

class SimpleGraphConv(nn.Module):
    """
    Graph convolution layer for node features:
    H' = activation( A @ (H W) + b )

    - in_dim: input feature dimension
    - out_dim: output feature dimension
    """
    def __init__(self, in_dim: int, out_dim: int, activation=nn.ReLU, use_bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None
        self.activation = activation()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(max(1, self.W.size(1)))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, A_sparsed: torch.sparse_coo_tensor, H: torch.Tensor) -> torch.Tensor:
        """
        H: [N, D_in]
        Returns: [N, D_out]
        """
        HW = H.matmul(self.W)  # [N, D_out]
        out = sparse_mm(A_sparsed, HW)  # [N, D_out]
        if self.bias is not None:
            out = out + self.bias
        return self.activation(out)

class TemporalGatedConv(nn.Module):
    """
    Gated temporal convolution (WaveNet-style).
    Input: [N, T, C_in] -> transpose -> Conv1d(C_in, C_out) -> gated activation -> [N, T, C_out]
    """
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        # two convs: filter and gate
        self.conv_f = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, dilation=dilation)
        self.conv_g = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, dilation=dilation)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [N, T, C_in]
        x = X.transpose(1, 2)  # [N, C_in, T]
        f = self.conv_f(x)[..., :X.shape[1]]  # allow for padding cropping
        g = self.conv_g(x)[..., :X.shape[1]]
        out = torch.tanh(f) * torch.sigmoid(g)  # gating
        return out.transpose(1, 2)  # [N, T, C_out]

class MLP(nn.Module):
    """
    Small multilayer perceptron used for embeddings.
    in_dim -> hidden_dims -> out_dim
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dims=(), activation=nn.ReLU, final_activation=None):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        if final_activation is not None:
            layers.append(final_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class SpatioTemporalFusionBlock(nn.Module):
    """
    One block that does:
      - spatial: graph conv (using fused adjacency A_f)
      - temporal: gated conv across time
      - residual connection + layernorm
    Input H: [N, T, D]
    """
    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int):
        super().__init__()
        self.gconv = SimpleGraphConv(in_dim, hidden_dim)
        self.temporal = TemporalGatedConv(hidden_dim, hidden_dim, kernel_size=3)
        self.res_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H: torch.Tensor, A_f: torch.sparse_coo_tensor) -> torch.Tensor:
        N, T, D = H.shape
        H_out_time = []
        # apply graph conv per time step (spatial propagation)
        for t in range(T):
            h_t = H[:, t, :]          # [N, D]
            g_t = self.gconv(A_f, h_t)  # [N, hidden_dim]
            H_out_time.append(g_t.unsqueeze(1))  # [N,1,hidden_dim]
        H_gc = torch.cat(H_out_time, dim=1)      # [N, T, hidden_dim]
        H_temp = self.temporal(H_gc)             # [N, T, hidden_dim]
        residual = self.res_proj(H).to(H_temp.dtype)
        return self.norm(H_temp + residual)

# -------------------------------------------------------------------
# Temporal adjacency learner + fusion (new components)
# -------------------------------------------------------------------
def normalize_sparse_adj(A: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    """
    Row-normalize a sparse adjacency A: each row sums to 1 (A_ij <- A_ij / row_sum_i)
    Returns a coalesced sparse tensor on the same device.
    """
    A = A.coalesce()
    indices = A.indices()
    values = A.values()
    n = A.shape[0]
    row = indices[0]
    row_sum = torch.zeros(n, dtype=values.dtype, device=values.device)
    # index_add to compute row sums
    row_sum = row_sum.index_add(0, row, values)
    # avoid division by zero
    row_sum[row_sum == 0] = 1.0
    inv = 1.0 / row_sum
    new_vals = values * inv[row]
    return torch.sparse_coo_tensor(indices, new_vals, A.shape, device=values.device).coalesce()

class TemporalAdjLearner(nn.Module):
    """
    Learns a temporal adjacency A_t from per-edge features U: [N, T, D].
    Produces a sparse (top-k) adjacency with row-normalization.
    """
    def __init__(self, in_dim: int, key_dim: int = 64, topk: int = 8, sparsify: bool = True):
        super().__init__()
        self.query = nn.Linear(in_dim, key_dim)
        self.key = nn.Linear(in_dim, key_dim)
        self.topk = topk
        self.sparsify = sparsify
        self.key_dim = key_dim

    def forward(self, U: torch.Tensor) -> torch.sparse_coo_tensor:
        """
        U: [N, T, D]
        Returns A_t: sparse (N x N) adjacency (row-normalized)
        """
        device = U.device
        N, T, D = U.shape

        # pool in time dimension to summarize each edge's behavior
        U_pool = U.mean(dim=1)   # [N, D]
        Q = self.query(U_pool)   # [N, K]
        K = self.key(U_pool)     # [N, K]

        # scaled dot-product affinity
        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(max(1.0, self.key_dim))

        # numerical stability
        scores = scores - scores.max(dim=1, keepdim=True)[0]
        attn = torch.softmax(scores, dim=1)  # [N, N] dense
        if not self.sparsify:
            # convert to sparse and normalize
            indices = torch.nonzero(attn > 0.0, as_tuple=False).t().contiguous().to(device)
            values = attn[indices[0], indices[1]]
            A_t = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
            return normalize_sparse_adj(A_t)

        # sparsify: top-k per row
        k = min(self.topk, attn.size(1))
        topk_vals, topk_idx = torch.topk(attn, k=k, dim=1)
        rows = torch.arange(N, device=device).unsqueeze(1).repeat(1, k).reshape(-1)
        cols = topk_idx.reshape(-1)
        vals = topk_vals.reshape(-1)
        indices = torch.stack([rows, cols], dim=0)
        A_t = torch.sparse_coo_tensor(indices, vals, (N, N), device=device).coalesce()
        return normalize_sparse_adj(A_t)

class FusionAdjacency(nn.Module):
    """
    Fuse static A_s and learned A_t using a scalar gamma:
      alpha = sigmoid(gamma)
      A_f = alpha * A_s + (1-alpha) * A_t
    Then row-normalize and return sparse.
    """
    def __init__(self, n_nodes: int, learnable_scalar: bool = True):
        super().__init__()
        if learnable_scalar:
            self.gamma = nn.Parameter(torch.tensor(0.0))  # 0 -> alpha = 0.5 at start
        else:
            self.register_buffer('gamma', torch.tensor(0.0))
        self.n = n_nodes

    def forward(self, A_s: torch.sparse_coo_tensor, A_t: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        # convert to dense
        device = A_s.device
        As = A_s.coalesce().to_dense().to(device)
        At = A_t.coalesce().to_dense().to(device)
        alpha = torch.sigmoid(self.gamma)
        Af = alpha * As + (1.0 - alpha) * At

        # row-normalize
        row_sum = Af.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        Af = Af / row_sum
        return Af.to_sparse().coalesce()
