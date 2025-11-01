import torch
import torch.nn as nn
import math

def sparse_mm(A: torch.sparse_coo_tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Perform sparse-dense matrix multiplication.

    Parameters
    ----------
    A : torch.sparse_coo_tensor
        Sparse adjacency matrix of shape [N, N].
    X : torch.Tensor
        Dense feature matrix of shape [N, D].

    Returns
    -------
    torch.Tensor
        Result of sparse matrix multiplication (A @ X) with shape [N, D].
    """
    return torch.spmm(A, X)


class SimpleGraphConv(nn.Module):
    """
    Basic graph convolution layer for spatial feature propagation.

    Implements the operation:
        H' = activation(A @ (H W) + b)

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    activation : nn.Module, optional
        Activation function applied after aggregation. Default is ReLU.
    use_bias : bool, optional
        Whether to include a learnable bias term. Default is True.
    """

    def __init__(self, in_dim: int, out_dim: int, activation=nn.ReLU, use_bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None
        self.activation = activation()
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using a uniform distribution based on output dimension."""
        stdv = 1.0 / math.sqrt(max(1, self.W.size(1)))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, A_sparsed: torch.sparse_coo_tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph convolution.

        Parameters
        ----------
        A_sparsed : torch.sparse_coo_tensor
            Sparse adjacency matrix [N, N].
        H : torch.Tensor
            Node feature matrix [N, D_in].

        Returns
        -------
        torch.Tensor
            Updated node features [N, D_out].
        """
        HW = H.matmul(self.W)
        out = sparse_mm(A_sparsed, HW)
        if self.bias is not None:
            out = out + self.bias
        return self.activation(out)


class TemporalGatedConv(nn.Module):
    """
    Gated temporal convolution (WaveNet-style) for temporal feature extraction.

    Parameters
    ----------
    in_dim : int
        Number of input channels.
    out_dim : int
        Number of output channels.
    kernel_size : int, optional
        Size of the temporal convolution kernel. Default is 3.
    dilation : int, optional
        Dilation factor for temporal convolution. Default is 1.
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # Two convolution branches for gating mechanism
        self.conv_f = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, dilation=dilation)
        self.conv_g = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, dilation=dilation)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of gated temporal convolution.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape [N, T, C_in].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, T, C_out].
        """
        x = X.transpose(1, 2)  # [N, C_in, T]
        f = self.conv_f(x)[..., :X.shape[1]]  # Crop to match input length
        g = self.conv_g(x)[..., :X.shape[1]]
        out = torch.tanh(f) * torch.sigmoid(g)
        return out.transpose(1, 2)


class MLP(nn.Module):
    """
    Multilayer Perceptron used for embedding and transformation.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    hidden_dims : tuple, optional
        Sequence of hidden layer sizes. Default is empty.
    activation : nn.Module, optional
        Activation function used between hidden layers. Default is ReLU.
    final_activation : nn.Module, optional
        Optional activation applied to the final layer. Default is None.
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
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [*, in_dim].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [*, out_dim].
        """
        return self.model(x)


class SpatioTemporalFusionBlock(nn.Module):
    """
    Combined spatial-temporal fusion block for feature refinement.

    Performs:
        1. Spatial graph convolution over nodes using fused adjacency (A_f)
        2. Temporal gated convolution across time dimension
        3. Residual connection followed by layer normalization

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden (output) feature dimension.
    """

    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int):
        super().__init__()
        self.gconv = SimpleGraphConv(in_dim, hidden_dim)
        self.temporal = TemporalGatedConv(hidden_dim, hidden_dim, kernel_size=3)
        self.res_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H: torch.Tensor, A_f: torch.sparse_coo_tensor) -> torch.Tensor:
        """
        Forward pass of the spatio-temporal fusion block.

        Parameters
        ----------
        H : torch.Tensor
            Input tensor of shape [N, T, D].
        A_f : torch.sparse_coo_tensor
            Fused adjacency matrix [N, N].

        Returns
        -------
        torch.Tensor
            Updated features of shape [N, T, hidden_dim].
        """
        N, T, D = H.shape
        H_out_time = []

        for t in range(T):  # Apply graph convolution per time step
            h_t = H[:, t, :]
            g_t = self.gconv(A_f, h_t)
            H_out_time.append(g_t.unsqueeze(1))

        H_gc = torch.cat(H_out_time, dim=1)
        H_temp = self.temporal(H_gc)
        residual = self.res_proj(H).to(H_temp.dtype)

        return self.norm(H_temp + residual)


def normalize_sparse_adj(A: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    """
    Row-normalize a sparse adjacency matrix.

    Parameters
    ----------
    A : torch.sparse_coo_tensor
        Sparse adjacency matrix [N, N].

    Returns
    -------
    torch.sparse_coo_tensor
        Row-normalized adjacency matrix.
    """
    A = A.coalesce()
    indices = A.indices()
    values = A.values()
    n = A.shape[0]
    row = indices[0]

    row_sum = torch.zeros(n, dtype=values.dtype, device=values.device)
    row_sum = row_sum.index_add(0, row, values)
    row_sum[row_sum == 0] = 1.0
    inv = 1.0 / row_sum
    new_vals = values * inv[row]

    return torch.sparse_coo_tensor(indices, new_vals, A.shape, device=values.device).coalesce()


class TemporalAdjLearner(nn.Module):
    """
    Learns a temporal adjacency matrix (A_t) from dynamic edge features.

    Computes pairwise attention between edges based on time-aggregated representations,
    and optionally sparsifies the matrix to retain top-k strongest connections.

    Parameters
    ----------
    in_dim : int
        Input feature dimension per edge.
    key_dim : int, optional
        Dimension of key/query embeddings used for attention. Default is 64.
    topk : int, optional
        Number of top attention connections to retain per node. Default is 8.
    sparsify : bool, optional
        Whether to sparsify the attention matrix. Default is True.
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
         Forward pass of the temporal adjacency learner.

         Parameters
         ----------
         U : torch.Tensor
             Input tensor of shape [N, T, D] representing temporal edge features.

         Returns
         -------
         torch.sparse_coo_tensor
             Learned sparse adjacency matrix A_t of shape [N, N].
         """
        device = U.device
        N, T, D = U.shape

        # Temporal pooling to get per-edge summary
        U_pool = U.mean(dim=1)
        Q = self.query(U_pool)
        K = self.key(U_pool)

        # Compute scaled dot-product similarity
        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(max(1.0, self.key_dim))
        scores = scores - scores.max(dim=1, keepdim=True)[0]
        attn = torch.softmax(scores, dim=1)

        if not self.sparsify:
            indices = torch.nonzero(attn > 0.0, as_tuple=False).t().contiguous().to(device)
            values = attn[indices[0], indices[1]]
            A_t = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
            return normalize_sparse_adj(A_t)

        # Retain only top-k connections per node
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
    Fuse static (A_s) and learned temporal (A_t) adjacency matrices.

    Combines the two using a learnable scalar gamma:
        α = sigmoid(γ)
        A_f = α * A_s + (1 - α) * A_t
    The result is row-normalized and returned as a sparse tensor.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    learnable_scalar : bool, optional
        Whether to make the fusion parameter γ learnable. Default is True.
    """

    def __init__(self, n_nodes: int, learnable_scalar: bool = True):
        super().__init__()
        if learnable_scalar:
            self.gamma = nn.Parameter(torch.tensor(0.0))  # 0 -> alpha = 0.5 at start
        else:
            self.register_buffer('gamma', torch.tensor(0.0))
        self.n = n_nodes

    def forward(self, A_s: torch.sparse_coo_tensor, A_t: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Forward pass of the fusion module.

        Parameters
        ----------
        A_s : torch.sparse_coo_tensor
            Static adjacency matrix [N, N].
        A_t : torch.sparse_coo_tensor
            Learned temporal adjacency matrix [N, N].

        Returns
        -------
        torch.sparse_coo_tensor
            Fused and row-normalized adjacency matrix [N, N].
        """
        device = A_s.device
        As = A_s.coalesce().to_dense().to(device)
        At = A_t.coalesce().to_dense().to(device)
        alpha = torch.sigmoid(self.gamma)
        Af = alpha * As + (1.0 - alpha) * At

        # Row-normalize
        row_sum = Af.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        Af = Af / row_sum
        return Af.to_sparse().coalesce()
