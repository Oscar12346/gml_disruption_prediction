from src.models.advanced_novel_model.layers import *
from typing import Optional
import torch


class E_STFGNN(nn.Module):
    """
    Edge-centric Spatio-Temporal Fusion Graph Neural Network (E-STFGNN).

    This model integrates spatial graph dependencies, temporal dynamics, and
    meteorological signals for disruption prediction. It learns a dynamic
    (temporal) adjacency matrix from temporal embeddings and fuses it with
    a static topological adjacency to create a unified spatio-temporal graph
    representation.

    Components:
        - Edge and weather feature embeddings
        - Temporal adjacency learner (learns A_t)
        - Fusion mechanism combining A_s (static) and A_t (learned)
        - Stacked spatio-temporal fusion blocks for joint reasoning
        - Prediction head for disruption duration

    Parameters
    ----------
    n_edges : int
        Number of edges (nodes in the line graph).
    in_feat_dim : int
        Dimensionality of the edge input features.
    weather_dim : int
        Dimensionality of the weather input features.
    d_model : int, optional
        Hidden dimension size of internal embeddings. Default is 64.
    n_blocks : int, optional
        Number of stacked SpatioTemporalFusionBlocks. Default is 2.
    key_dim : int, optional
        Dimensionality of the query/key vectors in TemporalAdjLearner.
        Defaults to d_model // 2.
    topk : int, optional
        Number of top connections retained per node in learned adjacency.
        Default is 8.
    """

    def __init__(self, n_edges:int, in_feat_dim:int, weather_dim:int, d_model:int = 64, n_blocks:int = 2,
                 key_dim: Optional[int] = None, topk:int = 8):
        super().__init__()
        self.n_edges = n_edges
        self.d_model = d_model
        key_dim = key_dim or (d_model // 2)

        # Embedding layers for edge and weather features
        self.edge_embed = MLP(in_feat_dim, d_model, hidden_dims=(max(8, d_model // 2),))
        self.weather_embed = MLP(weather_dim, d_model, hidden_dims=(max(8, d_model // 2),))
        self.combine = nn.Linear(d_model*2, d_model)

        # Temporal adjacency learner and fusion layer
        self.temporal_learner = TemporalAdjLearner(in_dim=d_model, key_dim=key_dim, topk=topk, sparsify=True)
        self.fusion = FusionAdjacency(n_nodes=n_edges, learnable_scalar=True)

        # Stacked spatio-temporal blocks and final prediction head
        self.blocks = nn.ModuleList([SpatioTemporalFusionBlock(n_edges, d_model, d_model) for _ in range(n_blocks)])
        self.pred_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))

    def forward(self, X_edges: torch.Tensor, X_weather_edges: torch.Tensor, A_s: torch.sparse_coo_tensor) -> torch.Tensor:
        """
        Forward pass of the E-STFGNN model.

        Parameters
        ----------
        X_edges : torch.Tensor
            Edge feature tensor of shape [N, T, in_feat_dim], where
            N is the number of edges and T is the number of time steps.
        X_weather_edges : torch.Tensor
            Weather feature tensor of shape [N, T, weather_dim].
        A_s : torch.sparse_coo_tensor
            Static adjacency matrix representing topological connectivity.

        Returns
        -------
        torch.Tensor
            Predicted disruption durations with shape [N, 1].
        """
        N, T, _ = X_edges.shape

        # Embed edge and weather sequences across all time steps
        Xe = self.edge_embed(X_edges.contiguous().reshape(-1, X_edges.size(-1))).reshape(N, T, -1)
        Xw = self.weather_embed(X_weather_edges.contiguous().reshape(-1, X_weather_edges.size(-1))).reshape(N, T, -1)
        H0 = torch.relu(self.combine(torch.cat([Xe, Xw], dim=-1)))

        # Compute learned temporal adjacency and fuse with static structure
        A_t = self.temporal_learner(H0)
        A_s_norm = normalize_sparse_adj(A_s.coalesce())
        A_f = self.fusion(A_s_norm, A_t)

        # Spatio-temporal propagation through stacked blocks
        H = H0
        for block in self.blocks:
            H = block(H, A_f)

        # Predict final disruption durations using last time step representation
        out = self.pred_head(H[:, -1, :])  # [N,1]
        return out
