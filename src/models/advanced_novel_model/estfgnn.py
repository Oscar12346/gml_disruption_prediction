from src.models.advanced_novel_model.layers import *
from typing import Optional
import torch


class E_STFGNN(nn.Module):
    """
    Full E-STFGNN:
      - embed edges & weather
      - compute temporal learned adjacency A_t (from H0)
      - fuse A_s and A_t -> A_f
      - stacked SpatioTemporalFusionBlock using A_f
      - predict final duration per edge
    """
    def __init__(self, n_edges:int, in_feat_dim:int, weather_dim:int, d_model:int = 64, n_blocks:int = 2,
                 key_dim: Optional[int] = None, topk:int = 8):
        super().__init__()
        self.n_edges = n_edges
        self.d_model = d_model
        key_dim = key_dim or (d_model // 2)
        # embeddings
        self.edge_embed = MLP(in_feat_dim, d_model, hidden_dims=(max(8, d_model // 2),))
        self.weather_embed = MLP(weather_dim, d_model, hidden_dims=(max(8, d_model // 2),))
        self.combine = nn.Linear(d_model*2, d_model)
        # learn adjacency and fuse
        self.temporal_learner = TemporalAdjLearner(in_dim=d_model, key_dim=key_dim, topk=topk, sparsify=True)
        self.fusion = FusionAdjacency(n_nodes=n_edges, learnable_scalar=True)
        # stacked blocks
        self.blocks = nn.ModuleList([SpatioTemporalFusionBlock(n_edges, d_model, d_model) for _ in range(n_blocks)])
        self.pred_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))

    def forward(self, X_edges: torch.Tensor, X_weather_edges: torch.Tensor, A_s: torch.sparse_coo_tensor) -> torch.Tensor:
        """
        X_edges: [N, T, in_feat_dim]
        X_weather_edges: [N, T, weather_dim]
        A_s: sparse (N x N) static adjacency
        returns: [N,1] predicted durations
        """
        N, T, _ = X_edges.shape
        # embed per time-step
        Xe = self.edge_embed(X_edges.contiguous().reshape(-1, X_edges.size(-1))).reshape(N, T, -1)
        Xw = self.weather_embed(X_weather_edges.contiguous().reshape(-1, X_weather_edges.size(-1))).reshape(N, T, -1)
        H0 = torch.relu(self.combine(torch.cat([Xe, Xw], dim=-1)))  # [N, T, d_model]

        # temporal adjacency & fusion
        A_t = self.temporal_learner(H0)                 # sparse (N x N)
        A_s_norm = normalize_sparse_adj(A_s.coalesce()) # ensure static adj normalized
        A_f = self.fusion(A_s_norm, A_t)                # fused sparse (row-normalized)

        # spatio-temporal processing using fused adjacency
        H = H0
        for block in self.blocks:
            H = block(H, A_f)
        out = self.pred_head(H[:, -1, :])  # [N,1]
        return out