import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random
import numpy as np

# -------------------------
# Preprocessing functions
# -------------------------
def build_minimal_graph(graph: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.graph = graph.graph.copy()

    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") != "TRAIN":
            continue
        ws_code = attrs.get("weather_station")
        ws_attrs = graph.nodes[ws_code]
        node_features = {feat: float(ws_attrs.get(feat, 0.0)) for feat in WEATHER_FEATURES}
        H.add_node(node, **node_features)

    for u, v, eattrs in graph.edges(data=True):
        if eattrs.get("type") != "WEATHER":
            duration = eattrs.get("duration", 0.0)
            H.add_edge(u, v, duration=duration)
    return H

def make_linegraph(G: nx.Graph) -> nx.Graph:
    LG = nx.line_graph(G)
    LG.graph = G.graph.copy()
    for (u, v) in LG.nodes:
        LG.nodes[(u, v)]['duration'] = G.edges[u, v].get('duration', 0.0)
    for (node_u, node_v) in LG.edges:
        shared_node = list(set(node_u) & set(node_v))[0]
        for feat in WEATHER_FEATURES:
            LG.edges[node_u, node_v][feat] = G.nodes[shared_node][feat]
    return LG

def build_edge_and_weather_tensors(LG, node2idx):
    N = len(node2idx)
    X_edges = torch.zeros(N, 1)
    X_weather_edges = torch.zeros(N, len(WEATHER_FEATURES))
    Y = torch.zeros(N)
    for node in LG.nodes:
        idx = node2idx[node]
        X_edges[idx, 0] = LG.nodes[node]['duration']
        weather_vector = []
        for feat in WEATHER_FEATURES:
            vals = [LG.edges[edge][feat] for edge in LG.edges(node)]
            val = any(vals) if isinstance(vals[0], bool) else float(sum(vals)/len(vals))
            weather_vector.append(val)
        X_weather_edges[idx] = torch.tensor(weather_vector)
        Y[idx] = LG.nodes[node]['duration']
    return X_edges, X_weather_edges, Y

# -------------------------
# E-STFGNN model components
# -------------------------
def sparse_mm(A: torch.sparse_coo_tensor, X: torch.Tensor) -> torch.Tensor:
    return torch.spmm(A, X)

class SimpleGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU, use_bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None
        self.activation = activation()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, A_sparsed: torch.sparse_coo_tensor, H: torch.Tensor):
        HW = H.matmul(self.W)
        out = sparse_mm(A_sparsed, HW)
        if self.bias is not None:
            out = out + self.bias
        return self.activation(out)

class TemporalGatedConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size-1) * dilation
        self.conv_f = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, dilation=dilation)
        self.conv_g = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding, dilation=dilation)

    def forward(self, X):
        x = X.transpose(1,2)
        f = self.conv_f(x)[..., :X.shape[1]]
        g = self.conv_g(x)[..., :X.shape[1]]
        out = torch.tanh(f) * torch.sigmoid(g)
        return out.transpose(1,2)

class MLP(nn.Module):
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

class SpatioTemporalFusionBlock(nn.Module):
    def __init__(self, n_nodes, in_dim, hidden_dim):
        super().__init__()
        self.gconv = SimpleGraphConv(in_dim, hidden_dim)
        self.temporal = TemporalGatedConv(hidden_dim, hidden_dim, kernel_size=3)
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
        return self.norm(H_temp + residual)

class E_STFGNN(nn.Module):
    def __init__(self, n_edges, in_feat_dim, weather_dim, d_model=64, n_blocks=2):
        super().__init__()
        self.edge_embed = MLP(in_feat_dim, d_model, hidden_dims=(d_model//2,))
        # If no weather features are provided, skip creating a weather embed and
        # adjust the combine projection accordingly.
        self.use_weather = (weather_dim is not None and weather_dim > 0)
        if self.use_weather:
            self.weather_embed = MLP(weather_dim, d_model, hidden_dims=(d_model//2,))
            self.combine = nn.Linear(d_model*2, d_model)
        else:
            self.weather_embed = None
            # project edge embedding to model dim (or identity-like projection)
            self.combine = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([SpatioTemporalFusionBlock(n_edges, d_model, d_model) for _ in range(n_blocks)])
        self.pred_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))

    def forward(self, X_edges, X_weather_edges, A_s: torch.sparse_coo_tensor):
        N, T, _ = X_edges.shape
        Xe = self.edge_embed(X_edges.contiguous().view(-1, X_edges.size(-1))).reshape(N, T, -1)
        if self.use_weather:
            Xw = self.weather_embed(X_weather_edges.contiguous().view(-1, X_weather_edges.size(-1))).reshape(N, T, -1)
            H0 = torch.relu(self.combine(torch.cat([Xe, Xw], dim=-1)))
        else:
            # No weather features: use only edge embedding
            H0 = torch.relu(self.combine(Xe))
        H = H0
        for block in self.blocks:
            H = block(H, A_s)
        H_last = H[:, -1, :]
        return self.pred_head(H_last)

# -------------------------
# Split dataset by percentage
# -------------------------
def split_dataset(X_seq, Xw_seq, Y_seq, train_ratio=0.6, val_ratio=0.2):
    T = X_seq.shape[0]
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)

    X_train = X_seq[:n_train]
    Xw_train = Xw_seq[:n_train]
    Y_train = Y_seq[:n_train]

    X_val = X_seq[n_train:n_train+n_val]
    Xw_val = Xw_seq[n_train:n_train+n_val]
    Y_val = Y_seq[n_train:n_train+n_val]

    X_test = X_seq[n_train+n_val:]
    Xw_test = Xw_seq[n_train+n_val:]
    Y_test = Y_seq[n_train+n_val:]

    return (X_train, Xw_train, Y_train,
            X_val, Xw_val, Y_val,
            X_test, Xw_test, Y_test)

# -------------------------
# Create sliding windows after split
# -------------------------
def slide_window(X_seq, Xw_seq, Y_seq, window_sizes):
    """
    Create sliding windows along the time axis.
    Returns lists of windows: each window is [N, W, F] for features.
    """
    T, N, F_edge = X_seq.shape
    _, _, F_weather = Xw_seq.shape
    X_e_list, X_w_list, Y_list = [], [], []

    for W in window_sizes:
        for t in range(W, T):
            # Slice time window [t-W:t], node dim stays full
            X_window_e = X_seq[t-W:t].transpose(0,1)  # [N, W, F_edge]
            X_window_w = Xw_seq[t-W:t].transpose(0,1)  # [N, W, F_weather]
            Y_next = Y_seq[t]  # [N]
            X_e_list.append(X_window_e)
            X_w_list.append(X_window_w)
            Y_list.append(Y_next)

    return X_e_list, X_w_list, Y_list

# -------------------------
# Training helpers
# -------------------------
def train_one_epoch(model, optim, X_list, Xw_list, Y_list):
    model.train()
    for X_seq_window, Xw_seq_window, Y_true in zip(X_list, Xw_list, Y_list):
        X_seq_window = X_seq_window.to(torch.float32).to(device)
        Xw_seq_window = Xw_seq_window.to(torch.float32).to(device)
        Y_true = Y_true.to(torch.float32).to(device)
        y_pred = model(X_seq_window, Xw_seq_window, A_s)
        loss = F.mse_loss(y_pred, Y_true.unsqueeze(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()

@torch.no_grad()
def evaluate(model, X_list, Xw_list, Y_list):
    model.eval()
    eval_loss = 0.0
    for X_seq_window, Xw_seq_window, Y_true in zip(X_list, Xw_list, Y_list):
        X_seq_window = X_seq_window.to(torch.float32).to(device)
        Xw_seq_window = Xw_seq_window.to(torch.float32).to(device)
        Y_true = Y_true.to(torch.float32).to(device)
        y_pred = model(X_seq_window, Xw_seq_window, A_s)
        loss = F.mse_loss(y_pred, Y_true.unsqueeze(-1))
        eval_loss += loss.item()
    return eval_loss / len(X_list)
    
if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # -------------------------
    # User-defined parameters
    # -------------------------
    # EPOCH = pd.Timestamp('2023-01-01 00:00:00')
    # HORIZON = pd.Timestamp('2023-02-01 00:00:00')
    WEATHER_FEATURES = ['wind', 'wind_max', 'temperature', 'rain', 'rain_duration', 'fog', 'snow', 'thunder', 'ice']

    # -------------------------
    # Placeholder for SNAPSHOTS
    # -------------------------
    from src.graph import SNAPSHOTS

    # -------------------------
    # Dataset split first
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Full tensor sequences
    X_edges_seq = []
    X_weather_edges_seq = []
    Y_seq = []

    # Use first snapshot to get edge superset
    first_snapshot = list(SNAPSHOTS.values())[0]
    G_min = build_minimal_graph(first_snapshot)
    LG = make_linegraph(G_min)
    node2idx = {n:i for i,n in enumerate(LG.nodes())}
    idx2node = {i:n for i,n in enumerate(LG.nodes())}
    num_edges = len(node2idx)

    # Sparse adjacency for edges
    rows, cols = [], []
    for (u,v) in LG.edges():
        i, j = node2idx[u], node2idx[v]
        rows += [i,j]
        cols += [j,i]

    for i in range(num_edges):
        rows.append(i)
        cols.append(i)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    A_s = torch.sparse_coo_tensor(indices, values, (num_edges, num_edges)).coalesce().to(device)

    # Build sequences for all snapshots
    for t in sorted(SNAPSHOTS.keys()):
        G = SNAPSHOTS[t]
        G_min = build_minimal_graph(G)
        LG = make_linegraph(G_min)
        X_e, X_w, Y = build_edge_and_weather_tensors(LG, node2idx)
        X_edges_seq.append(X_e)
        X_weather_edges_seq.append(X_w)
        Y_seq.append(Y)

    X_edges_seq = torch.stack(X_edges_seq).to(device)
    X_weather_edges_seq = torch.stack(X_weather_edges_seq).to(device)
    Y_seq = torch.stack(Y_seq).to(device)

    X_train, Xw_train, Y_train, X_val, Xw_val, Y_val, X_test, Xw_test, Y_test = split_dataset(
        X_edges_seq, X_weather_edges_seq, Y_seq
    )

    # window_sizes = [48, 24, 8, 4, 2]
    #
    # X_train_list, Xw_train_list, Y_train_list = slide_window(X_train, Xw_train, Y_train, window_sizes)
    # X_val_list, Xw_val_list, Y_val_list = slide_window(X_val, Xw_val, Y_val, window_sizes)
    # X_test_list, Xw_test_list, Y_test_list = slide_window(X_test, Xw_test, Y_test, window_sizes)

    # -------------------------
    # Instantiate and train model
    # -------------------------
    configs = [
        {"d_model": 32,  "n_blocks": 1, "lr": 1e-3, "window_sizes": [8, 4, 2]},
        {"d_model": 64,  "n_blocks": 2, "lr": 1e-3, "window_sizes": [24, 8, 4]},
        {"d_model": 64,  "n_blocks": 3, "lr": 5e-4, "window_sizes": [48, 24, 8]},
        {"d_model": 128, "n_blocks": 2, "lr": 1e-4, "window_sizes": [48, 24, 12]},
        {"d_model": 128, "n_blocks": 3, "lr": 5e-4, "window_sizes": [24, 12, 6]},
    ]

    results = []
    best_overall_val_loss = float('inf')
    best_overall_model_state = None
    best_overall_cfg = None

    for i, cfg in enumerate(configs, 1):
        print(f"\n=== Running config {i}: {cfg} ===")

        # Recreate model and optimizer for each run
        model = E_STFGNN(
            n_edges=len(LG.nodes()),
            in_feat_dim=1,
            weather_dim=len(WEATHER_FEATURES),
            d_model=cfg["d_model"],
            n_blocks=cfg["n_blocks"]
        ).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        # Rebuild sliding windows using the config's window sizes
        X_train_list, Xw_train_list, Y_train_list = slide_window(X_train, Xw_train, Y_train, cfg["window_sizes"])
        X_val_list, Xw_val_list, Y_val_list = slide_window(X_val, Xw_val, Y_val, cfg["window_sizes"])
        X_test_list, Xw_test_list, Y_test_list = slide_window(X_test, Xw_test, Y_test, cfg["window_sizes"])

        best_val_loss = float('inf')
        num_epochs = 20

        for epoch in range(num_epochs):
            train_one_epoch(model, optim, X_train_list, Xw_train_list, Y_train_list)
            val_loss = evaluate(model, X_val_list, Xw_val_list, Y_val_list)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")

        model.load_state_dict(best_state)
        test_loss = evaluate(model, X_test_list, Xw_test_list, Y_test_list)

        results.append({
            "config": cfg,
            "val_loss": best_val_loss,
            "test_loss": test_loss
        })

        print(f"Config {i}, -> Val: {best_val_loss:.4f}, Test: {test_loss:.4f}")

        # Update overall best model
        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_overall_model_state = best_state
            best_overall_cfg = cfg

    torch.save({
        "model_state_dict": best_overall_model_state,
        "config": best_overall_cfg,
    }, "best_estfgnn_model.pt")

    print("\n=== Summary of All Configs ===")
    for res in results:
        cfg = res["config"]
        print(f"d_model={cfg['d_model']}, n_blocks={cfg['n_blocks']}, lr={cfg['lr']}, windows={cfg['window_sizes']}"
            f"-> Val: {res['val_loss']:.4f}, Test: {res['test_loss']:.4f}")
else:
    print("run_estfgnn.py loaded")
