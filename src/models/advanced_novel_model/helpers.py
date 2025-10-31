from parameters import WEATHER_FEATURES

import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------------------------------------------------
# Preprocessing helpers (build minimal train graph -> line graph -> tensors)
# -------------------------------------------------------------------

def build_minimal_graph(graph: nx.Graph) -> nx.Graph:
    """
    Convert full BASE_GRAPH snapshot to a minimal TRAIN-only graph:
      - nodes: only TRAIN nodes (with attached weather features copied from their weather_station)
      - edges: only non-WEATHER edges (train track edges). If a disruption exists, 'duration' present,
               otherwise default 0.0 (missing disruptions -> duration 0).
    Returns a fresh networkx.Graph.
    """
    H = nx.Graph()
    H.graph = graph.graph.copy()

    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") != "TRAIN":
            continue
        ws_code = attrs.get("weather_station")
        ws_attrs = graph.nodes[ws_code]
        # copy weather features (fill 0.0 if missing)
        node_features = {feat: float(ws_attrs.get(feat, 0.0)) for feat in WEATHER_FEATURES}
        H.add_node(node, **node_features)

    # keep only train-track edges (type != 'WEATHER'), attach duration (0.0 default)
    for u, v, eattrs in graph.edges(data=True):
        if eattrs.get("type") != "WEATHER":
            duration = eattrs.get("duration", 0.0)
            H.add_edge(u, v, duration=duration)

    return H

def make_linegraph(G: nx.Graph) -> nx.Graph:
    """
    Convert the TRAIN graph into a line graph (edges -> nodes).
    For each edge-node in LG:
      - LG.nodes[(u,v)]['duration'] := duration of the corresponding edge in G (0 if missing)
    For each LG edge (which represents an adjacency between two original edges sharing a station),
      - copy the shared station's weather features into the LG edge attributes (helpful if needed).
    """
    LG = nx.line_graph(G)
    LG.graph = G.graph.copy()
    for (u, v) in LG.nodes:
        LG.nodes[(u, v)]['duration'] = G.edges[u, v].get('duration', 0.0)

    for (node_u, node_v) in LG.edges:
        shared_node = list(set(node_u) & set(node_v))[0]
        for feat in WEATHER_FEATURES:
            LG.edges[node_u, node_v][feat] = G.nodes[shared_node][feat]
    return LG

def build_edge_and_weather_tensors(LG: nx.Graph, node2idx: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build edge feature tensor (previous duration), per-edge weather tensor and target Y for one snapshot.

    Returns:
      X_edges: [N, 1] float tensor (previous duration; here same as target because we use last observed)
      X_weather_edges: [N, Fw] float tensor (aggregated weather for the edge)
      Y: [N] float tensor (target disruption duration, minutes, 0 if missing)
    Notes:
      - weather aggregation: for an edge-node in line graph, we compute values from incident edges' weather on LG.
        Numeric features are averaged; booleans are ORed (True if any neighbor has True).
    """
    N = len(node2idx)
    X_edges = torch.zeros(N, 1, dtype=torch.float32)
    X_weather_edges = torch.zeros(N, len(WEATHER_FEATURES), dtype=torch.float32)
    Y = torch.zeros(N, dtype=torch.float32)

    for node in LG.nodes:
        idx = node2idx[node]
        # edge feature: previous duration
        X_edges[idx, 0] = float(LG.nodes[node].get('duration', 0.0))
        # For each weather feature, gather values from LG.edges adjacent to this node (these edges hold weather of shared station)
        incident_edges = list(LG.edges(node))
        # If no incident edges (isolated edge), fallback to zeros
        if len(incident_edges) == 0:
            vals_per_feat = [0.0 for _ in WEATHER_FEATURES]
        else:
            vals_per_feat = []
            for feat_idx, feat in enumerate(WEATHER_FEATURES):
                vals = [LG.edges[e][feat] for e in incident_edges]
                first = vals[0]
                if isinstance(first, (bool, np.bool_)):
                    # boolean: treat as True if any neighbor True
                    val = any(vals)
                else:
                    # numeric: average
                    val = float(sum([float(v) for v in vals]) / len(vals))
                vals_per_feat.append(val)
        X_weather_edges[idx] = torch.tensor(vals_per_feat, dtype=torch.float32)
        Y[idx] = float(LG.nodes[node].get('duration', 0.0))

    return X_edges, X_weather_edges, Y

# -------------------------------------------------------------------
# Dataset split & sliding windows
# -------------------------------------------------------------------

def split_dataset(X_seq, Xw_seq, Y_seq, train_ratio: float = 0.6, val_ratio: float = 0.2):
    T = X_seq.shape[0]
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    X_train = X_seq[:n_train]; Xw_train = Xw_seq[:n_train]; Y_train = Y_seq[:n_train]
    X_val = X_seq[n_train:n_train + n_val]; Xw_val = Xw_seq[n_train:n_train + n_val]; Y_val = Y_seq[n_train:n_train + n_val]
    X_test = X_seq[n_train + n_val:]; Xw_test = Xw_seq[n_train + n_val:]; Y_test = Y_seq[n_train + n_val:]
    return X_train, Xw_train, Y_train, X_val, Xw_val, Y_val, X_test, Xw_test, Y_test

def slide_window(X_seq, Xw_seq, Y_seq, window_sizes: List[int]):
    """
    Create sliding windows across time axis, after data split.
    Each window entry has shape:
      X_window_e: [N, W, Fe]
      X_window_w: [N, W, Fw]
      Y_next: [N]
    Returns lists of those tensors (keeps them on CPU for now).
    """
    T, N, F_edge = X_seq.shape
    _, _, F_weather = Xw_seq.shape
    X_e_list, X_w_list, Y_list = [], [], []
    for W in window_sizes:
        for t in range(W, T):
            X_window_e = X_seq[t-W:t].transpose(0, 1).contiguous()    # [N, W, Fe]
            X_window_w = Xw_seq[t-W:t].transpose(0, 1).contiguous()  # [N, W, Fw]
            Y_next = Y_seq[t].contiguous()                            # [N]
            X_e_list.append(X_window_e)
            X_w_list.append(X_window_w)
            Y_list.append(Y_next)
    return X_e_list, X_w_list, Y_list

# -------------------------------------------------------------------
# Train / evaluate helpers
# -------------------------------------------------------------------

def train_one_epoch(model: nn.Module, optim: torch.optim.Optimizer, X_list, Xw_list, Y_list, A_s):
    model.train()
    for X_seq_window, Xw_seq_window, Y_true in zip(X_list, Xw_list, Y_list):
        # move per-window to device
        X_seq_window = X_seq_window.to(device).to(torch.float32)
        Xw_seq_window = Xw_seq_window.to(device).to(torch.float32)
        Y_true = Y_true.to(device).to(torch.float32)
        y_pred = model(X_seq_window, Xw_seq_window, A_s)  # forward expects [N, W, F]
        loss = F.mse_loss(y_pred, Y_true.unsqueeze(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()

@torch.no_grad()
def evaluate(model: nn.Module, X_list, Xw_list, Y_list, A_s) -> float:
    model.eval()
    eval_loss = 0.0
    for X_seq_window, Xw_seq_window, Y_true in zip(X_list, Xw_list, Y_list):
        X_seq_window = X_seq_window.to(device).to(torch.float32)
        Xw_seq_window = Xw_seq_window.to(device).to(torch.float32)
        Y_true = Y_true.to(device).to(torch.float32)
        y_pred = model(X_seq_window, Xw_seq_window, A_s)
        loss = F.mse_loss(y_pred, Y_true.unsqueeze(-1))
        eval_loss += loss.item()
    return eval_loss / max(1, len(X_list))