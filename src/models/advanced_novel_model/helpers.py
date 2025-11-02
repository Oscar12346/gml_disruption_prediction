from parameters import WEATHER_FEATURES

import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------------------------------------------------
# Preprocessing helpers (graph construction and tensor conversion)
# -------------------------------------------------------------------

def build_minimal_graph(graph: nx.Graph) -> nx.Graph:
    """
    Build a minimal TRAIN-only graph from a full snapshot.

    This removes WEATHER nodes and retains only TRAIN nodes and track edges.
    Weather features from each TRAIN node’s associated weather station are copied
    onto the node attributes. Missing values default to 0.0.

    Parameters
    ----------
    graph : nx.Graph
        Original snapshot graph containing TRAIN and WEATHER nodes.

    Returns
    -------
    nx.Graph
        Minimal graph containing only TRAIN nodes and non-WEATHER edges.
    """
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
    """
    Convert a TRAIN graph into a line graph representation.

    Each edge in the original graph becomes a node in the line graph (LG),
    and edges in LG connect pairs of original edges that share a common station.

    Parameters
    ----------
    G : nx.Graph
        Minimal TRAIN-only graph.

    Returns
    -------
    nx.Graph
        Line graph where nodes represent train track segments (edges from G).
        Each LG node has a 'duration' attribute, and LG edges carry the shared
        station's weather attributes.
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
    Build edge feature, weather feature, and target tensors from a line graph.

    For each edge-node in the line graph:
      - The 'duration' becomes the edge feature and target.
      - Weather features are aggregated from adjacent LG edges
        (averaged if numeric, OR'ed if boolean).

    Parameters
    ----------
    LG : nx.Graph
        Line graph for one snapshot.
    node2idx : dict
        Mapping from edge-node to numerical index.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        X_edges : [N, 1] tensor of edge durations.
        X_weather_edges : [N, Fw] tensor of aggregated weather features.
        Y : [N] tensor of target disruption durations.
    """
    N = len(node2idx)
    X_edges = torch.zeros(N, 1, dtype=torch.float32)
    X_weather_edges = torch.zeros(N, len(WEATHER_FEATURES), dtype=torch.float32)
    Y = torch.zeros(N, dtype=torch.float32)

    for node in LG.nodes:
        idx = node2idx[node]
        X_edges[idx, 0] = float(LG.nodes[node].get('duration', 0.0))

        incident_edges = list(LG.edges(node))
        if len(incident_edges) == 0:
            vals_per_feat = [0.0 for _ in WEATHER_FEATURES]
        else:
            vals_per_feat = []
            for feat_idx, feat in enumerate(WEATHER_FEATURES):
                vals = [LG.edges[e][feat] for e in incident_edges]
                first = vals[0]
                if isinstance(first, (bool, np.bool_)):
                    val = any(vals)
                else:
                    val = float(sum([float(v) for v in vals]) / len(vals))
                vals_per_feat.append(val)

        X_weather_edges[idx] = torch.tensor(vals_per_feat, dtype=torch.float32)
        Y[idx] = float(LG.nodes[node].get('duration', 0.0))

    return X_edges, X_weather_edges, Y

# -------------------------------------------------------------------
# Dataset split & sliding windows
# -------------------------------------------------------------------

def split_dataset(X_seq, Xw_seq, Y_seq, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """
    Split temporal sequence data into training, validation, and test sets.

    Parameters
    ----------
    X_seq : torch.Tensor
        Sequence of edge features [T, N, Fe].
    Xw_seq : torch.Tensor
        Sequence of weather features [T, N, Fw].
    Y_seq : torch.Tensor
        Sequence of targets [T, N].
    train_ratio : float, optional
        Proportion of time steps used for training. Default is 0.6.
    val_ratio : float, optional
        Proportion used for validation. Default is 0.2.

    Returns
    -------
    tuple
        Split tensors for training, validation, and testing.
    """
    T = X_seq.shape[0]
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)

    X_train = X_seq[:n_train]
    Xw_train = Xw_seq[:n_train]
    Y_train = Y_seq[:n_train]

    X_val = X_seq[n_train:n_train + n_val]
    Xw_val = Xw_seq[n_train:n_train + n_val]
    Y_val = Y_seq[n_train:n_train + n_val]

    X_test = X_seq[n_train + n_val:]
    Xw_test = Xw_seq[n_train + n_val:]
    Y_test = Y_seq[n_train + n_val:]

    return X_train, Xw_train, Y_train, X_val, Xw_val, Y_val, X_test, Xw_test, Y_test

def slide_window(X_seq, Xw_seq, Y_seq, window_sizes: List[int]):
    """
    Create temporal sliding windows from sequential data.

    Each window captures the last W time steps as input and the next step as target.

    Parameters
    ----------
    X_seq : torch.Tensor
        Edge features sequence [T, N, Fe].
    Xw_seq : torch.Tensor
        Weather features sequence [T, N, Fw].
    Y_seq : torch.Tensor
        Target sequence [T, N].
    window_sizes : list of int
        List of window lengths to generate.

    Returns
    -------
    tuple of lists
        X_e_list : list of [N, W, Fe] edge feature windows.
        X_w_list : list of [N, W, Fw] weather feature windows.
        Y_list : list of [N] target vectors.
    """
    T, N, F_edge = X_seq.shape
    _, _, F_weather = Xw_seq.shape
    X_e_list, X_w_list, Y_list = [], [], []

    for W in window_sizes:
        for t in range(W, T):
            X_window_e = X_seq[t-W:t].transpose(0, 1).contiguous()
            X_window_w = Xw_seq[t-W:t].transpose(0, 1).contiguous()
            Y_next = Y_seq[t].contiguous()
            X_e_list.append(X_window_e)
            X_w_list.append(X_window_w)
            Y_list.append(Y_next)

    return X_e_list, X_w_list, Y_list

# -------------------------------------------------------------------
# Train / evaluate helpers
# -------------------------------------------------------------------

def train_one_epoch(model: nn.Module, optim: torch.optim.Optimizer, X_list, Xw_list, Y_list, A_s):
    """
    Train the model for one epoch using provided sliding windows.

    Parameters
    ----------
    model : nn.Module
        E-STFGNN model instance.
    optim : torch.optim.Optimizer
        Optimizer used for training.
    X_list, Xw_list, Y_list : list of torch.Tensor
        Input and target sequences.
    A_s : torch.sparse_coo_tensor
        Static adjacency matrix.
    """
    model.train()
    for X_seq_window, Xw_seq_window, Y_true in zip(X_list, Xw_list, Y_list):
        X_seq_window = X_seq_window.to(device).to(torch.float32)
        Xw_seq_window = Xw_seq_window.to(device).to(torch.float32)
        Y_true = Y_true.to(device).to(torch.float32)

        y_pred = model(X_seq_window, Xw_seq_window, A_s)
        loss = F.mse_loss(y_pred, Y_true.unsqueeze(-1))

        optim.zero_grad()
        loss.backward()
        optim.step()

@torch.no_grad()
def evaluate(model: nn.Module, X_list, Xw_list, Y_list, A_s) -> float:
    """
    Evaluate model performance over a validation or test set.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    X_list, Xw_list, Y_list : list of torch.Tensor
        Input and target sequences.
    A_s : torch.sparse_coo_tensor
        Static adjacency matrix.

    Returns
    -------
    float
        Average MSE loss across all windows.
    """
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
