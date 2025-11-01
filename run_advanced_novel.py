"""
Driver script for training, evaluation, and interpretability of the E-STFGNN model.

This script:
  - Builds graph snapshots and corresponding line-graph datasets
  - Trains and validates the E-STFGNN model with hyperparameter configurations
  - Saves the best-performing model checkpoint
  - Performs interpretability analyses (saliency, integrated gradients, permutation importance)
  - Visualizes weather feature importance as a bar plot
"""

import random
import numpy as np
import torch

from src.graph import SNAPSHOTS
from src.models.advanced_novel_model.helpers import *
from src.models.advanced_novel_model.estfgnn import *
from src.models.advanced_novel_model.interpretability import *


# -------------------------------------------------------------------
# Setup and environment
# -------------------------------------------------------------------

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# -------------------------------------------------------------------
# Build minimal graph and line graph representation
# -------------------------------------------------------------------

first_snapshot = list(SNAPSHOTS.values())[0]
G_min = build_minimal_graph(first_snapshot)
LG = make_linegraph(G_min)
node2idx = {n: i for i, n in enumerate(LG.nodes())}
idx2node = {i: n for n, i in node2idx.items()}
num_edges = len(node2idx)
print(f"Number of edges (line-graph nodes): {num_edges}")

# Construct static adjacency (A_s)
rows, cols = [], []
for (u, v) in LG.edges():
    i, j = node2idx[u], node2idx[v]
    rows += [i, j]
    cols += [j, i]
# Add self-loops
for i in range(num_edges):
    rows.append(i); cols.append(i)

indices = torch.tensor([rows, cols], dtype=torch.long)
values = torch.ones(len(rows), dtype=torch.float32)
A_s = torch.sparse_coo_tensor(indices, values, (num_edges, num_edges)).coalesce().to(device)

# -------------------------------------------------------------------
# Build time sequences across all snapshots
# -------------------------------------------------------------------

X_edges_seq = []
X_weather_edges_seq = []
Y_seq = []

for t in sorted(SNAPSHOTS.keys()):
    G = SNAPSHOTS[t]
    G_min = build_minimal_graph(G)
    LG = make_linegraph(G_min)
    X_e, X_w, Y = build_edge_and_weather_tensors(LG, node2idx)
    X_edges_seq.append(X_e)
    X_weather_edges_seq.append(X_w)
    Y_seq.append(Y)

X_edges_seq = torch.stack(X_edges_seq)
X_weather_edges_seq = torch.stack(X_weather_edges_seq)
Y_seq = torch.stack(Y_seq)
print("Sequence shapes (T, N, F):", X_edges_seq.shape, X_weather_edges_seq.shape, Y_seq.shape)

# -------------------------------------------------------------------
# Dataset splitting
# -------------------------------------------------------------------

X_train, Xw_train, Y_train, X_val, Xw_val, Y_val, X_test, Xw_test, Y_test = split_dataset(X_edges_seq, X_weather_edges_seq, Y_seq)

# -------------------------------------------------------------------
# Model configurations and training loop
# -------------------------------------------------------------------

configs = [
    # Best tuned configuration
    {"d_model": 128, "n_blocks": 2, "lr": 1e-4, "window_sizes": [48, 24, 12], "topk": 8}
]

results = []
best_overall_val_loss = float('inf')
best_overall_model_state = None
best_overall_cfg = None

for i, cfg in enumerate(configs, 1):
    print(f"\n=== Running config {i}: {cfg} ===")

    # instantiate model
    model = E_STFGNN(n_edges=num_edges,
                     in_feat_dim=1,
                     weather_dim=len(WEATHER_FEATURES),
                     d_model=cfg["d_model"],
                     n_blocks=cfg["n_blocks"],
                     topk=cfg.get("topk", 8)).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Build sliding windows for this config
    X_train_list, Xw_train_list, Y_train_list = slide_window(X_train, Xw_train, Y_train, cfg["window_sizes"])
    X_val_list, Xw_val_list, Y_val_list = slide_window(X_val, Xw_val, Y_val, cfg["window_sizes"])
    X_test_list, Xw_test_list, Y_test_list = slide_window(X_test, Xw_test, Y_test, cfg["window_sizes"])

    print("Train windows:", len(X_train_list), "Val windows:", len(X_val_list), "Test windows:", len(X_test_list))

    best_val_loss = float('inf')
    best_state = None
    num_epochs = 20

    for epoch in range(num_epochs):
        train_one_epoch(model, optim, X_train_list, Xw_train_list, Y_train_list, A_s)
        val_loss = evaluate(model, X_val_list, Xw_val_list, Y_val_list, A_s)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.6f}")

    # Evaluate & save per-config best
    model.load_state_dict(best_state)
    test_loss = evaluate(model, X_test_list, Xw_test_list, Y_test_list, A_s)
    results.append({"config": cfg, "val_loss": best_val_loss, "test_loss": test_loss})
    print(f"Config {i}, Val: {best_val_loss:.6f}, Test: {test_loss:.6f}")

    # Update global best
    if best_val_loss < best_overall_val_loss:
        best_overall_val_loss = best_val_loss
        best_overall_model_state = best_state
        best_overall_cfg = cfg

# Save best overall model
if best_overall_model_state is not None:
    torch.save({
        "model_state_dict": best_overall_model_state,
        "config": best_overall_cfg,
        "node2idx": node2idx,
        "idx2node": idx2node,
    }, "output/best_estfgnn_model_interpretability.pt")
    print(f"Saved best overall model -> best_estfgnn_model_interpretability.pt (Val Loss {best_overall_val_loss:.6f})")

# -------------------------------------------------------------------
# Interpretability analyses
# -------------------------------------------------------------------

checkpoint = torch.load("output/best_estfgnn_model_interpretability.pt", map_location=device)
best_cfg = checkpoint["config"]
print("Loaded checkpoint config:", best_cfg)

model = E_STFGNN(n_edges=num_edges,
                 in_feat_dim=1,
                 weather_dim=len(WEATHER_FEATURES),
                 d_model=best_cfg["d_model"],
                 n_blocks=best_cfg["n_blocks"],
                 topk=best_cfg.get("topk", 8)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Choose a validation window to inspect
if len(X_val_list) > 0:
    example_Xe = X_val_list[0]    
    example_Xw = Xw_val_list[0]

    # 1) Gradient saliency
    edge_idx = 10 if num_edges > 10 else 0
    sal_e, sal_w = gradient_saliency(model, example_Xe, example_Xw, A_s, target_edge=edge_idx)
    print(f"Saliency (weather) shape: {sal_w.shape}; per-feature sum:", sal_w[edge_idx].sum(axis=0))

    # 2) Integrated gradients
    ig_e, ig_w = integrated_gradients(model, example_Xe, example_Xw, A_s, steps=40, target_edge=edge_idx)
    print("IG per-weather-feature (sum across time) for chosen edge:", ig_w[edge_idx].sum(axis=0))

    # 3) Permutation importance on a subset of validation windows
    subset = min(40, len(X_val_list))
    perm_df = permutation_importance(model, X_val_list[:subset], Xw_val_list[:subset], Y_val_list[:subset], A_s, n_repeats=4)
    print("Permutation importance (top features):")
    print(perm_df.head())

    # **Plot feature importance**
    plot_feature_importance(perm_df, top_n=10)

    # 4) Save top-k edges by IG
    ig_edge_scores = (np.abs(ig_e).sum(axis=(1,2)))
    topk = topk_edges_from_importance(ig_edge_scores, k=20)
    save_topk_edges_csv(topk, idx2node, "output/topk_ig_edges.csv")
else:
    print("No validation windows available to run interpretability examples.")
