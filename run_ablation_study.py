"""
E-STFGNN Ablation Study: Weather Feature Inclusion

This script conducts an ablation study on the E-STFGNN model by varying the
inclusion of different weather feature sets. These configurations include:
  - All weather features
  - No weather features
  - Continuous weather features (wind, temperature, rain)
  - Boolean weather features (fog, snow, thunder, ice)
"""

import argparse

# -------------------------------------------------------------------
# Parse command-line arguments for weather feature ablation
# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description="E-STFGNN Training")
parser.add_argument('--weather_features', type=str, choices=['all', 'none', 'continuous', 'boolean'], required=True,
                    help="Specify which weather features to include: 'all', 'none', 'continuous', or 'boolean'")
args = parser.parse_args()

import parameters

if args.weather_features == "all":
    parameters.WEATHER_FEATURES = ['wind', 'wind_max', 'temperature', 'rain', 'rain_duration', 'fog', 'snow', 'thunder', 'ice']
elif args.weather_features == "none":
    parameters.WEATHER_FEATURES = []
elif args.weather_features == "continuous":
    parameters.WEATHER_FEATURES = ['wind', 'wind_max', 'temperature', 'rain', 'rain_duration']
elif args.weather_features == "boolean":
    parameters.WEATHER_FEATURES = ['fog', 'snow', 'thunder', 'ice']

print("Included Weather Features:")
if parameters.WEATHER_FEATURES:
    for feature in parameters.WEATHER_FEATURES:
        print(f"  - {feature}")
else:
    print("  (no weather features selected)")

import random
import numpy as np
import torch

from src.models.advanced_novel_model.helpers import *
from src.models.advanced_novel_model.estfgnn import *
from src.models.advanced_novel_model.interpretability import *

# -------------------------------------------------------------------
# Setup and environment
# -------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from src.graph import SNAPSHOTS

# -------------------------------------------------------------------
# Build minimal graph and line graph representation
# -------------------------------------------------------------------

first_snapshot = list(SNAPSHOTS.values())[0]
G_min = build_minimal_graph(first_snapshot)
LG = make_linegraph(G_min)
node2idx = {n: i for i, n in enumerate(LG.nodes())}
idx2node = {i: n for n, i in node2idx.items()}
num_edges = len(node2idx)
print(f"\nNumber of edges (line-graph nodes): {num_edges}")

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

    print("Train windows:", len(X_train_list), "Val windows:", len(X_val_list), "Test windows:", len(X_test_list), "\n")

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
    print(f"\nConfig {i}, Val: {best_val_loss:.6f}, Test: {test_loss:.6f}")

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
    }, f"output/estfgnn_ablation_study_weather_{args.weather_features}.pt")
print(f"Saved best overall model -> estfgnn_ablation_study_weather_{args.weather_features}.pt (Val Loss {best_overall_val_loss:.6f})")
