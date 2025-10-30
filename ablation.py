import argparse

# First, parse command-line arguments
parser = argparse.ArgumentParser(description="E-STFGNN Training")
parser.add_argument('--weather_features', type=str, choices=['all', 'none', 'continuous'], required=True,
                    help="Specify which weather features to include: 'all', 'none', or 'continuous'")
args = parser.parse_args()

import parameters

if args.weather_features == "all":
    parameters.WEATHER_FEATURES = ['wind', 'wind_max', 'temperature', 'rain', 'rain_duration', 'fog', 'snow', 'thunder', 'ice']
elif args.weather_features == "none":
    parameters.WEATHER_FEATURES = []
elif args.weather_features == "continuous":
    parameters.WEATHER_FEATURES = ['wind', 'wind_max', 'temperature', 'rain', 'rain_duration']

import random
import numpy as np
import torch
from run_estfgnn import E_STFGNN, build_minimal_graph, evaluate, make_linegraph, build_edge_and_weather_tensors, slide_window, split_dataset, train_one_epoch
import run_estfgnn

# Use GPU if available
if torch.cuda.is_available():
    print("\nCUDA is available. Using GPU.")
    run_estfgnn.device = torch.device("cuda")
else:
    print("\nCUDA is not available. Using CPU.")
    run_estfgnn.device = torch.device("cpu")

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from src.graph import SNAPSHOTS

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
run_estfgnn.A_s = torch.sparse_coo_tensor(indices, values, (num_edges, num_edges)).coalesce().to(run_estfgnn.device)

# Build sequences for all snapshots
for t in sorted(SNAPSHOTS.keys()):
    G = SNAPSHOTS[t]
    G_min = build_minimal_graph(G)
    LG = make_linegraph(G_min)
    X_e, X_w, Y = build_edge_and_weather_tensors(LG, node2idx)
    X_edges_seq.append(X_e)
    X_weather_edges_seq.append(X_w)
    Y_seq.append(Y)

X_edges_seq = torch.stack(X_edges_seq).to(run_estfgnn.device)
X_weather_edges_seq = torch.stack(X_weather_edges_seq).to(run_estfgnn.device)
Y_seq = torch.stack(Y_seq).to(run_estfgnn.device)

X_train, Xw_train, Y_train, X_val, Xw_val, Y_val, X_test, Xw_test, Y_test = split_dataset(
    X_edges_seq, X_weather_edges_seq, Y_seq
)

config = {"d_model": 128, "n_blocks": 2, "lr": 1e-4, "window_sizes": [48, 24, 12]}

model = E_STFGNN(
    n_edges=len(LG.nodes()),
    in_feat_dim=1,
    weather_dim=len(parameters.WEATHER_FEATURES),
    d_model=config["d_model"],
    n_blocks=config["n_blocks"]
).to(run_estfgnn.device)

print("\nModel Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
print("\nIncluded Weather Features:")
if parameters.WEATHER_FEATURES:
    for feature in parameters.WEATHER_FEATURES:
        print(f"  - {feature}")
else:
    print("  (no weather features selected)")
print("\nTraining model...\n")

optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

# Rebuild sliding windows using the config's window sizes
X_train_list, Xw_train_list, Y_train_list = slide_window(X_train, Xw_train, Y_train, config["window_sizes"])
X_val_list, Xw_val_list, Y_val_list = slide_window(X_val, Xw_val, Y_val, config["window_sizes"])
X_test_list, Xw_test_list, Y_test_list = slide_window(X_test, Xw_test, Y_test, config["window_sizes"])

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

torch.save({
    "model_state_dict": model.state_dict(),
    "config": config,
}, f"estfgnn_ablation_weather_{args.weather_features}.pt")

print(f"\nFinal Test Loss with weather features '{args.weather_features}': {test_loss:.4f}")
