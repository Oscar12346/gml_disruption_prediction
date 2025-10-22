import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graph import SNAPSHOTS


def moving_average_baseline(graph_dict, train_split_idx=10, window=10, horizon=168):
    """
    Recursive moving-average baseline that predicts exactly `horizon` hours into the future.

    Args:
        graph_dict: dict[int, nx.Graph], graph with keys for timesteps. nodes are stations, edges are tracks with weight value for disruptions
        train_split_idx: int, the point where you stop 'training' and start 'testing'. Moving average does not actually train on data, but
                              this makes it easier to compare to other methods (probably). After this split every step in the future will be recursively calculated
                              and the real data will not be used anymore
        window: int, number of previous timesteps to use for averaging
        horizon: int, how many hours into the future to predict

    Returns:
        dict with y_true, y_pred, edges, times, MAE, RMSE
    """
    times = sorted(graph_dict.keys())
    edges = list(graph_dict[times[0]].edges())
    n_edges = len(edges)

    # Convert snapshots to DataFrame
    df = pd.DataFrame([
        [graph_dict[t][u][v].get('weight', 0.0) for (u, v) in edges]
        for t in times
    ], index=times, columns=edges)

    # Initialize history from last `window` timesteps of train
    df_train = df.iloc[:train_split_idx + 1]
    history = df_train.iloc[-window:].copy()

    # Determine test times
    test_times = times[train_split_idx + 1 : train_split_idx + 1 + horizon]
    if not test_times:
        raise ValueError("Horizon extends beyond available data. Reduce horizon or extend snapshots.")

    y_pred = pd.DataFrame(index=test_times, columns=edges, dtype=float)
    y_true = df.loc[test_times]

    # Recursive prediction
    for t in test_times:
        y_pred.loc[t] = history.mean()
        history = pd.concat([history, y_pred.loc[[t]]])
        if len(history) > window:
            history = history.iloc[-window:]

    # Flatten for evaluation
    y_true_flat = y_true.to_numpy().ravel()
    y_pred_flat = y_pred.to_numpy().ravel()

    mae = np.nanmean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(np.nanmean((y_true_flat - y_pred_flat)**2))

    print(f"Recursive Moving Average (window={window}, horizon={horizon}h): MAE={mae:.3f}, RMSE={rmse:.3f}")

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'edges': edges,
        'times': test_times,
        'MAE': mae,
        'RMSE': rmse
    }

# Here we use all the data and try to predict k steps ahead from our window
# This can logically perform really well with low k values (e.g. predict disruption for next hour knowing previous 6)

def moving_average_baseline_k_steps(graph_dict, window=6, k=1):
    """
    Predict k steps ahead using a moving average of previous 'window' hours.
    """
    import pandas as pd
    import numpy as np

    times = sorted(graph_dict.keys())
    edges = list(graph_dict[times[0]].edges())

    # Collect edge disruptions into DataFrame
    data = []
    for t in times:
        g = graph_dict[t]
        row = [g[u][v].get('weight', 0.0) for (u, v) in edges]
        data.append(row)
    df = pd.DataFrame(data, index=times, columns=edges)

    # Moving average prediction
    preds = df.rolling(window=window, min_periods=1).mean().shift(k)  # shift by k steps ahead
    y_true = df.iloc[window+k-1:]  # drop first few rows with incomplete history
    y_pred = preds.iloc[window+k-1:]

    y_true_flat = y_true.to_numpy().ravel()
    y_pred_flat = y_pred.to_numpy().ravel()
    mae = np.nanmean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(np.nanmean((y_true_flat - y_pred_flat)**2))

    print(f"Moving Average Baseline (window={window}, horizon={k}): MAE={mae:.3f}, RMSE={rmse:.3f}")

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'edges': edges,
        'times': y_true.index,
        'MAE': mae,
        'RMSE': rmse
    }


def plot_baseline_results(results, edge=None, avg_over_all=True, savepath=None):
    y_true = results['y_true']
    y_pred = results['y_pred']
    times = results.get('times', y_true.index)

    plt.figure(figsize=(10,5))

    if edge is not None and not avg_over_all:
        if edge not in y_true.columns:
            raise ValueError(f"Edge {edge} not found in results['edges']")
        plt.plot(times, y_true[edge], label='True', color='C0')
        plt.plot(times, y_pred[edge], label='Predicted', color='C1', linestyle='--')
        plt.title(f"Disruption over time for edge {edge}")
    else:
        plt.plot(times, y_true.mean(axis=1), label='True (avg)', color='C0')
        plt.plot(times, y_pred.mean(axis=1), label='Predicted (avg)', color='C1', linestyle='--')
        plt.title("Average Disruption over Time (All Edges)")

    plt.xlabel("Time (hour index)")
    plt.ylabel("Disruption (minutes)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

# Run moving avg
# Better to do this in one main/runnable file but easier for now to test for me
if __name__ == "__main__":

    EPOCH = pd.Timestamp('2023-01-01 00:00:00')
    WINDOW = 6  # moving average window in hours
    HORIZON = 10

    print("Getting snapshots...")
    graph_dict = {t: g.copy() for t, g in SNAPSHOTS.items()}

    # Save predictions + truths (wide format)
    out_dir = "outputs/baseline_moving_average"
    print("Total snapshots:", len(graph_dict))
    print("Max snapshot key:", max(graph_dict.keys()))

    print("Running recursive moving-average baseline...")
    # results = moving_average_baseline(graph_dict,
    #                                    window=WINDOW, horizon=HORIZON)
    results = moving_average_baseline(graph_dict, train_split_idx=4000)
    # Save wide format CSVs
    results['y_true'].to_csv(os.path.join(out_dir, "y_true_wide.csv"))
    results['y_pred'].to_csv(os.path.join(out_dir, "y_pred_wide.csv"))

    # Save flattened CSV for evaluation/plotting
    times = results['y_true'].index.values
    edges = results['edges']
    n_times = len(times)
    n_edges = len(edges)
    edges_str = [f"{u}-{v}" for (u, v) in edges]

    flat = pd.DataFrame({
        'time': np.repeat(times, n_edges),
        'edge': np.tile(edges_str, n_times),
        'y_true': results['y_true'].to_numpy().ravel(),
        'y_pred': results['y_pred'].to_numpy().ravel()
    })
    flat.to_csv(os.path.join(out_dir, "predictions_flat.csv"), index=False)

    # Plot
    plot_baseline_results(results, avg_over_all=True,
                          savepath=os.path.join(out_dir, "avg_disruption_vs_pred.png"))

    # Example edge plot
    if len(edges) > 0:
        example_edge = edges[0]
        try:
            plot_baseline_results(results, edge=example_edge, avg_over_all=False,
                                  savepath=os.path.join(out_dir, f"edge_{example_edge[0]}_{example_edge[1]}.png"))
        except Exception as e:
            print("Could not plot example edge:", e)

    print("Done.")