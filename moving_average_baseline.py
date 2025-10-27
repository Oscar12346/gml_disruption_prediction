import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graph import SNAPSHOTS


def snapshots_to_df(graph_dict):
    times = sorted(graph_dict.keys())
    edges = sorted(list(graph_dict[times[0]].edges()))
    data = [
        [graph_dict[t][u][v].get('weight', 0.0) for (u, v) in edges]
        for t in times
    ]
    df = pd.DataFrame(data, index=times, columns=edges)
    return df, edges, times

def weighted_rolling_mean(series, weights):
    """Apply weighted moving average to a 1D pandas series."""
    window = len(weights)
    def apply_weights(x):
        w = weights[-len(x):]  # adjust for first few rows
        return np.sum(x * w) / np.sum(w)
    return series.rolling(window, min_periods=1).apply(apply_weights, raw=True)

def one_step_moving_average(df, window, weighted=False):
    """Compute one-step-ahead MA (or WMA) over full timeline."""
    if weighted:
        weights = np.arange(1, window + 1)
        preds = df.apply(lambda col: weighted_rolling_mean(col, weights))
    else:
        preds = df.rolling(window=window, min_periods=1).mean()
    preds = preds.shift(1).fillna(0.0)
    y_true = df.copy()
    y_pred = preds.copy()
    mae = np.nanmean(np.abs(y_true.to_numpy().ravel() - y_pred.to_numpy().ravel()))
    rmse = np.sqrt(np.nanmean((y_true.to_numpy().ravel() - y_pred.to_numpy().ravel())**2))
    return y_true, y_pred, mae, rmse


def plot_avg(y_true, y_pred, title, outpath=None):
    plt.figure(figsize=(10,4))
    plt.plot(y_true.index, y_true.mean(axis=1), label='True (avg)', linewidth=1.2)
    plt.plot(y_pred.index, y_pred.mean(axis=1), label='Pred (avg)', linewidth=1.0, linestyle='--')
    plt.xlabel("time")
    plt.ylabel("Disruption (minutes)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.close()

if __name__ == "__main__":
    windows = {
        'day': 24,
        'week': 24 * 7,
        'month': 24 * 30,
        'year': 24 * 364
    }

    df_all, edges, times = snapshots_to_df(SNAPSHOTS)
    summary = []
    out_dir = "outputs/moving_average_baseline"
    weighted = False #Change to true for weighted average

    for name, window in windows.items():
        print(f"Computing moving average with window {name} ({window} hours)...")
        y_true, y_pred, mae, rmse = one_step_moving_average(df_all, window, weighted=weighted)

        prefix = os.path.join(out_dir, f"{name}")
        os.makedirs(prefix, exist_ok=True)
        y_true.to_csv(os.path.join(prefix, "y_true_wide.csv"))
        y_pred.to_csv(os.path.join(prefix, "y_pred_wide.csv"))

        # flattened version
        edges_str = [f"{u}-{v}" for (u, v) in edges]
        flat = pd.DataFrame({
            'time': np.repeat(y_true.index, len(edges)),
            'edge': np.tile(edges_str, len(y_true)),
            'y_true': y_true.to_numpy().ravel(),
            'y_pred': y_pred.to_numpy().ravel()
        })
        flat.to_csv(os.path.join(prefix, "predictions_flat.csv"), index=False)

        # plot average over all edges
        plot_avg(y_true, y_pred, title=f"MA one-step ({name})", outpath=os.path.join(prefix, "avg_plot.png"))

        print(f"{name}  MAE={mae:.4f}, RMSE={rmse:.4f}")
        summary.append({'window_name': name, 'window_hours': window, 'MAE': float(mae), 'RMSE': float(rmse)})


    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)

    # plot all different windows
    plt.figure(figsize=(6, 4))
    plt.bar(summary_df['window_name'], summary_df['MAE'], color='C0', alpha=0.7, label='MAE')
    plt.bar(summary_df['window_name'], summary_df['RMSE'], color='C1', alpha=0.4, label='RMSE')
    plt.xlabel("Moving Average Window")
    plt.ylabel("Error (minutes)")
    plt.title("Baseline Moving Average Performance by Window Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mae_rmse_by_window.png"), dpi=150)
    plt.show()
    print("\nSummary of all windows:")
    print(summary_df)

