import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.graph import SNAPSHOTS


def snapshots_to_df(graph_dict):
    # Takes long time
    times = sorted(graph_dict.keys())
    edges = list(graph_dict[times[0]].edges())

    edges = sorted(edges, key=lambda x: (str(x[0]), str(x[1])))
    data = [
        [graph_dict[t][u][v].get('duration', 0.0) for (u, v) in edges]
        for t in times
    ]
    df = pd.DataFrame(data, index=times, columns=edges)  # plain tuples
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
    print(outpath)
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

    n_last = 24 * 30

    # pick the last n_last rows
    df_last_month = df_all.iloc[-n_last:]

    print(f"Using last {n_last} timesteps for evaluation.")
    print(f"Time range: {df_last_month.index[0]} -> {df_last_month.index[-1]}")

    summary = []
    out_dir = os.path.join(os.getcwd(), "outputs", "moving_average_baseline")  # absolute path
    os.makedirs(out_dir, exist_ok=True)
    weighted = False #Change to true for weighted average

    for name, window in windows.items():
        print(f"Computing moving average with window {name} ({window} hours)...")
        # Predictions use full history, but metrics/plots only for last month
        y_true_full, y_pred_full, mae_full, rmse_full = one_step_moving_average(df_all, window)
        y_true_month = y_true_full.iloc[-n_last:]
        y_pred_month = y_pred_full.iloc[-n_last:]

        # Save wide CSVs
        prefix = os.path.join(out_dir, name)
        os.makedirs(prefix, exist_ok=True)
        y_true_month.to_csv(os.path.join(prefix, "y_true_last_month.csv"))
        y_pred_month.to_csv(os.path.join(prefix, "y_pred_last_month.csv"))

        # Flattened CSV
        edges_str = [f"{u}-{v}" for (u, v) in edges]
        flat = pd.DataFrame({
            'time': np.repeat(y_true_month.index, len(edges)),
            'edge': np.tile(edges_str, len(y_true_month)),
            'y_true': y_true_month.to_numpy().ravel(),
            'y_pred': y_pred_month.to_numpy().ravel()
        })
        flat.to_csv(os.path.join(prefix, "predictions_flat_last_month.csv"), index=False)

        # Plot average over edges
        plot_avg(y_true_month, y_pred_month,
                 title=f"MA one-step ({name}) last 30 days",
                 outpath=os.path.join(prefix, "avg_plot_last_month.png"))

        print(f"{name}: MAE={mae_full:.4f}, RMSE={rmse_full:.4f}")
        summary.append({'window_name': name, 'window_hours': window, 'MAE': float(mae_full), 'RMSE': float(rmse_full)})


        # Summary table
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "summary_metrics_last_month.csv"), index=False)
    # summary plot across windows (MAE and RMSE)
    plt.figure(figsize=(6, 4))
    plt.bar(summary_df['window_name'], summary_df['MAE'], color='C0', alpha=0.7, label='MAE')
    plt.bar(summary_df['window_name'], summary_df['RMSE'], color='C1', alpha=0.4, label='RMSE')
    plt.xlabel("Moving Average Window")
    plt.ylabel("Error (minutes)")
    plt.title(f"MAE/RMSE by window for last month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mae_rmse_by_window_month.png"), dpi=150)
    plt.show()
    print("\nSummary of all windows for last month:")
    print(summary_df)

