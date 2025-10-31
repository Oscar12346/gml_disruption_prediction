from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from parameters import WEATHER_FEATURES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _prep_inputs_for_attrib(X_e: torch.Tensor, X_w: torch.Tensor):
    """
    Helper: clone inputs, move to device and set requires_grad for attribution.
    Inputs are expected as [N, W, F]
    """
    Xe = X_e.clone().detach().to(device).float().requires_grad_(True)
    Xw = X_w.clone().detach().to(device).float().requires_grad_(True)
    return Xe, Xw

def gradient_saliency(model: nn.Module, X_e: torch.Tensor, X_w: torch.Tensor, A_s: torch.sparse_coo_tensor,
                      target_edge: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient * input saliency for a single window (X_e, X_w).
    Returns numpy arrays (sal_e, sal_w) with same shapes as X_e/X_w.
    If target_edge is None, the sum over outputs is used to produce gradients (global).
    """
    model.eval()
    Xe, Xw = _prep_inputs_for_attrib(X_e, X_w)
    y = model(Xe, Xw, A_s)  # [N,1]
    if target_edge is None:
        score = y.sum()
    else:
        score = y[target_edge, 0]
    model.zero_grad()
    score.backward()
    sal_e = (Xe.grad * Xe).detach().cpu().numpy()
    sal_w = (Xw.grad * Xw).detach().cpu().numpy()
    return sal_e, sal_w

def integrated_gradients(model: nn.Module, X_e: torch.Tensor, X_w: torch.Tensor, A_s: torch.sparse_coo_tensor,
                         baseline_e: Optional[torch.Tensor] = None, baseline_w: Optional[torch.Tensor] = None,
                         steps: int = 50, target_edge: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrated gradients approximation (Riemann sum) for single window.
    steps: number of interpolation points
    """
    model.eval()
    if baseline_e is None:
        baseline_e = torch.zeros_like(X_e)
    if baseline_w is None:
        baseline_w = torch.zeros_like(X_w)

    Xe = X_e.clone().detach().to(device).float()
    Xw = X_w.clone().detach().to(device).float()
    baseline_e = baseline_e.to(device).float()
    baseline_w = baseline_w.to(device).float()

    total_grad_e = torch.zeros_like(Xe, device=device)
    total_grad_w = torch.zeros_like(Xw, device=device)
    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        Xstep_e = (baseline_e + alpha * (Xe - baseline_e)).requires_grad_(True)
        Xstep_w = (baseline_w + alpha * (Xw - baseline_w)).requires_grad_(True)
        y = model(Xstep_e, Xstep_w, A_s)
        if target_edge is None:
            score = y.sum()
        else:
            score = y[target_edge, 0]
        model.zero_grad()
        score.backward()
        total_grad_e += Xstep_e.grad.detach()
        total_grad_w += Xstep_w.grad.detach()
    avg_grad_e = total_grad_e / steps
    avg_grad_w = total_grad_w / steps
    ig_e = ((Xe - baseline_e) * avg_grad_e).detach().cpu().numpy()
    ig_w = ((Xw - baseline_w) * avg_grad_w).detach().cpu().numpy()
    return ig_e, ig_w

def permutation_importance(model: nn.Module,
                           X_windows: List[torch.Tensor],
                           Xw_windows: List[torch.Tensor],
                           Y_windows: List[torch.Tensor],
                           A_s: torch.sparse_coo_tensor,
                           metric_fn = lambda y_true, y_pred: float(((y_true - y_pred)**2).mean()),  # MSE
                           n_repeats: int = 10) -> pd.DataFrame:
    """
    Compute permutation importance for weather features using provided windows.
    Each entry in X_windows is one window: [N,W,F]
    Returns a DataFrame with delta metric
    """
    model.eval()
    preds = []
    trues = []
    for Xe, Xw, Y in zip(X_windows, Xw_windows, Y_windows):
        y_pred = model(Xe.to(device).float(), Xw.to(device).float(), A_s).detach().cpu()
        preds.append(y_pred.squeeze(-1))
        trues.append(Y.cpu())
    base_pred = torch.cat(preds, dim=0).numpy()
    base_true = torch.cat(trues, dim=0).numpy()
    base_metric = metric_fn(base_true, base_pred)

    results = []
    Fw = Xw_windows[0].shape[-1]
    for feat_idx in range(Fw):
        metrics = []
        for _ in range(n_repeats):
            shuffled_preds = []
            for Xe, Xw, Y in zip(X_windows, Xw_windows, Y_windows):
                Xw_perm = Xw.clone()
                # flatten and permute global
                flat = Xw_perm[..., feat_idx].reshape(-1)
                perm = flat[torch.randperm(flat.shape[0])]
                Xw_perm[..., feat_idx] = perm.reshape(Xw_perm[..., feat_idx].shape)
                y_pred = model(Xe.to(device).float(), Xw_perm.to(device).float(), A_s).detach().cpu()
                shuffled_preds.append(y_pred.squeeze(-1))
            shuffled_pred = torch.cat(shuffled_preds, dim=0).numpy()
            metrics.append(metric_fn(base_true, shuffled_pred))
        results.append({
            "feature": WEATHER_FEATURES[feat_idx],
            "base_metric": base_metric,
            "perm_mean_metric": float(np.mean(metrics)),
            "perm_std_metric": float(np.std(metrics)),
            "delta": float(np.mean(metrics) - base_metric)
        })
    df = pd.DataFrame(results).sort_values("delta", ascending=False)
    return df

def topk_edges_from_importance(importance_matrix: np.ndarray, k: int = 20) -> List[Tuple[int, float]]:
    """
    Aggregate importance for edges (abs sum across time & features) and return top-k (edge_idx, score)
    """
    if importance_matrix.ndim == 1:
        scores = importance_matrix
    else:
        scores = np.abs(importance_matrix).sum(axis=tuple(range(1, importance_matrix.ndim)))
    idx_sorted = np.argsort(scores)[::-1]
    return [(int(i), float(scores[i])) for i in idx_sorted[:k]]

def save_topk_edges_csv(topk: List[Tuple[int, float]], idx2node: Dict[int, Tuple], filename: str):
    rows = []
    for i, score in topk:
        edge = idx2node[i]
        rows.append({"edge_idx": i, "edge": str(edge), "score": score})
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"Saved top-k edges to {filename}")

def plot_feature_importance(df_perm: pd.DataFrame, top_n: int = 10):
    df_plot = df_perm.head(top_n).sort_values("delta")
    plt.figure(figsize=(8, max(4, 0.5*top_n)))
    plt.barh(df_plot["feature"], df_plot["delta"])
    plt.xlabel("Increase in metric (worse = more important)")
    plt.title("Permutation importance (weather features)")
    plt.tight_layout()
    plt.show()