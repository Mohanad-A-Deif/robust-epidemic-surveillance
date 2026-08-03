"""State, graph, delay, uncertainty, and outbreak-detection metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

EPS = 1e-12


def _paired_finite(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def state_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    true, pred = _paired_finite(y_true, y_pred)
    if true.size == 0:
        return {}
    error = pred - true
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    medae = float(np.median(np.abs(error)))
    denom_range = float(np.ptp(true))
    denom_std = float(np.std(true, ddof=0))
    denom_mean = float(np.mean(np.abs(true)))
    mape = float(np.mean(np.abs(error) / np.maximum(np.abs(true), EPS)))
    smape = float(np.mean(2.0 * np.abs(error) / np.maximum(np.abs(true) + np.abs(pred), EPS)))
    pearson = float(stats.pearsonr(true, pred).statistic) if true.size > 2 and np.std(true) > 0 and np.std(pred) > 0 else np.nan
    spearman = float(stats.spearmanr(true, pred).statistic) if true.size > 2 else np.nan
    r2 = float(r2_score(true, pred)) if true.size > 1 else np.nan
    bias = float(np.mean(error))
    maxae = float(np.max(np.abs(error)))
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "median_ae": medae,
        "max_ae": maxae,
        "nrmse_range": rmse / denom_range if denom_range > EPS else np.nan,
        "nrmse_std": rmse / denom_std if denom_std > EPS else np.nan,
        "nmae_mean": mae / denom_mean if denom_mean > EPS else np.nan,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "bias": bias,
    }
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def state_metrics_by_node(y_true: np.ndarray, y_pred: np.ndarray, node_names: Iterable[str]) -> list[dict[str, float | str]]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError("State arrays must have matching shape (n_nodes, n_time).")
    rows = []
    for i, name in enumerate(node_names):
        row: dict[str, float | str] = {"node": str(name)}
        row.update(state_metrics(true[i], pred[i]))
        rows.append(row)
    return rows


def graph_metrics(w_true: np.ndarray, w_pred: np.ndarray, threshold: float = 1e-6) -> dict[str, float]:
    true = np.asarray(w_true, dtype=float)
    pred = np.asarray(w_pred, dtype=float)
    if true.shape != pred.shape or true.ndim != 2 or true.shape[0] != true.shape[1]:
        raise ValueError("Graph matrices must have matching square shape.")
    mask = ~np.eye(true.shape[0], dtype=bool)
    yt = true[mask]
    yp = pred[mask]
    bt = (yt > threshold).astype(int)
    bp = (yp > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(bt, bp, average="binary", zero_division=0)
    auc = float(roc_auc_score(bt, yp)) if np.unique(bt).size == 2 else np.nan
    auprc = float(average_precision_score(bt, yp)) if np.unique(bt).size == 2 else np.nan
    fro = float(np.linalg.norm(pred - true, ord="fro"))
    norm_true = float(np.linalg.norm(true, ord="fro"))
    edge_corr = float(stats.pearsonr(yt, yp).statistic) if np.std(yt) > 0 and np.std(yp) > 0 else np.nan
    shd = float(np.sum(bt != bp))
    density_true = float(bt.mean())
    density_pred = float(bp.mean())
    return {
        "graph_frobenius": fro,
        "graph_normalized_frobenius": fro / norm_true if norm_true > EPS else np.nan,
        "graph_support_precision": float(precision),
        "graph_support_recall": float(recall),
        "graph_support_f1": float(f1),
        "graph_auroc": auc,
        "graph_auprc": auprc,
        "graph_structural_hamming": shd,
        "graph_edge_weight_r": edge_corr,
        "graph_density_true": density_true,
        "graph_density_pred": density_pred,
        "graph_density_abs_error": abs(density_pred - density_true),
    }


def reference_graph_agreement(w_pred: np.ndarray, adjacency_reference: np.ndarray, threshold: float = 1e-6) -> dict[str, float]:
    ref = np.asarray(adjacency_reference, dtype=float)
    pred = np.asarray(w_pred, dtype=float)
    if ref.shape != pred.shape:
        raise ValueError("Reference adjacency and predicted graph differ in shape.")
    return {f"reference_{k.replace('graph_', '')}": v for k, v in graph_metrics(ref, pred, threshold).items()}


def graph_stability(graphs: list[np.ndarray], threshold: float = 1e-6) -> dict[str, float]:
    if len(graphs) < 2:
        return {"graph_stability_weight_r": np.nan, "graph_stability_jaccard": np.nan, "graph_stability_cv": np.nan}
    correlations, jaccards = [], []
    arrays = [np.asarray(g, dtype=float) for g in graphs]
    mask = ~np.eye(arrays[0].shape[0], dtype=bool)
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            a, b = arrays[i][mask], arrays[j][mask]
            correlations.append(stats.pearsonr(a, b).statistic if np.std(a) > 0 and np.std(b) > 0 else np.nan)
            sa, sb = a > threshold, b > threshold
            union = np.sum(sa | sb)
            jaccards.append(float(np.sum(sa & sb) / union) if union else 1.0)
    stack = np.stack(arrays)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0, ddof=0)
    active = np.abs(mean) > threshold
    cv = float(np.mean(std[active] / np.maximum(np.abs(mean[active]), EPS))) if np.any(active) else np.nan
    return {
        "graph_stability_weight_r": float(np.nanmean(correlations)),
        "graph_stability_jaccard": float(np.nanmean(jaccards)),
        "graph_stability_cv": cv,
    }


def delay_metrics(true_delays: np.ndarray, predicted_delays: np.ndarray) -> dict[str, float]:
    true, pred = _paired_finite(true_delays, predicted_delays)
    if true.size == 0:
        return {}
    err = pred - true
    return {
        "delay_mae": float(np.mean(np.abs(err))),
        "delay_rmse": float(np.sqrt(np.mean(err**2))),
        "delay_bias": float(np.mean(err)),
        "delay_exact_accuracy": float(np.mean(np.rint(pred) == true)),
        "delay_within_one_accuracy": float(np.mean(np.abs(pred - true) <= 1.0)),
    }


def posterior_delay_metrics(true_delays: np.ndarray, posterior: list[np.ndarray], feasible_delays: list[np.ndarray]) -> dict[str, float]:
    masses, means, entropies = [], [], []
    for delay, probs, support in zip(true_delays, posterior, feasible_delays, strict=True):
        p = np.asarray(probs, dtype=float)
        d = np.asarray(support, dtype=float)
        p = p / max(float(p.sum()), EPS)
        match = np.where(d == delay)[0]
        masses.append(float(p[match[0]]) if match.size else 0.0)
        means.append(float(np.sum(p * d)))
        entropies.append(float(-np.sum(p * np.log(np.maximum(p, EPS)))))
    result = delay_metrics(np.asarray(true_delays), np.asarray(means))
    result.update(
        {
            "delay_true_posterior_mass": float(np.mean(masses)) if masses else np.nan,
            "delay_posterior_entropy": float(np.mean(entropies)) if entropies else np.nan,
        }
    )
    return result


def uncertainty_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = np.isfinite(true) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        return {}
    coverage = np.mean((true[mask] >= lo[mask]) & (true[mask] <= hi[mask]))
    width = np.mean(hi[mask] - lo[mask])
    winkler = (hi[mask] - lo[mask]).copy()
    alpha = 0.05
    below = true[mask] < lo[mask]
    above = true[mask] > hi[mask]
    winkler[below] += (2 / alpha) * (lo[mask][below] - true[mask][below])
    winkler[above] += (2 / alpha) * (true[mask][above] - hi[mask][above])
    return {"interval_coverage_95": float(coverage), "interval_mean_width": float(width), "winkler_score_95": float(np.mean(winkler))}


def outbreak_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.9) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    threshold = float(np.nanquantile(true, quantile))
    true_event = (true >= threshold).astype(int).ravel()
    pred_event = (pred >= threshold).astype(int).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(true_event, pred_event, average="binary", zero_division=0)
    return {
        "outbreak_threshold": threshold,
        "outbreak_precision": float(precision),
        "outbreak_recall": float(recall),
        "outbreak_f1": float(f1),
    }
