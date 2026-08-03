"""Publication-ready result plots generated only from computed result files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import GRAY_LEVELS, method_style, panel_labels, save_png, style_axis


def _mean_sd(frame: pd.DataFrame, metric: str, group: str = "method") -> pd.DataFrame:
    return frame.groupby(group, as_index=False)[metric].agg(["mean", "std"]).reset_index().fillna(0.0)


def plot_main_benchmark(seed_results: pd.DataFrame, output: Path, scenario: str | None = None) -> Path | None:
    frame = seed_results.copy()
    if scenario is not None:
        frame = frame[frame["scenario"] == scenario]
    metrics = [m for m in ["rmse_test", "mae_test", "r2_test"] if m in frame.columns]
    if not metrics or frame.empty:
        return None
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.2, 2.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    methods = frame["method"].drop_duplicates().tolist()
    x = np.arange(len(methods))
    for ax, metric in zip(axes, metrics, strict=True):
        summary = frame.groupby("method")[metric].agg(["mean", "std"]).reindex(methods)
        bars = ax.bar(x, summary["mean"], yerr=summary["std"].fillna(0), capsize=3, color=[GRAY_LEVELS[(i + 1) % len(GRAY_LEVELS)] for i in range(len(methods))], edgecolor="black", linewidth=0.8)
        ax.set_ylabel(metric.replace("_test", "").upper())
        style_axis(ax)
        for bar in bars:
            bar.set_linewidth(0.8)
    axes[-1].set_xticks(x, methods, rotation=35, ha="right")
    panel_labels(axes)
    return save_png(fig, output)


def plot_robustness(seed_results: pd.DataFrame, family: str, x_col: str, metric: str, output: Path) -> Path | None:
    frame = seed_results[seed_results["family"] == family].copy()
    if frame.empty or x_col not in frame or metric not in frame:
        return None
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for i, (method, group) in enumerate(frame.groupby("method", sort=False)):
        summary = group.groupby(x_col)[metric].agg(["mean", "std"]).sort_index()
        if summary.empty:
            continue
        x = summary.index.to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        sd = summary["std"].fillna(0).to_numpy(dtype=float)
        style = method_style(i)
        ax.plot(x, y, label=method, **style)
        ax.fill_between(x, y - sd, y + sd, color=style["color"], alpha=0.12, linewidth=0)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_test", "").upper())
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    return save_png(fig, output)


def plot_trajectory_panel(truth: np.ndarray, predictions: dict[str, np.ndarray], node_names: list[str], output: Path, indices: np.ndarray | None = None, max_nodes: int = 4) -> Path | None:
    if truth.ndim != 2 or not predictions:
        return None
    nodes = min(max_nodes, truth.shape[0])
    indices = np.arange(truth.shape[1]) if indices is None else np.asarray(indices)
    fig, axes = plt.subplots(nodes, 1, figsize=(7.3, 2.0 * nodes), sharex=True)
    if nodes == 1:
        axes = [axes]
    for node, ax in enumerate(axes):
        ax.plot(indices, truth[node, indices], color="0.55", linestyle="-", linewidth=1.6, label="Ground truth")
        for i, (method, pred) in enumerate(predictions.items()):
            style = method_style(i)
            ax.plot(indices, pred[node, indices], label=method, markevery=max(1, len(indices) // 12), **style)
        ax.set_ylabel(node_names[node])
        style_axis(ax)
    axes[-1].set_xlabel("Time index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=min(4, len(labels)))
    fig._tight_bottom = 0.11
    panel_labels(axes)
    return save_png(fig, output)


def plot_objective_histories(histories: pd.DataFrame, output: Path) -> Path | None:
    if histories.empty or not {"method", "iteration", "objective"}.issubset(histories.columns):
        return None
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for i, (method, group) in enumerate(histories.groupby("method", sort=False)):
        summary = group.groupby("iteration")["objective"].agg(["mean", "std"])
        style = method_style(i)
        x = summary.index.to_numpy()
        y = summary["mean"].to_numpy()
        sd = summary["std"].fillna(0).to_numpy()
        ax.plot(x, y, label=method, **style)
        ax.fill_between(x, y - sd, y + sd, color=style["color"], alpha=0.12)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Objective value")
    ax.set_yscale("symlog", linthresh=1e-2)
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    return save_png(fig, output)


def plot_graph_heatmaps(w_true: np.ndarray | None, estimates: dict[str, np.ndarray], labels: list[str], output: Path, max_methods: int = 5) -> Path | None:
    matrices: list[tuple[str, np.ndarray]] = []
    if w_true is not None:
        matrices.append(("Ground truth", np.asarray(w_true)))
    matrices.extend(list(estimates.items())[:max_methods])
    if not matrices:
        return None
    n = len(matrices)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes_array = plt.subplots(nrows, ncols, figsize=(8.2, 3.7 * nrows), squeeze=False)
    axes = axes_array.ravel().tolist()
    if w_true is not None:
        vmax = max(float(np.nanmax(np.asarray(w_true))), 1e-12)
    else:
        positive = np.concatenate([matrix[np.isfinite(matrix) & (matrix > 0)].ravel() for _, matrix in matrices if np.any(np.isfinite(matrix) & (matrix > 0))])
        vmax = float(np.quantile(positive, 0.95)) if positive.size else 1.0
    for ax, (title, matrix) in zip(axes, matrices, strict=False):
        image = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=max(vmax, 1e-12))
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_title(title)
        style_axis(ax, grid=False)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    for ax in axes[len(matrices):]:
        ax.set_visible(False)
    panel_labels(axes[:len(matrices)])
    return save_png(fig, output)


def plot_runtime_accuracy(seed_results: pd.DataFrame, output: Path, metric: str = "rmse_test") -> Path | None:
    if not {metric, "runtime_seconds", "method"}.issubset(seed_results.columns):
        return None
    summary = seed_results.groupby("method")[[metric, "runtime_seconds"]].mean().dropna()
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for i, (method, row) in enumerate(summary.iterrows()):
        style = method_style(i)
        ax.scatter(row["runtime_seconds"], row[metric], marker=style["marker"], facecolors="white", edgecolors=style["color"], linewidths=1.1, s=55)
        ax.annotate(method, (row["runtime_seconds"], row[metric]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Runtime per instance (s)")
    ax.set_ylabel(metric.replace("_test", "").upper())
    ax.set_xscale("log")
    style_axis(ax)
    return save_png(fig, output)


def plot_average_ranks(ranks: pd.DataFrame, output: Path) -> Path | None:
    if ranks.empty or not {"method", "average_rank"}.issubset(ranks.columns):
        return None
    ranks = ranks.sort_values("average_rank")
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    y = np.arange(len(ranks))
    ax.errorbar(ranks["average_rank"], y, xerr=ranks.get("rank_std", pd.Series(np.zeros(len(ranks)))), fmt="o", color="black", markerfacecolor="white", capsize=3)
    ax.set_yticks(y, ranks["method"])
    ax.invert_yaxis()
    ax.set_xlabel("Average rank (lower is better)")
    style_axis(ax)
    return save_png(fig, output)


def plot_significance_matrix(pairwise: pd.DataFrame, output: Path, p_col: str = "wilcoxon_p_holm") -> Path | None:
    if pairwise.empty or p_col not in pairwise:
        return None
    methods = sorted(set(pairwise["method_a"]).union(pairwise["method_b"]))
    matrix = np.ones((len(methods), len(methods)))
    lookup = {name: i for i, name in enumerate(methods)}
    for row in pairwise.itertuples(index=False):
        i, j = lookup[row.method_a], lookup[row.method_b]
        p = float(getattr(row, p_col))
        matrix[i, j] = matrix[j, i] = p
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(-np.log10(np.maximum(matrix, 1e-12)), cmap="Greys")
    ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=7, color="white" if matrix[i,j] < 0.01 else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$-\log_{10}(p_{Holm})$")
    style_axis(ax, grid=False)
    return save_png(fig, output)


def generate_result_figures(
    seed_results: pd.DataFrame,
    output_dir: str | Path,
    histories: pd.DataFrame | None = None,
    ranks: pd.DataFrame | None = None,
    pairwise: pd.DataFrame | None = None,
    reference_scenario: str = "reference_moderate",
) -> list[str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    calls = [
        plot_main_benchmark(seed_results, root / "fig_results_01_main_benchmark.png", reference_scenario),
        plot_runtime_accuracy(seed_results[seed_results["scenario"] == reference_scenario], root / "fig_results_02_runtime_accuracy.png"),
        plot_robustness(seed_results, "missingness", "drop_probability", "rmse_test", root / "fig_results_03_missingness_rmse.png"),
        plot_robustness(seed_results, "delay", "mean_delay", "rmse_test", root / "fig_results_04_delay_rmse.png"),
        plot_robustness(seed_results, "outlier", "outlier_probability", "rmse_test", root / "fig_results_05_outlier_rmse.png"),
        plot_robustness(seed_results, "prior_mismatch", "prior_mismatch_level", "rmse_test", root / "fig_results_06_prior_mismatch_rmse.png"),
    ]
    if histories is not None:
        calls.append(plot_objective_histories(histories, root / "fig_results_07_convergence.png"))
    if ranks is not None:
        calls.append(plot_average_ranks(ranks, root / "fig_results_08_average_ranks.png"))
    if pairwise is not None:
        calls.append(plot_significance_matrix(pairwise, root / "fig_results_09_significance_matrix.png"))
    for item in calls:
        if item is not None:
            generated.append(str(item))
    return generated
