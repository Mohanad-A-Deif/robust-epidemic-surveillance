"""Exploratory data analysis tables and Nature/IEEE grayscale figures."""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller, kpss

from .io_utils import ProcessedBundle, ensure_output_tree, save_table, write_json
from .style import GRAY_LEVELS, method_style, panel_labels, save_png, style_axis


def _safe_adf(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 12 or np.std(x) < 1e-12:
        return {"adf_stat": np.nan, "adf_p": np.nan}
    try:
        result = adfuller(x, autolag="AIC")
        return {"adf_stat": float(result[0]), "adf_p": float(result[1])}
    except Exception:
        return {"adf_stat": np.nan, "adf_p": np.nan}


def _safe_kpss(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 12 or np.std(x) < 1e-12:
        return {"kpss_stat": np.nan, "kpss_p": np.nan}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(x, regression="c", nlags="auto")
        return {"kpss_stat": float(result[0]), "kpss_p": float(result[1])}
    except Exception:
        return {"kpss_stat": np.nan, "kpss_p": np.nan}


def _peak_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 14) -> tuple[int, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    best_lag, best_corr = 0, np.nan
    best_abs = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[:lag]
        elif lag > 0:
            aa, bb = a[:-lag], b[lag:]
        else:
            aa, bb = a, b
        mask = np.isfinite(aa) & np.isfinite(bb)
        if mask.sum() < 4 or np.std(aa[mask]) == 0 or np.std(bb[mask]) == 0:
            continue
        corr = float(stats.pearsonr(aa[mask], bb[mask]).statistic)
        if abs(corr) > best_abs:
            best_lag, best_corr, best_abs = lag, corr, abs(corr)
    return best_lag, best_corr


def _descriptive_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in frame.columns:
        x = pd.to_numeric(frame[col], errors="coerce")
        finite = x.dropna()
        q1, median, q3 = finite.quantile([0.25, 0.5, 0.75]) if len(finite) else (np.nan, np.nan, np.nan)
        rows.append(
            {
                "node": col,
                "n": int(len(x)),
                "valid": int(finite.size),
                "missing": int(x.isna().sum()),
                "missing_fraction": float(x.isna().mean()),
                "zero_fraction": float((finite == 0).mean()) if len(finite) else np.nan,
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                "min": float(finite.min()) if len(finite) else np.nan,
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "max": float(finite.max()) if len(finite) else np.nan,
                "iqr": float(q3 - q1),
                "skewness": float(stats.skew(finite, bias=False)) if len(finite) > 2 else np.nan,
                "excess_kurtosis": float(stats.kurtosis(finite, fisher=True, bias=False)) if len(finite) > 3 else np.nan,
                "coefficient_variation": float(finite.std(ddof=1) / finite.mean()) if len(finite) > 1 and abs(finite.mean()) > 1e-12 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _trend_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = np.arange(len(frame), dtype=float)
    for col in frame.columns:
        y = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() > 2:
            slope, intercept, r, p, stderr = stats.linregress(t[mask], y[mask])
        else:
            slope = intercept = r = p = stderr = np.nan
        acf_values = acf(y[mask], nlags=min(14, max(1, mask.sum() // 3)), fft=True, missing="drop") if mask.sum() > 5 else np.array([np.nan])
        row = {
            "node": col,
            "linear_slope_per_day": float(slope),
            "trend_r": float(r),
            "trend_p": float(p),
            "trend_slope_se": float(stderr),
            "acf_lag1": float(acf_values[1]) if len(acf_values) > 1 else np.nan,
            "acf_lag7": float(acf_values[7]) if len(acf_values) > 7 else np.nan,
            "peak_date": frame.index[int(np.nanargmax(y))].strftime("%Y-%m-%d") if mask.any() else "",
            "peak_value": float(np.nanmax(y)) if mask.any() else np.nan,
        }
        row.update(_safe_adf(y))
        row.update(_safe_kpss(y))
        rows.append(row)
    return pd.DataFrame(rows)


def _cross_lag_tables(frame: pd.DataFrame, max_lag: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = frame.columns.tolist()
    lags = pd.DataFrame(0, index=cols, columns=cols, dtype=float)
    corrs = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                continue
            lag, corr = _peak_lag(frame[a].to_numpy(), frame[b].to_numpy(), max_lag=max_lag)
            lags.loc[a, b] = lag
            corrs.loc[a, b] = corr
    return lags, corrs


def _graph_table(adjacency: np.ndarray | None, node_names: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if adjacency is None:
        return pd.DataFrame(), {}
    a = np.asarray(adjacency, dtype=float)
    graph = nx.from_numpy_array((a > 0).astype(int), create_using=nx.Graph)
    mapping = {i: node_names[i] for i in range(len(node_names))}
    graph = nx.relabel_nodes(graph, mapping)
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    rows = []
    for name in node_names:
        rows.append({"node": name, "degree": degree.get(name, 0), "clustering": clustering.get(name, 0.0)})
    summary = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "connected": nx.is_connected(graph) if graph.number_of_nodes() else False,
        "average_degree": float(np.mean(list(degree.values()))) if degree else 0.0,
        "average_clustering": nx.average_clustering(graph) if graph.number_of_nodes() else np.nan,
        "diameter": nx.diameter(graph) if graph.number_of_nodes() and nx.is_connected(graph) else None,
    }
    return pd.DataFrame(rows), summary


def _plot_trajectories(frame: pd.DataFrame, output: Path, title: str, ylabel: str, splits: pd.Series | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, col in enumerate(frame.columns):
        style = method_style(i)
        ax.plot(frame.index, frame[col], label=col, markevery=max(1, len(frame) // 15), **style)
    if splits is not None and len(splits):
        changes = np.where(splits.astype(str).to_numpy()[1:] != splits.astype(str).to_numpy()[:-1])[0]
        for index in changes:
            ax.axvline(frame.index[index + 1], color="0.2", linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    style_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    save_png(fig, output)


def _plot_distribution(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    data = [frame[col].dropna().to_numpy() for col in frame.columns]
    box = ax.boxplot(data, tick_labels=frame.columns, patch_artist=True, showfliers=True, whis=1.5)
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(GRAY_LEVELS[(i + 2) % len(GRAY_LEVELS)])
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for key in ["whiskers", "caps", "medians"]:
        for artist in box[key]:
            artist.set_color("black")
            artist.set_linewidth(0.8)
    for flier in box["fliers"]:
        flier.set(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=3)
    ax.set_ylabel("Daily epidemic state")
    ax.tick_params(axis="x", rotation=30)
    style_axis(ax)
    save_png(fig, output)


def _heatmap(matrix: pd.DataFrame | np.ndarray, labels: list[str], output: Path, title: str, cbar_label: str, fmt: str = ".2f", vmin: float | None = None, vmax: float | None = None) -> None:
    values = matrix.to_numpy(dtype=float) if isinstance(matrix, pd.DataFrame) else np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.3, 4.5))
    image = ax.imshow(values, cmap="Greys", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)
    threshold = (np.nanmin(values) + np.nanmax(values)) / 2 if np.isfinite(values).any() else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=7.5, color="white" if values[i, j] > threshold else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    style_axis(ax, grid=False)
    save_png(fig, output)


def _plot_acf(frame: pd.DataFrame, output: Path, max_lag: int = 28) -> None:
    n = len(frame.columns)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, max(2.2, 1.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (ax, col) in enumerate(zip(axes, frame.columns, strict=True)):
        values = frame[col].dropna().to_numpy(dtype=float)
        nlags = min(max_lag, max(1, len(values) - 2))
        result = acf(values, nlags=nlags, fft=True) if len(values) > 3 else np.full(nlags + 1, np.nan)
        markerline, stemlines, baseline = ax.stem(range(len(result)), result, linefmt="k-", markerfmt="ko", basefmt="k-")
        plt.setp(stemlines, linewidth=0.8)
        plt.setp(markerline, markersize=3, markerfacecolor="white")
        conf = 1.96 / math.sqrt(max(len(values), 1))
        ax.axhline(conf, color="0.4", linestyle=":", linewidth=0.8)
        ax.axhline(-conf, color="0.4", linestyle=":", linewidth=0.8)
        ax.set_ylabel(col)
        style_axis(ax)
    axes[-1].set_xlabel("Lag (days)")
    panel_labels(axes)
    save_png(fig, output)


def _plot_weekly(frame: pd.DataFrame, output: Path) -> None:
    weekday = frame.copy()
    weekday["weekday"] = weekday.index.dayofweek
    means = weekday.groupby("weekday").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, col in enumerate(frame.columns):
        ax.plot(range(7), means[col], label=col, **method_style(i))
    ax.set_xticks(range(7), labels=labels)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Mean daily state")
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(3, len(frame.columns)))
    save_png(fig, output)


def _plot_graph(adjacency: np.ndarray, node_names: list[str], output: Path) -> None:
    a = np.asarray(adjacency, dtype=float)
    graph = nx.from_numpy_array(a, create_using=nx.Graph)
    mapping = {i: node_names[i] for i in range(len(node_names))}
    graph = nx.relabel_nodes(graph, mapping)
    positions = nx.spring_layout(graph, seed=23, weight="weight")
    fig, ax = plt.subplots(figsize=(5.7, 4.7))
    weights = [max(0.7, 2.0 * float(graph[u][v].get("weight", 1.0))) for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, positions, ax=ax, width=weights, edge_color="0.35", arrows=False)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color="0.85", edgecolors="black", linewidths=0.9, node_size=900)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_family="serif", font_size=9)
    ax.set_axis_off()
    ax.set_title("Reference spatial network")
    save_png(fig, output)


def scenario_eda(message_files: list[Path], output_root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    if not message_files:
        return {"message_file_count": 0}
    summary_rows, delay_rows = [], []
    for path in message_files:
        frame = pd.read_csv(path)
        for col in ["received", "dropped", "right_censored", "is_outlier"]:
            if col in frame:
                frame[col] = frame[col].astype(str).str.lower().isin(["true", "1"])
        received = frame[frame["received"]] if "received" in frame else frame
        summary_rows.append(
            {
                "file": str(path),
                "scenario": frame["scenario"].iloc[0] if "scenario" in frame and len(frame) else path.parent.name,
                "family": frame["family"].iloc[0] if "family" in frame and len(frame) else path.parent.parent.name,
                "seed": int(frame["seed"].iloc[0]) if "seed" in frame and len(frame) else np.nan,
                "generated": len(frame),
                "received": int(frame["received"].sum()) if "received" in frame else len(frame),
                "drop_fraction": float(frame["dropped"].mean()) if "dropped" in frame else np.nan,
                "right_censored_fraction": float(frame["right_censored"].mean()) if "right_censored" in frame else np.nan,
                "outlier_fraction_received": float(received["is_outlier"].mean()) if len(received) and "is_outlier" in received else np.nan,
                "mean_delay_received": float(received["delay"].mean()) if len(received) and "delay" in received else np.nan,
                "max_delay_received": float(received["delay"].max()) if len(received) and "delay" in received else np.nan,
                "collision_rows": int(received.duplicated(["node_id", "arrival_index"], keep=False).sum()) if len(received) and {"node_id", "arrival_index"}.issubset(received.columns) else np.nan,
            }
        )
        if "delay" in received:
            counts = received["delay"].value_counts(normalize=True).sort_index()
            for delay, probability in counts.items():
                delay_rows.append({"scenario": summary_rows[-1]["scenario"], "seed": summary_rows[-1]["seed"], "delay": int(delay), "probability": float(probability)})
    summary = pd.DataFrame(summary_rows)
    delays = pd.DataFrame(delay_rows)
    save_table(summary, "eda_scenario_integrity", paths)
    if not delays.empty:
        save_table(delays, "eda_realized_delay_distribution", paths)

    grouped = summary.groupby(["family", "scenario"], as_index=False).agg(
        drop_fraction_mean=("drop_fraction", "mean"),
        outlier_fraction_mean=("outlier_fraction_received", "mean"),
        mean_delay=("mean_delay_received", "mean"),
        collision_rows_mean=("collision_rows", "mean"),
    )
    if not grouped.empty:
        fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True)
        x = np.arange(len(grouped))
        axes[0].bar(x, grouped["drop_fraction_mean"], color="0.75", edgecolor="black", linewidth=0.8)
        axes[1].bar(x, grouped["mean_delay"], color="0.55", edgecolor="black", linewidth=0.8)
        axes[2].bar(x, grouped["outlier_fraction_mean"], color="0.35", edgecolor="black", linewidth=0.8)
        axes[0].set_ylabel("Drop fraction")
        axes[1].set_ylabel("Mean delay")
        axes[2].set_ylabel("Outlier fraction")
        axes[2].set_xticks(x, grouped["scenario"], rotation=45, ha="right")
        for ax in axes:
            style_axis(ax)
        panel_labels(axes)
        save_png(fig, output_root / "fig_eda_11_scenario_characteristics.png")
    return {"message_file_count": len(message_files), "scenario_count": int(summary["scenario"].nunique()), "seed_count": int(summary["seed"].nunique())}


def run_eda(bundle: ProcessedBundle, output_dir: str | Path, message_files: list[Path] | None = None, max_cross_lag: int = 14) -> dict[str, Any]:
    paths = ensure_output_tree(output_dir)
    eda_figures = paths["eda"] / "figures_png"
    eda_figures.mkdir(parents=True, exist_ok=True)

    overview = pd.DataFrame(
        [
            {
                "start_date": bundle.x_raw.index.min().strftime("%Y-%m-%d"),
                "end_date": bundle.x_raw.index.max().strftime("%Y-%m-%d"),
                "n_days": len(bundle.x_raw),
                "n_nodes": bundle.x_raw.shape[1],
                "n_values": int(bundle.x_raw.size),
                "missing_values": int(bundle.x_raw.isna().sum().sum()),
                "overall_missing_fraction": float(bundle.x_raw.isna().mean().mean()),
                "train_days": int((bundle.splits == "train").sum()),
                "validation_days": int((bundle.splits == "validation").sum()),
                "test_days": int((bundle.splits == "test").sum()),
            }
        ]
    )
    save_table(overview, "eda_dataset_overview", paths)

    descriptive_raw = _descriptive_table(bundle.x_raw)
    descriptive_log = _descriptive_table(bundle.x_log1p)
    descriptive_std = _descriptive_table(bundle.x_standardized)
    save_table(descriptive_raw, "eda_descriptive_raw", paths)
    save_table(descriptive_log, "eda_descriptive_log1p", paths)
    save_table(descriptive_std, "eda_descriptive_standardized", paths)

    split_rows = []
    for split_name, idx in bundle.splits.groupby(bundle.splits).groups.items():
        section = bundle.x_raw.loc[idx]
        for col in section.columns:
            split_rows.append({"split": split_name, "node": col, "mean": float(section[col].mean()), "std": float(section[col].std(ddof=1)), "min": float(section[col].min()), "max": float(section[col].max()), "missing_fraction": float(section[col].isna().mean())})
    split_table = pd.DataFrame(split_rows)
    save_table(split_table, "eda_split_statistics", paths)

    trend = _trend_table(bundle.x_log1p)
    save_table(trend, "eda_trend_stationarity_autocorrelation", paths)

    pearson = bundle.x_log1p.corr(method="pearson")
    spearman = bundle.x_log1p.corr(method="spearman")
    save_table(pearson.reset_index(names="node"), "eda_pearson_correlation", paths)
    save_table(spearman.reset_index(names="node"), "eda_spearman_correlation", paths)

    lag_matrix, lag_corr = _cross_lag_tables(bundle.x_log1p, max_lag=max_cross_lag)
    save_table(lag_matrix.reset_index(names="source_node"), "eda_peak_cross_correlation_lags", paths)
    save_table(lag_corr.reset_index(names="source_node"), "eda_peak_cross_correlation_values", paths)

    graph_nodes, graph_summary = _graph_table(bundle.adjacency, bundle.x_raw.columns.tolist())
    if not graph_nodes.empty:
        save_table(graph_nodes, "eda_reference_graph_node_statistics", paths)
        write_json(graph_summary, paths["tables_csv"] / "eda_reference_graph_summary.json")

    _plot_trajectories(bundle.x_raw, eda_figures / "fig_eda_01_raw_trajectories.png", "Observed epidemic trajectories", "Daily epidemic state", bundle.splits)
    _plot_trajectories(bundle.x_log1p, eda_figures / "fig_eda_02_log1p_trajectories.png", "Log-transformed epidemic trajectories", r"$\log(1+x)$", bundle.splits)
    _plot_trajectories(bundle.x_standardized, eda_figures / "fig_eda_03_standardized_trajectories.png", "Leakage-safe standardized trajectories", "Standardized state", bundle.splits)
    _plot_distribution(bundle.x_raw, eda_figures / "fig_eda_04_raw_distribution_boxplots.png")
    _heatmap(pearson, pearson.columns.tolist(), eda_figures / "fig_eda_05_pearson_correlation_heatmap.png", "Interregional Pearson correlation", "Correlation", vmin=-1, vmax=1)
    _heatmap(lag_matrix, lag_matrix.columns.tolist(), eda_figures / "fig_eda_06_peak_lag_heatmap.png", "Peak cross-correlation lag", "Lag (days)", fmt=".0f", vmin=-max_cross_lag, vmax=max_cross_lag)
    _plot_acf(bundle.x_log1p, eda_figures / "fig_eda_07_autocorrelation.png")
    _plot_weekly(bundle.x_raw, eda_figures / "fig_eda_08_weekly_pattern.png")
    if bundle.adjacency is not None:
        _heatmap(bundle.adjacency, bundle.x_raw.columns.tolist(), eda_figures / "fig_eda_09_reference_adjacency_heatmap.png", "Reference adjacency matrix", "Edge weight", fmt=".1f")
        _plot_graph(bundle.adjacency, bundle.x_raw.columns.tolist(), eda_figures / "fig_eda_10_reference_network.png")

    scenario_summary = scenario_eda(message_files or [], eda_figures, paths)
    manifest = {
        "eda_complete": True,
        "tables": sorted(str(p.relative_to(paths["root"])) for p in paths["tables_csv"].glob("eda_*")),
        "figures": sorted(str(p.relative_to(paths["root"])) for p in eda_figures.glob("*.png")),
        "scenario_summary": scenario_summary,
        "graph_summary": graph_summary,
    }
    write_json(manifest, paths["root"] / "eda_manifest.json")
    return manifest
