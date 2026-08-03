"""End-to-end model execution, metrics, statistical tests, tables, and figures."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import (
    ArrivalAlignedInterpolation,
    BaselineConfig,
    DelayBackprojectionNowcast,
    GraphTemporalSmoother,
    LowRankMatrixCompletion,
    OracleTimestampInterpolation,
    RobustMedianSmoother,
)
from .io_utils import ProcessedBundle, ensure_output_tree, read_json, read_messages, save_table, write_json
from .metrics import (
    graph_metrics,
    graph_stability,
    outbreak_detection_metrics,
    posterior_delay_metrics,
    reference_graph_agreement,
    state_metrics,
    state_metrics_by_node,
)
from .proposed_model import DelayAwareRobustGraphInference, InferenceConfig, InferenceResult
from .result_plots import generate_result_figures, plot_graph_heatmaps, plot_trajectory_panel
from .statistics import average_ranks, friedman_test, pairwise_method_tests, summarize_seed_results


def _scenario_parameter_row(scenario: dict[str, Any]) -> dict[str, float | str]:
    pmf = np.asarray(scenario.get("delay_pmf", [1.0]), dtype=float)
    pmf /= max(pmf.sum(), 1e-12)
    mean_delay = float(np.sum(np.arange(len(pmf)) * pmf))
    return {
        "scenario": scenario["name"],
        "family": scenario.get("family", "unknown"),
        "drop_probability": float(scenario.get("drop_probability", np.nan)),
        "outlier_probability": float(scenario.get("outlier_probability", np.nan)),
        "max_delay": float(scenario.get("max_delay", len(pmf) - 1)),
        "mean_delay": mean_delay,
        "prior_mismatch_level": float(scenario.get("prior_mismatch_level", 0.0)),
    }


def _delay_pmf_by_slice(scenario: dict[str, Any], config: dict[str, Any], inference: bool = True) -> dict[str, np.ndarray]:
    key = "inference_delay_pmf" if inference and "inference_delay_pmf" in scenario else "delay_pmf"
    base = np.asarray(scenario[key], dtype=float)
    base /= max(base.sum(), 1e-12)
    bonus = float(config.get("scenario_generation", {}).get("critical_zero_delay_bonus", 0.0))
    critical = base.copy()
    critical[0] += bonus
    critical /= critical.sum()
    return {"critical": critical, "standard": base, "default": base}


def _method_suite(
    quick: bool,
    adjacency: np.ndarray | None,
    w_true: np.ndarray | None,
    candidate_mask: np.ndarray | None,
    base_cfg: InferenceConfig,
) -> dict[str, Any]:
    baseline_cfg = BaselineConfig(beta=base_cfg.beta, gamma=base_cfg.gamma, row_sum_cap=base_cfg.row_sum_cap, candidate_mask=candidate_mask)
    methods: dict[str, Any] = {
        "Proposed": DelayAwareRobustGraphInference(base_cfg),
        "No-delay ablation": DelayAwareRobustGraphInference(replace(base_cfg, no_delay=True)),
        "No-robustness ablation": DelayAwareRobustGraphInference(replace(base_cfg, huber_kappa=1e6)),
        "Known-delay oracle": DelayAwareRobustGraphInference(replace(base_cfg, oracle_delay=True)),
        "Arrival interpolation": ArrivalAlignedInterpolation(baseline_cfg),
        "Delay backprojection": DelayBackprojectionNowcast(baseline_cfg),
    }
    if quick:
        return methods
    methods.update(
        {
            "Robust median smoother": RobustMedianSmoother(window=5, config=baseline_cfg),
            "Robust low-rank completion": LowRankMatrixCompletion(rank=3, max_iter=80, config=baseline_cfg),
        }
    )
    if adjacency is not None:
        methods["Graph-temporal reconstruction"] = GraphTemporalSmoother(adjacency, config=baseline_cfg)
    fixed = w_true if w_true is not None else adjacency
    if fixed is not None:
        fixed = np.asarray(fixed, dtype=float)
        if w_true is None:
            row_sum = fixed.sum(axis=1, keepdims=True)
            fixed = np.divide(0.5 * fixed, np.maximum(row_sum, 1.0), where=np.ones_like(fixed, dtype=bool))
        methods["Fixed-graph oracle/reference"] = DelayAwareRobustGraphInference(replace(base_cfg, fixed_graph=fixed, update_graph=False))
    return methods


def _result_metrics(
    result: InferenceResult,
    truth: np.ndarray,
    test_indices: np.ndarray,
    node_names: list[str],
    messages: pd.DataFrame,
    w_true: np.ndarray | None,
    adjacency: np.ndarray | None,
    graph_threshold: float,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    row: dict[str, float] = {}
    row.update({f"{k}_all": v for k, v in state_metrics(truth, result.x_hat).items()})
    row.update({f"{k}_test": v for k, v in state_metrics(truth[:, test_indices], result.x_hat[:, test_indices]).items()})
    row.update({f"{k}_all": v for k, v in outbreak_detection_metrics(truth, result.x_hat).items()})
    row.update({f"{k}_test": v for k, v in outbreak_detection_metrics(truth[:, test_indices], result.x_hat[:, test_indices]).items()})
    if w_true is not None:
        row.update(graph_metrics(w_true, result.w_hat, threshold=graph_threshold))
    elif adjacency is not None:
        row.update(reference_graph_agreement(result.w_hat, adjacency, threshold=graph_threshold))

    if result.posterior:
        received = messages.copy()
        if "received" in received:
            if received["received"].dtype != bool:
                received["received"] = received["received"].astype(str).str.lower().isin(["true", "1"])
            received = received[received["received"]]
        true_delays = pd.to_numeric(received["delay"], errors="coerce").fillna(0).to_numpy(dtype=int)
        if len(true_delays) == len(result.posterior):
            row.update(posterior_delay_metrics(true_delays, result.posterior, result.feasible_delays))

    row.update(
        {
            "runtime_seconds": result.runtime_seconds,
            "peak_memory_mb": result.peak_memory_mb,
            "iterations": result.iterations,
            "converged": float(result.converged),
            "final_objective": float(result.objective_history[-1]) if result.objective_history else np.nan,
            "objective_nonincreasing": float(result.diagnostics.get("objective_nonincreasing", np.nan)),
            "learned_graph_density": float(result.diagnostics.get("graph_density", np.mean(result.w_hat > graph_threshold))),
            "learned_graph_max_row_sum": float(result.w_hat.sum(axis=1).max()) if result.w_hat.size else 0.0,
        }
    )
    node_rows = state_metrics_by_node(truth[:, test_indices], result.x_hat[:, test_indices], node_names)
    return row, node_rows


def run_experiments(
    bundle: ProcessedBundle,
    scenario_files: list[Path],
    config_path: str | Path,
    output_dir: str | Path,
    quick: bool = False,
    scenario_limit: set[str] | None = None,
    base_inference_config: InferenceConfig | None = None,
    graph_threshold: float = 1e-4,
) -> dict[str, Any]:
    paths = ensure_output_tree(output_dir)
    config = read_json(config_path, {}) or {}
    scenario_lookup = {scenario["name"]: scenario for scenario in config.get("scenario_generation", {}).get("scenarios", [])}
    truth = bundle.x_log1p.to_numpy(dtype=float).T
    node_names = bundle.x_log1p.columns.tolist()
    test_indices = np.where(bundle.splits.reindex(bundle.x_log1p.index).astype(str).to_numpy() == "test")[0]
    if not len(test_indices):
        test_indices = np.arange(max(0, truth.shape[1] - max(1, truth.shape[1] // 5)), truth.shape[1])

    w_true = None
    for candidate in [Path(config_path).parent / "processed" / "w_true.npy", Path(config_path).parent / "data" / "processed" / "w_true.npy"]:
        if candidate.exists():
            w_true = np.load(candidate)
            break
    candidate_mask = None
    if bundle.adjacency is not None:
        candidate_mask = np.asarray(bundle.adjacency) > 0
        candidate_mask = candidate_mask | candidate_mask.T
        np.fill_diagonal(candidate_mask, False)

    base_cfg = base_inference_config or InferenceConfig(
        candidate_mask=candidate_mask,
        max_outer_iter=5 if quick else 45,
        max_x_iter=2 if quick else 12,
        max_w_iter=1 if quick else 8,
        tol_objective=8e-4 if quick else 2e-5,
    )
    methods = _method_suite(quick, bundle.adjacency, w_true, candidate_mask, base_cfg)

    seed_rows: list[dict[str, Any]] = []
    node_rows_all: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    graph_store: dict[tuple[str, str], list[np.ndarray]] = {}
    reference_predictions: dict[str, np.ndarray] = {}
    reference_graphs: dict[str, np.ndarray] = {}
    reference_seed: int | None = None

    selected_files = []
    for file in scenario_files:
        scenario_name = file.parent.name
        if scenario_limit and scenario_name not in scenario_limit:
            continue
        selected_files.append(file)
    if quick:
        # A smoke run needs one deterministic seed per scenario; full mode uses all seeds.
        first_by_scenario: dict[str, Path] = {}
        for file in selected_files:
            first_by_scenario.setdefault(file.parent.name, file)
        selected_files = list(first_by_scenario.values())
    if not selected_files:
        raise ValueError("No scenario message files matched the requested experiment set.")

    for file in selected_files:
        messages = read_messages(file)
        scenario_name = str(messages["scenario"].iloc[0])
        family = str(messages["family"].iloc[0])
        seed = int(messages["seed"].iloc[0])
        scenario = scenario_lookup.get(scenario_name)
        if scenario is None:
            max_delay = int(pd.to_numeric(messages["delay"], errors="coerce").max())
            scenario = {"name": scenario_name, "family": family, "max_delay": max_delay, "delay_pmf": [1 / (max_delay + 1)] * (max_delay + 1), "drop_probability": np.nan, "outlier_probability": np.nan}
        params = _scenario_parameter_row(scenario)
        delay_pmf = _delay_pmf_by_slice(scenario, config, inference=True)
        max_delay = int(scenario["max_delay"])

        for method_name, estimator in methods.items():
            result = estimator.fit(messages=messages, n_nodes=truth.shape[0], n_time=truth.shape[1], max_delay=max_delay, delay_pmf_by_slice=delay_pmf)
            metrics, node_rows = _result_metrics(result, truth, test_indices, node_names, messages, w_true, bundle.adjacency, graph_threshold)
            seed_row = {**params, "seed": seed, "method": method_name, "message_file": str(file), **metrics}
            seed_rows.append(seed_row)
            for node_row in node_rows:
                node_rows_all.append({**params, "seed": seed, "method": method_name, **node_row})
            for iteration, objective in enumerate(result.objective_history):
                history_rows.append({"scenario": scenario_name, "family": family, "seed": seed, "method": method_name, "iteration": iteration, "objective": objective})
            key = (scenario_name, method_name)
            graph_store.setdefault(key, []).append(result.w_hat)
            array_name = f"{scenario_name}__seed_{seed}__{method_name.replace(' ', '_').replace('/', '_')}.npz"
            np.savez_compressed(paths["arrays"] / array_name, x_hat=result.x_hat, w_hat=result.w_hat, predicted_delays=result.predicted_delays, objective=np.asarray(result.objective_history))

            if scenario_name == "reference_moderate":
                if reference_seed is None or seed < reference_seed:
                    reference_seed = seed
                    reference_predictions = {}
                    reference_graphs = {}
                if seed == reference_seed:
                    reference_predictions[method_name] = result.x_hat
                    reference_graphs[method_name] = result.w_hat

    seed_results = pd.DataFrame(seed_rows)
    node_results = pd.DataFrame(node_rows_all)
    histories = pd.DataFrame(history_rows)
    seed_results.to_csv(paths["seed_results"] / "seed_level_metrics.csv", index=False)
    node_results.to_csv(paths["seed_results"] / "node_level_metrics.csv", index=False)
    histories.to_csv(paths["logs"] / "objective_histories.csv", index=False)

    metric_columns = [col for col in seed_results.columns if col.endswith("_test") or col in ["runtime_seconds", "peak_memory_mb", "iterations", "graph_support_f1", "graph_normalized_frobenius", "delay_mae", "delay_true_posterior_mass"]]
    summary = summarize_seed_results(seed_results, metric_columns)
    save_table(summary, "results_summary_long", paths)

    reference = seed_results[seed_results["scenario"] == "reference_moderate"].copy()
    main_metrics = [col for col in ["rmse_test", "mae_test", "r2_test", "pearson_r_test", "graph_support_f1", "delay_mae", "runtime_seconds"] if col in reference]
    if main_metrics:
        wide_rows = []
        for method, group in reference.groupby("method"):
            row: dict[str, Any] = {"method": method}
            for metric in main_metrics:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            wide_rows.append(row)
        save_table(pd.DataFrame(wide_rows), "results_main_benchmark", paths)

    stability_rows = []
    for (scenario_name, method_name), graphs in graph_store.items():
        stability_rows.append({"scenario": scenario_name, "method": method_name, **graph_stability(graphs, threshold=graph_threshold)})
    stability = pd.DataFrame(stability_rows)
    if not stability.empty:
        save_table(stability, "results_graph_stability", paths)

    pairwise = pairwise_method_tests(seed_results, "rmse_test", lower_is_better=True)
    pairwise.to_csv(paths["stats"] / "pairwise_rmse_tests.csv", index=False)
    if not pairwise.empty:
        (paths["stats"] / "pairwise_rmse_tests.tex").write_text(pairwise.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4g}"), encoding="utf-8")
    ranks = average_ranks(seed_results, "rmse_test", lower_is_better=True)
    ranks.to_csv(paths["stats"] / "average_ranks_rmse.csv", index=False)
    friedman = friedman_test(seed_results, "rmse_test")
    write_json(friedman, paths["stats"] / "friedman_rmse.json")

    figure_files = generate_result_figures(seed_results, paths["figures"], histories=histories, ranks=ranks, pairwise=pairwise)
    if reference_predictions:
        trajectory_path = plot_trajectory_panel(truth, reference_predictions, node_names, paths["figures"] / "fig_results_10_reference_trajectories.png", indices=test_indices, max_nodes=min(4, truth.shape[0]))
        if trajectory_path:
            figure_files.append(str(trajectory_path))
    if reference_graphs:
        graph_path = plot_graph_heatmaps(w_true, reference_graphs, node_names, paths["figures"] / "fig_results_11_graph_recovery_heatmaps.png")
        if graph_path:
            figure_files.append(str(graph_path))

    manifest = {
        "experiment_complete": True,
        "quick_mode": quick,
        "n_scenario_files": len(selected_files),
        "n_methods": len(methods),
        "methods": list(methods),
        "n_seed_level_rows": len(seed_results),
        "figures": [str(Path(item).relative_to(paths["root"])) if Path(item).is_relative_to(paths["root"]) else str(item) for item in figure_files],
        "tables_csv": sorted(str(path.relative_to(paths["root"])) for path in paths["tables_csv"].glob("*.csv")),
        "statistical_tests": sorted(str(path.relative_to(paths["root"])) for path in paths["stats"].glob("*")),
        "warnings": ["Synthetic demo outputs are pipeline validation only and must not be reported as real-data findings."] if "synthetic" in str(config.get("dataset_name", "")).lower() else [],
    }
    write_json(manifest, paths["root"] / "experiment_manifest.json")
    return manifest
