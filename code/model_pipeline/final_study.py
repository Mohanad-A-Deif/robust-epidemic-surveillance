#!/usr/bin/env python3
"""Leakage-safe real-data and semi-real benchmark study.

This script deliberately selects all hyperparameters on train/validation data,
locks the test interval, and only then evaluates retrospective reconstruction
and causal rolling nowcasting.  Real RKI epidemic trajectories are separated
from artificially injected communication corruptions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epidemic_results.io_utils import load_processed_bundle, read_messages
from epidemic_results.metrics import (
    graph_stability,
    posterior_delay_metrics,
    reference_graph_agreement,
    state_metrics,
    uncertainty_metrics,
)
from epidemic_results.proposed_model import DelayAwareRobustGraphInference, InferenceConfig
from epidemic_results.statistics import (
    average_ranks,
    bootstrap_ci,
    friedman_test,
    pairwise_method_tests,
    summarize_seed_results,
)
from epidemic_results.study_protocol import (
    build_method,
    causal_rolling_nowcast,
    config_from_jsonable,
    config_to_jsonable,
    received_messages_available_by,
    split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rki-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["tune", "reference", "causal", "robustness", "all"], default="all")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_table(frame: pd.DataFrame, stem: Path, caption: str = "", label: str = "") -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    tabular = frame.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.5g}")
    latex = "\\begin{table}[!htbp]\n\\centering\n"
    if caption:
        latex += f"\\caption{{{caption}}}\n"
    if label:
        latex += f"\\label{{{label}}}\n"
    latex += "\\resizebox{\\linewidth}{!}{%\n" + tabular + "}\n\\end{table}\n"
    stem.with_suffix(".tex").write_text(latex, encoding="utf-8")


def scenario_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in config["scenario_generation"]["scenarios"]}


def delay_pmfs(scenario: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    base = np.asarray(scenario.get("inference_delay_pmf", scenario["delay_pmf"]), dtype=float)
    max_delay = int(scenario["max_delay"])
    if len(base) < max_delay + 1:
        base = np.pad(base, (0, max_delay + 1 - len(base)))
    base = np.maximum(base[: max_delay + 1], 0.0)
    base /= max(float(base.sum()), 1e-12)
    critical = base.copy()
    critical[0] += float(config["scenario_generation"].get("critical_zero_delay_bonus", 0.0))
    critical /= critical.sum()
    return {"critical": critical, "standard": base, "default": base}


def candidate_configs(adjacency: np.ndarray, quick: bool) -> list[tuple[str, InferenceConfig]]:
    common = dict(
        geographic_reference=adjacency,
        candidate_mask=None,
        initialization="delay_backprojection",
        warmup_outer_iter=2,
        graph_stage_outer_iter=1,
        max_outer_iter=5 if quick else 8,
        max_x_iter=2 if quick else 3,
        max_w_iter=1 if quick else 2,
        tol_objective=2e-4,
        tol_x=5e-5,
        tol_w=5e-5,
    )
    specs = [
        ("C01_default", {}),
        ("C02_temp075", {"posterior_temperature": 0.75}),
        ("C03_temp125", {"posterior_temperature": 1.25}),
        ("C04_transition002", {"delay_transition_strength": 0.02}),
        ("C05_transition008", {"delay_transition_strength": 0.08}),
        ("C06_sigma025", {"sigma_w": 0.25}),
        ("C07_sigma050", {"sigma_w": 0.50}),
        ("C08_huber100", {"huber_kappa": 1.0}),
        ("C09_huber200", {"huber_kappa": 2.0}),
        ("C10_low_regularization", {"lambda_g": 0.005, "lambda_w": 0.005, "lambda_row": 0.003, "geo_prior_weight": 0.003}),
        ("C11_high_regularization", {"lambda_g": 0.03, "lambda_w": 0.03, "lambda_row": 0.02, "geo_prior_weight": 0.02}),
        ("C12_slow_dynamics", {"beta": 0.08, "gamma": 0.05}),
        ("C13_fast_dynamics", {"beta": 0.16, "gamma": 0.10}),
        ("C14_combined_tempered", {"posterior_temperature": 1.25, "delay_transition_strength": 0.02, "huber_kappa": 1.0}),
    ]
    if quick:
        specs = specs[:6]
    return [(name, InferenceConfig(**common, **updates)) for name, updates in specs]


def _fit_tuning_task(task: dict[str, Any]) -> dict[str, Any]:
    messages = read_messages(task["message_file"])
    available = received_messages_available_by(messages, task["validation_end"])
    cfg = config_from_jsonable(task["config"])
    estimator = DelayAwareRobustGraphInference(cfg)
    result = estimator.fit(
        available,
        n_nodes=task["truth"].shape[0],
        n_time=task["validation_end"] + 1,
        max_delay=task["max_delay"],
        delay_pmf_by_slice={k: np.asarray(v, dtype=float) for k, v in task["delay_pmfs"].items()},
    )
    truth = np.asarray(task["truth"], dtype=float)
    val = np.asarray(task["validation_indices"], dtype=int)
    metrics = state_metrics(truth[:, val], result.x_hat[:, val])
    return {
        "candidate": task["candidate"],
        "seed": task["seed"],
        **metrics,
        "runtime_seconds": result.runtime_seconds,
        "iterations": result.iterations,
        "final_normalized_objective": result.diagnostics["final_normalized_objective"],
        "objective_nonincreasing": result.diagnostics["objective_nonincreasing"],
        "w_hat": result.w_hat,
    }


def tune(rki_root: Path, output: Path, workers: int, quick: bool) -> InferenceConfig:
    processed = rki_root / "data" / "processed"
    bundle = load_processed_bundle(processed, rki_root / "data" / "metadata")
    config = json.loads((rki_root / "config.json").read_text(encoding="utf-8"))
    splits = split_indices(bundle.splits, bundle.x_log1p.index)
    val_end = int(splits["validation"].max())
    truth = bundle.x_log1p.to_numpy(dtype=float).T[:, : val_end + 1]
    scenario = scenario_lookup(config)["reference_moderate"]
    pmfs = delay_pmfs(scenario, config)
    files = sorted((rki_root / "data" / "scenarios" / "main" / "reference_moderate").glob("messages_seed_*.csv.gz"))
    tuning_files = files[: (3 if quick else 5)]
    candidates = candidate_configs(np.asarray(bundle.adjacency, dtype=float), quick)
    tasks = []
    for name, cfg in candidates:
        for file in tuning_files:
            seed = int(file.stem.split("_")[-1].split(".")[0])
            tasks.append(
                {
                    "candidate": name,
                    "config": config_to_jsonable(cfg),
                    "message_file": str(file),
                    "seed": seed,
                    "truth": truth,
                    "validation_indices": splits["validation"],
                    "validation_end": val_end,
                    "max_delay": int(scenario["max_delay"]),
                    "delay_pmfs": {k: v.tolist() for k, v in pmfs.items()},
                }
            )
    rows: list[dict[str, Any]] = []
    graph_by_candidate: dict[str, list[np.ndarray]] = {}
    if workers == 1:
        for index, task in enumerate(tasks, 1):
            result = _fit_tuning_task(task)
            graph_by_candidate.setdefault(result["candidate"], []).append(result.pop("w_hat"))
            rows.append(result)
            if index % max(1, len(tasks) // 10) == 0 or index == len(tasks):
                print(f"[tuning] {index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(_fit_tuning_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                graph_by_candidate.setdefault(result["candidate"], []).append(result.pop("w_hat"))
                rows.append(result)
    detail = pd.DataFrame(rows).sort_values(["candidate", "seed"])
    save_table(detail, output / "tuning" / "validation_candidate_seed_metrics", "Validation-only hyperparameter results; the test interval remained locked.", "tab:validation_tuning_seed")
    summary = detail.groupby("candidate", as_index=False).agg(
        validation_rmse_mean=("rmse", "mean"),
        validation_rmse_sd=("rmse", "std"),
        validation_mae_mean=("mae", "mean"),
        runtime_mean=("runtime_seconds", "mean"),
        objective_nonincreasing_rate=("objective_nonincreasing", "mean"),
    ).sort_values(["validation_rmse_mean", "runtime_mean"])
    save_table(summary, output / "tuning" / "validation_candidate_summary", "Hyperparameter selection using train/validation data only.", "tab:validation_tuning")
    selected_name = str(summary.iloc[0]["candidate"])
    selected = dict(candidates)[selected_name]

    # Graph threshold is selected from validation-seed stability only, without
    # using test outcomes or treating geography as transmission ground truth.
    graphs = graph_by_candidate[selected_name]
    thresholds = np.unique(np.r_[1e-5, np.logspace(-4, -1, 16)])
    threshold_rows = []
    n = graphs[0].shape[0]
    mask = ~np.eye(n, dtype=bool)
    for threshold in thresholds:
        stability = graph_stability(graphs, threshold=float(threshold))
        density = float(np.mean([np.mean(graph[mask] > threshold) for graph in graphs]))
        eligible = 0.05 <= density <= 0.50
        score = stability["graph_stability_jaccard"] if eligible else -np.inf
        threshold_rows.append({"threshold": threshold, "mean_density": density, **stability, "eligible": eligible, "selection_score": score})
    threshold_frame = pd.DataFrame(threshold_rows)
    eligible = threshold_frame[np.isfinite(threshold_frame["selection_score"])]
    if eligible.empty:
        selected_threshold = float(np.median([np.quantile(g[mask][g[mask] > 0], 0.5) for g in graphs if np.any(g[mask] > 0)]))
    else:
        selected_threshold = float(eligible.sort_values(["selection_score", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
    save_table(threshold_frame, output / "tuning" / "validation_graph_threshold", "Graph support threshold selected from validation-seed stability, not test accuracy.", "tab:graph_threshold")

    final = replace(
        selected,
        max_outer_iter=8 if quick else 12,
        max_x_iter=3 if quick else 4,
        max_w_iter=2 if quick else 3,
        tol_objective=1e-4,
    )
    protocol = {
        "selected_candidate": selected_name,
        "selected_config": config_to_jsonable(final),
        "selected_graph_threshold": selected_threshold,
        "selection_metric": "mean validation RMSE across paired corruption seeds",
        "tuning_seeds": [int(path.stem.split("_")[-1].split(".")[0]) for path in tuning_files],
        "train_indices": [int(v) for v in splits["train"]],
        "validation_indices": [int(v) for v in splits["validation"]],
        "test_indices_locked_during_tuning": [int(v) for v in splits["test"]],
        "validation_message_cutoff": val_end,
        "test_accessed_during_selection": False,
        "causal_rolling_stopping_protocol": {
            "first_test_day_outer_iterations": 8 if quick else 10,
            "subsequent_day_outer_iterations": 2 if quick else 3,
            "warm_start": True,
        },
    }
    write_json(protocol, output / "protocol" / "locked_test_protocol.json")
    return final


def load_selected(output: Path) -> tuple[InferenceConfig, float, dict[str, Any]]:
    protocol = json.loads((output / "protocol" / "locked_test_protocol.json").read_text(encoding="utf-8"))
    return config_from_jsonable(protocol["selected_config"]), float(protocol["selected_graph_threshold"]), protocol


def method_names_reference() -> list[str]:
    return [
        "Proposed revised",
        "Original proposed ablation",
        "Known-delay proposed oracle",
        "No-delay ablation",
        "No-robustness ablation",
        "No-graph ablation",
        "Fixed geographic graph",
        "Arrival interpolation",
        "Oracle timestamp interpolation",
        "Delay backprojection",
        "Kalman/RTS smoother (internal)",
        "Delay-aware state-space (internal)",
        "Robust median smoother",
        "Robust low-rank completion",
        "Graph-temporal reconstruction",
    ]


def _fit_retro_task(task: dict[str, Any]) -> dict[str, Any]:
    messages = read_messages(task["message_file"])
    config = config_from_jsonable(task["selected_config"])
    adjacency = np.asarray(task["adjacency"], dtype=float)
    estimator = build_method(task["method"], config, adjacency, quick=task.get("quick", False))
    result = estimator.fit(
        messages=messages,
        n_nodes=task["truth"].shape[0],
        n_time=task["truth"].shape[1],
        max_delay=task["max_delay"],
        delay_pmf_by_slice={k: np.asarray(v, dtype=float) for k, v in task["delay_pmfs"].items()},
    )
    truth = np.asarray(task["truth"], dtype=float)
    test = np.asarray(task["test_indices"], dtype=int)
    metrics = state_metrics(truth[:, test], result.x_hat[:, test])
    row: dict[str, Any] = {
        "scenario": task["scenario"],
        "family": task["family"],
        "seed": task["seed"],
        "method": task["method"],
        "evaluation": "retrospective reconstruction",
        **metrics,
        "runtime_seconds": result.runtime_seconds,
        "peak_memory_mb": result.peak_memory_mb,
        "iterations": result.iterations,
        "converged": result.converged,
        "learned_density": result.diagnostics.get("graph_density", float(np.mean(result.w_hat > task["graph_threshold"]))),
        "maximum_row_sum": float(result.w_hat.sum(axis=1).max()) if result.w_hat.size else 0.0,
        "objective_nonincreasing": result.diagnostics.get("objective_nonincreasing", np.nan),
    }
    if task["method"].startswith("Proposed") or "ablation" in task["method"] or "graph" in task["method"].lower() or "oracle" in task["method"].lower():
        row.update(reference_graph_agreement(result.w_hat, adjacency, threshold=task["graph_threshold"]))
    if result.posterior:
        received = messages[messages["received"].astype(bool)] if messages["received"].dtype == bool else messages[messages["received"].astype(str).str.lower().isin(["true", "1"])]
        true_delays = pd.to_numeric(received["delay"], errors="coerce").fillna(0).to_numpy(dtype=int)
        if len(true_delays) == len(result.posterior):
            row.update(posterior_delay_metrics(true_delays, result.posterior, result.feasible_delays))
    return {
        "row": row,
        "x_hat": result.x_hat if task["method"] == "Proposed revised" else None,
        "w_hat": result.w_hat if task["method"] == "Proposed revised" else None,
        "objective": np.asarray(result.normalized_objective_history, dtype=float) if task["method"] == "Proposed revised" else None,
    }


def run_parallel(tasks: list[dict[str, Any]], worker: Any, workers: int, progress_label: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(tasks)
    completed = 0
    started = time.perf_counter()
    if workers == 1:
        for task in tasks:
            results.append(worker(task))
            completed += 1
            if completed % max(1, total // 10) == 0 or completed == total:
                elapsed = time.perf_counter() - started
                print(f"[{progress_label}] {completed}/{total} completed in {elapsed:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            future_map = {executor.submit(worker, task): task for task in tasks}
            for future in as_completed(future_map):
                results.append(future.result())
                completed += 1
                if completed % max(1, total // 10) == 0 or completed == total:
                    elapsed = time.perf_counter() - started
                    print(f"[{progress_label}] {completed}/{total} completed in {elapsed:.1f}s", flush=True)
    return results


def summarize_and_stats(rows: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    metrics = [name for name in ["rmse", "mae", "r2", "pearson_r", "runtime_seconds", "peak_memory_mb", "delay_mae"] if name in rows]
    summary = summarize_seed_results(rows, metrics, group_cols=["scenario", "family", "method", "evaluation"])
    save_table(summary, output_dir / f"{prefix}_summary_long", "Seed-level summary with bootstrap 95% confidence intervals.", f"tab:{prefix}_summary")
    tests = []
    ranks = []
    friedman_rows = []
    for scenario, group in rows.groupby("scenario"):
        if group["seed"].nunique() < 2 or group["method"].nunique() < 2:
            continue
        pair = pairwise_method_tests(group, "rmse", lower_is_better=True, group_cols=("seed",))
        if not pair.empty:
            pair.insert(0, "scenario", scenario)
            tests.append(pair)
        rank = average_ranks(group, "rmse", lower_is_better=True, index_cols=("seed",))
        rank.insert(0, "scenario", scenario)
        ranks.append(rank)
        friedman_rows.append({"scenario": scenario, **friedman_test(group, "rmse", index_cols=("seed",))})
    if tests:
        save_table(pd.concat(tests, ignore_index=True), output_dir / f"{prefix}_paired_tests", "Paired tests within each scenario; Holm correction is applied within scenario.", f"tab:{prefix}_tests")
    if ranks:
        save_table(pd.concat(ranks, ignore_index=True), output_dir / f"{prefix}_average_ranks", "Average ranks computed with independent seeds as blocks within each scenario.", f"tab:{prefix}_ranks")
    if friedman_rows:
        save_table(pd.DataFrame(friedman_rows), output_dir / f"{prefix}_friedman", "Friedman tests using seeds as independent blocks within each scenario.", f"tab:{prefix}_friedman")


def run_reference(rki_root: Path, output: Path, workers: int, quick: bool) -> None:
    bundle = load_processed_bundle(rki_root / "data" / "processed", rki_root / "data" / "metadata")
    config_json = json.loads((rki_root / "config.json").read_text(encoding="utf-8"))
    selected, threshold, _ = load_selected(output)
    splits = split_indices(bundle.splits, bundle.x_log1p.index)
    truth = bundle.x_log1p.to_numpy(dtype=float).T
    scenario = scenario_lookup(config_json)["reference_moderate"]
    pmfs = delay_pmfs(scenario, config_json)
    files = sorted((rki_root / "data" / "scenarios" / "main" / "reference_moderate").glob("messages_seed_*.csv.gz"))
    if quick:
        files = files[:3]
    tasks = []
    for file in files:
        seed = int(file.stem.split("_")[-1].split(".")[0])
        for method in method_names_reference():
            tasks.append({
                "message_file": str(file), "scenario": "reference_moderate", "family": "main", "seed": seed,
                "method": method, "selected_config": config_to_jsonable(selected), "adjacency": np.asarray(bundle.adjacency).tolist(),
                "truth": truth, "test_indices": splits["test"], "max_delay": int(scenario["max_delay"]),
                "delay_pmfs": {k: v.tolist() for k, v in pmfs.items()}, "graph_threshold": threshold, "quick": quick,
            })
    results = run_parallel(tasks, _fit_retro_task, workers, "reference")
    rows = pd.DataFrame([item["row"] for item in results]).sort_values(["method", "seed"])
    save_table(rows, output / "reference" / "retrospective_reconstruction_seed_metrics", "Retrospective reconstruction on the locked RKI test interval.", "tab:retro_seed")
    summarize_and_stats(rows, output / "reference", "retrospective_reconstruction")

    proposed = [(item["row"]["seed"], item["x_hat"], item["w_hat"], item["objective"]) for item in results if item["x_hat"] is not None]
    proposed.sort(key=lambda z: z[0])
    arrays_dir = output / "reference" / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for seed, x_hat, w_hat, objective in proposed:
        np.savez_compressed(arrays_dir / f"proposed_seed_{seed}.npz", x_hat=x_hat, w_hat=w_hat, normalized_objective=objective)
    if proposed:
        x_stack = np.stack([item[1] for item in proposed])
        w_stack = np.stack([item[2] for item in proposed])
        lower, median, upper = np.quantile(x_stack, [0.025, 0.5, 0.975], axis=0)
        interval = uncertainty_metrics(truth[:, splits["test"]], lower[:, splits["test"]], upper[:, splits["test"]])
        write_json({"interpretation": "Empirical uncertainty across fixed corruption seeds, not a Bayesian posterior interval.", **interval}, output / "reference" / "state_uncertainty_metrics.json")
        node_names = bundle.x_log1p.columns.tolist()
        edge_rows = []
        for i, source in enumerate(node_names):
            for j, target in enumerate(node_names):
                if i == j:
                    continue
                values = w_stack[:, i, j]
                lo, hi = bootstrap_ci(values, n_boot=3000)
                edge_rows.append({
                    "source": source, "target": target, "mean_weight": float(values.mean()), "sd_weight": float(values.std(ddof=1)),
                    "selection_frequency": float(np.mean(values > threshold)), "bootstrap95_low": lo, "bootstrap95_high": hi,
                })
        save_table(pd.DataFrame(edge_rows), output / "reference" / "edge_uncertainty_and_selection_frequency", "Learned directed edge uncertainty and selection frequency across seeds.", "tab:edge_uncertainty")
        stability = graph_stability([item[2] for item in proposed], threshold=threshold)
        write_json(stability, output / "reference" / "graph_stability.json")


def robustness_methods() -> list[str]:
    return [
        "Proposed revised",
        "Arrival interpolation",
        "Delay backprojection",
        "Kalman/RTS smoother (internal)",
        "Delay-aware state-space (internal)",
        "Robust median smoother",
    ]


def run_robustness(rki_root: Path, output: Path, workers: int, quick: bool) -> None:
    bundle = load_processed_bundle(rki_root / "data" / "processed", rki_root / "data" / "metadata")
    config_json = json.loads((rki_root / "config.json").read_text(encoding="utf-8"))
    lookup = scenario_lookup(config_json)
    selected, threshold, _ = load_selected(output)
    splits = split_indices(bundle.splits, bundle.x_log1p.index)
    truth = bundle.x_log1p.to_numpy(dtype=float).T
    all_files = sorted((rki_root / "data" / "scenarios").glob("**/messages_seed_*.csv.gz"))
    if quick:
        all_files = [f for f in all_files if int(f.stem.split("_")[-1].split(".")[0]) in [1101, 1102]]
    tasks = []
    for file in all_files:
        messages_head = pd.read_csv(file, nrows=1)
        scenario_name = str(messages_head.loc[0, "scenario"])
        family = str(messages_head.loc[0, "family"])
        seed = int(messages_head.loc[0, "seed"])
        scenario = lookup[scenario_name]
        pmfs = delay_pmfs(scenario, config_json)
        for method in robustness_methods():
            tasks.append({
                "message_file": str(file), "scenario": scenario_name, "family": family, "seed": seed,
                "method": method, "selected_config": config_to_jsonable(selected), "adjacency": np.asarray(bundle.adjacency).tolist(),
                "truth": truth, "test_indices": splits["test"], "max_delay": int(scenario["max_delay"]),
                "delay_pmfs": {k: v.tolist() for k, v in pmfs.items()}, "graph_threshold": threshold, "quick": quick,
            })
    results = run_parallel(tasks, _fit_retro_task, workers, "robustness")
    rows = pd.DataFrame([item["row"] for item in results]).sort_values(["family", "scenario", "method", "seed"])
    save_table(rows, output / "robustness" / "robustness_seed_metrics", "Robustness and sensitivity experiments on nested corruption templates.", "tab:robustness_seed")
    summarize_and_stats(rows, output / "robustness", "robustness")
    # Compact sensitivity table for the proposed method.
    proposed = rows[rows["method"] == "Proposed revised"]
    sensitivity = proposed.groupby(["family", "scenario"], as_index=False).agg(
        n_seeds=("seed", "nunique"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), runtime_mean=("runtime_seconds", "mean"), learned_density_mean=("learned_density", "mean"),
    )
    save_table(sensitivity, output / "robustness" / "sensitivity_table", "One-factor-at-a-time sensitivity using nested seed-level corruption templates.", "tab:sensitivity")


def _causal_task(task: dict[str, Any]) -> dict[str, Any]:
    messages = read_messages(task["message_file"])
    config = config_from_jsonable(task["selected_config"])
    adjacency = np.asarray(task["adjacency"], dtype=float)
    method = task["method"]

    def factory(position: int, initial_x: np.ndarray | None, initial_w: np.ndarray | None):
        if method == "Proposed revised":
            rolling = replace(
                config,
                max_outer_iter=(task["first_outer"] if position == 0 else task["subsequent_outer"]),
                max_x_iter=(3 if position == 0 else 2),
                max_w_iter=(2 if position == 0 else 1),
                warmup_outer_iter=(2 if position == 0 else 0),
                graph_stage_outer_iter=(1 if position == 0 else 0),
            )
            return DelayAwareRobustGraphInference(rolling)
        return build_method(method, config, adjacency, quick=task.get("quick", False))

    estimates, results = causal_rolling_nowcast(
        factory,
        messages,
        n_nodes=task["truth"].shape[0],
        test_indices=np.asarray(task["test_indices"], dtype=int),
        max_delay=task["max_delay"],
        delay_pmf_by_slice={k: np.asarray(v, dtype=float) for k, v in task["delay_pmfs"].items()},
    )
    truth = np.asarray(task["truth"], dtype=float)
    test = np.asarray(task["test_indices"], dtype=int)
    metrics = state_metrics(truth[:, test], estimates)
    return {
        "row": {
            "scenario": "reference_moderate", "family": "main", "seed": task["seed"], "method": method,
            "evaluation": "causal rolling nowcasting", **metrics,
            "runtime_seconds": float(sum(result.runtime_seconds for result in results)),
            "peak_memory_mb": float(max((result.peak_memory_mb for result in results), default=0.0)),
            "n_nowcast_days": len(test),
        },
        "estimates": estimates if method == "Proposed revised" else None,
    }


def run_causal(rki_root: Path, output: Path, workers: int, quick: bool) -> None:
    bundle = load_processed_bundle(rki_root / "data" / "processed", rki_root / "data" / "metadata")
    config_json = json.loads((rki_root / "config.json").read_text(encoding="utf-8"))
    scenario = scenario_lookup(config_json)["reference_moderate"]
    pmfs = delay_pmfs(scenario, config_json)
    selected, _, protocol = load_selected(output)
    splits = split_indices(bundle.splits, bundle.x_log1p.index)
    truth = bundle.x_log1p.to_numpy(dtype=float).T
    files = sorted((rki_root / "data" / "scenarios" / "main" / "reference_moderate").glob("messages_seed_*.csv.gz"))
    if quick:
        files = files[:2]
    methods = [
        "Proposed revised",
        "Arrival interpolation",
        "Oracle timestamp interpolation",
        "Delay backprojection",
        "Kalman/RTS smoother (internal)",
        "Delay-aware state-space (internal)",
        "Robust median smoother",
    ]
    stopping = protocol["causal_rolling_stopping_protocol"]
    tasks = []
    for file in files:
        seed = int(file.stem.split("_")[-1].split(".")[0])
        for method in methods:
            tasks.append({
                "message_file": str(file), "seed": seed, "method": method, "selected_config": config_to_jsonable(selected),
                "adjacency": np.asarray(bundle.adjacency).tolist(), "truth": truth, "test_indices": splits["test"],
                "max_delay": int(scenario["max_delay"]), "delay_pmfs": {k: v.tolist() for k, v in pmfs.items()},
                "first_outer": stopping["first_test_day_outer_iterations"], "subsequent_outer": stopping["subsequent_day_outer_iterations"],
                "quick": quick,
            })
    results = run_parallel(tasks, _causal_task, workers, "causal")
    rows = pd.DataFrame([item["row"] for item in results]).sort_values(["method", "seed"])
    save_table(rows, output / "causal" / "causal_nowcasting_seed_metrics", "Causal rolling nowcasting: day t uses only messages received by day t.", "tab:causal_seed")
    summarize_and_stats(rows, output / "causal", "causal_nowcasting")
    arrays = output / "causal" / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    for item in results:
        if item["estimates"] is not None:
            np.save(arrays / f"proposed_causal_seed_{item['row']['seed']}.npy", item["estimates"])


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.phase in ("tune", "all"):
        selected = tune(args.rki_root, args.output, args.workers, args.quick)
        print("Selected configuration:", config_to_jsonable(selected), flush=True)
    if args.phase in ("reference", "all"):
        run_reference(args.rki_root, args.output, args.workers, args.quick)
    if args.phase in ("robustness", "all"):
        run_robustness(args.rki_root, args.output, args.workers, args.quick)
    if args.phase in ("causal", "all"):
        run_causal(args.rki_root, args.output, args.workers, args.quick)
    write_json({"phase": args.phase, "quick": args.quick, "elapsed_seconds": time.perf_counter() - start}, args.output / f"run_{args.phase}_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
