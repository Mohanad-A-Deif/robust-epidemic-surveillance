"""Deterministic synthetic validation data used only to test the full pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import write_json


def _stable_graph() -> np.ndarray:
    w = np.array(
        [
            [0.0, 0.25, 0.0, 0.0, 0.10, 0.0],
            [0.15, 0.0, 0.20, 0.0, 0.0, 0.0],
            [0.0, 0.12, 0.0, 0.18, 0.0, 0.08],
            [0.0, 0.0, 0.14, 0.0, 0.20, 0.0],
            [0.08, 0.0, 0.0, 0.16, 0.0, 0.12],
            [0.0, 0.0, 0.10, 0.0, 0.15, 0.0],
        ],
        dtype=float,
    )
    return w


def generate_latent(n_time: int = 100, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = _stable_graph()
    n = w.shape[0]
    beta, gamma = 0.12, 0.08
    degree = np.diag(w.sum(axis=1))
    a = (1 - gamma) * np.eye(n) - beta * (degree - w)
    x = np.zeros((n, n_time), dtype=float)
    x[:, 0] = np.array([0.5, 0.7, 0.4, 0.55, 0.65, 0.45])
    for t in range(n_time - 1):
        seasonal = 0.04 * (1 + np.sin(2 * np.pi * t / 14 + np.arange(n)))
        intervention = -0.06 if 55 <= t < 70 else 0.0
        innovation = rng.normal(0, 0.025, size=n)
        x[:, t + 1] = np.maximum(a @ x[:, t] + seasonal + intervention + innovation, 0.01)
    # Add local outbreak pulses while respecting short smoke-test horizons.
    def add_segment(node: int, start: int, values: np.ndarray) -> None:
        if start >= n_time:
            return
        end = min(n_time, start + len(values))
        x[node, start:end] += values[: end - start]

    add_segment(1, 28, np.linspace(0.0, 0.7, 10))
    add_segment(1, 38, np.linspace(0.7, 0.0, 9))
    add_segment(4, 72, np.linspace(0.0, 0.55, 8))
    add_segment(4, 80, np.linspace(0.55, 0.0, 8))
    return x, w


def default_demo_scenarios() -> list[dict]:
    moderate = [0.30, 0.25, 0.18, 0.12, 0.08, 0.05, 0.02]
    return [
        {"name": "reference_moderate", "family": "main", "max_delay": 6, "delay_pmf": moderate, "drop_probability": 0.10, "outlier_probability": 0.05, "prior_mismatch_level": 0.0},
        {"name": "missing_00", "family": "missingness", "max_delay": 6, "delay_pmf": moderate, "drop_probability": 0.00, "outlier_probability": 0.05, "prior_mismatch_level": 0.0},
        {"name": "missing_30", "family": "missingness", "max_delay": 6, "delay_pmf": moderate, "drop_probability": 0.30, "outlier_probability": 0.05, "prior_mismatch_level": 0.0},
        {"name": "missing_50", "family": "missingness", "max_delay": 6, "delay_pmf": moderate, "drop_probability": 0.50, "outlier_probability": 0.05, "prior_mismatch_level": 0.0},
        {"name": "delay_light", "family": "delay", "max_delay": 4, "delay_pmf": [0.55, 0.25, 0.12, 0.06, 0.02], "drop_probability": 0.10, "outlier_probability": 0.05, "prior_mismatch_level": 0.0},
        {"name": "delay_severe", "family": "delay", "max_delay": 8, "delay_pmf": [0.14, 0.14, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08, 0.06], "drop_probability": 0.10, "outlier_probability": 0.05, "prior_mismatch_level": 0.0},
        {"name": "outlier_00", "family": "outlier", "max_delay": 6, "delay_pmf": moderate, "drop_probability": 0.10, "outlier_probability": 0.00, "prior_mismatch_level": 0.0},
        {"name": "outlier_30", "family": "outlier", "max_delay": 6, "delay_pmf": moderate, "drop_probability": 0.10, "outlier_probability": 0.30, "prior_mismatch_level": 0.0},
        {"name": "prior_mismatch_40", "family": "prior_mismatch", "max_delay": 6, "delay_pmf": moderate, "inference_delay_pmf": [0.10, 0.12, 0.14, 0.17, 0.17, 0.16, 0.14], "drop_probability": 0.10, "outlier_probability": 0.05, "prior_mismatch_level": 0.40},
    ]


def _messages_for_scenario(x: np.ndarray, nodes: pd.DataFrame, scenario: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, t_count = x.shape
    pmf = np.asarray(scenario["delay_pmf"], dtype=float)
    pmf /= pmf.sum()
    outlier_scale = 5.0 * float(np.std(x))
    rows = []
    counter = 0
    for node in range(n):
        slice_id = nodes.loc[node, "slice_id"]
        node_pmf = pmf.copy()
        if slice_id == "critical":
            node_pmf[0] += 0.1
            node_pmf /= node_pmf.sum()
        drop = scenario["drop_probability"] * (0.75 if slice_id == "critical" else 1.0)
        for generation in range(t_count):
            counter += 1
            delay = int(rng.choice(np.arange(len(node_pmf)), p=node_pmf))
            dropped = bool(rng.random() < drop)
            arrival = generation + delay
            right_censored = bool((not dropped) and arrival >= t_count)
            received = not dropped and not right_censored
            is_outlier = bool(received and rng.random() < scenario["outlier_probability"])
            addition = float(rng.standard_t(df=3) * outlier_scale) if is_outlier else 0.0
            rows.append(
                {
                    "message_id": f"{scenario['name']}_{seed}_{counter}",
                    "scenario": scenario["name"],
                    "family": scenario["family"],
                    "seed": seed,
                    "node_id": node,
                    "node_code": nodes.loc[node, "code"],
                    "state": nodes.loc[node, "state"],
                    "slice_id": slice_id,
                    "generation_index": generation,
                    "generation_date": "",
                    "clean_value": float(x[node, generation]),
                    "delay": delay,
                    "arrival_index": arrival if received else "",
                    "arrival_date": "",
                    "dropped": dropped,
                    "right_censored": right_censored,
                    "received": received,
                    "is_outlier": is_outlier,
                    "outlier_addition": addition if received else "",
                    "observed_value": float(x[node, generation] + addition) if received else "",
                    "true_delay_probability": float(node_pmf[delay]),
                }
            )
    return pd.DataFrame(rows)


def create_demo_dataset(root: str | Path, n_time: int = 100, seeds: list[int] | None = None) -> dict:
    root = Path(root)
    processed = root / "processed"
    scenarios_root = root / "scenarios"
    processed.mkdir(parents=True, exist_ok=True)
    scenarios_root.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [1101, 1102, 1103]
    x, w = generate_latent(n_time=n_time)
    names = ["Region A", "Region B", "Region C", "Region D", "Region E", "Region F"]
    codes = ["A", "B", "C", "D", "E", "F"]
    dates = pd.date_range("2020-02-20", periods=n_time, freq="D")
    x_log = pd.DataFrame(x.T, index=dates, columns=names)
    x_raw = pd.DataFrame(np.expm1(x).T, index=dates, columns=names)
    train_end, val_end = int(0.6 * n_time), int(0.8 * n_time)
    means = x_log.iloc[:train_end].mean()
    stds = x_log.iloc[:train_end].std(ddof=0).replace(0, 1.0)
    x_std = (x_log - means) / stds
    splits = pd.Series(["train"] * train_end + ["validation"] * (val_end - train_end) + ["test"] * (n_time - val_end), index=dates, name="split")
    nodes = pd.DataFrame({"node_id": range(6), "code": codes, "state": names, "slice_id": ["critical"] * 3 + ["standard"] * 3})
    adjacency = ((w + w.T) > 0).astype(float)
    np.fill_diagonal(adjacency, 0.0)

    x_raw.to_csv(processed / "x_raw.csv")
    x_log.to_csv(processed / "x_log1p.csv")
    x_std.to_csv(processed / "x_standardized.csv")
    splits.to_frame().to_csv(processed / "splits.csv")
    nodes.to_csv(processed / "nodes.csv", index=False)
    pd.DataFrame(adjacency, index=names, columns=names).to_csv(processed / "adjacency_reference.csv")
    np.save(processed / "w_true.npy", w)
    np.save(processed / "x_true.npy", x)

    scenarios = default_demo_scenarios()
    summaries = []
    for scenario in scenarios:
        folder = scenarios_root / scenario["family"] / scenario["name"]
        folder.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            frame = _messages_for_scenario(x, nodes, scenario, seed)
            frame.to_csv(folder / f"messages_seed_{seed}.csv.gz", index=False, compression="gzip")
            received = frame[frame["received"]]
            summary = {
                "scenario": scenario["name"],
                "family": scenario["family"],
                "seed": seed,
                "n_generated": len(frame),
                "n_received": int(frame["received"].sum()),
                "realized_drop_fraction": float(frame["dropped"].mean()),
                "realized_outlier_fraction_among_received": float(received["is_outlier"].mean()),
                "mean_delay_received": float(received["delay"].mean()),
            }
            summaries.append(summary)
            write_json(summary, folder / f"summary_seed_{seed}.json")
    pd.DataFrame(summaries).to_csv(scenarios_root / "scenario_summary.csv", index=False)
    config = {
        "dataset_name": "Synthetic validation fixture for pipeline testing only",
        "selected_states": names,
        "scenario_generation": {"seeds": seeds, "critical_zero_delay_bonus": 0.1, "critical_drop_multiplier": 0.75, "scenarios": scenarios},
        "nodes": nodes.to_dict(orient="records"),
    }
    write_json(config, root / "config.json")
    write_json({"synthetic_demo": True, "n_nodes": 6, "n_time": n_time, "seeds": seeds}, root / "metadata.json")
    return {"root": str(root), "processed": str(processed), "scenarios": str(scenarios_root), "config": str(root / "config.json")}
