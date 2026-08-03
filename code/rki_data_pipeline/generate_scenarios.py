#!/usr/bin/env python3
"""Generate nested, reproducible message-level corruption scenarios.

For each seed, independent random templates are created for delays, drops,
outlier selection, and outlier magnitude.  Every scenario reuses the same
seed-level templates, so changing one sweep factor does not redraw the others.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--matrix-file", default="x_log1p.csv", help="Nonnegative latent-state matrix used to generate messages.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/scenarios"))
    parser.add_argument("--scenario", action="append", default=None, help="Generate only named scenario(s).")
    parser.add_argument("--seed", type=int, action="append", default=None, help="Override configured seeds.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pmf(values: list[float], expected_length: int, field: str) -> np.ndarray:
    pmf = np.asarray(values, dtype=float)
    if len(pmf) != expected_length:
        raise ValueError(f"{field} length must be {expected_length}, observed {len(pmf)}")
    if np.any(pmf < 0) or not np.isfinite(pmf).all():
        raise ValueError(f"{field} must contain finite nonnegative values")
    total = float(pmf.sum())
    if total <= 0:
        raise ValueError(f"{field} must have positive mass")
    return pmf / total


def slice_adjusted_pmf(base: np.ndarray, slice_id: str, zero_delay_bonus: float) -> np.ndarray:
    adjusted = base.copy()
    if slice_id == "critical":
        adjusted[0] += float(zero_delay_bonus)
    return adjusted / adjusted.sum()


def inverse_cdf_draw(uniform: float, pmf: np.ndarray) -> int:
    return int(min(np.searchsorted(np.cumsum(pmf), uniform, side="right"), len(pmf) - 1))


def build_corruption_template(n_nodes: int, n_time: int, seed: int) -> dict[str, np.ndarray]:
    """Create independent random streams, fixed across all scenario levels."""
    seed_sequence = np.random.SeedSequence(int(seed))
    delay_ss, drop_ss, outlier_ss, magnitude_ss = seed_sequence.spawn(4)
    shape = (n_nodes, n_time)
    return {
        "delay_uniform": np.random.default_rng(delay_ss).random(shape),
        "drop_uniform": np.random.default_rng(drop_ss).random(shape),
        "outlier_uniform": np.random.default_rng(outlier_ss).random(shape),
        "outlier_t3": np.random.default_rng(magnitude_ss).standard_t(df=3, size=shape),
    }


def template_sha256(template: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(template):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(template[key]).tobytes())
    return digest.hexdigest()


def generate_one(
    matrix: pd.DataFrame,
    nodes: pd.DataFrame,
    scenario: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    template: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = pd.DatetimeIndex(matrix.index)
    n_time = len(dates)
    max_delay = int(scenario["max_delay"])
    generation_pmf = validate_pmf(scenario["delay_pmf"], max_delay + 1, "delay_pmf")
    inference_values = scenario.get("inference_delay_pmf", scenario["delay_pmf"])
    if len(inference_values) < max_delay + 1:
        inference_values = list(inference_values) + [0.0] * (max_delay + 1 - len(inference_values))
    inference_pmf = validate_pmf(inference_values[: max_delay + 1], max_delay + 1, "inference_delay_pmf")

    generation = config["scenario_generation"]
    zero_delay_bonus = float(generation.get("critical_zero_delay_bonus", 0.0))
    critical_drop_multiplier = float(generation.get("critical_drop_multiplier", 1.0))
    outlier_scale = float(generation.get("outlier_scale", 6.0))
    global_scale = float(np.nanstd(matrix.to_numpy(dtype=float), ddof=0))
    if not np.isfinite(global_scale) or global_scale <= 1e-12:
        global_scale = 1.0

    node_by_state = nodes.set_index("state")
    n_nodes = len(nodes)
    template = template or build_corruption_template(n_nodes, n_time, seed)
    expected_shape = (n_nodes, n_time)
    for key, values in template.items():
        if np.asarray(values).shape != expected_shape:
            raise ValueError(f"Template {key} has shape {np.asarray(values).shape}; expected {expected_shape}")

    rows: list[dict[str, Any]] = []
    message_counter = 0

    for state in matrix.columns:
        metadata = node_by_state.loc[state]
        node_id = int(metadata["node_id"])
        slice_id = str(metadata["slice_id"])
        pmf = slice_adjusted_pmf(generation_pmf, slice_id, zero_delay_bonus)
        inference_slice_pmf = slice_adjusted_pmf(inference_pmf, slice_id, zero_delay_bonus)
        drop_probability = float(scenario["drop_probability"])
        if slice_id == "critical":
            drop_probability *= critical_drop_multiplier
        drop_probability = min(max(drop_probability, 0.0), 1.0)

        clean_values = matrix[state].to_numpy(dtype=float)
        for generation_index, clean_value in enumerate(clean_values):
            message_counter += 1
            delay = inverse_cdf_draw(float(template["delay_uniform"][node_id, generation_index]), pmf)
            dropped = bool(template["drop_uniform"][node_id, generation_index] < drop_probability)
            arrival_index_raw = generation_index + delay
            right_censored = bool((not dropped) and arrival_index_raw >= n_time)
            received = bool((not dropped) and (not right_censored))
            is_outlier = bool(
                received
                and template["outlier_uniform"][node_id, generation_index]
                < float(scenario["outlier_probability"])
            )
            addition = 0.0
            observed_value: float | None = None
            if received:
                if is_outlier:
                    addition = float(
                        template["outlier_t3"][node_id, generation_index]
                        * outlier_scale
                        * global_scale
                    )
                observed_value = float(clean_value + addition)
            arrival_date = dates[arrival_index_raw].strftime("%Y-%m-%d") if received else ""
            rows.append(
                {
                    "message_id": f"{scenario['name']}_s{seed}_m{message_counter:07d}",
                    "template_id": f"seed_{seed}",
                    "scenario": scenario["name"],
                    "family": scenario["family"],
                    "seed": seed,
                    "node_id": node_id,
                    "node_code": str(metadata["code"]),
                    "state": state,
                    "slice_id": slice_id,
                    "generation_index": generation_index,
                    "generation_date": dates[generation_index].strftime("%Y-%m-%d"),
                    "clean_value": float(clean_value),
                    "delay": delay,
                    "arrival_index": arrival_index_raw if received else "",
                    "arrival_date": arrival_date,
                    "dropped": dropped,
                    "right_censored": right_censored,
                    "received": received,
                    "is_outlier": is_outlier,
                    "outlier_addition": addition if received else "",
                    "observed_value": observed_value if received else "",
                    "true_delay_probability": float(pmf[delay]),
                    "inference_delay_probability": float(inference_slice_pmf[delay]),
                    "delay_uniform_template": float(template["delay_uniform"][node_id, generation_index]),
                    "drop_uniform_template": float(template["drop_uniform"][node_id, generation_index]),
                    "outlier_uniform_template": float(template["outlier_uniform"][node_id, generation_index]),
                }
            )

    frame = pd.DataFrame(rows)
    received_frame = frame[frame["received"]]
    collision_count = int(received_frame.duplicated(["node_id", "arrival_index"], keep=False).sum())
    summary = {
        "scenario": scenario["name"],
        "family": scenario["family"],
        "seed": seed,
        "template_sha256": template_sha256(template),
        "n_generated": int(len(frame)),
        "n_received": int(frame["received"].sum()),
        "n_dropped": int(frame["dropped"].sum()),
        "n_right_censored": int(frame["right_censored"].sum()),
        "n_outliers_received": int(frame["is_outlier"].sum()),
        "realized_drop_fraction": float(frame["dropped"].mean()),
        "realized_received_fraction": float(frame["received"].mean()),
        "realized_outlier_fraction_among_received": float(frame.loc[frame["received"], "is_outlier"].mean())
        if int(frame["received"].sum())
        else 0.0,
        "mean_delay_all_generated": float(frame["delay"].mean()),
        "mean_delay_received": float(received_frame["delay"].mean()) if len(received_frame) else None,
        "rows_participating_in_same_node_same_arrival_collisions": collision_count,
        "generation_delay_pmf_base": generation_pmf.tolist(),
        "inference_delay_pmf_base": inference_pmf.tolist(),
        "nested_sweep_note": "Delay, drop, outlier-selection, and outlier-magnitude random templates are fixed per seed and reused across scenarios.",
    }
    return frame, summary


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    matrix_path = args.processed_dir / args.matrix_file
    nodes_path = args.processed_dir / "nodes.csv"
    if not matrix_path.exists() or not nodes_path.exists():
        raise SystemExit("Processed data are missing. Run prepare_rki_dataset.py first.")
    matrix = pd.read_csv(matrix_path, index_col=0, parse_dates=True)
    nodes = pd.read_csv(nodes_path)
    if matrix.columns.tolist() != config["selected_states"]:
        raise ValueError("Processed matrix columns do not match configured state order.")

    scenarios = config["scenario_generation"]["scenarios"]
    if args.scenario:
        requested = set(args.scenario)
        scenarios = [scenario for scenario in scenarios if scenario["name"] in requested]
        missing = requested.difference({scenario["name"] for scenario in scenarios})
        if missing:
            raise SystemExit(f"Unknown scenario name(s): {sorted(missing)}")
    seeds = args.seed or config["scenario_generation"]["seeds"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = args.output_dir / "corruption_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    template_manifest: list[dict[str, Any]] = []
    for seed in seeds:
        template = build_corruption_template(len(nodes), len(matrix), int(seed))
        template_path = template_dir / f"template_seed_{int(seed)}.npz"
        np.savez_compressed(template_path, **template)
        template_manifest.append(
            {
                "seed": int(seed),
                "file": str(template_path.relative_to(args.output_dir)),
                "sha256": template_sha256(template),
            }
        )
        for scenario in scenarios:
            family_dir = args.output_dir / scenario["family"] / scenario["name"]
            family_dir.mkdir(parents=True, exist_ok=True)
            frame, summary = generate_one(matrix, nodes, scenario, int(seed), config, template=template)
            frame.to_csv(family_dir / f"messages_seed_{int(seed)}.csv.gz", index=False, compression="gzip")
            (family_dir / f"summary_seed_{int(seed)}.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summaries.append(summary)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "scenario_summary.csv", index=False)
    (args.output_dir / "scenario_manifest.json").write_text(
        json.dumps(
            {
                "matrix_source": str(matrix_path),
                "scenario_count": len(scenarios),
                "seeds": [int(seed) for seed in seeds],
                "files_are_message_level": True,
                "multiple_same_time_arrivals_preserved": True,
                "corruption_templates": template_manifest,
                "nested_sweeps": True,
                "summary_records": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
