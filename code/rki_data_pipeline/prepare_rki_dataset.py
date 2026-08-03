#!/usr/bin/env python3
"""Prepare a reproducible semi-real RKI epidemic trajectory dataset.

The script downloads (or accepts a local copy of) the DOI-pinned RKI StopptCOVID
symptom-onset dataset, verifies its checksum, aggregates age groups to daily state
trajectories, applies leakage-safe transformations, and writes model-ready files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ALIASES = {
    "Baden-Wuerttemberg": "Baden-Württemberg",
    "Baden Wurttemberg": "Baden-Württemberg",
    "North Rhine-Westphalia": "Nordrhein-Westfalen",
    "Lower Saxony": "Niedersachsen",
    "Rhineland-Palatinate": "Rheinland-Pfalz",
    "Bavaria": "Bayern",
    "Hesse": "Hessen",
    "achsen-Anhalt": "Sachsen-Anhalt",
    "Thueringen": "Thüringen",
}


@dataclass(frozen=True)
class SplitBounds:
    train_end: int
    validation_end: int
    total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=None,
        help="Optional local source CSV. If omitted, the DOI-pinned file is downloaded.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--skip-checksum", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324 - used only for source integrity matching
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_retries(url: str, destination: Path, attempts: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "RKI-epidemic-reproducibility-pipeline/1.0"}
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(20, 180), headers=headers) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            temporary.replace(destination)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts: {last_error}")


def obtain_source(config: dict[str, Any], raw_file: Path | None, raw_dir: Path) -> Path:
    source = config["source"]
    destination = raw_dir / source["file_name"]
    if raw_file is not None:
        if not raw_file.exists():
            raise SystemExit(f"Raw source file not found: {raw_file}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        if raw_file.resolve() != destination.resolve():
            shutil.copy2(raw_file, destination)
        else:
            destination = raw_file
    elif not destination.exists():
        urls = [source["download_url"], *source.get("fallback_urls", [])]
        errors: list[str] = []
        for url in urls:
            try:
                print(f"Downloading RKI source from {url} to {destination} ...", file=sys.stderr)
                download_with_retries(url, destination)
                break
            except RuntimeError as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError("All configured source downloads failed:\n" + "\n".join(errors))
    return destination


def normalize_state_name(value: str) -> str:
    cleaned = str(value).strip()
    return ALIASES.get(cleaned, cleaned)


def load_source(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    frame.columns = [str(column).strip() for column in frame.columns]
    if "Altersgrupppe" in frame.columns and "Altersgruppe" not in frame.columns:
        frame = frame.rename(columns={"Altersgrupppe": "Altersgruppe"})

    required = ["Bundesland", "Datum", "Altersgruppe", "EW_Fallzahl"]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Source file is missing required columns: {missing}")

    frame = frame[required].copy()
    frame["Bundesland"] = frame["Bundesland"].map(normalize_state_name)
    frame["Datum"] = pd.to_datetime(frame["Datum"], errors="coerce", format="mixed")
    frame["EW_Fallzahl"] = pd.to_numeric(frame["EW_Fallzahl"], errors="coerce")
    frame["Altersgruppe"] = frame["Altersgruppe"].astype(str).str.strip()

    bad_dates = int(frame["Datum"].isna().sum())
    bad_values = int(frame["EW_Fallzahl"].isna().sum())
    if bad_dates or bad_values:
        raise ValueError(
            f"Invalid source values: {bad_dates} unparsable dates and {bad_values} nonnumeric case values."
        )
    if (frame["EW_Fallzahl"] < 0).any():
        raise ValueError("EW_Fallzahl contains negative values, which are not valid for this pipeline.")
    return frame


def compute_splits(total: int, split_config: dict[str, float]) -> SplitBounds:
    train_fraction = float(split_config["train_fraction"])
    validation_fraction = float(split_config["validation_fraction"])
    test_fraction = float(split_config["test_fraction"])
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Train/validation/test fractions must sum to one.")
    if total < 5:
        # Offline fixture: preserve chronological ordering while keeping all labels represented when possible.
        train_end = max(1, total - 2)
        validation_end = max(train_end + 1, total - 1)
    else:
        train_end = max(1, int(np.floor(total * train_fraction)))
        validation_count = max(1, int(np.floor(total * validation_fraction)))
        validation_end = train_end + validation_count
        if validation_end >= total:
            validation_end = total - 1
    return SplitBounds(train_end=train_end, validation_end=validation_end, total=total)


def build_reference_adjacency(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(config["nodes"]).sort_values("node_id").reset_index(drop=True)
    selected = config["selected_states"]
    nodes = nodes[nodes["state"].isin(selected)].copy()
    if nodes["state"].tolist() != selected:
        nodes = nodes.set_index("state").loc[selected].reset_index()
    codes = nodes["code"].tolist()
    index = {code: idx for idx, code in enumerate(codes)}
    adjacency = np.zeros((len(codes), len(codes)), dtype=np.int8)
    edge_rows: list[dict[str, str]] = []
    for source, target in config.get("undirected_reference_edges", []):
        if source not in index or target not in index:
            continue
        i, j = index[source], index[target]
        adjacency[i, j] = 1
        adjacency[j, i] = 1
        edge_rows.append({"source_code": source, "target_code": target, "edge_type": "shared_border"})
    adjacency_frame = pd.DataFrame(adjacency, index=codes, columns=codes)
    edges = pd.DataFrame(edge_rows, columns=["source_code", "target_code", "edge_type"])
    return nodes, adjacency_frame


def prepare_dataset(config: dict[str, Any], source_path: Path, output_dir: Path, metadata_dir: Path) -> dict[str, Any]:
    source = load_source(source_path)
    selected_states: list[str] = config["selected_states"]
    start = pd.Timestamp(config["date_start"])
    end = pd.Timestamp(config["date_end"])
    population_config = config.get("population", {})
    population_values = population_config.get("values", {})
    missing_population = [state for state in selected_states if state not in population_values]
    if missing_population:
        raise ValueError(f"Population denominators are missing for: {missing_population}")
    population = {state: float(population_values[state]) for state in selected_states}
    if any((not np.isfinite(value) or value <= 0) for value in population.values()):
        raise ValueError("Population denominators must be finite positive values.")

    filtered = source[
        source["Bundesland"].isin(selected_states)
        & source["Datum"].between(start, end, inclusive="both")
    ].copy()
    if filtered.empty:
        raise ValueError("No rows remain after applying state and date filters.")

    observed_states = set(filtered["Bundesland"].unique())
    missing_states = [state for state in selected_states if state not in observed_states]
    if missing_states:
        raise ValueError(f"Selected states absent from source/date range: {missing_states}")

    age_values = sorted(filtered["Altersgruppe"].unique().tolist())
    daily = (
        filtered.groupby(["Datum", "Bundesland"], as_index=False, observed=True)["EW_Fallzahl"]
        .sum()
        .rename(columns={"Datum": "date", "Bundesland": "state", "EW_Fallzahl": "expected_cases"})
    )

    full_dates = pd.date_range(start, end, freq="D")
    expected_index = pd.MultiIndex.from_product([full_dates, selected_states], names=["date", "state"])
    daily = daily.set_index(["date", "state"]).reindex(expected_index)
    missing_cells = int(daily["expected_cases"].isna().sum())
    if missing_cells:
        if config.get("fill_missing_zero", False):
            daily["expected_cases"] = daily["expected_cases"].fillna(0.0)
        else:
            missing_preview = daily[daily["expected_cases"].isna()].head(10).index.tolist()
            raise ValueError(
                f"The filtered source has {missing_cells} missing state-date cells. "
                f"Examples: {missing_preview}. Set fill_missing_zero only if substantively justified."
            )
    daily = daily.reset_index()
    daily["state"] = pd.Categorical(daily["state"], categories=selected_states, ordered=True)
    daily = daily.sort_values(["date", "state"]).reset_index(drop=True)
    daily["population"] = daily["state"].astype(str).map(population).astype(float)
    daily["incidence_per_100k"] = 100000.0 * daily["expected_cases"].astype(float) / daily["population"]

    raw_cases = daily.pivot(index="date", columns="state", values="expected_cases").reindex(columns=selected_states)
    incidence = daily.pivot(index="date", columns="state", values="incidence_per_100k").reindex(columns=selected_states)
    if raw_cases.isna().any().any() or incidence.isna().any().any():
        raise AssertionError("Unexpected missing values after completed-grid validation.")
    log_values = np.log1p(incidence.astype(float))

    bounds = compute_splits(len(raw_cases), config["split"])
    train_log = log_values.iloc[: bounds.train_end]
    means = train_log.mean(axis=0)
    stds = train_log.std(axis=0, ddof=0)
    floor = float(config["transform"].get("standard_deviation_floor", 1e-8))
    stds = stds.mask(stds < floor, 1.0)
    standardized = (log_values - means) / stds

    labels = np.full(len(raw_cases), "test", dtype=object)
    labels[: bounds.train_end] = "train"
    labels[bounds.train_end : bounds.validation_end] = "validation"
    splits = pd.DataFrame({"date": raw_cases.index, "time_index": np.arange(len(raw_cases)), "split": labels})

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    nodes, adjacency = build_reference_adjacency(config)
    nodes["population"] = nodes["state"].map(population).astype(int)
    nodes["population_reference_date"] = str(population_config.get("reference_date", ""))

    daily.to_csv(output_dir / "daily_state_cases_long.csv", index=False, date_format="%Y-%m-%d")
    raw_cases.to_csv(output_dir / "x_raw.csv", date_format="%Y-%m-%d", float_format="%.8f")
    raw_cases.to_csv(output_dir / "x_raw_cases.csv", date_format="%Y-%m-%d", float_format="%.8f")
    incidence.to_csv(output_dir / "x_incidence_per_100k.csv", date_format="%Y-%m-%d", float_format="%.10f")
    log_values.to_csv(output_dir / "x_log1p.csv", date_format="%Y-%m-%d", float_format="%.10f")
    standardized.to_csv(output_dir / "x_standardized.csv", date_format="%Y-%m-%d", float_format="%.10f")
    splits.to_csv(output_dir / "splits.csv", index=False, date_format="%Y-%m-%d")
    nodes.to_csv(output_dir / "nodes.csv", index=False)
    adjacency.to_csv(output_dir / "adjacency_reference.csv")
    nodes.to_csv(metadata_dir / "nodes.csv", index=False)
    adjacency.to_csv(metadata_dir / "adjacency_reference.csv")

    normalization = pd.DataFrame(
        {
            "state": selected_states,
            "train_log_mean": means.reindex(selected_states).to_numpy(),
            "train_log_std": stds.reindex(selected_states).to_numpy(),
        }
    )
    normalization.to_csv(output_dir / "normalization_parameters.csv", index=False, float_format="%.12f")

    np.savez_compressed(
        output_dir / "dataset.npz",
        x_raw=raw_cases.to_numpy(dtype=float).T,
        x_raw_cases=raw_cases.to_numpy(dtype=float).T,
        x_incidence_per_100k=incidence.to_numpy(dtype=float).T,
        x_log1p=log_values.to_numpy(dtype=float).T,
        x_standardized=standardized.to_numpy(dtype=float).T,
        dates=raw_cases.index.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        states=np.asarray(selected_states, dtype="U64"),
        node_codes=nodes["code"].to_numpy(dtype="U4"),
        split_labels=labels.astype("U10"),
        adjacency_reference=adjacency.to_numpy(dtype=np.int8),
    )

    manifest = {
        "dataset_name": config["dataset_name"],
        "source_file": str(source_path),
        "source_md5_observed": md5sum(source_path),
        "source_rows_total": int(len(source)),
        "source_rows_selected": int(len(filtered)),
        "source_age_groups_selected": age_values,
        "selected_states": selected_states,
        "date_start": raw_cases.index.min().strftime("%Y-%m-%d"),
        "date_end": raw_cases.index.max().strftime("%Y-%m-%d"),
        "n_nodes": int(raw_cases.shape[1]),
        "n_time_points": int(raw_cases.shape[0]),
        "matrix_orientation_npz": "nodes_by_time",
        "matrix_orientation_csv": "dates_by_states",
        "missing_state_date_cells_before_policy": missing_cells,
        "split_counts": {key: int(value) for key, value in pd.Series(labels).value_counts().to_dict().items()},
        "latent_state_definition": "log1p of daily symptom-onset expected-case incidence per 100,000 population",
        "transform": "daily expected cases divided by a documented state population denominator, multiplied by 100,000, then log1p; optional per-state z-scoring is fitted on the training interval only",
        "population_reference_date": population_config.get("reference_date"),
        "population_source": population_config.get("source"),
        "population_values": {state: int(population[state]) for state in selected_states},
        "real_data_component": "RKI symptom-onset expected epidemic trajectories",
        "injected_components": ["message missingness", "reporting delay", "outlier contamination"],
        "reference_graph_note": "Geographic adjacency is a structural reference/soft prior, not a known directed ground-truth transmission graph.",
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    source_path = obtain_source(config, args.raw_file, args.raw_dir)
    expected_md5 = str(config["source"].get("md5", "")).lower()
    observed_md5 = md5sum(source_path)
    if expected_md5 and not args.skip_checksum and observed_md5 != expected_md5:
        raise SystemExit(
            "Source checksum mismatch. "
            f"Expected {expected_md5}, observed {observed_md5}. "
            "Use --skip-checksum only for the documented offline fixture or after auditing a new source version."
        )
    manifest = prepare_dataset(config, source_path, args.output_dir, args.metadata_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
