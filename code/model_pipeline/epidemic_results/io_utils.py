"""Input/output utilities and schema validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ProcessedBundle:
    x_raw: pd.DataFrame
    x_log1p: pd.DataFrame
    x_standardized: pd.DataFrame
    nodes: pd.DataFrame
    splits: pd.Series
    adjacency: np.ndarray | None
    metadata: dict[str, Any]


def read_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(data: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output


def read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.apply(pd.to_numeric, errors="coerce")


def load_processed_bundle(processed_dir: str | Path, metadata_dir: str | Path | None = None) -> ProcessedBundle:
    root = Path(processed_dir)
    required = ["x_raw.csv", "x_log1p.csv", "x_standardized.csv", "nodes.csv"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed files in {root}: {missing}")

    x_raw = read_matrix(root / "x_raw.csv")
    x_log = read_matrix(root / "x_log1p.csv")
    x_std = read_matrix(root / "x_standardized.csv")
    nodes = pd.read_csv(root / "nodes.csv")

    if not (x_raw.index.equals(x_log.index) and x_raw.index.equals(x_std.index)):
        raise ValueError("Processed matrices have different date indices.")
    if not (x_raw.columns.tolist() == x_log.columns.tolist() == x_std.columns.tolist()):
        raise ValueError("Processed matrices have different node columns.")

    split_path = root / "splits.csv"
    if split_path.exists():
        split_frame = pd.read_csv(split_path, index_col=0, parse_dates=True)
        if split_frame.shape[1] == 1:
            splits = split_frame.iloc[:, 0].astype(str)
        elif "split" in split_frame.columns:
            splits = split_frame["split"].astype(str)
        else:
            splits = split_frame.iloc[:, 0].astype(str)
    else:
        n = len(x_raw)
        a, b = int(round(0.6 * n)), int(round(0.8 * n))
        labels = np.array(["train"] * a + ["validation"] * (b - a) + ["test"] * (n - b))
        splits = pd.Series(labels, index=x_raw.index, name="split")

    adjacency = None
    for name in ["adjacency_reference.npy", "reference_adjacency.npy", "adjacency.npy"]:
        if (root / name).exists():
            adjacency = np.load(root / name)
            break
    if adjacency is None:
        for name in ["adjacency_reference.csv", "reference_adjacency.csv", "adjacency.csv"]:
            if (root / name).exists():
                adjacency = pd.read_csv(root / name, index_col=0).to_numpy(dtype=float)
                break

    metadata_root = Path(metadata_dir) if metadata_dir else root.parent / "metadata"
    metadata = {}
    for name in ["dataset_manifest.json", "processed_manifest.json", "metadata.json"]:
        candidate = metadata_root / name
        if candidate.exists():
            metadata.update(read_json(candidate, {}) or {})

    return ProcessedBundle(x_raw, x_log, x_std, nodes, splits, adjacency, metadata)


def discover_message_files(scenario_dir: str | Path) -> list[Path]:
    root = Path(scenario_dir)
    return sorted(root.glob("**/messages_seed_*.csv.gz")) + sorted(root.glob("**/messages_seed_*.csv"))


def read_messages(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "scenario",
        "seed",
        "node_id",
        "generation_index",
        "clean_value",
        "delay",
        "received",
        "is_outlier",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Message file {path} is missing columns: {missing}")
    for col in ["received", "dropped", "right_censored", "is_outlier"]:
        if col in frame:
            frame[col] = frame[col].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)
    return frame


def ensure_output_tree(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    paths = {
        "root": base,
        "figures": base / "figures_png",
        "tables_csv": base / "tables_csv",
        "tables_latex": base / "tables_latex",
        "seed_results": base / "seed_level_results",
        "stats": base / "statistical_tests",
        "arrays": base / "prediction_arrays",
        "logs": base / "runtime_logs",
        "eda": base / "eda",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_table(frame: pd.DataFrame, name: str, paths: dict[str, Path], index: bool = False) -> tuple[Path, Path]:
    csv_path = paths["tables_csv"] / f"{name}.csv"
    tex_path = paths["tables_latex"] / f"{name}.tex"
    frame.to_csv(csv_path, index=index)
    latex = frame.to_latex(index=index, escape=False, float_format=lambda x: f"{x:.4g}")
    tex_path.write_text(latex, encoding="utf-8")
    return csv_path, tex_path
