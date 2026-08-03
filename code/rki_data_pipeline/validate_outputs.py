#!/usr/bin/env python3
"""Validate processed matrices and message-level scenario files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--scenario-dir", type=Path, default=Path("data/scenarios"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw = pd.read_csv(args.processed_dir / "x_raw.csv", index_col=0, parse_dates=True)
    standardized = pd.read_csv(args.processed_dir / "x_standardized.csv", index_col=0, parse_dates=True)
    splits = pd.read_csv(args.processed_dir / "splits.csv", parse_dates=["date"])
    adjacency = pd.read_csv(args.processed_dir / "adjacency_reference.csv", index_col=0)

    assert raw.columns.tolist() == config["selected_states"], "State order mismatch"
    assert standardized.shape == raw.shape, "Transformed matrix shape mismatch"
    assert not raw.isna().any().any(), "Raw matrix contains missing values"
    assert not standardized.isna().any().any(), "Standardized matrix contains missing values"
    assert (raw.to_numpy() >= 0).all(), "Raw epidemic counts contain negative values"
    assert len(splits) == len(raw), "Split file length mismatch"
    assert set(splits["split"]) == {"train", "validation", "test"}, "Split labels incomplete"
    assert adjacency.shape == (len(raw.columns), len(raw.columns)), "Adjacency shape mismatch"
    assert np.array_equal(adjacency.to_numpy(), adjacency.to_numpy().T), "Reference adjacency is not symmetric"
    assert np.all(np.diag(adjacency.to_numpy()) == 0), "Reference adjacency has self loops"

    scenario_files = sorted(args.scenario_dir.glob("**/messages_seed_*.csv.gz"))
    for path in scenario_files:
        messages = pd.read_csv(path)
        required = {
            "message_id", "generation_index", "delay", "dropped", "right_censored", "received",
            "is_outlier", "observed_value", "arrival_index", "state", "seed", "scenario"
        }
        assert required.issubset(messages.columns), f"Missing columns in {path}"
        assert messages["message_id"].is_unique, f"Duplicate message IDs in {path}"
        assert len(messages) == raw.shape[0] * raw.shape[1], f"Unexpected message count in {path}"
        assert ((messages["received"] & messages["dropped"]) == False).all(), f"Dropped messages marked received in {path}"
        assert ((messages["received"] & messages["right_censored"]) == False).all(), f"Censored messages marked received in {path}"
        assert messages.loc[messages["received"], "observed_value"].notna().all(), f"Received values missing in {path}"
        assert messages.loc[~messages["received"], "observed_value"].isna().all(), f"Unreceived values present in {path}"
        received = messages[messages["received"]]
        assert (received["arrival_index"] == received["generation_index"] + received["delay"]).all(), f"Arrival arithmetic mismatch in {path}"

    print(
        json.dumps(
            {
                "status": "ok",
                "processed_shape_dates_by_states": list(raw.shape),
                "scenario_files_checked": len(scenario_files),
                "states": raw.columns.tolist(),
                "date_start": raw.index.min().strftime("%Y-%m-%d"),
                "date_end": raw.index.max().strftime("%Y-%m-%d"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
