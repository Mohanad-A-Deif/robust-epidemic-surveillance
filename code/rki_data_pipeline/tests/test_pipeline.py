#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
OUT = TESTS / "out"
RAW = TESTS / "raw"
SCENARIOS = TESTS / "scenarios"
METADATA = TESTS / "metadata"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    for directory in (OUT, RAW, SCENARIOS, METADATA):
        shutil.rmtree(directory, ignore_errors=True)

    run(
        [
            sys.executable,
            str(ROOT / "prepare_rki_dataset.py"),
            "--config",
            str(TESTS / "config_fixture.json"),
            "--raw-file",
            str(TESTS / "rki_fixture.csv"),
            "--raw-dir",
            str(RAW),
            "--output-dir",
            str(OUT),
            "--metadata-dir",
            str(METADATA),
            "--skip-checksum",
        ]
    )

    raw = pd.read_csv(OUT / "x_raw.csv", index_col=0)
    incidence = pd.read_csv(OUT / "x_incidence_per_100k.csv", index_col=0)
    expected = np.array(
        [
            [1.124967, 2.675033],
            [3.350000, 2.041630],
            [2.425033, 3.566630],
        ]
    )
    assert raw.shape == (3, 2), raw.shape
    assert np.allclose(raw.to_numpy(), expected, atol=1e-6), raw
    expected_incidence = expected.copy()
    expected_incidence[:, 1] /= 2.0  # fixture population for Bayern is 200,000
    assert np.allclose(incidence.to_numpy(), expected_incidence, atol=1e-6), incidence

    manifest = json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_nodes"] == 2
    assert manifest["n_time_points"] == 3
    assert manifest["split_counts"] == {"train": 1, "validation": 1, "test": 1}

    run(
        [
            sys.executable,
            str(ROOT / "generate_scenarios.py"),
            "--config",
            str(TESTS / "config_fixture.json"),
            "--processed-dir",
            str(OUT),
            "--output-dir",
            str(SCENARIOS),
        ]
    )
    message_file = SCENARIOS / "test" / "fixture_scenario" / "messages_seed_42.csv.gz"
    messages = pd.read_csv(message_file)
    assert len(messages) == 6
    assert messages["message_id"].is_unique
    assert set(messages["state"]) == {"Baden-Württemberg", "Bayern"}
    received = messages[messages["received"]]
    assert (received["arrival_index"] == received["generation_index"] + received["delay"]).all()
    assert {"delay_uniform_template", "drop_uniform_template", "outlier_uniform_template"}.issubset(messages.columns)
    template_files = list((SCENARIOS / "corruption_templates").glob("template_seed_*.npz"))
    assert len(template_files) == 1

    run(
        [
            sys.executable,
            str(ROOT / "validate_outputs.py"),
            "--config",
            str(TESTS / "config_fixture.json"),
            "--processed-dir",
            str(OUT),
            "--scenario-dir",
            str(SCENARIOS),
        ]
    )
    print("All offline fixture tests passed.")
    return 0


def test_offline_pipeline() -> None:
    """Pytest entry point for the complete offline fixture integration test."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
