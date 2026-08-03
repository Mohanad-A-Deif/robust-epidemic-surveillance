#!/usr/bin/env python3
"""Run EDA and the complete experiment-results pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from epidemic_results.demo_data import create_demo_dataset
from epidemic_results.eda import run_eda
from epidemic_results.experiment import run_experiments
from epidemic_results.io_utils import discover_message_files, load_processed_bundle, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset package root containing processed/ and scenarios/ or data/processed and data/scenarios.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--mode", choices=["eda", "experiments", "all", "demo"], default="all")
    parser.add_argument("--quick", action="store_true", help="Shorter optimization and fewer baselines for smoke testing.")
    parser.add_argument("--scenario", action="append", default=None, help="Restrict experiments to named scenario(s).")
    parser.add_argument("--demo-days", type=int, default=100)
    return parser.parse_args()


def resolve_layout(root: Path, config_override: Path | None) -> tuple[Path, Path, Path]:
    processed_candidates = [root / "processed", root / "data" / "processed"]
    scenario_candidates = [root / "scenarios", root / "data" / "scenarios"]
    processed = next((path for path in processed_candidates if path.exists()), processed_candidates[0])
    scenarios = next((path for path in scenario_candidates if path.exists()), scenario_candidates[0])
    config = config_override or root / "config.json"
    if not config.exists() and (root / "config" / "config.json").exists():
        config = root / "config" / "config.json"
    return processed, scenarios, config


def main() -> int:
    args = parse_args()
    if args.mode == "demo":
        demo_root = args.output / "demo_dataset"
        layout = create_demo_dataset(demo_root, n_time=args.demo_days)
        data_root = demo_root
        mode = "all"
    else:
        if args.data_root is None:
            raise SystemExit("--data-root is required unless --mode demo is used.")
        data_root = args.data_root
        mode = args.mode

    processed, scenarios, config = resolve_layout(data_root, args.config)
    if not processed.exists():
        raise SystemExit(f"Processed directory not found: {processed}")
    if not config.exists():
        raise SystemExit(f"Configuration file not found: {config}")

    bundle = load_processed_bundle(processed)
    message_files = discover_message_files(scenarios) if scenarios.exists() else []
    output = args.output / "results" if args.mode == "demo" else args.output
    output.mkdir(parents=True, exist_ok=True)
    master = {"data_root": str(data_root), "processed": str(processed), "scenarios": str(scenarios), "config": str(config), "mode": mode, "quick": args.quick}

    if mode in ["eda", "all"]:
        master["eda"] = run_eda(bundle, output, message_files=message_files)
    if mode in ["experiments", "all"]:
        if not message_files:
            raise SystemExit(f"No scenario message files found under {scenarios}")
        master["experiments"] = run_experiments(
            bundle,
            message_files,
            config,
            output,
            quick=args.quick,
            scenario_limit=set(args.scenario) if args.scenario else None,
        )
    write_json(master, output / "master_manifest.json")
    print(json.dumps(master, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
