#!/usr/bin/env python3
"""Run source preparation, scenario generation, and validation in sequence."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--raw-file", type=Path, default=None)
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--scenario", action="append", default=None)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    python = sys.executable
    prepare = [python, str(root / "prepare_rki_dataset.py"), "--config", str(args.config)]
    if args.raw_file:
        prepare += ["--raw-file", str(args.raw_file)]
    if args.skip_checksum:
        prepare += ["--skip-checksum"]
    run(prepare)

    generate = [python, str(root / "generate_scenarios.py"), "--config", str(args.config)]
    for seed in args.seed or []:
        generate += ["--seed", str(seed)]
    for scenario in args.scenario or []:
        generate += ["--scenario", scenario]
    run(generate)
    run([python, str(root / "validate_outputs.py"), "--config", str(args.config)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
