#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CODE = ROOT / "code" / "rki_data_pipeline"
CONFIG = ROOT / "configs" / "data_config.json"
RAW = ROOT / "data" / "raw" / "IfSG_COVID-19_Erkrankungsbeginn_Erwartungswert.csv"

def run(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Generate only the reference scenario and seed 1101.")
    args = parser.parse_args()
    py = sys.executable
    run([py, str(DATA_CODE / "prepare_rki_dataset.py"), "--config", str(CONFIG), "--raw-file", str(RAW),
         "--raw-dir", str(ROOT / "data/raw"), "--output-dir", str(ROOT / "data/processed"),
         "--metadata-dir", str(ROOT / "data/metadata")])
    scenario_dir = ROOT / ("outputs_quick/data_scenarios" if args.quick else "data/scenarios")
    if args.quick:
        import shutil
        shutil.rmtree(scenario_dir, ignore_errors=True)
    generate = [py, str(DATA_CODE / "generate_scenarios.py"), "--config", str(CONFIG),
                "--processed-dir", str(ROOT / "data/processed"), "--output-dir", str(scenario_dir)]
    if args.quick:
        generate += ["--scenario", "reference_moderate", "--seed", "1101"]
    run(generate)
    run([py, str(DATA_CODE / "validate_outputs.py"), "--config", str(CONFIG),
         "--processed-dir", str(ROOT / "data/processed"), "--scenario-dir", str(scenario_dir)])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
