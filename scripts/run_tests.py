#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)

def main() -> int:
    py = sys.executable
    run([py, "tests/test_pipeline.py"], ROOT / "code/rki_data_pipeline")
    run([py, "-m", "pytest", "-q", "tests/test_core.py"], ROOT / "code/model_pipeline")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
