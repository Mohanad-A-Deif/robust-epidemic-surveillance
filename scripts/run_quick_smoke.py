#!/usr/bin/env python3
"""Run the minimal end-to-end model smoke test."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code" / "model_pipeline"

def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_core.py::test_proposed_model_smoke"]
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=MODEL)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
