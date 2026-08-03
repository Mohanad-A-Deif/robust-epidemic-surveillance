#!/usr/bin/env python3
from __future__ import annotations
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.txt"

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024*1024): h.update(chunk)
    return h.hexdigest()

def main() -> int:
    files=[]
    for p in ROOT.rglob("*"):
        if not p.is_file() or p == MANIFEST or ".git" in p.parts or "outputs_quick" in p.parts:
            continue
        files.append(p)
    lines=[f"{digest(p)}  {p.relative_to(ROOT).as_posix()}" for p in sorted(files)]
    MANIFEST.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {MANIFEST}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
