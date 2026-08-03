#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
python "$REPO/scripts/reproduce_data.py"
python "$HERE/run_all.py" --data-root "$REPO" --mode all --output "$REPO/outputs/recomputed_full"
