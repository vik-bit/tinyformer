#!/usr/bin/env bash
# Run scripts/retrieval.py under WSL, where vecdex builds (see vecdex README).
# Usage (from Windows): wsl -d Ubuntu -- bash scripts/wsl_retrieval.sh pretrained scratch
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${VECDEX_PY:-$HOME/.venvs/vecdex/bin/python}"
"$PY" -c "import vecdex, numpy; print('vecdex', vecdex.__version__, vecdex.simd)"
"$PY" scripts/retrieval.py "$@"
