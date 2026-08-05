#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${BILLMV2_PYTHON:-python}"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT_DIR"
exec "$PYTHON_BIN" run_ptq.py "$@"
