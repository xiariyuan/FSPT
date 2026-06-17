#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python if your environment lives elsewhere." >&2
  exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -u projects/mmp_tracker/train_mmp.py "$@"
