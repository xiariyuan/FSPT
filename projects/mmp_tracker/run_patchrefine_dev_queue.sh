#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

WAIT_PID="${1:?usage: run_patchrefine_dev_queue.sh <wait_pid> <config> <seed>}"
CONFIG_PATH="${2:?usage: run_patchrefine_dev_queue.sh <wait_pid> <config> <seed>}"
SEED="${3:?usage: run_patchrefine_dev_queue.sh <wait_pid> <config> <seed>}"

cd "$REPO_ROOT"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/mmp_localglobal_nocommit_patchrefine_dev_seed${SEED}_${STAMP}"
LOG_PATH="outputs/patchrefine_queue_.log"

echo "[$(date '+%F %T')] Launching patchrefine seed=${SEED} -> ${OUTPUT_DIR}" >> "$LOG_PATH"
"$PYTHON_BIN" -u projects/mmp_tracker/train_mmp.py \
  --config "$CONFIG_PATH" \
  --seed "$SEED" \
  --output "$OUTPUT_DIR" >> "$LOG_PATH" 2>&1

