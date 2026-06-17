#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="/gemini/code/datasets/megadepth"
EXTRACT_ROOT="$DATA_ROOT/extracted"
MEGA1500_ROOT="$EXTRACT_ROOT/megadepth1500"
STATUS_LOG="${STATUS_LOG:-$REPO_ROOT/outputs/megadepth1500_stage0_prep.status}"
PAIR_OUT="${PAIR_OUT:-$REPO_ROOT/data/two_view/megadepth1500_pairs_calibrated.jsonl}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$STATUS_LOG"
}

mkdir -p "$REPO_ROOT/outputs" "$REPO_ROOT/data/two_view" "$EXTRACT_ROOT"

if [[ ! -d "$MEGA1500_ROOT" ]]; then
  log "Extracting megadepth1500.zip"
  /root/miniconda3/bin/python - <<'PY'
import zipfile
from pathlib import Path

zip_path = Path("/gemini/code/datasets/megadepth/megadepth1500.zip")
extract_root = Path("/gemini/code/datasets/megadepth/extracted")
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(extract_root)
print("done")
PY
else
  log "megadepth1500 already extracted"
fi

log "Generating MegaDepth-1500 calibrated pair manifest"
/root/miniconda3/bin/python "$REPO_ROOT/scripts/prepare_megadepth1500_pairs.py" \
  --root "$MEGA1500_ROOT" \
  --output "$PAIR_OUT" \
  --relative-paths

log "MegaDepth-1500 stage0 prep finished"
