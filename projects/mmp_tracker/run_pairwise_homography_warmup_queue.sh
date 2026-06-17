#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATUS_LOG="${STATUS_LOG:-$REPO_ROOT/outputs/pairwise_homography_warmup_queue.status}"
STAMP="$(date '+%Y%m%d_%H%M%S')"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$STATUS_LOG"
}

run_train() {
  local config="$1"
  local seed="$2"
  local out_dir="$3"
  /root/miniconda3/bin/python -u scripts/train_pairwise_homography_stage0.py \
    --config "$config" \
    --seed "$seed" \
    --output "$out_dir"
}

cd "$REPO_ROOT"
mkdir -p outputs

SMOKE_OUT="outputs/anchorbank_pairwise_homography_warmup_smoke_seed42_${STAMP}"
LONG_OUT="outputs/anchorbank_pairwise_homography_warmup_long_seed42_${STAMP}"

log "Starting homography warm-up smoke"
run_train "projects/mmp_tracker/configs/anchorbank_pairwise_homography_warmup_smoke.yaml" 42 "$SMOKE_OUT"
log "Smoke finished: $SMOKE_OUT"

log "Starting long homography warm-up run"
run_train "projects/mmp_tracker/configs/anchorbank_pairwise_homography_warmup_long.yaml" 42 "$LONG_OUT"
log "Long warm-up finished: $LONG_OUT"
