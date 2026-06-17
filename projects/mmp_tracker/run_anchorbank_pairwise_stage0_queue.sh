#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_remote_train.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "Runner not found: $RUNNER" >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p outputs

STAMP="$(date '+%Y%m%d_%H%M%S')"
SMOKE_SEED="${SMOKE_SEED:-42}"
FULL_SEEDS="${FULL_SEEDS:-42}"
STATUS_LOG="${STATUS_LOG:-outputs/anchorbank_stage0_queue.status}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$STATUS_LOG"
}

summarize_run() {
  local run_dir="$1"
  /root/miniconda3/bin/python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
path = run_dir / "epoch_metrics.jsonl"
if not path.exists():
    raise SystemExit(f"Missing metrics file: {path}")

rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def score(rec):
    return float(rec.get("AJ_longocc20", 0.0) or 0.0) + float(rec.get("AJ_longocc30", 0.0) or 0.0) - 0.001 * (
        float(rec.get("reapp_error_longocc20_px", 0.0) or 0.0)
        + float(rec.get("reapp_error_longocc30_px", 0.0) or 0.0)
    )
best = max(rows, key=score)
summary = {
    "run": run_dir.name,
    "best_epoch": int(best.get("epoch", -1)),
    "AJ": float(best.get("AJ", 0.0) or 0.0),
    "AJ_longocc20": float(best.get("AJ_longocc20", 0.0) or 0.0),
    "AJ_longocc30": float(best.get("AJ_longocc30", 0.0) or 0.0),
    "reapp20": float(best.get("reapp_error_longocc20_px", 0.0) or 0.0),
    "reapp30": float(best.get("reapp_error_longocc30_px", 0.0) or 0.0),
    "best_longocc_score": score(best),
}
out_path = run_dir / "best_summary.json"
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY
}

log "Starting AnchorBank Stage-0 queue"

SMOKE_OUT="outputs/mmp_anchorbank_pairwise_coarse_smoke_dev_seed${SMOKE_SEED}_${STAMP}"
log "Launching AnchorBank smoke seed=${SMOKE_SEED} -> ${SMOKE_OUT}"
bash "$RUNNER" \
  --config projects/mmp_tracker/configs/anchorbank_pairwise_coarse_smoke_dev.yaml \
  --seed "$SMOKE_SEED" \
  --output "$SMOKE_OUT"
log "Smoke finished: $(summarize_run "$SMOKE_OUT")"

for seed in $FULL_SEEDS; do
  FULL_OUT="outputs/mmp_anchorbank_pairwise_coarse_night_full_seed${seed}_${STAMP}"
  log "Launching AnchorBank full seed=${seed} -> ${FULL_OUT}"
  bash "$RUNNER" \
    --config projects/mmp_tracker/configs/anchorbank_pairwise_coarse_night_full.yaml \
    --seed "$seed" \
    --output "$FULL_OUT"
  log "Full finished: $(summarize_run "$FULL_OUT")"
done

log "AnchorBank Stage-0 queue finished"

