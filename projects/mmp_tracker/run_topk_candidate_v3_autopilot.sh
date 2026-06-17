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

WAIT_PID="${1:-}"
STAMP="$(date '+%Y%m%d_%H%M%S')"

log() {
  echo "[$(date '+%F %T')] $*"
}

wait_for_existing() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    log "Wait PID $pid is not running; continuing immediately"
    return 0
  fi
  log "Waiting for existing PID=$pid to finish"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 300
  done
  log "Existing PID=$pid finished"
}

summarize_and_decide() {
  STAMP_IN="$1" /root/miniconda3/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path("outputs")
stamp = os.environ["STAMP_IN"]
prefix_map = {
    "local": "mmp_local_dev",
    "nocommit": "mmp_localglobal_nocommit_dev",
    "topk_v3": "mmp_localglobal_topk_abstain_v3_dev",
}
seeds = [42, 43]

def score(rec):
    return float(rec.get("AJ_longocc20", 0.0) or 0.0) + float(rec.get("AJ_longocc30", 0.0) or 0.0) - 0.001 * (
        float(rec.get("reapp_error_longocc20_px", 0.0) or 0.0)
        + float(rec.get("reapp_error_longocc30_px", 0.0) or 0.0)
    )

summary = {}
for seed in seeds:
    summary[seed] = {}
    for name, prefix in prefix_map.items():
        run = root / f"{prefix}_seed{seed}_{stamp}"
        path = run / "epoch_metrics.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        best = max(rows, key=score)
        summary[seed][name] = {
            "run": run.name,
            "best_epoch": int(best.get("epoch", -1)),
            "best_score": score(best),
            "AJ": float(best.get("AJ", 0.0) or 0.0),
            "AJ_longocc20": float(best.get("AJ_longocc20", 0.0) or 0.0),
            "AJ_longocc30": float(best.get("AJ_longocc30", 0.0) or 0.0),
            "reapp20": float(best.get("reapp_error_longocc20_px", 0.0) or 0.0),
            "reapp30": float(best.get("reapp_error_longocc30_px", 0.0) or 0.0),
            "candidate_selected_oracle_rate": float(best.get("candidate_selected_oracle_rate", 0.0) or 0.0),
            "candidate_gate_when_global_better": float(best.get("candidate_gate_when_global_better", 0.0) or 0.0),
            "candidate_rank_oracle_rate": float(best.get("candidate_rank_oracle_rate", 0.0) or 0.0),
        }

def mean_score(name):
    vals = [summary[s][name]["best_score"] for s in seeds if name in summary[s]]
    return sum(vals) / len(vals) if vals else None

local_mean = mean_score("local")
nocommit_mean = mean_score("nocommit")
topk_mean = mean_score("topk_v3")
run_topk_full = False
if topk_mean is not None and nocommit_mean is not None and local_mean is not None:
    run_topk_full = topk_mean >= nocommit_mean and topk_mean >= (local_mean - 0.003)

out = {
    "stamp": stamp,
    "summary": summary,
    "local_mean_score": local_mean,
    "nocommit_mean_score": nocommit_mean,
    "topk_v3_mean_score": topk_mean,
    "run_topk_v3_full": run_topk_full,
}
(root / "topk_v3_autopilot_latest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
(root / f"topk_v3_autopilot_decision_{stamp}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
print(f"RUN_TOPK_V3_FULL={1 if run_topk_full else 0}")
PY
}

wait_for_existing "$WAIT_PID"

log "Starting top-k v3 dev matrix"
bash "$SCRIPT_DIR/run_topk_candidate_v3_matrix.sh" 42 43

log "Summarizing v3 dev evidence and deciding full run"
decision_output="$(summarize_and_decide "$STAMP")"
echo "$decision_output"
run_topk_full="$(echo "$decision_output" | awk -F= '/RUN_TOPK_V3_FULL=/{print $2}' | tail -n 1)"

if [[ "$run_topk_full" == "1" ]]; then
  topk_full_out="outputs/mmp_localglobal_topk_abstain_v3_night_full_seed42_${STAMP}"
  log "Launching full top-k v3 mainline -> $topk_full_out"
  bash "$RUNNER" --config projects/mmp_tracker/configs/localglobal_topk_abstain_v3_night_full.yaml --seed 42 --output "$topk_full_out"
else
  log "Dev evidence does not justify v3 full run; stopping after dev matrix"
fi

log "Top-k v3 autopilot finished"
