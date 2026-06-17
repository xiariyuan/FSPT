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
  log "Waiting for existing queue PID=$pid to finish"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  log "Existing queue PID=$pid finished"
}

summarize_and_decide() {
  /root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('outputs')
prefix_map = {
    'local': 'mmp_local_dev',
    'nocommit': 'mmp_localglobal_nocommit_dev',
    'deferred': 'mmp_localglobal_top1_deferredcommit_dev',
}
seeds = [42, 43]

def score(rec):
    return float(rec.get('AJ_longocc20', 0.0) or 0.0) + float(rec.get('AJ_longocc30', 0.0) or 0.0) - 0.001 * (
        float(rec.get('reapp_error_longocc20_px', 0.0) or 0.0) + float(rec.get('reapp_error_longocc30_px', 0.0) or 0.0)
    )

def latest_run(prefix: str, seed: int):
    candidates = [p for p in root.glob(f'{prefix}_seed{seed}_*') if p.is_dir() and (p / 'epoch_metrics.jsonl').exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

summary = {}
for seed in seeds:
    summary[seed] = {}
    for name, prefix in prefix_map.items():
        run = latest_run(prefix, seed)
        if run is None:
            continue
        lines = [json.loads(line) for line in (run / 'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if not lines:
            continue
        best = max(lines, key=score)
        summary[seed][name] = {
            'run': run.name,
            'best_epoch': int(best.get('epoch', -1)),
            'best_score': score(best),
            'AJ': float(best.get('AJ', 0.0) or 0.0),
            'AJ_longocc20': float(best.get('AJ_longocc20', 0.0) or 0.0),
            'AJ_longocc30': float(best.get('AJ_longocc30', 0.0) or 0.0),
            'reapp20': float(best.get('reapp_error_longocc20_px', 0.0) or 0.0),
            'reapp30': float(best.get('reapp_error_longocc30_px', 0.0) or 0.0),
        }

def mean_score(name: str):
    vals = [summary[s][name]['best_score'] for s in seeds if name in summary[s]]
    return sum(vals) / len(vals) if vals else None

local_mean = mean_score('local')
nocommit_mean = mean_score('nocommit')
deferred_mean = mean_score('deferred')
run_deferred_full = False
if deferred_mean is not None and nocommit_mean is not None and local_mean is not None:
    run_deferred_full = deferred_mean >= nocommit_mean and deferred_mean >= (local_mean - 0.003)

out = {
    'summary': summary,
    'local_mean_score': local_mean,
    'nocommit_mean_score': nocommit_mean,
    'deferred_mean_score': deferred_mean,
    'run_deferred_full': run_deferred_full,
}
out_path = root / 'overnight_autopilot_decision.json'
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))
print(f'RUN_DEFERRED_FULL={1 if run_deferred_full else 0}')
PY
}

wait_for_existing "$WAIT_PID"

log "Starting seed43 dev evidence chain"
bash "$SCRIPT_DIR/run_controlled_commit_matrix.sh" 43

log "Summarizing dev evidence and deciding full queue"
decision_output="$(summarize_and_decide)"
echo "$decision_output"
run_deferred_full="$(echo "$decision_output" | awk -F= '/RUN_DEFERRED_FULL=/{print $2}' | tail -n 1)"

local_full_out="outputs/mmp_local_night_full_${STAMP}"
log "Launching full local baseline -> $local_full_out"
bash "$RUNNER" --config projects/mmp_tracker/configs/local_night_full.yaml --seed 42 --output "$local_full_out"

if [[ "$run_deferred_full" == "1" ]]; then
  deferred_full_out="outputs/mmp_localglobal_top1_deferredcommit_night_full_${STAMP}"
  log "Launching full deferred mainline -> $deferred_full_out"
  bash "$RUNNER" --config projects/mmp_tracker/configs/localglobal_top1_deferredcommit_night_full.yaml --seed 42 --output "$deferred_full_out"
else
  log "Dev evidence does not justify deferred full run; stopping after local full baseline"
fi

log "Overnight autopilot finished"
