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
SEEDS=("$@")
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  SEEDS=(42 43)
fi

log() {
  echo "[$(date '+%F %T')] $*"
}

summarize_pair() {
  local stamp="$1"
  /root/miniconda3/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path("outputs")
stamp = os.environ.get("STAMP_OUT", "")
seed_env = os.environ.get("SEEDS_OUT", "").strip()
seeds = [int(part) for part in seed_env.split() if part.strip()] if seed_env else [42, 43]

def score(rec):
    return float(rec.get("AJ_longocc20", 0.0) or 0.0) + float(rec.get("AJ_longocc30", 0.0) or 0.0) - 0.001 * (
        float(rec.get("reapp_error_longocc20_px", 0.0) or 0.0)
        + float(rec.get("reapp_error_longocc30_px", 0.0) or 0.0)
    )

def best_metrics(run_dir: Path):
    path = run_dir / "epoch_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    best = max(rows, key=score)
    return {
        "run": run_dir.name,
        "best_epoch": int(best.get("epoch", -1)),
        "best_score": score(best),
        "AJ": float(best.get("AJ", 0.0) or 0.0),
        "AJ_longocc20": float(best.get("AJ_longocc20", 0.0) or 0.0),
        "AJ_longocc30": float(best.get("AJ_longocc30", 0.0) or 0.0),
        "reapp20": float(best.get("reapp_error_longocc20_px", 0.0) or 0.0),
        "reapp30": float(best.get("reapp_error_longocc30_px", 0.0) or 0.0),
    }

summary = {}
for seed in seeds:
    summary[seed] = {}
    local_dir = root / f"mmp_local_night_full_seed{seed}_{stamp}"
    nocommit_dir = root / f"mmp_localglobal_nocommit_night_full_seed{seed}_{stamp}"
    if local_dir.exists() and (local_dir / "epoch_metrics.jsonl").exists():
        summary[seed]["local"] = best_metrics(local_dir)
    if nocommit_dir.exists() and (nocommit_dir / "epoch_metrics.jsonl").exists():
        summary[seed]["nocommit"] = best_metrics(nocommit_dir)

def mean_score(name: str):
    vals = [summary[s][name]["best_score"] for s in seeds if name in summary[s]]
    return sum(vals) / len(vals) if vals else None

out = {
    "stamp": stamp,
    "summary": summary,
    "local_mean_score": mean_score("local"),
    "nocommit_mean_score": mean_score("nocommit"),
}
path = root / f"full_lockin_local_vs_nocommit_{stamp}.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
}

for seed in "${SEEDS[@]}"; do
  local_out="outputs/mmp_local_night_full_seed${seed}_${STAMP}"
  nocommit_out="outputs/mmp_localglobal_nocommit_night_full_seed${seed}_${STAMP}"

  log "Launching full local seed=${seed} -> ${local_out}"
  bash "$RUNNER" --config projects/mmp_tracker/configs/local_night_full.yaml --seed "$seed" --output "$local_out"

  log "Launching full nocommit seed=${seed} -> ${nocommit_out}"
  bash "$RUNNER" --config projects/mmp_tracker/configs/localglobal_nocommit_night_full.yaml --seed "$seed" --output "$nocommit_out"
done

log "Summarizing lock-in full comparison"
STAMP_OUT="$STAMP" SEEDS_OUT="${SEEDS[*]}" summarize_pair "$STAMP"
log "Full lock-in finished"
