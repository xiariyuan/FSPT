#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_remote_train.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "Runner not found: $RUNNER" >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  echo "Usage: bash projects/mmp_tracker/run_topk_candidate_v3_matrix.sh <seed> [seed ...]" >&2
  exit 1
fi

cd "$REPO_ROOT"

MMP_PARALLEL="${MMP_PARALLEL:-0}"

launch_one() {
  local config_path="$1"
  local tag="$2"
  local seed="$3"
  local ts="$4"
  local out_dir="outputs/${tag}_seed${seed}_${ts}"
  local log_path="${out_dir}.log"
  echo "[$(date '+%F %T')] Launching ${tag} seed=${seed}"
  if [[ "$MMP_PARALLEL" == "1" ]]; then
    bash "$RUNNER" --config "$config_path" --seed "$seed" --output "$out_dir" > "$log_path" 2>&1 &
    echo "$!"
  else
    bash "$RUNNER" --config "$config_path" --seed "$seed" --output "$out_dir" > "$log_path" 2>&1
  fi
}

ts="$(date '+%Y%m%d_%H%M%S')"
for seed in "$@"; do
  launch_one "projects/mmp_tracker/configs/local_dev.yaml" "mmp_local_dev" "$seed" "$ts"
  launch_one "projects/mmp_tracker/configs/localglobal_nocommit_dev.yaml" "mmp_localglobal_nocommit_dev" "$seed" "$ts"
  launch_one "projects/mmp_tracker/configs/localglobal_topk_abstain_v3_dev.yaml" "mmp_localglobal_topk_abstain_v3_dev" "$seed" "$ts"
done

if [[ "$MMP_PARALLEL" == "1" ]]; then
  wait
fi
