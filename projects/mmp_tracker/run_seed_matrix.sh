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

launch_pair() {
  local seed="$1"
  local ts="$2"
  local base_out="outputs/mmp_local_dev_seed${seed}_${ts}"
  local relax_out="outputs/mmp_localglobal_top1_relaxedcommit_dev_seed${seed}_${ts}"

  echo "[$(date '+%F %T')] Launching seed ${seed} pair"
  bash "$RUNNER" --config projects/mmp_tracker/configs/local_dev.yaml --seed "$seed" --output "$base_out" > "${base_out}.log" 2>&1 &
  local base_pid=$!
  bash "$RUNNER" --config projects/mmp_tracker/configs/localglobal_top1_relaxedcommit_dev.yaml --seed "$seed" --output "$relax_out" > "${relax_out}.log" 2>&1 &
  local relax_pid=$!
  echo "seed=${seed} base_pid=${base_pid} relax_pid=${relax_pid}"
  wait "$base_pid"
  wait "$relax_pid"
  echo "[$(date '+%F %T')] Seed ${seed} pair finished"
}

ts="$(date '+%Y%m%d_%H%M%S')"
for seed in "$@"; do
  launch_pair "$seed" "$ts"
done
