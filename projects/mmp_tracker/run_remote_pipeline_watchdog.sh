#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FSPT}"
cd "$REPO_ROOT"

STATUS_LOG="${STATUS_LOG:-outputs/remote_pipeline_watchdog.status}"
AUTOPILOT_PROJECT="${AUTOPILOT_PROJECT:-projects/mmp_tracker/run_remote_autopilot_after_current_queue.sh}"
AUTOPILOT_OUTPUT="${AUTOPILOT_OUTPUT:-outputs/remote_autopilot_after_current_queue.sh}"
AUTOPILOT_NOHUP="${AUTOPILOT_NOHUP:-outputs/remote_autopilot_after_current_queue.nohup.log}"
AUTOPILOT_STATUS="${AUTOPILOT_STATUS:-outputs/remote_autopilot_after_current_queue.status}"
POST_LOG="${POST_LOG:-outputs/post_lockin_coarse_vs_refine_autopilot.log}"
POLL_SECONDS="${POLL_SECONDS:-300}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$STATUS_LOG"
}

autopilot_running() {
  pgrep -af "bash outputs/remote_autopilot_after_current_queue.sh|bash projects/mmp_tracker/run_remote_autopilot_after_current_queue.sh" >/dev/null 2>&1
}

active_training_or_queue() {
  pgrep -af "python.*train_mmp|run_full_lockin_local_vs_nocommit.sh 44 45|run_full_lockin_local_vs_coarse_vs_refine.sh|run_post_lockin_coarse_vs_refine_queue.sh" >/dev/null 2>&1
}

pipeline_finished() {
  [[ -f "$AUTOPILOT_STATUS" ]] && grep -q "Post-lockin coarse-vs-refine queue finished successfully" "$AUTOPILOT_STATUS"
}

pick_autopilot_script() {
  if [[ -f "$AUTOPILOT_OUTPUT" ]]; then
    echo "$AUTOPILOT_OUTPUT"
  else
    echo "$AUTOPILOT_PROJECT"
  fi
}

restart_autopilot() {
  local script
  script="$(pick_autopilot_script)"
  if [[ ! -f "$script" ]]; then
    log "Autopilot script missing: $script"
    return 1
  fi
  bash -n "$script"
  nohup bash "$script" > "$AUTOPILOT_NOHUP" 2>&1 < /dev/null &
  disown || true
  log "Restarted autopilot using $script"
}

log "Remote pipeline watchdog started"

while true; do
  if pipeline_finished; then
    log "Pipeline already finished; watchdog exiting"
    exit 0
  fi

  if autopilot_running; then
    sleep "$POLL_SECONDS"
    continue
  fi

  if active_training_or_queue; then
    log "Autopilot missing while training/queue is active; restarting autopilot"
    restart_autopilot || true
    sleep "$POLL_SECONDS"
    continue
  fi

  log "No active autopilot and no active training detected; restarting autopilot"
  restart_autopilot || true
  sleep "$POLL_SECONDS"
done
